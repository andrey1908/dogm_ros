/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "dogm_ros/dogm_node.h"
#include "dogm_ros/dogm_ros.h"

#include <dogm/dogm.h>
#include <dogm/dogm_types.h>

#include <dogm_msgs/DynamicOccupancyGrid.h>

#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Transform.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "time_measurer.h"

namespace dogm_ros
{

DOGMRos::DOGMRos(ros::NodeHandle nh, ros::NodeHandle private_nh) 
	: nh_(nh), private_nh_(private_nh), tf_buffer_(), tf_listener_(tf_buffer_), is_first_measurement_(true)
{
	private_nh_.param("map/size", params_.size, 50.0f);
	private_nh_.param("map/resolution", params_.resolution, 0.2f);
	private_nh_.param("particles/particle_count", params_.particle_count, 3 * static_cast<int>(10e5));
	private_nh_.param("particles/new_born_particle_count", params_.new_born_particle_count, 3 * static_cast<int>(10e4));
	private_nh_.param("particles/persistence_probability", params_.persistence_prob, 0.99f);
	private_nh_.param("particles/process_noise_position", params_.stddev_process_noise_position, 0.1f);
	private_nh_.param("particles/process_noise_velocity", params_.stddev_process_noise_velocity, 1.0f);
	private_nh_.param("particles/birth_probability", params_.birth_prob, 0.02f);
	private_nh_.param("particles/velocity_persistent", params_.stddev_velocity, 30.0f);
	private_nh_.param("particles/velocity_birth", params_.init_max_velocity, 30.0f);

	private_nh_.param("robot_frame_id", robot_frame_id_, std::string("base_link"));

	private_nh_.param("opencv_visualization", opencv_visualization_, false);
	private_nh_.param("vis_occupancy_threshold", vis_occupancy_threshold_, 0.6f);
	private_nh_.param("vis_mahalanobis_distance", vis_mahalanobis_distance_, 1.0f);
	private_nh_.param("vis_image_size", vis_image_size_, int(400));

	grid_map_.reset(new dogm::DOGM(params_));
	meas_grid_.resize(grid_map_->grid_cell_count);

	subscriber_ = nh_.subscribe("static_map", 1, &DOGMRos::process, this);
	publisher_ = nh_.advertise<dogm_msgs::DynamicOccupancyGrid>("dynamic_map", 1);
}

void DOGMRos::process(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid)
{
	MEASURE_TIME_FROM_HERE(OccupancyGrid2MeasurementGrid);
	projectOccupancyGrid(occupancy_grid);
	STOP_TIME_MESUREMENT(OccupancyGrid2MeasurementGrid);
	
	MEASURE_TIME_FROM_HERE(UpdateDynamicMap);
	ros::Time time_stamp = occupancy_grid->header.stamp;
	if (!is_first_measurement_)
	{
		float dt = (time_stamp - last_time_stamp_).toSec();
		grid_map_->updateGrid(meas_grid_.data(), new_x_, new_y_, 0, dt, false);
	}
	else
	{
		grid_map_->updateGrid(meas_grid_.data(), new_x_, new_y_, 0, 0, false);
		is_first_measurement_ = false;
	}
	last_time_stamp_ = time_stamp;
	STOP_TIME_MESUREMENT(UpdateDynamicMap);
	
	MEASURE_TIME_FROM_HERE(DynamicMap2ROSMessage);
	dogm_msgs::DynamicOccupancyGrid message;
    dogm_ros::DOGMRosConverter::toDOGMMessage(*grid_map_, message, occupancy_grid->header.frame_id);
	STOP_TIME_MESUREMENT(DynamicMap2ROSMessage);
    
	publisher_.publish(message);

	if (opencv_visualization_)
	{
		MEASURE_TIME_FROM_HERE(Visualization);
		cv::Mat occupancy_image = grid_map_->getOccupancyImage();
		grid_map_->drawVelocities(occupancy_image, vis_image_size_, 1., vis_occupancy_threshold_, vis_mahalanobis_distance_);
		cv::namedWindow("occupancy_image", cv::WINDOW_NORMAL);
		cv::imshow("occupancy_image", occupancy_image);
		cv::waitKey(1);
		STOP_TIME_MESUREMENT(Visualization);
	}
}

void DOGMRos::projectOccupancyGrid(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid, float occupancy_threshold /* 0.5 */)
{
	// get transform from map to measurement grid
	geometry_msgs::TransformStamped map_to_robot =
		tf_buffer_.lookupTransform(occupancy_grid->header.frame_id, robot_frame_id_, occupancy_grid->header.stamp, ros::Duration(0.15));
	cv::Mat opencv_map_to_measurement_grid(cv::Mat::eye(cv::Size(3, 3), CV_32F));
	opencv_map_to_measurement_grid.at<float>(0, 2) = map_to_robot.transform.translation.x / params_.resolution - grid_map_->grid_size / 2.;
	opencv_map_to_measurement_grid.at<float>(1, 2) = map_to_robot.transform.translation.y / params_.resolution - grid_map_->grid_size / 2.;
	new_x_ = opencv_map_to_measurement_grid.at<float>(0, 2) * params_.resolution;
	new_y_ = opencv_map_to_measurement_grid.at<float>(1, 2) * params_.resolution;

	// get scale transform from measurement grid to occupancy grid
	cv::Mat scale_measurement_grid_to_occupancy_grid(cv::Mat::eye(cv::Size(3, 3), CV_32F));
	float scale = occupancy_grid->info.resolution / params_.resolution;
	scale_measurement_grid_to_occupancy_grid.at<float>(0, 0) *= scale;
	scale_measurement_grid_to_occupancy_grid.at<float>(1, 1) *= scale;

	// get transform from map to occupancy grid
	Eigen::Quaternionf eigen_map_to_occupancy_grid_rotation_quaternion;
	eigen_map_to_occupancy_grid_rotation_quaternion.x() = occupancy_grid->info.origin.orientation.x;
	eigen_map_to_occupancy_grid_rotation_quaternion.y() = occupancy_grid->info.origin.orientation.y;
	eigen_map_to_occupancy_grid_rotation_quaternion.z() = occupancy_grid->info.origin.orientation.z;
	eigen_map_to_occupancy_grid_rotation_quaternion.w() = occupancy_grid->info.origin.orientation.w;
	Eigen::Matrix3f eigen_map_to_occupancy_grid_rotation = eigen_map_to_occupancy_grid_rotation_quaternion.normalized().toRotationMatrix();
	Eigen::Matrix2f eigen_map_to_occupancy_grid_rotation_2d = eigen_map_to_occupancy_grid_rotation.block<2, 2>(0, 0);
	cv::Mat opencv_map_to_occupancy_grid(cv::Mat::eye(cv::Size(3, 3), CV_32F));
	cv::eigen2cv(eigen_map_to_occupancy_grid_rotation_2d, opencv_map_to_occupancy_grid(cv::Range(0, 2), cv::Range(0, 2)));
	opencv_map_to_occupancy_grid.at<float>(0, 2) = occupancy_grid->info.origin.position.x / occupancy_grid->info.resolution;
	opencv_map_to_occupancy_grid.at<float>(1, 2) = occupancy_grid->info.origin.position.y / occupancy_grid->info.resolution;

	// get transform from measurement grid to occupancy grid
	cv::Mat measurement_grid_to_occupancy_grid =
		opencv_map_to_measurement_grid.inv() * scale_measurement_grid_to_occupancy_grid * opencv_map_to_occupancy_grid;

	// transform occupancy grid to measurement grid system
	std::vector<signed char> occupancy_grid_data(occupancy_grid->data);
	cv::Mat occupancy_grid_host(cv::Size(occupancy_grid->info.width, occupancy_grid->info.height), CV_8S, occupancy_grid_data.data());
	occupancy_grid_host.convertTo(occupancy_grid_host, CV_32S);
	cv::cuda::GpuMat occupancy_grid_device;
	occupancy_grid_device.upload(occupancy_grid_host);
	cv::cuda::warpAffine(occupancy_grid_device, occupancy_grid_device, measurement_grid_to_occupancy_grid(cv::Range(0, 2), cv::Range(0, 3)),
		cv::Size(grid_map_->grid_size, grid_map_->grid_size), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(-1));
	occupancy_grid_device.download(occupancy_grid_host);
	
	const float eps = 0.0001;
	for (int x = 0; x < grid_map_->grid_size; x++)
	{
		for (int y = 0; y < grid_map_->grid_size; y++)
		{
			int meas_idx = x + y * grid_map_->grid_size;
			float occ = occupancy_grid_host.at<int>(y, x) / 100.;
			if (occ < 0)
			{
				meas_grid_[meas_idx].free_mass = eps;
				meas_grid_[meas_idx].occ_mass = eps;
			}
			else if (occ < occupancy_threshold)
			{
				meas_grid_[meas_idx].free_mass = std::max(eps, std::min(1 - eps, 1 - occ));
				meas_grid_[meas_idx].occ_mass = eps;
			}
			else
			{
				meas_grid_[meas_idx].free_mass = eps;
				meas_grid_[meas_idx].occ_mass = std::max(eps, std::min(1 - eps, occ));
			}
			meas_grid_[meas_idx].likelihood = 1.0f;
			meas_grid_[meas_idx].p_A = 1.0f;
		}
	}
}

} // namespace dogm_ros
