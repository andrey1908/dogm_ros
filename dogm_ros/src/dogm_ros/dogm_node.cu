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

__global__ void setUnknownAsFree(cv::cuda::PtrStepSz<signed char> occupancy_grid);
__global__ void fillMeasurementGrid(dogm::MeasurementCell* __restrict__ measurement_grid, const cv::cuda::PtrStepSzi source,
									float occupancy_threshold);

DOGMRos::DOGMRos(ros::NodeHandle nh, ros::NodeHandle private_nh) 
	: nh_(nh), private_nh_(private_nh), tf_buffer_(), tf_listener_(tf_buffer_), is_first_measurement_(true)
{
	private_nh_.param("map/size", params_.size, 50.0f);
	private_nh_.param("map/resolution", params_.resolution, 0.2f);
	private_nh_.param("particles/particle_count", params_.particle_count, 3 * static_cast<int>(1e6));
	private_nh_.param("particles/new_born_particle_count", params_.new_born_particle_count, 3 * static_cast<int>(1e5));
	private_nh_.param("particles/persistence_probability", params_.persistence_prob, 0.99f);
	private_nh_.param("particles/process_noise_position", params_.stddev_process_noise_position, 0.1f);
	private_nh_.param("particles/process_noise_velocity", params_.stddev_process_noise_velocity, 1.0f);
	private_nh_.param("particles/birth_probability", params_.birth_prob, 0.02f);
	private_nh_.param("particles/velocity_persistent", params_.stddev_velocity, 30.0f);
	private_nh_.param("particles/velocity_birth", params_.init_max_velocity, 30.0f);

	private_nh_.param("frame_id", frame_id_, std::string("base_link"));

	private_nh_.param("opencv_visualization", opencv_visualization_, false);
	private_nh_.param("vis_occupancy_threshold", vis_occupancy_threshold_, 0.6f);
	private_nh_.param("vis_mahalanobis_distance", vis_mahalanobis_distance_, 6.0f);
	private_nh_.param("vis_image_size", vis_image_size_, int(400));

	dogm_map_.reset(new dogm::DOGM(params_));
	CHECK_ERROR(cudaMalloc(&measurement_grid_, dogm_map_->grid_cell_count * sizeof(dogm::MeasurementCell)));

	subscriber_ = nh_.subscribe("static_map", 1, &DOGMRos::process, this);
	publisher_ = nh_.advertise<dogm_msgs::DynamicOccupancyGrid>("dynamic_map", 1);
}

DOGMRos::~DOGMRos()
{
	CHECK_ERROR(cudaFree(measurement_grid_));
}

void DOGMRos::process(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid)
{
	MEASURE_TIME_FROM_HERE(OccupancyGrid2MeasurementGrid);
	occupancyGridToMeasurementGrid(occupancy_grid);
	STOP_TIME_MEASUREMENT(OccupancyGrid2MeasurementGrid);
	
	MEASURE_TIME_FROM_HERE(UpdateDynamicMap);
	ros::Time time_stamp = occupancy_grid->header.stamp;
	if (!is_first_measurement_)
	{
		float dt = (time_stamp - last_time_stamp_).toSec();
		dogm_map_->updateGrid(measurement_grid_, new_x_, new_y_, 0, dt);
	}
	else
	{
		dogm_map_->updateGrid(measurement_grid_, new_x_, new_y_, 0, 0);
		is_first_measurement_ = false;
	}
	last_time_stamp_ = time_stamp;
	STOP_TIME_MEASUREMENT(UpdateDynamicMap);
	
	MEASURE_TIME_FROM_HERE(DynamicMap2ROSMessage);
	dogm_msgs::DynamicOccupancyGrid message;
    dogm_ros::DOGMRosConverter::toDOGMMessage(*dogm_map_, message, occupancy_grid->header.frame_id);
	STOP_TIME_MEASUREMENT(DynamicMap2ROSMessage);
    
	publisher_.publish(message);

	if (opencv_visualization_)
	{
		MEASURE_TIME_FROM_HERE(Visualization);
		cv::Mat occupancy_image = dogm_map_->getOccupancyImage();
		dogm_map_->drawVelocities(occupancy_image, vis_image_size_, 1., vis_occupancy_threshold_, vis_mahalanobis_distance_);
		cv::namedWindow("occupancy_image", cv::WINDOW_NORMAL);
		cv::imshow("occupancy_image", occupancy_image);
		cv::waitKey(1);
		STOP_TIME_MEASUREMENT(Visualization);
	}
}

void DOGMRos::occupancyGridToMeasurementGrid(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid, float occupancy_threshold /* 0.5 */)
{
	geometry_msgs::TransformStamped odom_to_robot =
		tf_buffer_.lookupTransform(occupancy_grid->header.frame_id, frame_id_, occupancy_grid->header.stamp, ros::Duration(0.15));
	cv::Mat odom_to_measurement_grid(cv::Mat::eye(cv::Size(3, 3), CV_32F));
	odom_to_measurement_grid.at<float>(0, 2) = odom_to_robot.transform.translation.x / params_.resolution - dogm_map_->grid_size / 2.;
	odom_to_measurement_grid.at<float>(1, 2) = odom_to_robot.transform.translation.y / params_.resolution - dogm_map_->grid_size / 2.;
	new_x_ = odom_to_measurement_grid.at<float>(0, 2) * params_.resolution;
	new_y_ = odom_to_measurement_grid.at<float>(1, 2) * params_.resolution;

	cv::Mat scale_measurement_grid(cv::Mat::eye(cv::Size(3, 3), CV_32F));
	float scale = occupancy_grid->info.resolution / params_.resolution;
	scale_measurement_grid.at<float>(0, 0) *= scale;
	scale_measurement_grid.at<float>(1, 1) *= scale;

	// strange bug: eigen matrices here should have different types (float or double) with
	// eigen matrices in dogm.cu (dogm repository) in function drawVelocities(), otherwise
	// the program will crash
	Eigen::Quaternionf eigen_odom_to_occupancy_grid_quaternion;
	eigen_odom_to_occupancy_grid_quaternion.x() = occupancy_grid->info.origin.orientation.x;
	eigen_odom_to_occupancy_grid_quaternion.y() = occupancy_grid->info.origin.orientation.y;
	eigen_odom_to_occupancy_grid_quaternion.z() = occupancy_grid->info.origin.orientation.z;
	eigen_odom_to_occupancy_grid_quaternion.w() = occupancy_grid->info.origin.orientation.w;
	Eigen::Matrix3f eigen_odom_to_occupancy_grid_rotation = eigen_odom_to_occupancy_grid_quaternion.normalized().toRotationMatrix();
	Eigen::Matrix2f eigen_odom_to_occupancy_grid_rotation_2d = eigen_odom_to_occupancy_grid_rotation.block<2, 2>(0, 0);
	cv::Mat odom_to_occupancy_grid(cv::Mat::eye(cv::Size(3, 3), CV_32F));
	cv::eigen2cv(eigen_odom_to_occupancy_grid_rotation_2d, odom_to_occupancy_grid(cv::Range(0, 2), cv::Range(0, 2)));
	odom_to_occupancy_grid.at<float>(0, 2) = occupancy_grid->info.origin.position.x / occupancy_grid->info.resolution;
	odom_to_occupancy_grid.at<float>(1, 2) = occupancy_grid->info.origin.position.y / occupancy_grid->info.resolution;

	cv::Mat measurement_grid_to_occupancy_grid = odom_to_measurement_grid.inv() * scale_measurement_grid * odom_to_occupancy_grid;

	dim3 blocks(1, 1);
	dim3 threads(16, 16);
	std::vector<signed char> occupancy_grid_data(occupancy_grid->data);
	cv::Mat occupancy_grid_host(cv::Size(occupancy_grid->info.width, occupancy_grid->info.height), CV_8S, occupancy_grid_data.data());
	cv::cuda::GpuMat occupancy_grid_device;
	occupancy_grid_device.upload(occupancy_grid_host);
	setUnknownAsFree<<<blocks, threads>>>(occupancy_grid_device);
	occupancy_grid_device.convertTo(occupancy_grid_device, CV_32S);

	cv::Mat measurement_grid;
    cv::cuda::GpuMat measurement_grid_device;
	cv::cuda::warpAffine(occupancy_grid_device, measurement_grid_device, measurement_grid_to_occupancy_grid(cv::Range(0, 2), cv::Range(0, 3)),
		cv::Size(dogm_map_->grid_size, dogm_map_->grid_size), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
	fillMeasurementGrid<<<blocks, threads>>>(measurement_grid_, measurement_grid_device, occupancy_threshold);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
}

__global__ void setUnknownAsFree(cv::cuda::PtrStepSz<signed char> occupancy_grid)
{
	int start_row = blockIdx.y * blockDim.y + threadIdx.y;
	int start_col = blockIdx.x * blockDim.x + threadIdx.x;
	int step_row = blockDim.y * gridDim.y;
	int step_col = blockDim.x * gridDim.x;
	for (int row = start_row; row < occupancy_grid.rows; row += step_row)
	{
		for (int col = start_col; col < occupancy_grid.cols; col += step_col)
		{
			if (occupancy_grid(row, col) < 0)
			{
				occupancy_grid(row, col) = 0;
			}
		}
	}
}

__device__ float clip(float x, float min, float max)
{
	assert(min <= max);
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

__global__ void fillMeasurementGrid(dogm::MeasurementCell* __restrict__ measurement_grid, const cv::cuda::PtrStepSzi source,
									float occupancy_threshold)
{
	int start_row = blockIdx.y * blockDim.y + threadIdx.y;
	int start_col = blockIdx.x * blockDim.x + threadIdx.x;
	int step_row = blockDim.y * gridDim.y;
	int step_col = blockDim.x * gridDim.x;
	const float eps = 0.0001f;
	for (int row = start_row; row < source.rows; row += step_row)
	{
		for (int col = start_col; col < source.cols; col += step_col)
		{
			int index = col + row * source.cols;
			float occ = source(row, col) / 100.f;
			if (occ < occupancy_threshold)
			{
				measurement_grid[index].free_mass = clip(1 - occ, eps, 1 - eps);
				measurement_grid[index].occ_mass = eps;
			}
			else
			{
				measurement_grid[index].free_mass = eps;
				measurement_grid[index].occ_mass = clip(occ, eps, 1 - eps);
			}
			measurement_grid[index].likelihood = 1.0f;
			measurement_grid[index].p_A = 1.0f;
		}
	}
}

} // namespace dogm_ros
