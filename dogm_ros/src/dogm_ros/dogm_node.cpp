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

#include "time_measurer.h"

namespace dogm_ros
{

DOGMRos::DOGMRos(ros::NodeHandle nh, ros::NodeHandle private_nh) 
	: nh_(nh), private_nh_(private_nh), tf_buffer_(), tf_listener_(tf_buffer_)
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

	private_nh_.param("laser/fov", laser_params_.fov, 120.0f);
	private_nh_.param("laser/max_range", laser_params_.max_range, 50.0f);
	laser_params_.resolution = params_.resolution;  // TODO make independent of grid_params.resolution

	length_in_cells_ = static_cast<int>(params_.size / params_.resolution);
	int num_grid_cells = length_in_cells_ * length_in_cells_;
	meas_grid_.resize(num_grid_cells);
	
	grid_map_.reset(new dogm::DOGM(params_));

	grid_generator_.reset(new LaserMeasurementGrid(laser_params_, params_.size, params_.resolution));
	
	is_first_measurement_ = true;
	
	// subscriber_ = nh_.subscribe("velodyne/scan", 1, &DOGMRos::process, this);
	subscriber_ = nh_.subscribe("grid_map", 1, &DOGMRos::process, this);
	publisher_ = nh_.advertise<dogm_msgs::DynamicOccupancyGrid>("dogm/map", 1);
}

void DOGMRos::process(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid)
{
	float time_stamp = occupancy_grid->header.stamp.toSec();

	MEASURE_TIME_FROM_HERE(OccupancyGrid2StaticMap);
	geometry_msgs::TransformStamped robot_pose;
    try {
    	robot_pose = tf_buffer_.lookupTransform(occupancy_grid->header.frame_id, "base_link", ros::Time(0), ros::Duration(0.2));
    }
    catch (tf2::TransformException &ex) {
		ROS_WARN("%s", ex.what());
		exit(0);
    }

	Eigen::Isometry3d eigen_robot_pose = tf2::transformToEigen(robot_pose);

	geometry_msgs::Transform trans;
	trans.rotation = occupancy_grid->info.origin.orientation;
	trans.translation.x = occupancy_grid->info.origin.position.x;
	trans.translation.y = occupancy_grid->info.origin.position.y;
	trans.translation.z = occupancy_grid->info.origin.position.z;
	Eigen::Isometry3d eigen_grid_pose = tf2::transformToEigen(trans);

	Eigen::Isometry3d grid_to_robot = eigen_grid_pose.inverse() * eigen_robot_pose;
	
	for (int x = 0; x < length_in_cells_; x++) {
		for (int y = 0; y < length_in_cells_; y++) {
			double robot_x = x - length_in_cells_ / 2. + 0.5;
			double robot_y = y - length_in_cells_ / 2. + 0.5;
			robot_x *= params_.resolution;
			robot_y *= params_.resolution;
			Eigen::Vector3d robot_coord = {robot_x, robot_y, 0};
			Eigen::Vector3d grid_coord = grid_to_robot * robot_coord;
			int grid_x = static_cast<int>(grid_coord(0) / occupancy_grid->info.resolution);
			int grid_y = static_cast<int>(grid_coord(1) / occupancy_grid->info.resolution);
			int meas_idx = x + y * length_in_cells_;
			if (grid_x < 0 || grid_y < 0 || grid_x >= occupancy_grid->info.width || grid_y >= occupancy_grid->info.height) {
				meas_grid_[meas_idx].free_mass = 0.00001;
				meas_grid_[meas_idx].occ_mass = 0.00001;
				continue;
			}
			int idx = grid_x + grid_y * occupancy_grid->info.width;
			int8_t occ = occupancy_grid->data[idx];
			if (occ == -1) {
				meas_grid_[meas_idx].free_mass = 0.00001;
				meas_grid_[meas_idx].occ_mass = 0.00001;
			} else if (occ < 50) {
				meas_grid_[meas_idx].free_mass = 1 - occ / 100.;
				meas_grid_[meas_idx].occ_mass = 0.00001;
			} else {
				meas_grid_[meas_idx].free_mass = 0.00001;
				meas_grid_[meas_idx].occ_mass = occ / 100;
			}
			meas_grid_[meas_idx].likelihood = 1.0f;
			meas_grid_[meas_idx].p_A = 1.0f;
		}
	}
	STOP_TIME_MESUREMENT(OccupancyGrid2StaticMap);
	
	MEASURE_TIME_FROM_HERE(UpdateDynamicMap);
	if (!is_first_measurement_)
	{
		float dt = time_stamp - last_time_stamp_;
		grid_map_->updateGrid(meas_grid_.data(), -params_.size / 2, -params_.size / 2, 0, dt, false);
	}
	else
	{
		grid_map_->updateGrid(meas_grid_.data(), -params_.size / 2, -params_.size / 2, 0, 0, false);
		is_first_measurement_ = false;
	}
	STOP_TIME_MESUREMENT(UpdateDynamicMap);
	
	MEASURE_TIME_FROM_HERE(DynamicMap2ROSMessage);
	dogm_msgs::DynamicOccupancyGrid message;
    dogm_ros::DOGMRosConverter::toDOGMMessage(*grid_map_, message);
	STOP_TIME_MESUREMENT(DynamicMap2ROSMessage);
    
	publisher_.publish(message);
	
	last_time_stamp_ = time_stamp;
}

/*
void DOGMRos::process(const sensor_msgs::LaserScan::ConstPtr& scan)
{
	float time_stamp = scan->header.stamp.toSec();
	
	MEASURE_TIME_FROM_HERE(LaserScan2StaticMap);
	dogm::MeasurementCell* meas_grid = grid_generator_->generateGrid(std::vector<float>(scan->ranges.data(),
			scan->ranges.data() + scan->ranges.size()));
	STOP_TIME_MESUREMENT(LaserScan2StaticMap);
	
	MEASURE_TIME_FROM_HERE(UpdateDynamicMap);
	if (!is_first_measurement_)
	{
		float dt = time_stamp - last_time_stamp_;
		grid_map_->updateGrid(meas_grid, -10, -10, 0, dt);
	}
	else
	{
		grid_map_->updateGrid(meas_grid, -10, -10, 0, 0);
		is_first_measurement_ = false;
	}
	STOP_TIME_MESUREMENT(UpdateDynamicMap);
	
	MEASURE_TIME_FROM_HERE(DynamicMap2ROSMessage);
	dogm_msgs::DynamicOccupancyGrid message;
    dogm_ros::DOGMRosConverter::toDOGMMessage(*grid_map_, message);
	STOP_TIME_MESUREMENT(DynamicMap2ROSMessage);
    
	publisher_.publish(message);
	
	last_time_stamp_ = time_stamp;
}
*/

} // namespace dogm_ros
