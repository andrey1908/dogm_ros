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
#pragma once

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/OccupancyGrid.h>

#include <dogm/dogm.h>
#include <dogm/dogm_types.h>

#include <vector>

namespace dogm_ros
{

class DOGMRos
{
public:
    DOGMRos(ros::NodeHandle nh, ros::NodeHandle private_nh);
    virtual ~DOGMRos();
    void process(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid);

private:
    void occupancyGridToMeasurementGrid(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid, float occupancy_threshold = 0.5);

private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    ros::Subscriber subscriber_;
    ros::Publisher publisher_;

    std::string frame_id_;
    bool opencv_visualization_;
    float vis_occupancy_threshold_;
    float vis_mahalanobis_distance_;
    int vis_image_size_;

    ros::Time last_time_stamp_;
    bool is_first_measurement_;

    boost::shared_ptr<dogm::DOGM> dogm_map_;
    dogm::MeasurementCell* measurement_grid_;
    float new_x_;
    float new_y_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

} // namespace dogm_ros
