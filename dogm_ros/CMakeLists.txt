cmake_minimum_required(VERSION 3.0.2)
project(dogm_ros LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  dogm_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_eigen
)

find_package(dogm REQUIRED)
find_package(time_measurer REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    dogm_msgs
    nav_msgs
    std_msgs
  DEPENDS
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${DOGM_INCLUDE_DIRS}
  ${time_measurer_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a cpp library
add_library(${PROJECT_NAME}
  src/dogm_ros/dogm_ros_converter.cpp
  src/dogm_ros/dogm_node.cu
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${DOGM_LIBRARIES}
  ${time_measurer_LIBRARIES}
  ${OpenCV_LIBS}
)

add_dependencies(${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)

message( ${DOGM_INCLUDE_DIRS} )
message( ${DOGM_LIBRARIES} )

add_library(${PROJECT_NAME}_nodelet src/dogm_ros_nodelet.cpp)
target_link_libraries(${PROJECT_NAME}_nodelet ${PROJECT_NAME} ${DOGM_LIBRARIES} ${catkin_LIBRARIES})

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/dogm_ros_node.cpp)
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${DOGM_LIBRARIES} ${catkin_LIBRARIES})

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
