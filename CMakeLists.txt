cmake_minimum_required(VERSION 3.8)
project(lidar_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Include directories
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /opt/ros/humble/include  # Add the ROS2 include directory explicitly
)

# Add the DBSCAN library
add_library(dbscan_lib include/dbscan/dbscan.cpp)
ament_target_dependencies(dbscan_lib rclcpp sensor_msgs pcl_conversions PCL)

# Add the DBSCAN node
add_executable(dbscan_node src/dbscan_node.cpp)
ament_target_dependencies(dbscan_node rclcpp sensor_msgs pcl_conversions PCL visualization_msgs)
target_link_libraries(dbscan_node dbscan_lib)

# Add the DBSCAN detection node
add_executable(dbscan_detection_node src/dbscan_detection_node.cpp)
ament_target_dependencies(dbscan_detection_node rclcpp sensor_msgs pcl_conversions PCL visualization_msgs)
target_link_libraries(dbscan_detection_node dbscan_lib)

add_executable(radar_detection_node src/dbscan_detection_node.cpp)
ament_target_dependencies(radar_detection_node rclcpp sensor_msgs pcl_conversions PCL visualization_msgs)
target_link_libraries(radar_detection_node dbscan_lib)

# Install targets
install(TARGETS
  dbscan_node
  dbscan_lib
  dbscan_detection_node
  radar_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
