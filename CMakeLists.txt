cmake_minimum_required(VERSION 2.8.3)
project(lsd_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Lines.msg
  Line.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
catkin_package(
 CATKIN_DEPENDS rospy std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
