cmake_minimum_required(VERSION 2.8.3)
project(neural_pkg)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  genmsg  
  #message_generation
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

find_package(PCL 1.7 REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} )

#set(PCL_BUILD_TYPE Release)

add_message_files(
   FILES
   Descriptor.msg
   XYZObjects.msg
)

add_service_files(
  FILES
  StartGestureRecognition.srv
  GetXYZObjects.srv
  EmptyService.srv
  PostureRequestService.srv
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package()

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

include_directories(include ${catkin_INCLUDE_DIRS})
#include_directories(/opt/ros/groovy/include/pcl-1.6/pcl)
#include_directories(/opt/ros/groovy/include/pcl-1.6)
include_directories(/usr/include/eigen3)

add_executable(get_pc2 src/get_pc2.cpp)
target_link_libraries(get_pc2 ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(get_pc2 mlp_pkg_generate_messages_cpp)

add_executable(segment_cluster src/logsFile.cpp src/pointCloudMethods.cpp src/viewMethods.cpp src/segment_cluster.cpp)
target_link_libraries(segment_cluster ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}) 
add_dependencies(segment_cluster neural_pkg_generate_messages_cpp)

add_executable(get_gestures src/logsFile.cpp src/pointCloudMethods.cpp src/viewMethods.cpp src/get_gestures.cpp)
target_link_libraries(get_gestures ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}) 
#add_dependencies(segment_cluster mlp_pkg_generate_messages_cpp)
add_dependencies(get_gestures neural_pkg_generate_messages_cpp)

add_executable(clientService src/clientService.cpp)
target_link_libraries(clientService ${catkin_LIBRARIES}) 
#add_dependencies(clientService mlp_pkg_generate_messages_cpp)
add_dependencies(clientService neural_pkg_generate_messages_cpp)

