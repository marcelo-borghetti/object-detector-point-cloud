#include "ros/ros.h"
#include "std_msgs/String.h"
#include "../include/mlp_pkg/rosHandler.h"
/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
void LabelCallback(const std_msgs::String::ConstPtr& msg)
{
  ROS_INFO("I heard: [%s]", msg->data.c_str());
  //cout << "I heard" << msg->data.c_str()  << "\n";
}

/*
void rosHandler::initNode()
{
  int argc;
  char** argv;
  ros::init(argc, argv, "C_node");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("Label", 1000, LabelCallback);

  *
   * ros::spin() will enter a loop, pumping callbacks.  With this version, all
   * callbacks will be called from within this thread (the main one).  ros::spin()
   * will exit when Ctrl-C is pressed, or the node is shutdown by the master.

  //ros::spin();
}
*/
