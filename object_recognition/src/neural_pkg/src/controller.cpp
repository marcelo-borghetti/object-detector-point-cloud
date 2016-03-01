#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Twist.h"
#include "std_srvs/Empty.h"
#include <sstream>

void apply_control(ros::Publisher controller_pub , geometry_msgs::Twist control)
{
    controller_pub.publish(control);
    //float angle = atan(goal_y/goal_x);
}

/*void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
 	nav_msgs::Odometry odom_msg;
 	//msg->
    ROS_INFO("I heard: [%s]", msg->data.c_str());
}*/

void chatterCallback(const nav_msgs::OdometryConstPtr& msg)
{
 	//msg->
    //ROS_INFO("I heard: [%s]", msg->data.c_str());
 	//ROS_INFO("I heard: [%s]", msg);
	ROS_INFO("I heard: [%.2f %.2f %.2f]",msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.orientation.z);
}

int main(int argc, char **argv)
{
	  if ( argc < 2 )
	  {
		  printf("Informe vx, vy e vtheta\n");
		  return 0;
	  }

      ros::init(argc, argv, "controller");
      ros::NodeHandle n;

      ros::Publisher controller_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);
      ros::ServiceClient client = n.serviceClient<std_srvs::Empty>("/body_stiffness/enable");

      ros::Rate loop_rate(10);
      int count = 0;

      std_srvs::Empty srv;
      client.call(srv);

      geometry_msgs::Twist control;
	  ros::Subscriber sub = n.subscribe("/odom", 1000, chatterCallback);

      while (ros::ok())
      {
    	  control.linear.x = atof(argv[1]);
    	  control.linear.y = atof(argv[2]);
    	  control.angular.z = atof(argv[3]);
    	  apply_control(controller_pub, control);
	
    	  ros::spinOnce();

    	  loop_rate.sleep();
    	  ++count;
		  //ros::spin();
     }
     printf("cheguei\n");
     control.linear.x = 0;
     control.linear.y = 0;
     control.angular.z = 0;
     apply_control(controller_pub, control);

     //client = n.serviceClient<std_srvs::Empty>("/body_stiffness/disable");
     //std_srvs::Empty srv;
     //client.call(srv);
     return 0;
}
