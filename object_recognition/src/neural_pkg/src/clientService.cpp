#include "ros/ros.h"
#include "neural_pkg/EmptyService.h"
#include <cstdlib>
#include "std_msgs/String.h"
#include "neural_pkg/XYZObjects.h"

void writePoints(const neural_pkg::XYZObjects::ConstPtr& msg)
{
    /*printf("[%d (%.2f, %.2f, %.2f)]\n", msg->id[0], msg->x[0], msg->y[0], msg->z[0]);
    printf("[%d (%.2f, %.2f, %.2f)]\n", msg->id[1], msg->x[1], msg->y[1], msg->z[1]);
    printf("[%d (%.2f, %.2f, %.2f)]\n", msg->id[2], msg->x[2], msg->y[2], msg->z[2]);
    printf("\n");*/
    printf("[Banana (%.2f, %.2f, %.2f)]\n", msg->x[0], msg->y[0], msg->z[0]);
    printf("[Box (%.2f, %.2f, %.2f)]\n", msg->x[1], msg->y[1], msg->z[1]);
    printf("[Cup (%.2f, %.2f, %.2f)]\n", msg->x[2], msg->y[2], msg->z[2]);
    printf("\n");
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "TestStartGestureRecognition");
  
    ros::NodeHandle n;
    
    ros::Publisher  pub = n.advertise<std_msgs::String>("/StartObjectsRecog", 1000);
    ros::Subscriber sub = n.subscribe("/ObjectPoints", 1000, writePoints);
    
    std_msgs::String msg;
    msg.data = "Go";
    int count = 0;
    
    while (ros::ok())
    {  
        if ( count == 0 )
	{
	   pub.publish(msg);
	}
	//count++;
        ros::spinOnce();
    }
     
    /*ros::NodeHandle n;
    ros::ServiceClient client = n.serviceClient<neural_pkg::EmptyService>("StartGestureRecognition");
    neural_pkg::EmptyService srv;
    if (client.call(srv))
    {
      //ROS_INFO("Sum: %ld", (long int)srv.response.x[0]);
      printf("Sucess to call service StartGestureRecognition!\n");
      
    }
    else
    {
      ROS_ERROR("Failed to call service StartGestureRecognition!\n");
      return 1;
    }*/

    return 0;	
}