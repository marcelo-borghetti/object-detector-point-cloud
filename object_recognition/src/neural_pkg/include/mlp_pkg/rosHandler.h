#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "ros/ros.h"
#include "std_msgs/String.h"

using namespace std;

class rosHandler
{
private:

public:
	void LabelCallback(const std_msgs::String::ConstPtr& msg);
	//void initNode();
};
