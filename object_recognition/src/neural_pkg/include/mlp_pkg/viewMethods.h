#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
using namespace pcl;

class viewMethods
{
private:

public:
	void segmentClustersView(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table, pcl::ModelCoefficients::Ptr coefficients);
	pcl::ModelCoefficients::Ptr planeSegmentationView(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table);
	//void grabberCallback(const PointCloud<PointXYZRGBA>::ConstPtr& cloud, char* directory, ofstream* outFile, LogsFile* logfile, ros::Publisher pub);
	//void image_cb_ (const boost::shared_ptr<openni_wrapper::Image> &image, char* directory);
};
