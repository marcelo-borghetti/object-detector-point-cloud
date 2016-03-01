#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
//#include "logsFile.h"
using namespace std;

class pointCloudMethods
{
private:
	//std::stringstream createStructuredDir(char **argv, int counter);
public:
	void createVoxels(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered);

	pcl::ModelCoefficients::Ptr planeSegmentation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table, char* dir, LogsFile& logFile, bool &recordFlag);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr segmentClusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table, pcl::ModelCoefficients::Ptr coefficients, char* dir, LogsFile& logFile, bool &recordFlag);

	std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> findClusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2, std::vector<pcl::PointIndices> &cluster_indices, char* dir, LogsFile& logFile, bool &recordFlag);
	std::vector<pcl::PointIndices> findClustersIndices(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2);

	vector<float> findDescriptorVFH(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object, ofstream& outFile);
	vector<float> findDescriptorVFHFromSelectedClusterInTheFullPC(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::vector<pcl::PointIndices> cluster_indices, int indexCluster);
	vector<float> findDescriptorOURCVFH(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object, ofstream& outFile);

	void injectNoise(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);
	void applyRotationAndTranslation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, char* dir, ofstream& outFile, LogsFile& logFile);
	void selectCluster(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::vector<pcl::PointIndices> cluster_indices, int indexCluster);

	void convertPointCloudToPNG(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized, std::string file);
	void convertPointCloudToPNG2(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized, std::string file);
	void getSlicesFromPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_slice, double z, double delta);
	void saveAsPNG(std::string path, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);
	void saveAsPNG2(std::string path, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);
	std::string buildPath(char** argv, int view, int idSlice);
};
