#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni_grabber.h>
//#include <pcl/io/png_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/vfh.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/eigen.h>
#include <pcl/console/parse.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32MultiArray.h"
//#include "../include/mlp_pkg/rosHandler.h"
#include "neural_pkg/GetXYZObjects.h"
#include "neural_pkg/XYZObjects.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/mlp_pkg/logsFile.h"
#include "../include/mlp_pkg/pointCloudMethods.h"
//#include "../include/mlp_pkg/viewMethods.h"

#include<sys/socket.h>
#include<arpa/inet.h> //inet_addr
#include<unistd.h>    //write
#include<pthread.h> //for threading , link with lpthread

//#include <thread>
//#include <mutex>
/*
#include <pcl/io/io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/pcd_io.h>
*/

using namespace pcl;
struct termios orig_termios;
 
LogsFile logfile;
pointCloudMethods pcMethods;
//viewMethods vMethods;

PointCloud<PointXYZRGBA>::Ptr cloudptr(new PointCloud<PointXYZRGBA>); // A cloud that will store color info.
PointCloud<PointXYZ>::Ptr fallbackCloud(new PointCloud<PointXYZ>);    // A fallback cloud with just depth data.
boost::shared_ptr<visualization::CloudViewer> viewer;                 // Point cloud viewer object.
boost::shared_ptr<visualization::ImageViewer> image_viewer;
Grabber* OpenniGrabber;                                               // OpenNI grabber that takes data from the device.
unsigned int cloudFilesSaved = 0;                                          // For the numbering of the clouds saved to disk.
unsigned int imageFilesSaved = 0;                                          // For the numbering of the clouds saved to disk.
bool saveCloud(false), saveImage(false), noColor(false), segmentFlag(false), recordFlag(false), testFlag(false), clusterFlag(false), writingViewsFlag(false), runningFlag(false);                                // Program control.
bool noLabel(true);
int indexCluster = 0, writeFlag = 0;
int numClusters = 0;
std::vector<pcl::PointIndices> cluster_indices;
std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> cluster_list, cluster_list_before_red_paint;
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr globalSegmented ( new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::PointXYZRGB pointText, pointText_to_service;
std::string msgText;
std::ostringstream lastMsg;
std_msgs::String idObject;
neural_pkg::XYZObjects msg_pos;

bool startFlag = false;
boost::mutex g_pages_mutex;
boost::mutex text_mutex;
boost::mutex remove_text_mutex;

//the thread function
void *connection_handler(void *);

void reset_terminal_mode()
{
    tcsetattr(0, TCSANOW, &orig_termios);
}

void set_conio_terminal_mode()
{
    struct termios new_termios;

    /* take two copies - one for now, one for later */
    tcgetattr(0, &orig_termios);
    memcpy(&new_termios, &orig_termios, sizeof(new_termios));

    /* register cleanup handler, and set the new terminal mode */
    atexit(reset_terminal_mode);
    cfmakeraw(&new_termios);
    tcsetattr(0, TCSANOW, &new_termios);
}

void startObjectsRecog(const std_msgs::String::ConstPtr& msg)
{
	startFlag = true;	
	
	testFlag = true;
	
	reset_terminal_mode();
	//printf("Signal Received\n");
	cout << "message:" << msg->data << "\n";
	set_conio_terminal_mode();	
}

bool GetXYZObjects(neural_pkg::GetXYZObjects::Request  &req,
         	 	   neural_pkg::GetXYZObjects::Response &res)
{
  startFlag = true;	

  testFlag = true;

  //res.x.push_back(3);
  
  return true;
}



void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    //viewer.setBackgroundColor(0.7, 0.7, 0.7);
	viewer.setBackgroundColor(0, 0, 0);
}

void addText(pcl::visualization::PCLVisualizer& viewer)
{
	//viewer.addText3D("Teste", point);
	lastMsg << msgText;
	reset_terminal_mode();
	viewer.addText3D (msgText, pointText, 0.05, 1.0, 1.0, 0.0, lastMsg.str());
	set_conio_terminal_mode();
	text_mutex.unlock();
}

void removeText(pcl::visualization::PCLVisualizer& viewer)
{
	/*//viewer.addText3D("Teste", point);
	lastMsg << msgText;
	reset_terminal_mode();
	viewer.removeText3D (lastMsg.str());
	set_conio_terminal_mode();
	remove_text_mutex.unlock();*/
	//viewer.removeAllPointClouds();
	viewer.removeAllShapes();
}

int kbhit()
{
    struct timeval tv = { 0L, 0L };
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    return select(1, &fds, NULL, NULL, &tv);
}

int getch()
{
    int r;
    unsigned char c;
    if ((r = read(0, &c, sizeof(c))) < 0) {
        return r;
    } else {
        return c;
    }
}

void printUsage(const char* programName)
{
	reset_terminal_mode();
	cout << "Usage: " << programName << " [options]"
		 << endl
		 << endl
		 << "Options:\n"
		 << endl
		 << "\t-off <-mlp/cnn/gng> <DATABASE>				Process a database to format the input to the selected neural approach.\n"
		 << "\t-on  <-mlp/cnn/gng> <LABEL> <OUTPUT_DIR>		start capturing from an OpenNI device and test using the selected neural approach.\n"
		 << "\t-on -v FILE						visualize the given .pcd file.\n"
		 << "\t-h							shows this help.\n";
	set_conio_terminal_mode();
}

int count_files(std::string directory, std::string ext)
{
	namespace fs = boost::filesystem;
	fs::path Path(directory);
	int Nb_ext = 0;
	fs::directory_iterator end_iter; // Default constructor for an iterator is the end iterator

	for (fs::directory_iterator iter(Path); iter != end_iter; ++iter)
		if (iter->path().extension() == ext)
			++Nb_ext;

	return Nb_ext;
}

void image_cb_ (const boost::shared_ptr<openni_wrapper::Image> &image, char* directory)
 {
         unsigned int height = image->getHeight();
         unsigned int width = image->getWidth();
         cv::Mat frameBGR=cv::Mat(image->getHeight(),image->getWidth(),CV_8UC3);
         cv::Mat frame =cv::Mat(image->getHeight(),image->getWidth(),CV_8UC3);

         image->fillRGB(frameBGR.cols,frameBGR.rows,frameBGR.data,frameBGR.step);

         vector<int> compression_params;
         compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
         compression_params.push_back(9);
         //-- 3. Apply the classifier to the frame
         if( !frameBGR.empty() )
         {
                 unsigned char *rgb_buffer;
                 rgb_buffer = (unsigned char *) malloc(sizeof (char)*(width*height*3));
                 for(int j=0; j<height; j++)
                 {
                         for(int i=0; i<width; i++)
                         {
                                 rgb_buffer[(j*width + i)*3+0] = frameBGR.at<cv::Vec3b>(j,i)[0];  // B
                                 rgb_buffer[(j*width + i)*3+1] = frameBGR.at<cv::Vec3b>(j,i)[1];  // G
                                 rgb_buffer[(j*width + i)*3+2] = frameBGR.at<cv::Vec3b>(j,i)[2];  // R
                                 //std::cout << (j*width + i)*3+0 << "," << (j*width + i)*3+1 << "," << (j*width + i)*3+2 << "," << std::endl;

                                 frame.at<cv::Vec3b>(j,i)[2] = frameBGR.at<cv::Vec3b>(j,i)[0];  // B
                                 frame.at<cv::Vec3b>(j,i)[1] = frameBGR.at<cv::Vec3b>(j,i)[1];  // G
                                 frame.at<cv::Vec3b>(j,i)[0] = frameBGR.at<cv::Vec3b>(j,i)[2];  // R
                         }
                 }

                 //if(!image_viewer->wasStopped())
                 //        image_viewer->showRGBImage(rgb_buffer,width,height);

                 free(rgb_buffer);
              	 if (saveImage)
             	 {
              		 std::stringstream subdir;
              		 subdir << "/new_trainning_data/";
              		 std::stringstream ss;
              		 ss << directory << subdir.str();
					 imageFilesSaved = count_files(ss.str(), ".jpg");
            		 ss << "inputRBG" << imageFilesSaved << ".jpg";
             		 string filename = ss.str();
           	 		 //imageFilesSaved++;
          			 reset_terminal_mode();
         			 printf("\t[Saved %s]\n", filename.c_str());
         			 set_conio_terminal_mode();

             		 cv::imwrite(filename, frame);

               		 saveImage = false;
             	 }
         }
         else
         { printf(" --(!) No captured frame -- Break!"); }

}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr changeOrientation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, char** argv, int counter, float angleX, float angleY, float angleZ)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::copyPointCloud(*cloud, *cloud_tmp);

	std::stringstream s;
	s.str(std::string());
	s << "original.pcd";
	pcl::io::savePCDFileASCII (s.str(), *cloud_tmp);

	//1. Find centroid
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid (*cloud_tmp, centroid);

	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

	// Define a translation of 2.5 meters on the x axis.
	transform_2.translation() << -centroid[0], -centroid[1], -centroid[2];
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);

	//3. Appy rotations

	transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (angleX, Eigen::Vector3f::UnitX()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);
	transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (angleY, Eigen::Vector3f::UnitY()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);
	//transform_2 = Eigen::Affine3f::Identity();
	//transform_2.rotate (Eigen::AngleAxisf (angleZ, Eigen::Vector3f::UnitZ()));
	//pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);

	/*transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (angleX, Eigen::Vector3f::UnitX()));
	transform_2.rotate (Eigen::AngleAxisf (angleY, Eigen::Vector3f::UnitY()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);
    float roll,pitch,yaw;
    pcl::getEulerAngles(transform_2,roll,pitch,yaw);
	transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (-yaw, Eigen::Vector3f::UnitZ()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);*/

	s.str(std::string());
	s << "rot.pcd";
	pcl::io::savePCDFileASCII (s.str(), *cloud_tmp);

	//5. translate back to original position
	transform_2 = Eigen::Affine3f::Identity();
	transform_2.translation() << centroid[0], centroid[1], centroid[2];
	// The same rotation matrix as before; tetha radians arround X axis
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);

	//##################
	std::stringstream ss;
	std::string inFilename;
	ss << argv[3];
	ss >> inFilename; //argv[1]

	//Directory number inside "Results"
	std::stringstream ss2;
	std::string directory;
	ss2 << argv[4];
	ss2 >> directory;

	//Label
	std::stringstream ss3;
	std::string outFilename;
	ss3 << argv[5];
	ss3 >> outFilename;

	//spliting diretory + filename
	std::string path = directory;

	unsigned found = inFilename.find_last_of("//");
	std::string file = inFilename.substr(found + 1);

	unsigned foundI = outFilename.find_last_of("//");
	unsigned foundF = outFilename.find_last_of(".");
	std::string label = outFilename.substr(foundI + 1, ( foundF - foundI - 1));
	std::string objects = "CNN/images";


	//creating diretory for the for the object
	std::stringstream objDir;
	objDir << path << "/" << objects << "/";
	boost::filesystem::path dir1(objDir.str());
	if (boost::filesystem::create_directory(dir1)) {
		//std::cout << "Success" << "\n";
	}

	std::stringstream labelDir;
	labelDir << path << "/" << objects << "/" << label << "/";
	boost::filesystem::path dir2(labelDir.str());
	if (boost::filesystem::create_directory(dir2)) {
		//std::cout << "Success" << "\n";
	}

	//creating diretory for the for the the example
	std::stringstream baseDir;
	//baseDir << path << "/" << objects << "/" << label << "/" << file << "_" << angleX << "/"; // take out this angleX
	baseDir << path << "/" << objects << "/" << label << "/"; // take out this angleX
	boost::filesystem::path dir3(baseDir.str());
	if (boost::filesystem::create_directory(dir3)) {
		//std::cout << "Success" << "\n";
	}
	std::stringstream stream;
	stream << baseDir.str() << "/" << file << "_" << counter << ".png";

	pcMethods.saveAsPNG2(stream.str(), cloud_tmp);

	s.str(std::string());
	s << "trans_back.pcd";
	pcl::io::savePCDFileASCII (s.str(), *cloud_tmp);

	s.str("");
	s << counter;
	logfile.add(s.str() + " - View in file " + stream.str()+ "\n");

	return cloud_tmp;
}


pcl::PointCloud<pcl::PointXYZRGBA>::Ptr changeOrientation_CNN(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, char** argv, int counter, float angleX, float angleY, float angleZ)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::copyPointCloud(*cloud, *cloud_tmp);

	std::stringstream s;
	s.str(std::string());
	s << "original.pcd";
	pcl::io::savePCDFileASCII (s.str(), *cloud_tmp);

	//1. Find centroid
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid (*cloud_tmp, centroid);

	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

	// Define a translation of 2.5 meters on the x axis.
	transform_2.translation() << -centroid[0], -centroid[1], -centroid[2];
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);

	//3. Appy rotations

	transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (angleX, Eigen::Vector3f::UnitX()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);
	transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (angleY, Eigen::Vector3f::UnitY()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);
	//transform_2 = Eigen::Affine3f::Identity();
	//transform_2.rotate (Eigen::AngleAxisf (angleZ, Eigen::Vector3f::UnitZ()));
	//pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);

	/*transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (angleX, Eigen::Vector3f::UnitX()));
	transform_2.rotate (Eigen::AngleAxisf (angleY, Eigen::Vector3f::UnitY()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);
    float roll,pitch,yaw;
    pcl::getEulerAngles(transform_2,roll,pitch,yaw);
	transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (-yaw, Eigen::Vector3f::UnitZ()));
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);*/

	s.str(std::string());
	s << "rot.pcd";
	pcl::io::savePCDFileASCII (s.str(), *cloud_tmp);

	//5. translate back to original position
	transform_2 = Eigen::Affine3f::Identity();
	transform_2.translation() << centroid[0], centroid[1], centroid[2];
	// The same rotation matrix as before; tetha radians arround X axis
	pcl::transformPointCloud (*cloud_tmp, *cloud_tmp, transform_2);

	//##################
	std::stringstream ss;
	std::string inFilename;
	ss << argv[3];
	ss >> inFilename; //argv[1]

	//Directory number inside "Results"
	std::stringstream ss2;
	std::string directory;
	ss2 << argv[4];
	ss2 >> directory;

	//spliting diretory + filename
	std::string path = directory;

	unsigned found = inFilename.find_last_of("//");
	std::string file = inFilename.substr(found + 1);

	unsigned foundI = inFilename.find_last_of("//");
	unsigned foundF = inFilename.find_last_of(".");
	std::string label = inFilename.substr(foundI + 1, ( foundF - foundI - 1));
	std::string objects = "CNN/images";


	//creating diretory for the for the object
	std::stringstream objDir;
	objDir << path << "/" << objects << "/";
	boost::filesystem::path dir1(objDir.str());
	if (boost::filesystem::create_directory(dir1)) {
		//std::cout << "Success" << "\n";
	}

	std::stringstream labelDir;
	labelDir << path << "/" << objects << "/" << label << "/";
		
	boost::filesystem::path dir2(labelDir.str());
	if (boost::filesystem::create_directory(dir2)) {
		//std::cout << "Success" << "\n";
	}

	//creating diretory for the for the the example
	std::stringstream baseDir;
	//baseDir << path << "/" << objects << "/" << label << "/" << file << "_" << angleX << "/"; // take out this angleX
	baseDir << path << "/" << objects << "/" << label << "/"; // take out this angleX
	boost::filesystem::path dir3(baseDir.str());
	if (boost::filesystem::create_directory(dir3)) {
		//std::cout << "Success" << "\n";
	}
	
	counter = count_files(baseDir.str(), ".png");
	
	std::stringstream stream;
	//stream << baseDir.str() << "/" << file << "_" << counter << ".png";
	stream << baseDir.str() << "/" << counter << ".png";

	pcMethods.saveAsPNG2(stream.str(), cloud_tmp);

	s.str(std::string());
	s << "trans_back.pcd";
	pcl::io::savePCDFileASCII (s.str(), *cloud_tmp);

	s.str("");
	s << counter;
	logfile.add(s.str() + " - View in file " + stream.str()+ "\n");

	return cloud_tmp;
}

void LabelCallback(const std_msgs::String::ConstPtr& msg)
{
	reset_terminal_mode();
	printf("Received Label\n");
	set_conio_terminal_mode();	
	// In this method something happens when the message arrives
	//if (writeFlag == 0)
	{
		int id = 0;
		if ( msg->data == "Banana" ) id = 0;
		if ( msg->data == "Box" ) id = 1;
		if ( msg->data == "Cup" ) id = 2;
		
		msg_pos.x[id] = pointText.x; 
		msg_pos.y[id] = pointText.y;
		msg_pos.z[id] = pointText.z;
		msg_pos.id[id] = id;
		
		
		reset_terminal_mode();
		//cout << "Label from Manager to Interface: " << msg->data << "\n";
		//cout << "(" << id << ": " <<  pointText.x << ", " << pointText.y << ", " <<  pointText.z << ")\n";
		if ( msg->data == "Banana" ) msgText = "A";
		if ( msg->data == "Box" ) msgText = "B";
		if ( msg->data == "Cup" ) msgText = "C";
		
	    set_conio_terminal_mode();
	    
		viewer->runOnVisualizationThreadOnce (addText);
		text_mutex.lock();
		text_mutex.lock();
		text_mutex.unlock();
		
		writeFlag++;
		noLabel = false;
		
	}
	//sleep(5);
	//printf("unlocking mutex\n");
	g_pages_mutex.unlock();
	//printf("Finished call\n");
	//ros::Publisher pub_num = n.advertise<std_msgs::String>("/NumberMessage", 1000);
	/*ros::NodeHandle nTT;
	ros::Publisher pubTT = nTT.advertise<std_msgs::String>("/TextToSay", 1000); // PUT THIS AS A GLOBAL VARIABLE

	std_msgs::String str;
	str.data = "This is a "+msgText;
	for (unsigned int i=0; i < 100; i++)
	{
		pubTT.publish(str);
		//pub_num.publish(idObject);
	}*/
}

void LabelCallback2(const std_msgs::String::ConstPtr msg)
{
	// In this method something happens when the message arrives
	//if (writeFlag == 0)
	{
		int id = 0;
		if ( msg->data == "Banana" ) id = 0;
		if ( msg->data == "Box" ) id = 1;
		if ( msg->data == "Cup" ) id = 2;
		
		msg_pos.x[id] = pointText.x; 
		msg_pos.y[id] = pointText.y;
		msg_pos.z[id] = pointText.z;
		msg_pos.id[id] = id;
		
		
		reset_terminal_mode();
		cout << "Label from Manager to Interface: " << msg->data << "\n";
		msgText = msg->data;
	    set_conio_terminal_mode();
		viewer->runOnVisualizationThreadOnce (addText);
		writeFlag++;
		noLabel = false;
	}
	//ros::Publisher pub_num = n.advertise<std_msgs::String>("/NumberMessage", 1000);
	/*ros::NodeHandle nTT;
	ros::Publisher pubTT = nTT.advertise<std_msgs::String>("/TextToSay", 1000); // PUT THIS AS A GLOBAL VARIABLE

	std_msgs::String str;
	str.data = "This is a "+msgText;
	for (unsigned int i=0; i < 100; i++)
	{
		pubTT.publish(str);
		//pub_num.publish(idObject);
	}*/
}


// This function is called every time the device has new data.
void grabberCallback(const PointCloud<PointXYZRGBA>::ConstPtr& cloud, char* directory, ofstream* outFile, LogsFile* logfile, ros::Publisher pub)
{
	// I don't know why, but it seems that I cant change the cloud capture. I think this is the reason why I am working with other variables, plot and cloud2

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr result ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2 ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table ( new pcl::PointCloud<pcl::PointXYZRGBA>);

	pcl::copyPointCloud(*cloud, *cloud2);
	pcl::copyPointCloud(*cloud, *plot);

	/*  *****************************************************************************/
	/*  METHOD employed to correct the orientation of the cloud captured			*/
	/*  I think this correction does not disturb the other algorithms, but BE AWARE */
	/*  If you remove this part below, everything should return to the same state   */
	/*  *****************************************************************************/
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	// Define a translation of 2.5 meters on the x axis.
	transform_2.translation() << 0, 0.0, -2.0;
	// The same rotation matrix as before; tetha radians arround X axis
	transform_2.rotate (Eigen::AngleAxisf (M_PI, Eigen::Vector3f::UnitX()));
	pcl::transformPointCloud (*cloud2, *cloud2, transform_2);
	pcl::transformPointCloud (*plot, *plot, transform_2);


	//pcl::ModelCoefficients::Ptr coefficients = vMethods.planeSegmentationView(cloud2, plot, cloud_table);
	bool falseFlag = false;
	pcl::ModelCoefficients::Ptr coefficients = pcMethods.planeSegmentation(cloud2, plot, cloud_table, directory, (*logfile), falseFlag);

	if (segmentFlag)
	{
		//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
		// This flag is set to true when segmentation is performed
		// So, if false screen will show normal plot, constantly getting new readings from sensor
		// if true, screen will show a freezed plot, to easy the feature selection
	    if ( !clusterFlag )
	    {
	    	pcl::copyPointCloud(*globalSegmented, *plot);
	    }
	    else
	    {
	    	pcl::copyPointCloud(*cloud2, *plot); //deletar essa linha
	    }

		pcMethods.segmentClusters(plot, cloud_table, coefficients, directory, (*logfile), falseFlag);

		// perform the clusterization of each object in the point cloud previously segmented - ideally it should find one object
	    //std::vector<pcl::PointIndices> cluster_indices = pcMethods.findClustersIndices(plot);
		//std::vector<pcl::PointIndices> cluster_indices;
	    //std:vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> cluster_list, cluster_list_before_red_paint;

	    if ( clusterFlag )
	    {
	    		cluster_list_before_red_paint = pcMethods.findClusters(plot, cluster_indices, directory, (*logfile), recordFlag); //getSelectedCluster
	    		numClusters = cluster_indices.size();
	    		clusterFlag = false;
	    		pcl::copyPointCloud(*plot, *globalSegmented);
	    }

		pcMethods.selectCluster(plot, cluster_indices, indexCluster);

		Eigen::Vector4f centroid;
		pcl::compute3DCentroid (*cluster_list_before_red_paint[indexCluster], centroid);
		//pointText.x = centroid[0];
		//pointText.y = centroid[1];
		//pointText.z = centroid[2] + 0.5;


		if ( recordFlag )
	    {
	    	//cluster_list = pcMethods.findClusters(plot, cluster_indices, directory, (*logfile), recordFlag); //getSelectedCluster
			// find the descriptor for the point cloud
			reset_terminal_mode();
			cout << "\t" << "Saving descriptor...\n";
			set_conio_terminal_mode();
			(*outFile) << "online_capture" << " ";
			(*outFile) << cluster_list.size() << " ";
			pcMethods.findDescriptorVFH(cluster_list_before_red_paint[indexCluster], (*outFile));
			recordFlag = false;
			(*outFile).close();

			//saving the pcd
			reset_terminal_mode();
   		    printf("\t%d - ", cloudFilesSaved);
			set_conio_terminal_mode();
			std::stringstream subdir;
			subdir << "/new_trainning_data/";
			std::stringstream ss;
			ss << directory << subdir.str();
			(*logfile).createDir(ss.str());
			string ss2 = ss.str();
			if ( cloudFilesSaved == 0 ) cloudFilesSaved = count_files(ss.str(), ".pcd");
			ss << "inputCloud" << cloudFilesSaved << ".pcd";
			string filename = ss.str();
			//if (io::savePCDFile(filename, *cloud, true) == 0)
			if (io::savePCDFile(filename, *cluster_list_before_red_paint[indexCluster], true) == 0)
			{
				//cloudFilesSaved++;
				reset_terminal_mode();
				printf("\t[Saved %s]\n", filename.c_str());
				set_conio_terminal_mode();
				cloudFilesSaved = count_files(ss2, ".pcd");
			}
			else PCL_ERROR("Problem saving %s.\n", filename.c_str());
			reset_terminal_mode();
			cout << "\t" << "Data Saved Successfully!\n";
			set_conio_terminal_mode();
	    }

		if ( testFlag )
		{
			reset_terminal_mode();
			//cout << "Just after press `t`\n";
			set_conio_terminal_mode();
			vector<float> data;
			testFlag = false;
			std_msgs::Float32MultiArray msg;
			msg.data.clear();

	    	//cluster_list = pcMethods.findClusters(plot, directory, (*logfile), recordFlag);
			//data = pcMethods.findDescriptorVFH(cluster_list[indexCluster], (*outFile));
			// separacao do código com duas funcoes está muito ruim para findVFH!!!
			// Acho q esse metodo vai desaparecer
			data = pcMethods.findDescriptorVFHFromSelectedClusterInTheFullPC(plot, cluster_indices, indexCluster); //cluster_list[0]
			for (unsigned int i=0; i < data.size(); i++)
			{
				msg.data.push_back(data[i]);
			}
			for (unsigned int i=0; i < 500; i++)
			{
				pub.publish(msg);
			}
			writeFlag=0;
		}

	}
	if (! viewer->wasStopped())
		viewer->showCloud(plot);

	  plot->width = plot->points.size ();
	  plot->height = 1;
	  plot->is_dense = true;

	if (saveCloud)
	{
		std::stringstream subdir;
		subdir << "/new_trainning_data/";
		std::stringstream ss;
		ss << directory << subdir.str();
		(*logfile).createDir(ss.str());
		string ss2 = ss.str();
		if ( cloudFilesSaved == 0 ) cloudFilesSaved = count_files(ss.str(), ".pcd");
		ss << "inputCloud" << cloudFilesSaved << ".pcd";
		string filename = ss.str();
		if (io::savePCDFile(filename, *cloud, true) == 0)
		//if (io::savePCDFile(filename, *plot, true) == 0)
		{
			//cloudFilesSaved++;
			reset_terminal_mode();
			printf("\t[Saved %s]\n", filename.c_str());
			set_conio_terminal_mode();
			cloudFilesSaved = count_files(ss2, ".pcd");
		}
		else PCL_ERROR("Problem saving %s.\n", filename.c_str());

		saveCloud = false;
	}
}


void printLocation(neural_pkg::XYZObjects msg_pos)
{
	neural_pkg::XYZObjects map;
	int nObjects = 3;
	for (unsigned j = 0; j < nObjects; j++)
	{
		map.x.push_back(-1); 
		map.y.push_back(-1); 
		map.z.push_back(-1); 
		map.id.push_back(-1); 
	}	
	
	for (unsigned int j = 0; j < nObjects; j++)
	{
		float minX = 1000;
		int min_index=0;
		for (unsigned int i = 0; i < nObjects; i++)
		{
			if ( msg_pos.id[i] != -1 )
			{
				if ( msg_pos.x[i] < minX )
				{
					minX = msg_pos.x[i];
					min_index = i;
				}
			}
		}
		if ( msg_pos.id[min_index] != -1 )
		{
			map.x[j] = msg_pos.x[min_index];
			map.y[j] = msg_pos.y[min_index];
			map.z[j] = msg_pos.z[min_index];
			map.id[j] = msg_pos.id[min_index];
			msg_pos.id[min_index] = -1;
		}
	}
	reset_terminal_mode();
	cout << "["; 
	std::string ss = "";
	for (unsigned int j = 0; j < nObjects; j++)
	{
		ss = "Unknown";
		if ( map.id[j] == 0 ) ss = "Banana";
		if ( map.id[j] == 1 ) ss = "Box";
		if ( map.id[j] == 2 ) ss = "Cup";
		cout << ss << ": " <<  "(" <<  map.x[j] << ", " << map.y[j] << ", " <<  map.z[j] << ")   ";
	}
	cout << "]\n"; 	
	set_conio_terminal_mode();	
}

// This function is called every time the device has new data.
void grabberCallback_CNN(const PointCloud<PointXYZRGBA>::ConstPtr& cloud, char** argv, char* directory, ofstream* outFile, LogsFile* logfile, ros::Publisher pub, ros::Publisher pub_pos)
{
	// I don't know why, but it seems that I cant change the cloud capture. I think this is the reason why I am working with other variables, plot and cloud2

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot_tmp ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr result ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2 ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2_tmp ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table ( new pcl::PointCloud<pcl::PointXYZRGBA>);

	pcl::copyPointCloud(*cloud, *cloud2_tmp);
	pcl::copyPointCloud(*cloud, *plot_tmp);
	
	pcMethods.createVoxels(cloud2_tmp, cloud2);
	pcMethods.createVoxels(plot_tmp, plot);
	
	/*  *****************************************************************************/
	/*  METHOD employed to correct the orientation of the cloud captured			*/
	/*  I think this correction does not disturb the other algorithms, but BE AWARE */
	/*  If you remove this part below, everything should return to the same state   */
	/*  *****************************************************************************/
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	// Define a translation of 2.5 meters on the x axis.
	transform_2.translation() << 0, 0.0, -2.0;
	// The same rotation matrix as before; tetha radians arround X axis
	transform_2.rotate (Eigen::AngleAxisf (M_PI, Eigen::Vector3f::UnitX()));
	pcl::transformPointCloud (*cloud2, *cloud2, transform_2);
	pcl:transformPointCloud (*plot, *plot, transform_2);


	//pcl::ModelCoefficients::Ptr coefficients = vMethods.planeSegmentationView(cloud2, plot, cloud_table);
	bool falseFlag = false;
	pcl::ModelCoefficients::Ptr coefficients = pcMethods.planeSegmentation(cloud2, plot, cloud_table, directory, (*logfile), falseFlag);

	if (segmentFlag)
	{
		//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
		// This flag is set to true when segmentation is performed
		// So, if false screen will show normal plot, constantly getting new readings from sensor
		// if true, screen will show a freezed plot, to easy the feature selection
	    if ( !clusterFlag )
	    {
	    	pcl::copyPointCloud(*globalSegmented, *plot);
	    }
	    else
	    {
	    	pcl::copyPointCloud(*cloud2, *plot); //deletar essa linha
	    }

		pcMethods.segmentClusters(plot, cloud_table, coefficients, directory, (*logfile), falseFlag);

		// perform the clusterization of each object in the point cloud previously segmented - ideally it should find one object
	    //std::vector<pcl::PointIndices> cluster_indices = pcMethods.findClustersIndices(plot);
		//std::vector<pcl::PointIndices> cluster_indices;
	    //std:vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> cluster_list, cluster_list_before_red_paint;

	    if ( clusterFlag )
	    {
	    		cluster_list_before_red_paint = pcMethods.findClusters(plot, cluster_indices, directory, (*logfile), recordFlag); //getSelectedCluster
	    		numClusters = cluster_indices.size();
	    		clusterFlag = false;
	    		pcl::copyPointCloud(*plot, *globalSegmented);
	    }

		//pcMethods.selectCluster(plot, cluster_indices, indexCluster);

		Eigen::Vector4f centroid;
		pcl::compute3DCentroid (*cluster_list_before_red_paint[indexCluster], centroid);
		//reset_terminal_mode();
		//printf("(%.2f, %.2f, %.2f)\n", centroid[0], centroid[1], centroid[2]);
		//set_conio_terminal_mode();
		
		pointText.x = centroid[0];
		pointText.y = centroid[1];
		pointText.z = centroid[2] + 0.5;

		// I did this just to correct the image, before writing into the disc
		Eigen::Affine3f transform_record = Eigen::Affine3f::Identity();
		transform_record.translation() << 0, 0.0, -2.0;
		transform_record.rotate (Eigen::AngleAxisf (M_PI, Eigen::Vector3f::UnitX()));
		
		
		//===============================TEST
		/*double a = abs(coefficients->values[0]);
		double b = abs(coefficients->values[1]);
		double c = abs(coefficients->values[2]);
		double d = abs(coefficients->values[3]);
		Eigen::Vector3f x_axis (b / sqrt (a * a + b * b), -a / sqrt (a * a + b * b), 0.);
	    Eigen::Vector3f y_direction (a, b, c);
	    Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
	    rotation = pcl::getTransFromUnitVectorsXY (x_axis, y_direction);
	    float roll,pitch,yaw;
	    pcl::getEulerAngles(rotation,roll,pitch,yaw);
	    
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_service ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	    pcl::copyPointCloud(*cluster_list_before_red_paint[indexCluster], *cloud_to_service);
	    transformPointCloud(*cloud_to_service, *cloud_to_service, rotation);
		Eigen::Vector4f centroid_to_service;
		pcl::compute3DCentroid (*cloud_to_service, centroid_to_service);
		pointText_to_service.x = centroid_to_service[0];
		pointText_to_service.y = centroid_to_service[1];
		pointText_to_service.z = centroid_to_service[2];
		reset_terminal_mode();
		printf("(%.2f, %.2f, %.2f)\n", centroid_to_service[0], centroid_to_service[1], centroid_to_service[2]);
		set_conio_terminal_mode();*/		
		//===============================TEST
	    
		
		
		if ( ( recordFlag ) && ( !writingViewsFlag ) ) 
	    {
			writingViewsFlag = true;
			//The cloud view with inclination (non parallel) is saved
			//This loop is not valid in the default way: It is here just to create the DB of reorientaded images
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_tmp;
			//for (unsigned int i = 0; i < 10; i++)
			//{
			pcl::transformPointCloud (*cluster_list_before_red_paint[indexCluster], *cluster_list_before_red_paint[indexCluster], transform_record);
			cloud_tmp = changeOrientation_CNN(cluster_list_before_red_paint[indexCluster], argv, 0, 0, 0, 0);
			//}
			// and slices for this view are computed
			//saveSlices(cluster_list_with_inclination[0], argv, 0);
			std::stringstream cnnDir;
			cnnDir << directory << "/CNN/";
			//printf("Fail: %s,\n", cnnDir.str().c_str());
			boost::filesystem::path dir(cnnDir.str());
			if (boost::filesystem::create_directory(dir)) {
				//std::cout << "Success" << "\n";
			}
			
			int numViewsAxisX = 5;
			int numViewsAxisY = 10;
			int view = 1;
			float angle_grad = 60;
			float angle_rad = ( (M_PI*angle_grad) / 180);
			for (float angleY = -(angle_rad); angleY < - angle_rad + angle_rad*2; angleY = angleY + ( (2*angle_rad)/numViewsAxisY ) )
			{
				for (float angleX = (angle_rad); angleX < angle_rad + angle_rad*2; angleX = angleX + ( (2*angle_rad)/numViewsAxisX ) )
				{
					//Then cloud (that is parallel cloud to the surface) is reorientaded and stored in cloud_tmp
					//cloud_tmp = changeOrientation(cloud, argv, view, angleX, angleY, 0); //ORIGINAL
					pcl::transformPointCloud (*cluster_list_before_red_paint[indexCluster], *cluster_list_before_red_paint[indexCluster], transform_record);
					cloud_tmp = changeOrientation_CNN(cluster_list_before_red_paint[indexCluster], argv, view, angleX, angleY, 0); 
					//slices of the cloud_tmp (reorientaded are computed)
					//saveSlices(cloud_tmp, argv, view);
					view++;
				}
			}
			writingViewsFlag = false;
		}

		//if ( ( testFlag ) && ( numClusters > 0 ) )
		if ( testFlag ) 
		{
			//printf("Directory:%s\n", directory);
			testFlag = false;
			for (unsigned int i = 0; i < numClusters; i++)
			{
				//pcMethods.selectCluster(plot, cluster_indices, i);

				Eigen::Vector4f centroid;
				pcl::compute3DCentroid (*cluster_list_before_red_paint[i], centroid);
				//reset_terminal_mode();
				//printf("(%.2f, %.2f, %.2f)\n", centroid[0], centroid[1], centroid[2]);
				//set_conio_terminal_mode();
				
				pointText.x = centroid[0];
				pointText.y = centroid[1];
				pointText.z = centroid[2] + 0.5;
				if ( pointText.y < 0.1 )
				{

					Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
					// Define a translation of 2.5 meters on the x axis.
					transform_2.translation() << 0, 0.0, -2.0;
					// The same rotation matrix as before; tetha radians arround X axis
					transform_2.rotate (Eigen::AngleAxisf (M_PI, Eigen::Vector3f::UnitX()));
					//pcl::transformPointCloud (*cluster_list_before_red_paint[indexCluster], *cluster_list_before_red_paint[indexCluster], transform_2);
					pcl::transformPointCloud (*cluster_list_before_red_paint[i], *cluster_list_before_red_paint[i], transform_2);
					
					std::stringstream stream;			
					//stream << "/informatik/isr/wtm/home/borghetti/Desktop/TESTECNN/testImage/" << i << ".png";
					stream << "CNN_PYTHON/Database/testImage/1.png";
					//pcMethods.saveAsPNG2(stream.str(), cluster_list_before_red_paint[indexCluster]);
					pcMethods.saveAsPNG2(stream.str(), cluster_list_before_red_paint[i]);
					
					std_msgs::String msg;
					msg.data.clear();
					msg.data = "1";
		
					//cluster_list = pcMethods.findClusters(plot, directory, (*logfile), recordFlag);
					//data = pcMethods.findDescriptorVFH(cluster_list[indexCluster], (*outFile));
					// separacao do código com duas funcoes está muito ruim para findVFH!!!
					// Acho q esse metodo vai desaparecer
					//noLabel = true;
					//for (unsigned int j=0; j < 1; j++)
					//{
					pub.publish(msg);
					//}
					//sleep(5);
					
					
					std_msgs::StringConstPtr  m = ros::topic::waitForMessage<std_msgs::String>("/LabelFromManagerToInterface", ros::Duration(10));
					//printf("Finished wait for message\n");
					g_pages_mutex.lock();
					g_pages_mutex.lock();
					//printf("Finished wait for mutex\n");
					g_pages_mutex.unlock();
					//LabelCallback2(m);
					
					//ros::topic::waitForMessage("/LabelFromManagerToInterface");
					//while ( noLabel );
					writeFlag=0;
				}
			}
		    /*for (unsigned int k = 0; k < 3; k++)
		    {
				if ( msg_pos.id[k] == 0 ) msgText = "Banana";
				if ( msg_pos.id[k] == 1 ) msgText = "Box";
				if ( msg_pos.id[k] == 2 ) msgText = "Cup";
		    	viewer->runOnVisualizationThreadOnce (removeText);
		    }
			remove_text_mutex.lock();
			remove_text_mutex.lock();
			remove_text_mutex.unlock();*/
			//pcl::visualization::PCLVisualizer::removeAllPointClouds();
			viewer->runOnVisualizationThreadOnce( removeText );
			
			startFlag = true; // just to cancel the segmentation and start it again
			pub_pos.publish(msg_pos);
		    printLocation(msg_pos);
			msg_pos.x.clear();
			msg_pos.y.clear();
			msg_pos.z.clear();
			msg_pos.id.clear();
			int nObjects = 3;
			for (unsigned j = 0; j < nObjects; j++)
			{
				msg_pos.x.push_back(-1); 
				msg_pos.y.push_back(-1); 
				msg_pos.z.push_back(-1); 
				msg_pos.id.push_back(-1); 
			}
		}
		segmentFlag = !segmentFlag;
	}
	if (! viewer->wasStopped())
		viewer->showCloud(plot);

	  plot->width = plot->points.size ();
	  plot->height = 1;
	  plot->is_dense = true;

	if (saveCloud)
	{
		std::stringstream subdir;
		subdir << "/new_trainning_data/";
		std::stringstream ss;
		ss << directory << subdir.str();
		(*logfile).createDir(ss.str());
		string ss2 = ss.str();
		if ( cloudFilesSaved == 0 ) cloudFilesSaved = count_files(ss.str(), ".pcd");
		ss << "inputCloud" << cloudFilesSaved << ".pcd";
		string filename = ss.str();
		if (io::savePCDFile(filename, *cloud, true) == 0)
		//if (io::savePCDFile(filename, *plot, true) == 0)
		{
			//cloudFilesSaved++;
			reset_terminal_mode();
			printf("\t[Saved %s]\n", filename.c_str());
			set_conio_terminal_mode();
			cloudFilesSaved = count_files(ss2, ".pcd");
		}
		else PCL_ERROR("Problem saving %s.\n", filename.c_str());

		saveCloud = false;
	}
}

// For detecting when SPACE is pressed.
void keyboardEventOccurred(const visualization::KeyboardEvent& event,
					  void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown())
	{
		saveCloud = true;
		saveImage = true;
	}
}

// Creates, initializes and returns a new viewer.
boost::shared_ptr<visualization::CloudViewer> createViewer()
{
	boost::shared_ptr<visualization::CloudViewer> v
	(new visualization::CloudViewer("OpenNi viewer Cloud"));
	//v->registerKeyboardCallback(keyboardEventOccurred);

	return (v);
}

boost::shared_ptr<visualization::ImageViewer> createImageViewer()
{
	boost::shared_ptr<visualization::ImageViewer> v
		(new visualization::ImageViewer("OpenNi viewer Image"));
		v->registerKeyboardCallback(keyboardEventOccurred);
    return (v);
}


void *connection_handler(void *)
{
    int socket_desc , client_sock , d;
	struct sockaddr_in server , client;
	 
	
	//Create socket
	socket_desc = socket(AF_INET , SOCK_STREAM , 0);
	if (socket_desc == -1)
	{
		reset_terminal_mode();
		printf("Could not create socket");
		set_conio_terminal_mode();	
	}
	reset_terminal_mode();
	puts("Socket created");
	set_conio_terminal_mode();	
	 
	//Prepare the sockaddr_in structure
	server.sin_family = AF_INET;
	server.sin_addr.s_addr = INADDR_ANY;
	server.sin_port = htons( 54012 );
	 
	//Bind
	if( bind(socket_desc,(struct sockaddr *)&server , sizeof(server)) < 0)
	{
		//print the error message
		reset_terminal_mode();
		perror("bind failed. Error");
		set_conio_terminal_mode();	
	}
	reset_terminal_mode();
	puts("bind done");
	set_conio_terminal_mode();	
	 
	//Listen
	listen(socket_desc , 3);
	 
	//Accept and incoming connection
	reset_terminal_mode();
	puts("Waiting for incoming connections...");
	set_conio_terminal_mode();	
	d = sizeof(struct sockaddr_in);
	 
	 
	//Accept and incoming connection
	reset_terminal_mode();
	puts("Waiting for incoming connections...");
	set_conio_terminal_mode();	
	d = sizeof(struct sockaddr_in);
	
	while (1)
	{
	   while( (client_sock = accept(socket_desc, (struct sockaddr *)&client, (socklen_t*)&d)) )
	   {
			 
			startFlag = true;	
			
			testFlag = true;
			
			reset_terminal_mode();
			puts("Connection accepted");
			set_conio_terminal_mode();	
			
			
			//Now join the thread , so that we dont terminate before the thread
			//pthread_join( thread_id , NULL);
			puts("Handler assigned");
		    if (client_sock < 0)
		    {
		        perror("accept failed");
		    }
		}
	}
		
} 

int onlineProcedure(int argc, char** argv, int net)
{
	bool justVisualize(false);
	string filename;
	char *outFilename;
	char *directory;

	ros::init(argc, argv, "C_node");
	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("/LabelFromManagerToInterface", 100, LabelCallback);
	ros::Subscriber subStart = n.subscribe("/StartObjectsRecog", 1000, startObjectsRecog);
	
	
	ros::Publisher  pub = n.advertise<std_msgs::Float32MultiArray>("/DescriptorFromInterfaceToManager", 1000);
	ros::Publisher  pubCNN = n.advertise<std_msgs::String>("/RequestFromInterfaceToCNN", 1000);
	ros::Publisher  pub_pos = n.advertise<neural_pkg::XYZObjects>("/ObjectPoints", 1000);
    //ros::ServiceServer service = n.advertiseService("GetXYZObjects", GetXYZObjects);
	
	//*********************************************************
	// CODE FOR SOCKETS - THREADS
	//*********************************************************
	pthread_t thread_id;
    if ( pthread_create( &thread_id , NULL ,  connection_handler, NULL ) < 0)
    {
        perror("could not create thread");
	    return 1;
	}
	//*********************************************************
	// END CODE FOR SOCKETS - THREADS
	//*********************************************************
	
	

	ofstream outFile;
    if (console::find_argument(argc, argv, "-v") >= 0)
	{
		if (argc != 4)
		{
			printUsage(argv[0]);
			return 0;
		}

		filename = argv[3];
		justVisualize = true;
	}
	else if (console::find_argument(argc, argv, "-mlp") >= 0)
	{
		if (argc != 5)
		{
			printUsage(argv[0]);
			return 0;
		}
		outFilename = argv[3];
		directory = argv[4]; //output directory to store the results
	}
	else if (console::find_argument(argc, argv, "-cnn") >= 0)
	{
		if (argc != 5)
		{
			printUsage(argv[0]);
			return 0;
		}
		outFilename = argv[3];
		directory = argv[4]; //output directory to store the results
	}
	outFile.open (outFilename, ios::out | ios::app | ios::binary);
	outFile.close();

	// First mode, open and show a cloud from disk.
	if (justVisualize)
	{
		// Try with color information...
		try
		{
			io::loadPCDFile<PointXYZRGBA>(filename.c_str(), *cloudptr);
		}
		catch (PCLException e1)
		{
			try
			{
				// ...and if it fails, fall back to just depth.
				io::loadPCDFile<PointXYZ>(filename.c_str(), *fallbackCloud);
			}
			catch (PCLException e2)
			{
				return -1;
			}

			noColor = true;
		}
		reset_terminal_mode();
		cout << "Loaded " << filename << "." << endl;
		set_conio_terminal_mode();
		if (noColor)
		{
			reset_terminal_mode();
			cout << "This cloud has no RGBA color information present." << endl;
			set_conio_terminal_mode();
		}
		else
		{
			reset_terminal_mode();
			cout << "This cloud has RGBA color information present." << endl;
			set_conio_terminal_mode();
		}
	}
	// Second mode, start fetching and displaying frames from the device.
	else
	{
		reset_terminal_mode();
		OpenniGrabber = new OpenNIGrabber();
		set_conio_terminal_mode();
		if (OpenniGrabber == 0)
			return false;

		if ( net ==  0 )
		{
			boost::function<void (const PointCloud<PointXYZRGBA>::ConstPtr&)> fc =
					boost::bind(&grabberCallback, _1, directory, &outFile, &logfile, pub);
			OpenniGrabber->registerCallback(fc);

		}

		if ( net ==  1 )
		{
			boost::function<void (const PointCloud<PointXYZRGBA>::ConstPtr&)> fc =
					boost::bind(&grabberCallback_CNN, _1, argv, directory, &outFile, &logfile, pubCNN, pub_pos);
			OpenniGrabber->registerCallback(fc);
		}

		boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&)> fi =
                boost::bind (&image_cb_, _1, directory);
		OpenniGrabber->registerCallback (fi);

    }

	viewer = createViewer();
	//image_viewer = createImageViewer();
	viewer->runOnVisualizationThreadOnce (viewerOneOff);
	if (justVisualize)
	{
		/*  *****************************************************************************/
		/*  METHOD employed to correct the orientation of the cloud captured			*/
		/*  I think this correction does not disturb the other algorithms, but BE AWARE */
		/*  If you remove this part below, everything should return to the same state   */
		/*  *****************************************************************************/
		Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
		// Define a translation of 2.5 meters on the x axis.
		//transform_2.translation() << 0, 0.0, -2.0;
		// The same rotation matrix as before; tetha radians arround X axis
		transform_2.rotate (Eigen::AngleAxisf (M_PI, Eigen::Vector3f::UnitX()));

		if (noColor)
		{
			pcl::transformPointCloud (*fallbackCloud, *fallbackCloud, transform_2);
			viewer->showCloud(fallbackCloud);
		}
		else
		{
			pcl::transformPointCloud (*cloudptr, *cloudptr, transform_2);
			viewer->showCloud(cloudptr);
		}
	}
	else
	{
		OpenniGrabber->start();
	}

	// Main loop.
	char c;
	while (! viewer->wasStopped())
	{
		boost::this_thread::sleep(boost::posix_time::seconds(1));
		//if (kbhit()) 
		{
			//char c = getch();
			if ( startFlag )
			{
				c = 's';
			}
			else
			{
				c = 'X';
			}
			
			if (c == ' ' )
			{
			   	printf("%d - ", cloudFilesSaved);
		       	saveImage = true;
		       	saveCloud = true;
			}
			if (c == 's' )
			{
				if ( recordFlag )
				{
					reset_terminal_mode();
					printf("Please, stop recording first.\n");
					set_conio_terminal_mode();
				}
				else
				{
					//segmentFlag = !segmentFlag;
					segmentFlag = true;
					clusterFlag = true;
				}
				startFlag = false;
			}
			//cout << c;
		    if (c == 'q')
		    {
		    	if ( recordFlag )
				{
					reset_terminal_mode();
					printf("It is better for the reliability of your data, to cancel the recording mode first. So, press `r` now, please.\n");
					set_conio_terminal_mode();
				}
		    	else
		    	{
		    		break;
		    	}
		    }

		    if (c == 'r')
		    {
				if ( segmentFlag )
		    	{
					recordFlag = !recordFlag;
					if ( recordFlag )
					{
						reset_terminal_mode();
						cout << "Now recording!\n";
						set_conio_terminal_mode();
						outFile.open (outFilename, ios::out | ios::app | ios::binary);
					}
		    	}
				else
				{
					reset_terminal_mode();
					//printf("Comand allowed only in segmentation mode!\n");
					cout << "Are you trying recording before pressing `s`? Press `s` for segmenting first, please.\n";
					set_conio_terminal_mode();
				}
		    }

		    if (c == 't')
		    {
				if ( segmentFlag )
		    	{
					if ( recordFlag )
					{
						reset_terminal_mode();
						cout << "God! You should stop recording first! Press ´r´ first\n";
						set_conio_terminal_mode();
					}
					else
					{
						testFlag = true;
						std::stringstream index;
						index << indexCluster;
						idObject.data = index.str();
						if (!(lastMsg.str() == ""))
							viewer->runOnVisualizationThreadOnce (removeText);
					}

		    	}
				else
				{
					reset_terminal_mode();
					cout << "Are you trying test before pressing `s`? Press `s` for segmenting first, please.\n";
					set_conio_terminal_mode();
				}
		    }

		    if (c == '\t')
		    {
		    	if ( segmentFlag )
		    	{
		    		indexCluster =  (++indexCluster) % (numClusters)  ;
		    		viewer->runOnVisualizationThreadOnce (removeText);
		    	}
		    	else
		    	{
					reset_terminal_mode();
					printf("You won't being successed trying this... Press `s` for segmenting first, please.\n");
					set_conio_terminal_mode();
		    	}
		    }

		}
		ros::spinOnce();
	}

	if (! justVisualize)
	{
		OpenniGrabber->stop();
	}
    outFile.close();
	return 0;
}

int offlineProcedureMLP(char** argv)
{
	char *inFilename, *outFilename;
	char *directory;
	inFilename = argv[3]; // input point cloud
	directory = argv[4]; //output directory to store the results
	outFilename = argv[5]; //name of the file that will be output

	ofstream outFile;
	outFile.open (outFilename, ios::out | ios::app | ios::binary);

    // Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(inFilename, *cloud) != 0)
	{
		cout << "Inform a point cloud as input" << "\n";
	    return -1;
	}
	//s << "Point Cloud " << inFilename << " loaded.\n";
	//logfile.add(s.str());
	//s.str("");

	std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);

	// Transform de point cloud in voxels, afer downsampling
	pcMethods.createVoxels(cloud, cloud_filtered);
	//pcl::copyPointCloud(*cloud, *cloud_filtered);

	// Remove all planes: floor, walls, etc.
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::copyPointCloud(*cloud_filtered, *plot);
	recordFlag = true;
	pcl::ModelCoefficients::Ptr coefficients = pcMethods.planeSegmentation(cloud_filtered, plot, cloud_table, directory, logfile, recordFlag); // descomentar depois

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
	//pcl::copyPointCloud(*plot, *cloud_filtered2); //deletar essa linha
	pcl::copyPointCloud(*cloud_filtered, *cloud_filtered2); //deletar essa linha

	// Segment the are in which the object can be found
	pcMethods.segmentClusters(cloud_filtered2, cloud_table, coefficients, directory, logfile, recordFlag); // descomentar apos os testes

	// perform the clusterization of each object in the point cloud previously segmented - ideally it should find one object
	std::vector<pcl::PointIndices> cluster_indices;
	std:vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> cluster_list = pcMethods.findClusters(cloud_filtered2, cluster_indices, directory, logfile, recordFlag);

	// find the descriptor for the point cloud
	for (unsigned int i = 0; i < cluster_list.size(); i++)
	{
		outFile << inFilename << " ";
		outFile << cluster_list.size() << " ";
		reset_terminal_mode();
		//pcMethods.findDescriptorVFH(cluster_list[i], outFile);
		pcMethods.findDescriptorOURCVFH(cluster_list[i], outFile);
		//Rotation+translation
		//applyRotationAndTranslation(cloud_filtered2, directory);
		pcMethods.applyRotationAndTranslation(cluster_list[0], directory, outFile, logfile);
		set_conio_terminal_mode();
	}
	outFile.close();
	return 0;
}

void saveSlices(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, char** argv, int view)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_slice (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized(new pcl::PointCloud<pcl::PointXYZRGBA>);

    // Create a set of planar coefficients with X=Y=0,Z=1
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    coefficients->values.resize (4);
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;

	double maxZ = -100;
	double minZ = 100;
	for (size_t i = 0; ( i < cloud->points.size() ) ; ++i)
	{
	 if ( cloud->points[i].z > maxZ )
	 {
		maxZ = cloud->points[i].z;
	 }
	 if ( cloud->points[i].z < minZ )
	 {
		minZ = cloud->points[i].z;
	 }
	}

	//double tamSlice = 0.005;
	//int numberOfTotalSlices = ( maxZ - minZ ) / tamSlice;
	int numberOfTotalSlices = 10;
	double tamSlice  = ( maxZ - minZ ) / (double) numberOfTotalSlices;

	//int startingSlice = ( numberOfTotalSlices / 2 ) - ( numberOfMainSlices / 2 );
	//int startingSlice = ( numberOfTotalSlices / 2 ) - numberOfMainSlices;
	//int startingSlice = 40;
	//std::cout << "\nStarting Slice: " << startingSlice << "\n";
	//std::cout << "\ntamSlice: " << tamSlice << "\n";

	int idSlice = 0;
	//double delta = 0.08;
	double delta = 0.02;
	reset_terminal_mode();
	std::cerr << "Computing slices for view " << view << ":" << std::endl;
	set_conio_terminal_mode();
	for (double k = minZ; k < maxZ; k = k + tamSlice)
	{
	  if ( ( idSlice < numberOfTotalSlices) )
	  {
		  cloud_slice->clear();
		  cloud_slice->points.clear();
		  cloud_projected->clear();
		  cloud_projected->points.clear();
		  pcMethods.getSlicesFromPointCloud(cloud, cloud_slice, k, delta);

		  // Create the filtering object
		  pcl::ProjectInliers<pcl::PointXYZRGBA> proj;
		  proj.setModelType (pcl::SACMODEL_PLANE);
		  //proj.setInputCloud (cloud);
		  proj.setInputCloud (cloud_slice);
		  //std::cout << "cloud slice size = " << cloud_slice->points.size() << "\n";
		  proj.setModelCoefficients (coefficients);
		  proj.filter (*cloud_projected);

		  if ( cloud_projected->points.size() > 0 ) pcl::io::savePCDFileASCII ("projected.pcd", *cloud_projected);
		  reset_terminal_mode();
		  std::cerr << "\tSaving the sliced cloud: " << idSlice << "..." << std::endl;
		  set_conio_terminal_mode();
		  reset_terminal_mode();
		  pcMethods.convertPointCloudToPNG2(cloud_projected, cloud_organized, pcMethods.buildPath(argv, view, idSlice));
		  set_conio_terminal_mode();
		  idSlice++;
	  }
	}

	//repete the last image to be sure that you have 10 images
	//if more slices are required to fill the stack, use black images
	for (unsigned int i=0; i < cloud_projected->points.size(); i++)
	{
		cloud_projected->points[i].r = 0;
		cloud_projected->points[i].g = 0;
		cloud_projected->points[i].b = 0;
	}
	while ( idSlice < numberOfTotalSlices )
	{
	  pcMethods.convertPointCloudToPNG2(cloud_projected, cloud_organized, pcMethods.buildPath(argv, view, idSlice));
	  idSlice++;
	}
	stringstream tmp;
	tmp << idSlice;
	if ( idSlice < 20 ) logfile.add("\t Less than 20 slices saved.\n");

}

int offlineProcedureCNN(char** argv)
{
	char *inFilename, *outFilename;
	char *directory;
	inFilename = argv[3]; // input point cloud
	directory = argv[4]; //output directory to store the results
	outFilename = argv[5]; //name of the file that will be output

	std::stringstream s;
	logfile.createStructure(directory);


	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(argv[3], *cloud) != 0)
	{
	  std::cout << "Inform a point cloud as input" << "\n";
	  return -1;
	}

	std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);

	//TESTES#######################################################################################
	//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);

	// Transform de point cloud in voxels, afer downsampling
	//pcMethods.createVoxels(cloud, cloud_filtered);

	// Remove all planes: floor, walls, etc.
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::copyPointCloud(*cloud, *plot);
	recordFlag = true;
	pcl::ModelCoefficients::Ptr coefficients = pcMethods.planeSegmentation(cloud, plot, cloud_table, directory, logfile, recordFlag); // descomentar depois

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);
	//pcl::copyPointCloud(*plot, *cloud_filtered2); //deletar essa linha
	pcl::copyPointCloud(*cloud, *cloud_filtered); //deletar essa linha

	// Segment the are in which the object can be found
	// cloud_tmp is the cloud with Z parallel to the surface
	// cloud_filtered is the cloud non parallel to the surface
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_tmp = pcMethods.segmentClusters(cloud_filtered, cloud_table, coefficients, directory, logfile, recordFlag); // descomentar apos os testes
	// A list of cluster of the non parallel cloud is computed below
	std::vector<pcl::PointIndices> cluster_indices;
	std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> cluster_list_with_inclination = pcMethods.findClusters(cloud_filtered, cluster_indices, directory, logfile, recordFlag);

	// Then the parallel cloud is copied to the main cloud
	pcl::copyPointCloud(*cloud_tmp, *cloud_filtered);
	// And a list of cluster of the parallel cloud is computed below
	std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> cluster_list = pcMethods.findClusters(cloud_filtered, cluster_indices, directory, logfile, recordFlag);
	// The methods considers just one object
	pcl::copyPointCloud(*cluster_list[0], *cloud);
	//#######################################################################################################################

	std::stringstream cnnDir;
	cnnDir << directory << "/CNN/";
	boost::filesystem::path dir(cnnDir.str());
	if (boost::filesystem::create_directory(dir)) {
		//std::cout << "Success" << "\n";
	}

	std::stringstream tmp;
	tmp << cloud->points.size();
	logfile.add("cloud size = "+ tmp.str() + "\n");

	//The cloud view with inclination (non parallel) is saved
	//This loop is not valid in the default way: It is here just to create the DB of reorientaded images
	for (unsigned int i = 0; i < 10; i++)
	{
		cloud_tmp = changeOrientation(cluster_list_with_inclination[0], argv, i, 0, 0, 0);
	}
	// and slices for this view are computed
	//saveSlices(cluster_list_with_inclination[0], argv, 0);
	int numViewsAxisX = 5;
	int numViewsAxisY = 10;
	int view = 1;
	float angle_grad = 60;
	float angle_rad = ( (M_PI*angle_grad) / 180);
	for (float angleY = -(angle_rad); angleY < - angle_rad + angle_rad*2; angleY = angleY + ( (2*angle_rad)/numViewsAxisY ) )
	{
		for (float angleX = (angle_rad); angleX < angle_rad + angle_rad*2; angleX = angleX + ( (2*angle_rad)/numViewsAxisX ) )
		{
			//Then cloud (that is parallel cloud to the surface) is reorientaded and stored in cloud_tmp
			cloud_tmp = changeOrientation(cloud, argv, view, angleX, angleY, 0);
			//slices of the cloud_tmp (reorientaded are computed)
			//saveSlices(cloud_tmp, argv, view);
			view++;
		}
	}
	if ( view < 100 )
			logfile.add("Less than 100 views saved.\n");

	return (0);
}


int main(int argc, char** argv)
{
	//net 0 = MLP
	//net 1 = CNN
	
	int nObjects = 3;
	for (unsigned j = 0; j < nObjects; j++)
	{
		msg_pos.x.push_back(-1); 
		msg_pos.y.push_back(-1); 
		msg_pos.z.push_back(-1); 
		msg_pos.id.push_back(-1); 
	}
	
	set_conio_terminal_mode();

	if (console::find_argument(argc, argv, "-h") >= 0)
	{
		printUsage(argv[0]);
		return 0;
	} 
	else if (console::find_argument(argc, argv, "-on") >= 0)
	{
		if (console::find_argument(argc, argv, "-mlp") >= 0)
		{
			onlineProcedure(argc, argv, 0);
		}
		else if (console::find_argument(argc, argv, "-v") >= 0)
		{
			onlineProcedure(argc, argv, 0);
		}
		else if (console::find_argument(argc, argv, "-cnn") >= 0)
		{
			onlineProcedure(argc, argv, 1);
		}
	}
	else if (console::find_argument(argc, argv, "-off") >= 0)
	{
		if (console::find_argument(argc, argv, "-mlp") >= 0)
		{
			offlineProcedureMLP(argv);
		}
		else if (console::find_argument(argc, argv, "-cnn") >= 0)
		{
			offlineProcedureCNN(argv);
		}
	}
	return (0);
}
