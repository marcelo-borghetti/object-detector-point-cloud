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
//#include "neural_pkg/StartGestureRecognition.h"
#include "neural_pkg/EmptyService.h"
#include "neural_pkg/PostureRequestService.h"
//#include "neural_pkg/XYZObjects.h"

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
//neural_pkg::XYZObjects msg_pos;
int global_argc;
char** global_argv;

bool startFlag = false;
bool flagRemoveFiles = false;
boost::mutex g_pages_mutex;
boost::mutex text_mutex;
boost::mutex remove_text_mutex;

//the thread function
void *connection_handler(void *);
void *onlineProcedure(void *);

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
	//text_mutex.unlock();
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
              		 subdir << "/images_gestures/";
              		 std::stringstream ss;
              		 ss << directory << subdir.str();
					 //imageFilesSaved = count_files(ss.str(), ".jpg");
            		 ss << "inputRBG" << imageFilesSaved << ".jpg";
             		 string filename = ss.str();
           	 		 imageFilesSaved++;
          			 //reset_terminal_mode();
         			 printf("\t[Saved %s]\n", filename.c_str());
         			 //set_conio_terminal_mode();
             		 cv::imwrite(filename, frame);

               		 saveImage = false;
             	 }
         }
         else
         { printf(" --(!) No captured frame -- Break!"); }

}

bool EmptyService(neural_pkg::EmptyService::Request  &req,
         	 	   neural_pkg::EmptyService::Response &res)
{
	startFlag = true;	
	flagRemoveFiles = true;
	//res.x.push_back(3);
  
    return true;
}


//void *onlineProcedure(int argc, char** argv, int net)
void *onlineProcedure(void *)
{
	struct termios orig_termios;
	bool justVisualize(false);
	string filename;
	char *outFilename;
	char *directory;

	//ros::init(global_argc, global_argv, "Gesture_node");
	
	ofstream outFile;
    if (console::find_argument(global_argc, global_argv, "-v") >= 0)
	{
		if (global_argc != 4)
		{
			printUsage(global_argv[0]);
			return 0;
		}

		filename = global_argv[3];
		justVisualize = true;
	}
	else if (console::find_argument(global_argc, global_argv, "-mlp") >= 0)
	{
		if (global_argc != 5)
		{
			printUsage(global_argv[0]);
			return 0;
		}
		outFilename = global_argv[3];
		directory = global_argv[4]; //output directory to store the results
	}
	else if (console::find_argument(global_argc, global_argv, "-cnn") >= 0)
	{
		if (global_argc != 5)
		{
			printUsage(global_argv[0]);
			return 0;
		}
		outFilename = global_argv[3];
		directory = global_argv[4]; //output directory to store the results
	}
	outFile.open (outFilename, ios::out | ios::app | ios::binary);
	outFile.close();

	//reset_terminal_mode();
	OpenniGrabber = new OpenNIGrabber();
	//set_conio_terminal_mode();

	//reset_terminal_mode();
	boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&)> fi =
			boost::bind (&image_cb_, _1, directory);
	OpenniGrabber->registerCallback (fi);
	//set_conio_terminal_mode();
	
	OpenniGrabber->start();
	// Main loop.
	char c;
	int count_images = 0;
	//while (! viewer->wasStopped())
	//while ( count_images < 3 )
	
	ros::NodeHandle n2;
    ros::ServiceClient client = n2.serviceClient<neural_pkg::PostureRequestService>("HandPostureService");
    neural_pkg::PostureRequestService srv;
	
	std::stringstream subdir;
	subdir << "/images_gestures/";
	std::stringstream ss;
	ss << directory << subdir.str();
	int n = 20;
		
	while ( 1 )
	{
		if ( startFlag )
		{
			//if ( (count_files(ss.str(), ".jpg") == n ) )
			if ( flagRemoveFiles )
			{
				flagRemoveFiles = false;
				imageFilesSaved = 0;
				boost::filesystem::remove_all(ss.str());
				boost::filesystem::path dir(ss.str());
				boost::filesystem::create_directory(dir);
			}

			count_images = count_files(ss.str(), ".jpg");
		
			if (count_images < n )
			{
				/*if ( (count_files(ss.str(), ".jpg") == 0 ) )
				{
					imageFilesSaved = 0;
				}*/
				c = ' ';	

				if (c == ' ' )
				{
					saveImage = true;
					saveCloud = true;
				}
				
			}
			else
			{
				startFlag = false;
				printf("\n");
			        client.call(srv);
				// I have to call the Finish Service here!		
			}
		}
	}

	if (! justVisualize)
	{
		OpenniGrabber->stop();
	}
    outFile.close();
}


int main(int argc, char** argv)
{
	global_argc = argc;
	global_argv = argv;
	
	//*********************************************************
	// CODE FOR SOCKETS - THREADS
	//*********************************************************
	ros::init(argc, argv, "Gesture_node");
	ros::NodeHandle n;
	ros::ServiceServer service = n.advertiseService("StartGestureRecognition", EmptyService);
	
	pthread_t thread_id;
    //if ( pthread_create( &thread_id , NULL ,  connection_handler, NULL ) < 0)
	int net = 0;
	if ( pthread_create( &thread_id , NULL ,  onlineProcedure, NULL) < 0)
    {
        perror("could not create thread");
	    return 1;
	}
	//*********************************************************
	// END CODE FOR SOCKETS - THREADS
	//*********************************************************	
    //onlineProcedure(argc, argv, 0); // we should put this o a thread
	ros::spin();
	return (0);
}
