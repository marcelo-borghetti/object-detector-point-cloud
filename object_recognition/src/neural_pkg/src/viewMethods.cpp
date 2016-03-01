#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
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

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32MultiArray.h"
//#include "../include/mlp_pkg/rosHandler.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../include/mlp_pkg/logsFile.h"
#include "../include/mlp_pkg/pointCloudMethods.h"
#include "../include/mlp_pkg/viewMethods.h"

using namespace pcl;

void viewMethods::segmentClustersView(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table, pcl::ModelCoefficients::Ptr coefficients)
{
    // 4. Rotate scene s.t. mugs are oriented upright
	double a = abs(coefficients->values[0]);
	double b = abs(coefficients->values[1]);
	double c = abs(coefficients->values[2]);
	double d = abs(coefficients->values[3]);

	Eigen::Vector3f x_axis (b / sqrt (a * a + b * b), -a / sqrt (a * a + b * b), 0.);
    Eigen::Vector3f y_direction (a, b, c);
    Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
    rotation = pcl::getTransFromUnitVectorsXY (x_axis, y_direction);

    transformPointCloud(*plot, *plot, rotation);
    transformPointCloud(*cloud_table, *cloud_table, rotation);

    double x_mean = 0, z_mean = 0, delta = 0.6;
    double yf_min = 100, yf_max = 0, xf_max = 0, xf_min = 1000, zf_max = 0, zf_min = 1000;
	for ( unsigned int i = 0; i < cloud_table->points.size(); i++ )
	{
		x_mean = x_mean + cloud_table->points[i].x;
		z_mean = z_mean + cloud_table->points[i].z;


		if ( xf_min > cloud_table->points[i].x )
			 xf_min = cloud_table->points[i].x;
		if ( xf_max < cloud_table->points[i].x )
			 xf_max = cloud_table->points[i].x;

		if ( yf_min > cloud_table->points[i].y )
			yf_min = cloud_table->points[i].y;
		if ( yf_max < cloud_table->points[i].y )
			 yf_max = cloud_table->points[i].y;

		if ( zf_min > cloud_table->points[i].z )
			 zf_min = cloud_table->points[i].z;
		if ( zf_max < cloud_table->points[i].z )
			 zf_max = cloud_table->points[i].z;
	}
	x_mean = x_mean / cloud_table->points.size();
	z_mean = z_mean / cloud_table->points.size();

	for ( unsigned int i = 0; i < plot->points.size(); i++ )
	{
	   if ( ( plot->points[i].y > yf_min ) && ( plot->points[i].b == 255 ) ||
			( plot->points[i].y > yf_min ) ||
			( ( plot->points[i].x < x_mean - delta ) || ( plot->points[i].x > x_mean + delta ) ) ||
			( ( plot->points[i].z < z_mean - delta) || ( plot->points[i].z > z_mean + delta ) ) )
       {
    	   plot->points[i].r = 0;
    	   plot->points[i].b = 0;
    	   plot->points[i].g = 0;
    	   plot->points[i].x = 0;
    	   plot->points[i].y = 0;
    	   plot->points[i].z = 0;
	   }
	}
}


pcl::ModelCoefficients::Ptr viewMethods::planeSegmentationView(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table)
{
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZRGBA>);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot_aux (new pcl::PointCloud<pcl::PointXYZRGBA>);

	  // Create the segmentation object for the planar model and set all the parameters
	  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
	  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	  seg.setOptimizeCoefficients (true);

	  seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
	  //seg.setModelType (pcl::SACMODEL_PLANE);
	  seg.setMethodType (pcl::SAC_RANSAC);
	  //seg.setMaxIterations (100); --> default
	  //seg.setDistanceThreshold (0.02); --> deafult
	  seg.setMaxIterations (100);
	  seg.setDistanceThreshold (0.02);

	  int i=0, nr_points = (int) cloud_filtered->points.size ();
	  int count_plane = 0;
	  int max_planes = 1;
	  double minX = -0.10, maxX = 0.05;
	  double minY = -0.96, maxY = 0.90;
	  double minZ = -0.70, maxZ = 0.80;

	  while ( ( count_plane < max_planes ) ) //&& (cloud_filtered->points.size () > 0.6 * nr_points) )
	  {
	     // Segment the largest planar component from the remaining cloud
	     seg.setInputCloud (cloud_filtered);
	 	 pcl::copyPointCloud(*cloud_filtered, *plot_aux);
	 	 //reset_terminal_mode();
	     seg.segment (*inliers, *coefficients);
	     //set_conio_terminal_mode();

	     if (inliers->indices.size () == 0)
	     {
	    	 //reset_terminal_mode();
	    	 std::cout << "Could not estimate a planar model for the given dataset.\n";
	    	 //set_conio_terminal_mode();
	    	 break;
	     }

	     // Extract the planar inliers from the input cloud
	     pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
	     extract.setInputCloud (cloud_filtered);
	     extract.setIndices (inliers);
	     extract.setNegative (false);

	     //std::cout << "Coeficientes: a=" << coefficients->values[0] << ", b=" << coefficients->values[1] << ", c=" << coefficients->values[2] << "\n";
	     //std::vector<double> yf;
	     //double y_mean ;
	     if   ( ( coefficients->values[0] > minX ) && ( coefficients->values[0] < maxX ) &&
	    	    ( coefficients->values[1] > minY ) && ( coefficients->values[1] < maxY ) &&
	    	    ( coefficients->values[2] > minZ ) && ( coefficients->values[2] < maxZ ) )
	     {
			 for ( unsigned int i = 0; i < inliers->indices.size(); i++ )
			 {
				 //plot_aux->points[inliers->indices[i]].r = 50 * (count_plane + 1);
				 //plot_aux->points[inliers->indices[i]].b = 0;
				 //plot_aux->points[inliers->indices[i]].g = 50 * (2 - count_plane);
				 plot_aux->points[inliers->indices[i]].r = 0;
				 plot_aux->points[inliers->indices[i]].b = 255;
				 plot_aux->points[inliers->indices[i]].g = 0;
				 plot->points.push_back(plot_aux->points[inliers->indices[i]]);
			 }
	     }

	     // Get the points associated with the planar surface
	     extract.filter (*cloud_plane);
	     if ( count_plane == 0 )
	    	 pcl::copyPointCloud(*cloud_plane, *cloud_table);

	     //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

	     // Remove the planar inliers, extract the rest
	     extract.setNegative (true);
	     extract.filter (*cloud_f);
	     *cloud_filtered = *cloud_f;
	     count_plane++;
	  }

	  return coefficients;
}

/*
void reset_terminal_mode()
{
    tcsetattr(0, TCSANOW, &orig_termios);
}

void set_conio_terminal_mode()
{
    struct termios new_termios;

     take two copies - one for now, one for later
    tcgetattr(0, &orig_termios);
    memcpy(&new_termios, &orig_termios, sizeof(new_termios));

     register cleanup handler, and set the new terminal mode
    atexit(reset_terminal_mode);
    cfmakeraw(&new_termios);
    tcsetattr(0, TCSANOW, &new_termios);
}

void viewMethods::image_cb_ (const boost::shared_ptr<openni_wrapper::Image> &image, char* directory)
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
            		 ss << "inputRBG" << imageFilesSaved << ".jpg";
             		 string filename = ss.str();
           	 		 imageFilesSaved++;
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


// This function is called every time the device has new data.
void viewMethods::grabberCallback(const PointCloud<PointXYZRGBA>::ConstPtr& cloud, char* directory, ofstream* outFile, LogsFile* logfile, ros::Publisher pub)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr result ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2 ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table ( new pcl::PointCloud<pcl::PointXYZRGBA>);

	pcl::copyPointCloud(*cloud, *cloud2);
	pcl::copyPointCloud(*cloud, *plot);
	//pcl::ModelCoefficients::Ptr coefficients = vMethods.planeSegmentationView(cloud2, plot, cloud_table);
	bool falseFlag = false;
	pcl::ModelCoefficients::Ptr coefficients = pcMethods.planeSegmentation(cloud2, plot, cloud_table, directory, (*logfile), falseFlag);

	if (segmentFlag)
	{
		//pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::copyPointCloud(*cloud2, *plot); //deletar essa linha
		pcMethods.segmentClusters(plot, cloud_table, coefficients, directory, (*logfile), falseFlag);

		// perform the clusterization of each object in the point cloud previously segmented - ideally it should find one object
	    std::vector<pcl::PointIndices> cluster_indices = pcMethods.findClustersIndices(plot);
	    std:vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> cluster_list;
	    numClusters = cluster_indices.size();
		pcMethods.selectCluster(plot, cluster_indices, indexCluster);

		if ( recordFlag )
	    {
	    	cluster_list = pcMethods.findClusters(plot, directory, (*logfile), recordFlag); //getSelectedCluster
			// find the descriptor for the point cloud
			reset_terminal_mode();
			cout << "\t" << "Saving descriptor...\n";
			set_conio_terminal_mode();
			(*outFile) << "online_capture" << " ";
			(*outFile) << cluster_list.size() << " ";
			pcMethods.findDescriptorVFH(cluster_list[indexCluster], (*outFile));
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
			ss << "inputCloud" << cloudFilesSaved << ".pcd";

			string filename = ss.str();
			//if (io::savePCDFile(filename, *cloud, true) == 0)
			if (io::savePCDFile(filename, *plot, true) == 0)
			{
				cloudFilesSaved++;
				reset_terminal_mode();
				printf("\t[Saved %s]\n", filename.c_str());
				set_conio_terminal_mode();
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
			data = pcMethods.findDescriptorVFHFromSelectedClusterInTheFullPC(plot, cluster_indices, indexCluster); //cluster_list[0]
			for (unsigned int i=0; i < data.size(); i++)
			{
				msg.data.push_back(data[i]);
			}
			for (unsigned int i=0; i < 500; i++)
			{
				pub.publish(msg);
			}
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
		ss << "inputCloud" << cloudFilesSaved << ".pcd";

		string filename = ss.str();
		//if (io::savePCDFile(filename, *cloud, true) == 0)
		if (io::savePCDFile(filename, *plot, true) == 0)
		{
			cloudFilesSaved++;
			reset_terminal_mode();
			printf("\t[Saved %s]\n", filename.c_str());
			set_conio_terminal_mode();
		}
		else PCL_ERROR("Problem saving %s.\n", filename.c_str());

		saveCloud = false;
	}
}
*/
