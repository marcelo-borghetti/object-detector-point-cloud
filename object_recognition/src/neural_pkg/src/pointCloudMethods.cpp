#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <time.h>
#include <stdio.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/vfh.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <boost/filesystem.hpp>
#include "../include/mlp_pkg/logsFile.h"
#include "../include/mlp_pkg/pointCloudMethods.h"

using namespace std; 

void pointCloudMethods::createVoxels(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered)
{
	  pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	  vg.setInputCloud (cloud);
	  vg.setLeafSize (0.01f, 0.01f, 0.01f);
	  //vg.setLeafSize (0.002f, 0.002f, 0.002f);
	  vg.filter (*cloud_filtered);
	  //std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*
}


pcl::ModelCoefficients::Ptr pointCloudMethods::planeSegmentation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table, char* dir, LogsFile& logfile, bool &recordFlag)
{
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZRGBA>);

	  // Create the segmentation object for the planar model and set all the parameters
	  pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
	  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	  seg.setOptimizeCoefficients (true);

	  seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
	  seg.setMethodType (pcl::SAC_RANSAC);
	  seg.setMaxIterations (100);
	  seg.setDistanceThreshold (0.02);

	  int i=0, nr_points = (int) cloud->points.size ();
	  int count_plane = 0;
	  int max_planes = 1;
	  double minX = -0.10, maxX = 0.05;
	  double minY = -0.96, maxY = 0.90;
	  double minZ = -0.70, maxZ = 0.80;

	  while ( ( count_plane < max_planes ) ) //&& (cloud_filtered->points.size () > 0.6 * nr_points) )
	  {
	     // Segment the largest planar component from the remaining cloud
	     seg.setInputCloud (cloud);
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
	     extract.setInputCloud (cloud);
	     extract.setIndices (inliers);
	     extract.setNegative (false);

	     if   ( ( coefficients->values[0] > minX ) && ( coefficients->values[0] < maxX ) &&
	    	    ( coefficients->values[1] > minY ) && ( coefficients->values[1] < maxY ) &&
	    	    ( coefficients->values[2] > minZ ) && ( coefficients->values[2] < maxZ ) )
	     {
			 for ( unsigned int i = 0; i < inliers->indices.size(); i++ )
			 {
				 plot->points[inliers->indices[i]].r = 0;
				 plot->points[inliers->indices[i]].b = 255;
				 plot->points[inliers->indices[i]].g = 0;
			 }
	     }

	     // Get the points associated with the planar surface
	     extract.filter (*cloud_table);

	     // Remove the planar inliers, extract the rest
	     extract.setNegative (true);
	     extract.filter (*cloud_f);
	     *cloud = *cloud_f;
	     count_plane++;
	  }
	  // saves the main point cloud without the biggest plan surface
	  if ( recordFlag )
	  {
		  std::stringstream subdir;
		  subdir << "/ransac/";
		  std::stringstream ss;
		  ss << dir << subdir.str();
		  logfile.createDir(ss.str());
		  ss << "/pc_ransac.pcd";
		  pcl::io::savePCDFileASCII (ss.str(), *cloud);

		  // saves the main point cloud with the biggest plan surface hightlighted in different color
		  std::stringstream ss2;
		  ss2 << dir << subdir.str();
		  logfile.createDir(ss2.str());
		  ss2 << "/pc_ransac_plot.pcd";
		  pcl::io::savePCDFileASCII (ss2.str(), *plot);
	  }

	  return coefficients;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloudMethods::segmentClusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table, pcl::ModelCoefficients::Ptr coefficients, char* dir, LogsFile& logfile, bool &recordFlag)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot_aux (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot_with_horizontal_obj (new pcl::PointCloud<pcl::PointXYZRGBA>);


    // 4. Rotate scene s.t. mugs are oriented upright
	double a = abs(coefficients->values[0]);
	double b = abs(coefficients->values[1]);
	double c = abs(coefficients->values[2]);
	double d = abs(coefficients->values[3]);

	Eigen::Vector3f x_axis (b / sqrt (a * a + b * b), -a / sqrt (a * a + b * b), 0.);
    Eigen::Vector3f y_direction (a, b, c);
    Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
    rotation = pcl::getTransFromUnitVectorsXY (x_axis, y_direction);

    float roll,pitch,yaw;
    pcl::getEulerAngles(rotation,roll,pitch,yaw);

    transformPointCloud(*plot, *plot, rotation);
    transformPointCloud(*cloud_table, *cloud_table, rotation);

    //double x_mean = 0, z_mean = 0, delta_x = .40, delta_z = 0.25; // THIS VALUES ARE SUPPOSED TO BE USED WITH OLD DATABASE-1
    double x_mean = 0, z_mean = 0, delta_x = .30, delta_z = 0.45; // THIS VALUES ARE SUPPOSED TO BE USED WITH NEW DATABASE-2 AND DATABASE-3
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

	for ( unsigned int i = 0; i < cloud_table->points.size(); i++ )
	{
		//if ( ( ( plot->points[i].x > x_mean - delta_x ) && ( plot->points[i].x < x_mean + delta_x ) ) &&
		//     ( ( plot->points[i].z > z_mean - delta_z ) && ( plot->points[i].z < z_mean + delta_z ) ) )
		// VC rodou a vida toda com o if de cima.. esse if novo abaixo está sob observação. Eu não entendo como o de cima funcionava.
		if ( ( ( cloud_table->points[i].x > x_mean - delta_x ) && ( cloud_table->points[i].x < x_mean + delta_x ) ) &&
		     ( ( cloud_table->points[i].z > z_mean - delta_z ) && ( cloud_table->points[i].z < z_mean + delta_z ) ) )
		{
			if ( yf_min > cloud_table->points[i].y )
				yf_min = cloud_table->points[i].y;
			if ( yf_max < cloud_table->points[i].y )
				yf_max = cloud_table->points[i].y;
		}
	}

	for ( unsigned int i = 0; i < plot->points.size(); i++ )
	{
	   if ( //( plot->points[i].y > yf_min ) && ( plot->points[i].b == 255 ) ||
			( plot->points[i].y < yf_min + 10 ) &&
			//( plot->points[i].y < yf_min ) && // this line is used with "plot" variable, but the method does not work properly
			( ( plot->points[i].x > x_mean - delta_x ) && ( plot->points[i].x < x_mean + delta_x ) ) &&
			( ( plot->points[i].z > z_mean - delta_z) && ( plot->points[i].z < z_mean + delta_z ) ) )
       {
    	   //plot->points[i].r = 255;
    	   //plot->points[i].b = 0;
    	   //plot->points[i].g = 0;
		   plot_aux->points.push_back (plot->points[i]);
	   }
	}

	plot_aux->width = plot_aux->points.size ();
	plot_aux->height = 1;
	plot_aux->is_dense = true;

	pcl::copyPointCloud(*plot_aux, *plot_with_horizontal_obj);

	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	transform_2.rotate (Eigen::AngleAxisf (-roll, Eigen::Vector3f::UnitX()));
	//transform_2.rotate (Eigen::AngleAxisf (pitch, Eigen::Vector3f::UnitX()));
	//transform_2.rotate (Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitX()));
    transformPointCloud(*plot_aux, *plot_aux, transform_2);

	if ( recordFlag )
	{
		// saves the objects that are inside the main sphere of segmentation (as only one cluster)
		std::stringstream subdir;
		subdir << "/ransac/";
		std::stringstream ss;
		ss << dir << subdir.str();
		logfile.createDir(ss.str());
		ss << "/pc_segmented.pcd";
		pcl::io::savePCDFileASCII (ss.str(), *plot_aux);
	}

	//cout << "Size plot_aux:" << plot_aux->points.size() << "\n";
	//cout << "Size plot:" << plot->points.size() << "\n";
	pcl::copyPointCloud(*plot_aux, *plot);

	return plot_with_horizontal_obj;
}

void pointCloudMethods::selectCluster(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::vector<pcl::PointIndices> cluster_indices, int indexCluster)
{
	   //for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	   //{
			std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin () + indexCluster;
		    //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
	        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
	        {
	           //clusterFile << cloud_filtered2->points[*pit].x << ", " << cloud_filtered2->points[*pit].y << ", " << cloud_filtered2->points[*pit].z << "\n";
	           cloud->points[*pit].r = 255;
	           cloud->points[*pit].g = 0;
	           cloud->points[*pit].b = 0;
	        }
	  //}
}


std::vector<pcl::PointIndices> pointCloudMethods::findClustersIndices(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2)
{
	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA>);
	tree->setInputCloud (cloud_filtered2);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
	ec.setClusterTolerance (0.05); // 2cm
	ec.setMinClusterSize (50);
	ec.setMaxClusterSize (50000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (cloud_filtered2);
	ec.extract (cluster_indices);

	return cluster_indices;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> pointCloudMethods::findClusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered2, 	std::vector<pcl::PointIndices> &cluster_indices, char* dir, LogsFile& logfile, bool &recordFlag)
{
	   cluster_indices = findClustersIndices(cloud_filtered2);
	   std:vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr>	cluster_list;
	   int j = 0;
	   //cout << "Cluster size:" << cluster_indices.size() << "\n";
	   for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	   {
	    	std::stringstream file;
		    //file << "cloud_cluster_" << j << ".txt";
		    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
	        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
	        {
	           cloud_cluster->points.push_back (cloud_filtered2->points[*pit]);
	        }
	        cloud_cluster->width = cloud_cluster->points.size ();
	        cloud_cluster->height = 1;
	        cloud_cluster->is_dense = true;

	        if ( recordFlag )
	        {
				//std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
				std::stringstream subdir;
				subdir << "/clusters/";
				std::stringstream ss;
				ss << dir << subdir.str();
				logfile.createDir(ss.str());
				ss << "cloud_cluster_" << j << ".pcd";
				pcl::PCDWriter writer;
				writer.write<pcl::PointXYZRGBA> (ss.str (), *cloud_cluster, false);
	        }
			j++;
			cluster_list.push_back(cloud_cluster);
	  }
  	  //recordFlag = false;
      return cluster_list;
}
/*void pointCloudMethods::findDescriptor(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, ofstream& outFile)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the FPFH descriptors for each point.
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());

	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	// FPFH estimation object.
	pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud(cloud);
	fpfh.setInputNormals(normals);
	fpfh.setSearchMethod(kdtree);
	// Search radius, to look for neighbors. Note: the value given here has to be
	// larger than the radius used to estimate the normals.
	fpfh.setRadiusSearch(0.05);

    fpfh.compute(*descriptors);

	outFile << cloud->points.size() << " ";
	for (unsigned int j = 0; j < cloud->points.size(); j++)
	{
		for (unsigned int i = 0; i < 33; i++)
		{
			outFile << descriptors->points[j].histogram[i] << " ";
		}
	}
	outFile << "\n";
 }*/

vector<float> pointCloudMethods::findDescriptorOURCVFH(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object, ofstream& outFile)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr objectXYZ (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*object, *objectXYZ);

	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the VFH descriptor.
	pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(objectXYZ);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

/*
	// VFH estimation object.
	pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(object);
	vfh.setInputNormals(normals);
	vfh.setSearchMethod(kdtree);
	// Optionally, we can normalize the bins of the resulting histogram,
	// using the total number of points.
	vfh.setNormalizeBins(true);
	// Also, we can normalize the SDC with the maximum size found between
	// the centroid and any of the cluster's points.
	vfh.setNormalizeDistance(false);
	// Compute the features
	vfh.compute(*descriptors);
*/
	// OUR-CVFH estimation object.
	pcl::OURCVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> ourcvfh;
	ourcvfh.setInputCloud(objectXYZ);
	ourcvfh.setInputNormals(normals);
	ourcvfh.setSearchMethod(kdtree);
	ourcvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
	ourcvfh.setCurvatureThreshold(1.0);
	ourcvfh.setNormalizeBins(false);
	// Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
	// this will decide if additional Reference Frames need to be created, if ambiguous.
	ourcvfh.setAxisRatio(0.03);
	// Compute the features
	ourcvfh.compute(*descriptors);

	outFile << objectXYZ->points.size() << " ";
	vector<float> data;
	//for (unsigned int j = 0; j < object->points.size(); j++)
	{
		for (unsigned int i = 0; i < 308; i++)       // This loop is very time consuming
		{
			outFile << descriptors->points[0].histogram[i] << " ";
			data.push_back(descriptors->points[0].histogram[i]);
		}
	}
	outFile << "\n";

	return data;
}

vector<float> pointCloudMethods::findDescriptorVFH(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object, ofstream& outFile)
{
	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the VFH descriptor.
	pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(object);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	// VFH estimation object.
	pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(object);
	vfh.setInputNormals(normals);
	vfh.setSearchMethod(kdtree);
	// Optionally, we can normalize the bins of the resulting histogram,
	// using the total number of points.
	vfh.setNormalizeBins(true);
	// Also, we can normalize the SDC with the maximum size found between
	// the centroid and any of the cluster's points.
	vfh.setNormalizeDistance(false);
	// Compute the features
	vfh.compute(*descriptors);

	outFile << object->points.size() << " ";
	vector<float> data;
	//for (unsigned int j = 0; j < object->points.size(); j++)
	{
		for (unsigned int i = 0; i < 308; i++)       // This loop is very time consuming
		{
			outFile << descriptors->points[0].histogram[i] << " ";
			data.push_back(descriptors->points[0].histogram[i]);
		}
	}
	outFile << "\n";

	return data;
}

vector<float> pointCloudMethods::findDescriptorVFHFromSelectedClusterInTheFullPC(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::vector<pcl::PointIndices> cluster_indices, int indexCluster)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object (new pcl::PointCloud<pcl::PointXYZRGBA>);
    std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin () + indexCluster;
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
    {
        object->points.push_back (cloud->points[*pit]);
    }

	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the VFH descriptor.
	pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(object);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	// VFH estimation object.
	pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(object);
	vfh.setInputNormals(normals);
	vfh.setSearchMethod(kdtree);
	// Optionally, we can normalize the bins of the resulting histogram,
	// using the total number of points.
	vfh.setNormalizeBins(true);
	// Also, we can normalize the SDC with the maximum size found between
	// the centroid and any of the cluster's points.
	vfh.setNormalizeDistance(false);

	// Compute the features
	vfh.compute(*descriptors);

	vector<float> data;
	for (unsigned int i = 0; i < 308; i++)       // This loop is very time consuming
	{
		data.push_back(descriptors->points[0].histogram[i]);
	}

	return data;
}

void pointCloudMethods::injectNoise(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
	//initialize random seed:
	srand (time(NULL));

	double ratio = 0.7;
	//cout << "number" << ratio*cloud->points.size() << "\n";
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_noisy_cloud;
	pcl::copyPointCloud(*cloud, *rotated_noisy_cloud);
	for (unsigned int j = 0; j < ratio*cloud->points.size(); j++)
	{
		//generate secret number
		int index = rand() % ( cloud->points.size() - 1 ) + 1;

		float dx = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
		float dy = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
		float dz = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
		printf("dx = %.2f, dy = %.2f,dz = %.2f\n", dx, dy, dz);
		cloud->points[index].x = cloud->points[index].x + dx;
		cloud->points[index].y = cloud->points[index].y + dy;
		cloud->points[index].z = cloud->points[index].z + dz;
	}

}

void pointCloudMethods::applyRotationAndTranslation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, char* dir, ofstream& outFile, LogsFile& logfile)
{
    std::stringstream subdir;
    subdir << "/transformed_cloud/";
    std::stringstream ss;
    ss << dir << subdir.str();
    logfile.createDir(ss.str());

	float x = 0;
	float y = 0;
    float z = 0;
	Eigen::Affine3f t;
	pcl::PointCloud<pcl::PointXYZRGBA> rotated_cloud;
	int i = 1;
	for (float roll = 0.7; roll < 2.5; roll = roll + 0.7 )
	//for (float roll = 0.7; roll < 1; roll = roll + 0.7 )
	{
		for (float pitch = 0.7; pitch < 2.5; pitch = pitch + 0.7 )
		//for (float pitch = 0.7; pitch < 1.0; pitch = pitch + 0.7 )
		{
			for (float yaw = 0.7; yaw < 2.5; yaw = yaw + 0.7 )
			//for (float yaw = 0.7; yaw < 1.0; yaw = yaw + 0.7 )
			{
				pcl::getTransformation(x, y, z, roll, pitch, yaw, t);
				pcl::transformPointCloud(*cloud, rotated_cloud, t);

				//initialize random seed:
				srand (time(NULL));
				double ratio = 0.1;
				double max = 0.01;
				double min = -0.01;
				//cout << "number" << ratio*cloud->points.size() << "\n";
				for (unsigned int j = 0; j < ratio*rotated_cloud.points.size(); j++)
				{
					//generate secret number
					int index = rand() % ( rotated_cloud.points.size() - 1 ) + 1;

					float dx = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX)) * (max -min)) + min;
					float dy = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX)) * (max -min)) + min;
					float dz = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX)) * (max -min)) + min;
					//printf("dx = %.2f, dy = %.2f,dz = %.2f\n", dx, dy, dz);
					rotated_cloud.points[index].x = rotated_cloud.points[index].x + dx;
					rotated_cloud.points[index].y = rotated_cloud.points[index].y + dy;
					rotated_cloud.points[index].z = rotated_cloud.points[index].z + dz;
				}
				//findDescriptorVFH(rotated_cloud.makeShared(), outFile);
				findDescriptorOURCVFH(rotated_cloud.makeShared(), outFile);
				ss.str("");
				ss << dir << subdir.str();
				ss << "/rotated_cloud_cluster_" << i << ".pcd";
				pcl::PCDWriter writer;
				writer.write<pcl::PointXYZRGBA> (ss.str (), rotated_cloud, false);
				i++;
			}
		}
	}
}

std::string pointCloudMethods::buildPath(char** argv, int view, int idSlice)
{
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
	std::string objects = "images";


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
	baseDir << path << "/" << objects << "/" << label << "/" << file << "_view-" << view << "/";
	boost::filesystem::path dir3(baseDir.str());
	if (boost::filesystem::create_directory(dir3)) {
		//std::cout << "Success" << "\n";
	}
	std::stringstream stream;

	stream << baseDir.str() << "/" << file << "_" << idSlice << ".png";

	return stream.str();
}


void pointCloudMethods::convertPointCloudToPNG(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized, std::string file)
{
	cloud_organized->clear();
	cloud_organized->points.clear();

	cloud_organized->width = 50;
    cloud_organized->height = 50;
	//cloud_organized->width = 640;
    //cloud_organized->height = 480;
	cloud_organized->is_dense = false;
	pcl::PointXYZRGBA p;

	for (size_t i = 0; ( i < cloud_organized->height ) ; i++)
	{
		for (size_t j = 0; ( j < cloud_organized->width ) ; j++)
		{
			p.x = j;
			p.y = i;
			p.z = 0;
			cloud_organized->points.push_back (p);
		}
	}
	//std::cerr << "control point 1: " << std::endl;


    double maxX = -100; double minX = 100; double maxY = -100; double minY = 100;
	for (size_t i = 0; ( i < cloud->points.size() ) ; ++i)
	{
		//if ( ( cloud->points[i].r > 0 ) || ( cloud->points[i].g > 0 ) || ( cloud->points[i].b > 0 ) )
		{
			if ( cloud->points[i].x > maxX )
				maxX = cloud->points[i].x;
			if ( cloud->points[i].x < minX )
				minX = cloud->points[i].x;

			if ( cloud->points[i].y > maxY )
				maxY = cloud->points[i].y;
			if ( cloud->points[i].y < minY )
				minY = cloud->points[i].y;
		}
	}
	//std::cerr << "control point 2: " << std::endl;

	//double maxX = 0.65; double minX = -0.65; double maxY = 0.65; double minY = -0.65;
	//double maxX = 640; double minX = 0; double maxY = 480; double minY = 0;

	//std::cerr << "minX=" << minX << ", maxX=" << maxX << ", minY=" << minY << ", maxY=" << maxY << ", points.size()=" << cloud->points.size() << std::endl;
	double newX;
	double newY;
	for (size_t i = 0; ( i < cloud->points.size() ) ; i++)
	{
	    newX = ( cloud_organized->width * ( cloud->points[i].x - minX ) ) / ( maxX  - minX );
	    newY = ( cloud_organized->height * ( cloud->points[i].y - minY ) ) / ( maxY  - minY );
		//std::cerr << "newX=" << newX << ", newY=" << newY << std::endl;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].r = cloud->points[i].r;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].g = cloud->points[i].g;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].b = cloud->points[i].b;
	}

	pcl::io::savePNGFile(file, *cloud_organized);
}

void pointCloudMethods::convertPointCloudToPNG2(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized, std::string file)
{
	cloud_organized->clear();
	cloud_organized->points.clear();

	cloud_organized->width = 100;
    cloud_organized->height = 100;
	//cloud_organized->width = 640;
    //cloud_organized->height = 480;
	cloud_organized->is_dense = false;
	pcl::PointXYZRGBA p;

	for (size_t i = 0; ( i < cloud_organized->height ) ; i++)
	{
		for (size_t j = 0; ( j < cloud_organized->width ) ; j++)
		{
			p.x = j;
			p.y = i;
			p.z = 0;
			cloud_organized->points.push_back (p);
		}
	}
	//std::cerr << "control point 1: " << std::endl;


    double maxX = -100; double minX = 100; double maxY = -100; double minY = 100;
	for (size_t i = 0; ( i < cloud->points.size() ) ; ++i)
	{
		//if ( ( cloud->points[i].r > 0 ) || ( cloud->points[i].g > 0 ) || ( cloud->points[i].b > 0 ) )
		{
			if ( cloud->points[i].x > maxX )
				maxX = cloud->points[i].x;
			if ( cloud->points[i].x < minX )
				minX = cloud->points[i].x;

			if ( cloud->points[i].y > maxY )
				maxY = cloud->points[i].y;
			if ( cloud->points[i].y < minY )
				minY = cloud->points[i].y;
		}
	}

	//double maxX = 0.65; double minX = -0.65; double maxY = 0.65; double minY = -0.65;
	double delta_x = ( maxX - minX );
	double delta_y = ( maxY - minY );
	//double maxX = 640; double minX = 0; double maxY = 480; double minY = 0;
    //cout << "delta_x, delta_Y" << delta_x << ", " << delta_y;
    double newX;
	double newY;
	//double newWidth = 50;
	double newWidth = delta_x*300;
	double newHeight = ( newWidth * delta_y) / delta_x;
	for (size_t i = 0; ( i < cloud->points.size() ) ; i++)
	{
	    //newX = ( cloud_organized->width * ( cloud->points[i].x - minX ) ) / ( maxX  - minX );
	    //newY = ( cloud_organized->height * ( cloud->points[i].y - minY ) ) / ( maxY  - minY );
	    newX = ( newWidth * ( cloud->points[i].x - minX ) ) / ( delta_x );
	    newY = ( newHeight * ( cloud->points[i].y - minY ) ) / ( delta_y );
		newX = newX + ( cloud_organized->width/2 - newWidth/2 );
		newY = newY + ( cloud_organized->height/2 - newHeight/2 );
		if ( newX < 0 ) newX = 0;
		if ( newY < 0 ) newY = 0;
		if ( newX >= cloud_organized->width) newX = cloud_organized->width;
		if ( newY >= cloud_organized->height) newY = cloud_organized->height;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].r = cloud->points[i].r;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].g = cloud->points[i].g;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].b = cloud->points[i].b;
	}
	pcl::io::savePNGFile(file, *cloud_organized); //I dont know if it is possible to save this
}

void pointCloudMethods::getSlicesFromPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_slice, double z, double delta)
{
	for (size_t i = 0; i < cloud->points.size(); i++)
	{
		if ( ( cloud->points[i].z ) > ( z - delta ) &&
			 ( cloud->points[i].z ) < ( z + delta ) )
			  cloud_slice->points.push_back (cloud->points[i]);
	}
}

void pointCloudMethods::saveAsPNG(std::string path, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized (new pcl::PointCloud<pcl::PointXYZRGBA>);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	coefficients->values.resize (4);
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;

	pcl::ProjectInliers<pcl::PointXYZRGBA> proj;
	proj.setModelType (pcl::SACMODEL_PLANE);
	//proj.setInputCloud (cloud);
	proj.setInputCloud (cloud);
	//std::cout << "cloud slice size = " << cloud_slice->points.size() << "\n";
	proj.setModelCoefficients (coefficients);
	proj.filter (*cloud_projected);

	cloud_organized->clear();
	cloud_organized->points.clear();

	cloud_organized->width = 50;
    cloud_organized->height = 50;
	//cloud_organized->width = 640;
    //cloud_organized->height = 480;
	cloud_organized->is_dense = false;
	pcl::PointXYZRGBA p;

	for (size_t i = 0; ( i < cloud_organized->height ) ; i++)
	{
		for (size_t j = 0; ( j < cloud_organized->width ) ; j++)
		{
			p.x = j;
			p.y = i;
			p.z = 0;
			cloud_organized->points.push_back (p);
		}
	}

    double maxX = -100; double minX = 100; double maxY = -100; double minY = 100;
	for (size_t i = 0; ( i < cloud->points.size() ) ; ++i)
	{
		//if ( ( cloud->points[i].r > 0 ) || ( cloud->points[i].g > 0 ) || ( cloud->points[i].b > 0 ) )
		{
			if ( cloud->points[i].x > maxX )
				maxX = cloud->points[i].x;
			if ( cloud->points[i].x < minX )
				minX = cloud->points[i].x;

			if ( cloud->points[i].y > maxY )
				maxY = cloud->points[i].y;
			if ( cloud->points[i].y < minY )
				minY = cloud->points[i].y;
		}
	}

	//double maxX = 0.65; double minX = -0.65; double maxY = 0.65; double minY = -0.65;
	//double maxX = 640; double minX = 0; double maxY = 480; double minY = 0;

	double newX;
	double newY;
	for (size_t i = 0; ( i < cloud->points.size() ) ; i++)
	{
	    newX = ( cloud_organized->width * ( cloud->points[i].x - minX ) ) / ( maxX  - minX );
	    newY = ( cloud_organized->height * ( cloud->points[i].y - minY ) ) / ( maxY  - minY );
	    //cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX] = cloud->points[i];
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].r = cloud->points[i].r;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].g = cloud->points[i].g;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].b = cloud->points[i].b;
	}
	pcl::io::savePNGFile(path, *cloud_organized); //I dont know if it is possible to save this
}

void pointCloudMethods::saveAsPNG2(std::string path, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized (new pcl::PointCloud<pcl::PointXYZRGBA>);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
	coefficients->values.resize (4);
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;

	pcl::ProjectInliers<pcl::PointXYZRGBA> proj;
	proj.setModelType (pcl::SACMODEL_PLANE);
	//proj.setInputCloud (cloud);
	proj.setInputCloud (cloud);
	//std::cout << "cloud slice size = " << cloud_slice->points.size() << "\n";
	proj.setModelCoefficients (coefficients);
	proj.filter (*cloud_projected);

	cloud_organized->clear();
	cloud_organized->points.clear();

	cloud_organized->width = 100;
    cloud_organized->height = 100;
	cloud_organized->is_dense = false;
	pcl::PointXYZRGBA p;

	for (size_t i = 0; ( i < cloud_organized->height ) ; i++)
	{
		for (size_t j = 0; ( j < cloud_organized->width ) ; j++)
		{
			p.x = j;
			p.y = i;
			p.z = 0;
			cloud_organized->points.push_back (p);
		}
	}

    double maxX = -100; double minX = 100; double maxY = -100; double minY = 100;
	for (size_t i = 0; ( i < cloud->points.size() ) ; ++i)
	{
		//if ( ( cloud->points[i].r > 0 ) || ( cloud->points[i].g > 0 ) || ( cloud->points[i].b > 0 ) )
		{
			if ( cloud->points[i].x > maxX )
				maxX = cloud->points[i].x;
			if ( cloud->points[i].x < minX )
				minX = cloud->points[i].x;

			if ( cloud->points[i].y > maxY )
				maxY = cloud->points[i].y;
			if ( cloud->points[i].y < minY )
				minY = cloud->points[i].y;
		}
	}
	//double maxX = 0.65; double minX = -0.65; double maxY = 0.65; double minY = -0.65;
	double delta_x = ( maxX - minX );
	double delta_y = ( maxY - minY );
	//double maxX = 640; double minX = 0; double maxY = 480; double minY = 0;
    //cout << "delta_x, delta_Y" << delta_x << ", " << delta_y;
    double newX;
	double newY;
	//double newWidth = 50;
	//double newWidth = delta_x*150;
	double newWidth = delta_x*300;
	double newHeight = ( newWidth * delta_y) / delta_x;
	for (size_t i = 0; ( i < cloud->points.size() ) ; i++)
	{
	    //newX = ( cloud_organized->width * ( cloud->points[i].x - minX ) ) / ( maxX  - minX );
	    //newY = ( cloud_organized->height * ( cloud->points[i].y - minY ) ) / ( maxY  - minY );
	    newX = ( newWidth * ( cloud->points[i].x - minX ) ) / ( delta_x );
	    newY = ( newHeight * ( cloud->points[i].y - minY ) ) / ( delta_y );
		newX = newX + ( cloud_organized->width/2 - newWidth/2 );
		newY = newY + ( cloud_organized->height/2 - newHeight/2 );
		if ( newX < 0 ) newX = 0;
		if ( newY < 0 ) newY = 0;
		if ( newX >= cloud_organized->width) newX = cloud_organized->width;
		if ( newY >= cloud_organized->height) newY = cloud_organized->height;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].r = cloud->points[i].r;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].g = cloud->points[i].g;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX].b = cloud->points[i].b;
	}
	pcl::io::savePNGFile(path, *cloud_organized); //I dont know if it is possible to save this
}
