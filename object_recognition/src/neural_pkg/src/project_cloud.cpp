#include <iostream>
#include <string.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/png_io.h>

void convertPointCloudToPNG(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized, int counter, char** argv)
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

    /*double maxX = -100; double minX = 100; double maxY = -100; double minY = 100;
	for (size_t i = 0; ( i < cloud->points.size() ) ; ++i)
	{
	   if ( cloud->points[i].x > maxX )
		   maxX = cloud->points[i].x;
	   if ( cloud->points[i].x < minX )
		   minX = cloud->points[i].x;

	   if ( cloud->points[i].y > maxY )
		   maxY = cloud->points[i].y;
	   if ( cloud->points[i].y < minY )
		   minY = cloud->points[i].y;
	}*/
	double maxX = 0.65; double minX = -0.65; double maxY = 0.65; double minY = -0.65;

	double newX;
	double newY;
	for (size_t i = 0; ( i < cloud->points.size() ) ; i++)
	{
	    newX = ( cloud_organized->width * ( cloud->points[i].x - minX ) ) / ( maxX  - minX );
	    newY = ( cloud_organized->height * ( cloud->points[i].y - minY ) ) / ( maxY  - minY );
	    //std::cout << newX << "," << newY << "\n";
	    //std::cout << newX << newY << std::endl;
	    cloud_organized->points[( (int) newY * cloud_organized->width ) + ( int) newX] = cloud->points[i];
	}
	//puting the path+filename in a string s
	std::stringstream ss;
	std::string s;
	ss << argv[1];
	ss >> s; //argv[1]

	std::stringstream ss2;
	std::string s2;
	ss2 << argv[2];
	ss2 >> s2; //argv[1]
	//spliting diretory + filename
	unsigned found = s.find_last_of("//");
	std::string path = s.substr(0, found);
	std::string file = s.substr(found + 1);

	//creating diretory for the results
	std::stringstream baseDir;
	baseDir << path << "/" << s2 << "/";
	boost::filesystem::path dir(baseDir.str());
	if (boost::filesystem::create_directory(dir)) {
		//std::cout << "Success" << "\n";
	}

	//creating directory for the example
	std::stringstream exampleDir;
	exampleDir << baseDir.str() << file;
	boost::filesystem::path dir2(exampleDir.str());
	if (boost::filesystem::create_directory(dir2)) {
		//std::cout << "Success" << "\n";
	}

	//creating the file to be saved
	std::stringstream stream;
	stream << exampleDir.str() << "/" << file << "_" << counter << ".png";

	std::cerr << "Saving the sliced cloud: " << stream.str() << std::endl;
	pcl::io::savePNGFile(stream.str(), *cloud_organized);

}

void getSlicesFromPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_slice, double z, double delta)
{
	for (size_t i = 0; i < cloud->points.size(); i++)
	{
		if ( ( cloud->points[i].z ) > ( z - delta ) &&
			 ( cloud->points[i].z ) < ( z + delta ) )
			  cloud_slice->points.push_back (cloud->points[i]);
	}
}


int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_slice (new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_organized(new pcl::PointCloud<pcl::PointXYZRGBA>);

  // Read a PCD file from disk.
  if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(argv[1], *cloud) != 0)
  {
      std::cout << "Inform a point cloud as input" << "\n";
	  return -1;
  }
  std::vector<int> mapping;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);

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
  std::cout << "\ntamSlice: " << tamSlice << "\n";

  int counter = 0;
  double delta = 0.08;
  for (double k = minZ; k < maxZ; k = k + tamSlice)
  {
	  if ( ( counter < numberOfTotalSlices) )
	  {
		  cloud_slice->clear();
		  cloud_slice->points.clear();
		  cloud_projected->clear();
		  cloud_projected->points.clear();
		  getSlicesFromPointCloud(cloud, cloud_slice, k, delta);

		  // Create the filtering object
		  pcl::ProjectInliers<pcl::PointXYZRGBA> proj;
		  proj.setModelType (pcl::SACMODEL_PLANE);
		  //proj.setInputCloud (cloud);
		  proj.setInputCloud (cloud_slice);
		  //std::cout << "cloud slice size = " << cloud_slice->points.size() << "\n";
		  proj.setModelCoefficients (coefficients);
		  proj.filter (*cloud_projected);

		  if ( cloud_projected->points.size() > 0 ) pcl::io::savePCDFileASCII ("projected.pcd", *cloud_projected);
		  convertPointCloudToPNG(cloud_projected, cloud_organized, counter, argv);
		  counter++;
	  }
  }

  //repete the last image to be sure that you have 10 images
  while ( counter < numberOfTotalSlices )
  {
	  convertPointCloudToPNG(cloud_projected, cloud_organized, counter, argv);
	  counter++;
  }

  return (0);
  //
}
