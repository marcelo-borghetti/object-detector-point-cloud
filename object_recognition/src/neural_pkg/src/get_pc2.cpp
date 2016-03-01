// Original code by Geoffrey Biggs, taken from the PCL tutorial in
// http://pointclouds.org/documentation/tutorials/pcl_visualizer.php
// Simple OpenNI viewer that also allows to write the current scene to a .pcd
// when pressing SPACE.

#include <iostream>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/console/parse.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>
#include <boost/format.hpp>
#include <pcl/common/transforms.h>
#include <pcl/common/eigen.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

using namespace std;
using namespace pcl;

PointCloud<PointXYZRGBA>::Ptr cloudptr(new PointCloud<PointXYZRGBA>); // A cloud that will store color info.
PointCloud<PointXYZ>::Ptr fallbackCloud(new PointCloud<PointXYZ>);    // A fallback cloud with just depth data.
boost::shared_ptr<visualization::CloudViewer> viewer;                 // Point cloud viewer object.
boost::shared_ptr<visualization::ImageViewer> image_viewer;
Grabber* OpenniGrabber;                                               // OpenNI grabber that takes data from the device.
unsigned int cloudFilesSaved = 0;                                          // For the numbering of the clouds saved to disk.
unsigned int imageFilesSaved = 0;                                          // For the numbering of the clouds saved to disk.
bool saveCloud(false), saveImage(false), noColor(false), segmentFlag(false);                                // Program control.

struct termios orig_termios;

void
viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    //viewer.setBackgroundColor(0.7, 0.7, 0.7);
	viewer.setBackgroundColor(0, 0, 0);
}

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

void
printUsage(const char* programName)
{
	cout << "Usage: " << programName << " [options]"
		 << endl
		 << endl
		 << "Options:\n"
		 << endl
		 << "\t<none>     start capturing from an OpenNI device.\n"
		 << "\t-v FILE    visualize the given .pcd file.\n"
		 << "\t-h         shows this help.\n";
}

void image_cb_ (const boost::shared_ptr<openni_wrapper::Image> &image)
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
              		 stringstream stream;
             		 stream << "inputRBG" << imageFilesSaved << ".jpg";
             		 string filename = stream.str();
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

void modifyPointCloud(PointCloud<PointXYZRGBA>::Ptr&  cloud)
{
	/* initialize random seed: */
	srand (time(NULL));
	double ratio = 0.1;
	double max = 0.10;
	double min = -0.10;
	//pcl::copyPointCloud(*cloud, *rotated_cloud);
	for (unsigned int j = 0; j < cloud->points.size(); j++)
	{
		float dx = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX)) * (max -min)) + min;
		float dy = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX)) * (max -min)) + min;
		float dz = static_cast <float> ((rand()) / (static_cast <float> (RAND_MAX)) * (max -min)) + min;

		/*cloud->points[j].x = cloud->points[j].x;// + dx;
		cloud->points[j].y = cloud->points[j].y;// + dy;
		cloud->points[j].z = cloud->points[j].z;// + dz;*/
		cloud->points[j].r = 130;
		cloud->points[j].g = 0;
		cloud->points[j].b = 0;
	}
}

void segmentClusters(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table, pcl::ModelCoefficients::Ptr coefficients)
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
	   if ( /*( plot->points[i].y > yf_min ) && ( plot->points[i].b == 255 ) ||*/
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


pcl::ModelCoefficients::Ptr planeSegmentation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table)
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
	 	 reset_terminal_mode();
	     seg.segment (*inliers, *coefficients);
	     set_conio_terminal_mode();

	     if (inliers->indices.size () == 0)
	     {
	    	 reset_terminal_mode();
	    	 std::cout << "Could not estimate a planar model for the given dataset.\n";
	    	 set_conio_terminal_mode();
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

// This function is called every time the device has new data.
void
grabberCallback(const PointCloud<PointXYZRGBA>::ConstPtr& cloud)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr plot ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr result ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2 ( new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_table ( new pcl::PointCloud<pcl::PointXYZRGBA>);

	pcl::copyPointCloud(*cloud, *cloud2);
	pcl::copyPointCloud(*cloud, *plot);
	pcl::ModelCoefficients::Ptr coefficients = planeSegmentation(cloud2, plot, cloud_table);

	if (segmentFlag)
	{
		segmentClusters(plot, cloud_table, coefficients);
	}
	if (! viewer->wasStopped())
		viewer->showCloud(plot);

	  plot->width = plot->points.size ();
	  plot->height = 1;
	  plot->is_dense = true;

	if (saveCloud)
	{
		stringstream stream;
		stream << "inputCloud" << cloudFilesSaved << ".pcd";
		//printf("inputCloud%d.pcd\n", cloudFilesSaved);
		string filename = stream.str();
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

// For detecting when SPACE is pressed.
void
keyboardEventOccurred(const visualization::KeyboardEvent& event,
					  void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown())
	{
		saveCloud = true;
		saveImage = true;
	}
}

// Creates, initializes and returns a new viewer.
boost::shared_ptr<visualization::CloudViewer>
createViewer()
{
	boost::shared_ptr<visualization::CloudViewer> v
	(new visualization::CloudViewer("OpenNi viewer Cloud"));
	//v->registerKeyboardCallback(keyboardEventOccurred);

	return (v);
}

boost::shared_ptr<visualization::ImageViewer>
createImageViewer()
{
	boost::shared_ptr<visualization::ImageViewer> v
		(new visualization::ImageViewer("OpenNi viewer Image"));
		v->registerKeyboardCallback(keyboardEventOccurred);
    return (v);
}

int
main(int argc, char** argv)
{
	if (console::find_argument(argc, argv, "-h") >= 0)
	{
		printUsage(argv[0]);
		return 0;
	}

	bool justVisualize(false);
	string filename;
	if (console::find_argument(argc, argv, "-v") >= 0)
	{
		if (argc != 3)
		{
			printUsage(argv[0]);
			return 0;
		}

		filename = argv[2];
		justVisualize = true;
	}
	else if (argc != 1)
	{
		printUsage(argv[0]);
		return 0;
	}

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

		cout << "Loaded " << filename << "." << endl;
		if (noColor)
			cout << "This cloud has no RGBA color information present." << endl;
		else cout << "This cloud has RGBA color information present." << endl;
	}
	// Second mode, start fetching and displaying frames from the device.
	else
	{
		OpenniGrabber = new OpenNIGrabber();
		if (OpenniGrabber == 0)
			return false;

        boost::function<void (const PointCloud<PointXYZRGBA>::ConstPtr&)> fc =
			boost::bind(&grabberCallback, _1);
		OpenniGrabber->registerCallback(fc);

		boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&)> fi =
                boost::bind (&image_cb_, _1);
        OpenniGrabber->registerCallback (fi);
    }

	viewer = createViewer();
	//image_viewer = createImageViewer();
	viewer->runOnVisualizationThreadOnce (viewerOneOff);
	if (justVisualize)
	{
		if (noColor)
			viewer->showCloud(fallbackCloud);
		else viewer->showCloud(cloudptr);
	}
	else
	{
		OpenniGrabber->start();
	}

	set_conio_terminal_mode();

	// Main loop.
	while (! viewer->wasStopped())
	//while (! image_viewer->wasStopped())
	{
		boost::this_thread::sleep(boost::posix_time::seconds(1));
		if (kbhit()) {
			char c = getch();
			if (c == ' ' )
			{
			   printf("%d - ", cloudFilesSaved);
		       saveImage = true;
		       saveCloud = true;
			}
			if (c == 's' )
			{
		       segmentFlag = !segmentFlag;
			}
			//cout << c;
		    if (c == 'q') break;
		}
	}

	if (! justVisualize)
	{
		OpenniGrabber->stop();
	}
}
