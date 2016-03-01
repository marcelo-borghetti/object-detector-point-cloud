extern LogsFile logfile;
extern pointCloudMethods pcMethods;
extern viewMethods vMethods;
extern PointCloud<PointXYZRGBA>::Ptr cloudptr(new PointCloud<PointXYZRGBA>); // A cloud that will store color info.
extern PointCloud<PointXYZ>::Ptr fallbackCloud(new PointCloud<PointXYZ>);    // A fallback cloud with just depth data.
extern boost::shared_ptr<visualization::CloudViewer> viewer;                 // Point cloud viewer object.
extern boost::shared_ptr<visualization::ImageViewer> image_viewer;
extern Grabber* OpenniGrabber;                                               // OpenNI grabber that takes data from the device.
extern unsigned int cloudFilesSaved = 0;                                          // For the numbering of the clouds saved to disk.
extern unsigned int imageFilesSaved = 0;                                          // For the numbering of the clouds saved to disk.
extern bool saveCloud(false), saveImage(false), noColor(false), segmentFlag(false), recordFlag(false), testFlag(false);                                // Program control.
extern int indexCluster = 0;
extern int numClusters = 0;
