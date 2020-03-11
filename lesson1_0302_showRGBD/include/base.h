#include<iostream>
#include<bits/stdc++.h> 
#include<string>
using namespace std;

#include<opencv2/opencv.hpp>
// using namespace cv;

#include<pcl-1.8/pcl/io/pcd_io.h>
#include<pcl-1.8/pcl/point_types.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct FRAME
{
    cv::Mat rgb, depth;
    cv::Mat desp;
    vector<cv::KeyPoint> kp;
};

struct CAMERA_PARAMETERS
{
    float fx,fy,cx,cy;
    cv::Mat K;
    int scale;
};

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

CAMERA_PARAMETERS LoadSettings(const string& SettingFilenames);

void mergeImage(cv::Mat &dst, vector<cv::Mat> &images);

PointCloud::Ptr img2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_PARAMETERS camera_p);
