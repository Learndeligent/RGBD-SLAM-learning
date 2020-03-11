#include<iostream>
#include<bits/stdc++.h> 
#include<string>
using namespace std;

#include<opencv2/opencv.hpp>

#include<pcl-1.8/pcl/io/pcd_io.h>
#include<pcl-1.8/pcl/point_types.h>

#include "base.h"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

CAMERA_PARAMETERS LoadSettings(const string& SettingFilenames)
{
    cv::FileStorage Settings(SettingFilenames, cv::FileStorage::READ);
    float fx = Settings["Camera.fx"];
    float fy = Settings["Camera.fy"];
    float cx = Settings["Camera.cx"];
    float cy = Settings["Camera.cy"];
    cv::Mat K=cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0)=fx;
    K.at<float>(1,1)=fy;
    K.at<float>(0,2)=cx;
    K.at<float>(1,2)=cy;
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0)=Settings["Camera.k1"];
    DistCoef.at<float>(1)=Settings["Camera.k2"];
    DistCoef.at<float>(2)=Settings["Camera.p1"];
    DistCoef.at<float>(3)=Settings["Camera.p2"];
    const float k3=Settings["Camera.k3"];
    if(k3!=0){
        DistCoef.resize(5);
        DistCoef.at<float>(4)=k3;
    }
    CAMERA_PARAMETERS camera_p = {fx, fy, cx, cy, K,DistCoef, 5000};
    return camera_p;
}

void mergeImage(cv::Mat &dst, vector<cv::Mat> &images)
{
    int nCount=images.size();
    if(nCount<=0){
        cout<<"the number of images is not enough"<<endl;
        return;
    }
    
    cout<<"the amount of images is " << nCount<<endl;
    
    int rows=images[0].rows;
    int cols=images[0].cols;
    
    dst.create(rows*nCount/2,cols*2,CV_8UC3);
    
    for(int i=0;i<nCount;i++){
        images[i].copyTo(dst(cv::Rect((i%2)*cols,(i/2)*rows,images[0].cols,images[0].rows)));
    }
    return;
}

cv::Point3f point2dTo3d(cv::Point3f& pt,CAMERA_PARAMETERS camera_p)
{
    cv::Point3f p;
    p.z=double(pt.z)/camera_p.scale;
    p.x=(pt.x-camera_p.cx)*p.z/camera_p.fx;
    p.y=(pt.y-camera_p.cy)*p.z/camera_p.fy;
    return p;
}

PointCloud::Ptr img2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_PARAMETERS camera_p)
{
        PointCloud::Ptr cloud(new PointCloud);
        for (int m = 0;m<depth.rows;m++) {
            for (int n = 0;n<depth.cols;n++) {
                ushort d = depth.ptr<ushort>(m)[ n];
                if (d == 0) continue;
                    
                PointT p;
                // location
                p.z = double(d)/camera_p.scale;
                p.x = (n-camera_p.cx) *p.z/camera_p.fx;
                p.y = (m-camera_p.cy) *p.z/camera_p.fy;
                // color
                p.b =rgb.ptr<uchar>(m)[n*3];
                p.g = rgb.ptr<uchar>(m)[n*3+1];
                p.r = rgb.ptr<uchar>(m)[n*3+2];
                
                cloud->points.push_back(p);
            }
        }
        return cloud;
}

