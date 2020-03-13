#include<iostream>
#include<bits/stdc++.h> 
#include<string>
using namespace std;

#include<opencv2/opencv.hpp>
#include<pcl-1.8/pcl/io/pcd_io.h>
#include<pcl-1.8/pcl/point_types.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

#include "base.h"

int main(int argc, char** argv)
{
    // 输入格式为【可执行文件，数据集路径，参数路径，关联序列文件路径】
    if (argc != 4) {
        cerr << endl << "Usage: ./showRGBD path_to_dataset path_to_settings" << endl;
        return 1;
    }
   
   // 加载rgb图片和depth图片的路径
    string datasetFilenames = string(argv[1]);
    string strAssociationFilename = string(argv[3]);
    vector<string> vstrRGBImageFilenames,  vstrDepthImageFilenames;
    vector<double> vTimestamps;
    LoadImages(strAssociationFilename, vstrRGBImageFilenames, vstrDepthImageFilenames, vTimestamps);
    int N = vstrRGBImageFilenames.size();
    cout <<"The amount of pairs(RGB+D) is " << N<< endl;
    
    // 读取参数类
    string SettingFilenames = string(argv[2]);
    CAMERA_PARAMETERS camera_p = LoadSettings(SettingFilenames);
    cout << "The camera intrisics matrix is " << endl<< camera_p.K << endl;
    
    FRAME frame;
    int n_images = 5;
    for (int i = 0;i<n_images;i++) {
        // 读取rgb图象和depth图象
        frame.rgb = cv::imread(datasetFilenames+"/"+vstrRGBImageFilenames[i],-1);
        frame.depth = cv::imread(datasetFilenames+"/"+vstrDepthImageFilenames[i], -1);
        cv::imshow("RGB", frame.rgb);
        cv::imshow("DEPTH", frame.depth);
        cv::waitKey(0);
    }
    
    
    return 0;
};
