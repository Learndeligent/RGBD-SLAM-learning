#include<iostream>
#include<bits/stdc++.h>
#include<string>
#include <vector>
#include <list>
#include <memory>
#include <set>
#include <unordered_map>
#include <map>
using namespace std;


#include<pcl-1.8/pcl/io/pcd_io.h>
#include<pcl-1.8/pcl/point_types.h>
#include<pcl-1.8/pcl/common/transforms.h>
#include<pcl-1.8/pcl/visualization/cloud_viewer.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

#include<Eigen/Core>
#include<Eigen/Geometry>

#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/core/eigen.hpp>

#include<opencv2/highgui.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

#include <sophus/se3.h>
#include <sophus/so3.h>


struct FRAME
{
    cv::Mat rgb, depth;
    cv::Mat desp;
    vector<cv::KeyPoint> kp;
    Sophus::SE3 T;
};

struct CAMERA_PARAMETERS
{
    float fx,fy,cx,cy;
    cv::Mat K;
    cv::Mat DistCoef;
    int scale;
};

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

CAMERA_PARAMETERS LoadSettings(const string& SettingFilenames);

void mergeImage(cv::Mat &dst, vector<cv::Mat> &images);

cv::Point3f point2dTo3d(cv::Point3f& pt,CAMERA_PARAMETERS camera_p);

PointCloud::Ptr img2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_PARAMETERS camera_p);


// only to optimize the pose, no point
class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeError();
    virtual void linearizeOplus();

    virtual bool read( std::istream& in ){}
    virtual bool write(std::ostream& os) const {};

    Eigen::Vector3d point_; //一个三维点
    CAMERA_PARAMETERS* camera_; //相机位姿
    //除了上述之外，还有_error,_measurement,_information
};
