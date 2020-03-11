
#include "base.h"


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


void EdgeProjectXYZ2UVPoseOnly::computeError()
{
    const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
    //pose这里是一个节点，pose->estimate()表示一个SE3的相机位姿
    Eigen::Vector3d point_c=pose->estimate().map(point_);
//    _error = _measurement - camera_->camera2pixel (
//        pose->estimate().map(point_) );
    _error = _measurement - Eigen::Vector2d(
                camera_->fx*point_c(0,0)/point_c(2,0)+camera_->cx,
                camera_->fy*point_c(1,0)/point_c(2,0)+camera_->cy
                );
    //.map(point_)将世界坐标系的3d点转换到相机坐标系下，仍是3d；
    //通过camera_->camera2pixel()将相机坐标系下的点转化为像素坐标系下的Vector2d的点
}

void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
{
    g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
    g2o::SE3Quat T ( pose->estimate() );
    Eigen::Vector3d xyz_trans = T.map ( point_ ); //这里的xyz_trans表示世界坐标系的点在相机坐标系下的坐标
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2]; //分别表示三个坐标
    double z_2 = z*z;

    _jacobianOplusXi ( 0,0 ) =  x*y/z_2 *camera_->fx;
    _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *camera_->fx;
    _jacobianOplusXi ( 0,2 ) = y/z * camera_->fx;
    _jacobianOplusXi ( 0,3 ) = -1./z * camera_->fx;
    _jacobianOplusXi ( 0,4 ) = 0;
    _jacobianOplusXi ( 0,5 ) = x/z_2 * camera_->fx;

    _jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *camera_->fy;
    _jacobianOplusXi ( 1,1 ) = -x*y/z_2 *camera_->fy;
    _jacobianOplusXi ( 1,2 ) = -x/z *camera_->fy;
    _jacobianOplusXi ( 1,3 ) = 0;
    _jacobianOplusXi ( 1,4 ) = -1./z *camera_->fy;
    _jacobianOplusXi ( 1,5 ) = y/z_2 *camera_->fy;
}

