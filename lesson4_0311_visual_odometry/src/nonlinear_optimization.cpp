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

#include "base.h"

int main(int argc, char** argv)
{
    // 输入格式为【可执行文件，数据集路径，参数路径，关联序列文件路径】
    if (argc != 4) {
        cerr << endl << "Usage: ./pc_registration path_to_dataset path_to_settings path_to_associations" << endl;
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
    
    FRAME frame1,frame2;
    int frame1_idx=200;
    int frame2_idx=205;
    
    //读取rgb图象和depth图象
    frame1.rgb=cv::imread(datasetFilenames+"/"+vstrRGBImageFilenames[frame1_idx],-1);
    frame1.depth=cv::imread(datasetFilenames+"/"+vstrDepthImageFilenames[frame1_idx],-1);
    frame2.rgb=cv::imread(datasetFilenames+"/"+vstrRGBImageFilenames[frame2_idx],-1);
    frame2.depth=cv::imread(datasetFilenames+"/"+vstrDepthImageFilenames[frame2_idx],-1);
    
    //分别提取出两张Frame的特征：SIFT/SURF/ORB
    cv::Ptr<cv::FeatureDetector> detector=cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor=cv::ORB::create();
    
    //第一步：检测Oriented Fast角点位置
    detector->detect(frame1.rgb,frame1.kp);
    detector->detect(frame2.rgb,frame2.kp);
    //第二步：根据角点位置计算BRIEF描述子
    descriptor->compute(frame1.rgb,frame1.kp,frame1.desp);
    descriptor->compute(frame2.rgb,frame2.kp,frame2.desp);
    //显示出来特征点的位置
//    cv::Mat Outimg1,Outimg2;
//    cv::drawKeypoints(frame1.rgb,frame1.kp,Outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    cv::drawKeypoints(frame1.rgb,frame2.kp,Outimg2,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    cv::imshow("keypoints1",Outimg1);
//    cv::imshow("keypoints2",Outimg2);
//    cv::waitKey(0);
    //第三步：对两张图象进行匹配
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher;
    matcher.match(frame1.desp,frame2.desp,matches);
    cout<<"the amount of matches is "<<matches.size()<<endl;
    //可视化粗匹配的特征
    cv::Mat imgMatches;
//    cv::drawMatches(frame1.rgb,frame1.kp,frame2.rgb,frame2.kp,matches,imgMatches);
//    cv::imshow("matches",imgMatches);
//    cv::waitKey(0);
    //第四步：初步筛选匹配的特征
    double min_distance=matches[0].distance;
    for(int i=1;i<matches.size();i++){
        min_distance=min(min_distance,double(matches[i].distance));
    }
    cout<<"the minimum distance of matches is "<<min_distance<<endl;
    vector<cv::DMatch> good_matches;
    for(auto match:matches){
        if(match.distance<min_distance*3) good_matches.push_back(match);
    }
    cout<<"the amount of good_matches is "<<good_matches.size()<<endl;
//    cv::drawMatches(frame1.rgb,frame1.kp,frame2.rgb,frame2.kp,good_matches,imgMatches);
//    cv::imshow("good matches",imgMatches);
//    cv::waitKey(0);

    //第五步：PnP计算相对运动R和t，主要用到cv::solvePnPRansac()
    //需要参考帧图象的三维点和当前帧的图象点
    vector<cv::Point3f> pt_ref;
    vector<cv::Point2f> pt_cur;
    for(auto good_match:good_matches){
        cv::Point2f p=frame1.kp[good_match.queryIdx].pt;
        ushort d=frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
        //if(d==0) continue;
        pt_cur.push_back(frame2.kp[good_match.trainIdx].pt);

        cv::Point3f pt(p.x,p.y,d);
        cv::Point3f pd=point2dTo3d(pt,camera_p);
        pt_ref.push_back(pd);
    }
    cv::Mat rvec,tvec,inliers;
    cv::solvePnPRansac(pt_ref,pt_cur,camera_p.K,camera_p.DistCoef,rvec,tvec,false,200,8,0.99,inliers);

    cout<<"the amount of inliers_matches is "<<inliers.rows<<endl;
    cout<<"rvec="<<endl<<rvec<<endl;
    cout<<"tvec="<<endl<<tvec<<endl;

    //可视化inliers的匹配
    vector<cv::DMatch> inliersMatch;
    for(int i=0;i<inliers.rows;i++){
        inliersMatch.push_back(good_matches[inliers.ptr<int>(i)[0]]);
    }
    cv::drawMatches(frame1.rgb,frame1.kp,frame2.rgb,frame2.kp,inliersMatch,imgMatches);
    cv::imshow("inliers matches",imgMatches);
    cv::waitKey(0);

    Sophus::SE3 T_pnp_estimated=Sophus::SE3(
                Sophus::SO3(rvec.at<double>(0,0),rvec.at<double>(1,0),rvec.at<double>(2,0)),
                Eigen::Vector3d(tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0))
                );

    typedef Eigen::Matrix<double,6,1> Vector6d;


    cout<<"pose before nonlinear optimization is " <<T_pnp_estimated<<endl;
//    Vector6d se3=T_pnp_estimated.log();
//    cout<<"se3 hat = "<<Sophus::SE3::hat(se3)<<endl;
//    cout<<"se3 hat vee = "<<Sophus::SE3::vee(Sophus::SE3::hat(se3)).transpose()<<endl<<endl;

    cout<<"SO3 = "<<endl<<T_pnp_estimated.rotation_matrix()<<endl<<endl;
    cout<<"translation = "<<endl<<T_pnp_estimated.translation()<<endl<<endl;
    //SE3平移在前，旋转在后

    //第一步：初始化求解器
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver=new g2o::LinearSolverDense<Block::PoseMatrixType>();


//    Block* solver_ptr=new Block(linearSolover);
//    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    Block* solver_ptr = new Block( unique_ptr<Block::LinearSolverType>(linearSolver) );      // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( unique_ptr<Block>(solver_ptr) );


    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);


    //第二步：添加顶点
    g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
         T_pnp_estimated.rotation_matrix(),
         T_pnp_estimated.translation()
    ));
    optimizer.addVertex(pose);

    //第三步：添加边
    for(int i=0;i<inliers.rows;i++){
        int index=inliers.at<int>(i,0);
        EdgeProjectXYZ2UVPoseOnly* edge=new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0,pose);
        edge->camera_=&camera_p;
        edge->point_=Eigen::Vector3d(pt_ref[index].x,pt_ref[index].y,pt_ref[index].z);
        edge->setMeasurement(Eigen::Vector2d(pt_cur[index].x,pt_cur[index].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    T_pnp_estimated = Sophus::SE3(
                pose->estimate().rotation(),
                pose->estimate().translation()
                );

    cout<<"pose after nonlinear optimization is "<<T_pnp_estimated<<endl;

//    Vector6d se3_1=T_pnp_estimated.log();
//    cout<<"se3 hat = "<<Sophus::SE3::hat(se3_1)<<endl;
//    cout<<"se3 hat vee = "<<Sophus::SE3::vee(Sophus::SE3::hat(se3_1)).transpose()<<endl;


    cout<<"SO3 = "<<endl<<T_pnp_estimated.rotation_matrix()<<endl<<endl;
    cout<<"translation = "<<endl<<T_pnp_estimated.translation()<<endl<<endl;



//    //求出变换矩阵
//    cv::Mat R;
//    cv::Rodrigues(rvec,R);
//    Eigen::Matrix3d r;
//    cv::Mat R;
    Eigen::Matrix3d r=Eigen::Matrix3d(T_pnp_estimated.rotation_matrix());
//    cv::cv2eigen(R,r);

    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(r);
    T=angle;
    Eigen::Vector3d translation=T_pnp_estimated.translation();
//    T(0,3)=tvec.at<double>(0,0);
//    T(1,3)=tvec.at<double>(0,1);
//    T(2,3)=tvec.at<double>(0,2);
    T(0,3)=translation(0);
    T(1,3)=translation(1);
    T(2,3)=translation(2);

    cout<<"T = "<<endl<<T.matrix()<<endl;

    //拼接点云
    cout<<"point cloud registration: "<<endl;
    PointCloud::Ptr cloud1=img2PointCloud(frame1.rgb,frame1.depth,camera_p);
    PointCloud::Ptr cloud2=img2PointCloud(frame2.rgb,frame2.depth,camera_p);
    PointCloud::Ptr cloud_all(new PointCloud());

    pcl::transformPointCloud(*cloud1,*cloud_all,T.matrix());

    *cloud_all+=*cloud2;

    pcl::visualization::CloudViewer viewer("point cloud registration");
    viewer.showCloud(cloud_all);
    while(!viewer.wasStopped()){}

    //求解出相机的四元数
    Eigen::Quaterniond q(r);
    string filename="Save2FrmaeTrajectory_"+to_string(frame1_idx)+"_"+to_string(frame2_idx)+".txt";
    ofstream f;
    f.open(filename.c_str());
    f<<setprecision(15)<<vTimestamps[frame1_idx]<<setprecision(7)<<" "<<"0"<<" "<<"0"<<" "<<\
       "0"<<" "<<"0" <<" "<<"0"<<" "<<"0"<<" "<<"0" <<endl;
    f<<setprecision(15)<<vTimestamps[frame2_idx]<<setprecision(7)<<" "<<tvec.at<double>(0)<<" "<<tvec.at<double>(1)<<" "<<\
       tvec.at<double>(2)<<" "<<q.x() <<" "<<q.y()<<" "<<q.z()<<" "<<q.w() <<endl;
    f.close();

    filename="Update_"+filename;
    f.open(filename.c_str());
    f<<setprecision(15)<<vTimestamps[frame1_idx]<<setprecision(7)<<" "<<"0"<<" "<<"0"<<" "<<\
       "0"<<" "<<"0" <<" "<<"0"<<" "<<"0"<<" "<<"0" <<endl;
    f<<setprecision(15)<<vTimestamps[frame2_idx]<<setprecision(7)<<" "<<translation(0)<<" "<<translation(1)<<" "<<\
       translation(2)<<" "<<q.x() <<" "<<q.y()<<" "<<q.z()<<" "<<q.w() <<endl;
    f.close();


//    cout<<"hello SLAM"<<endl;
//    cout<<tvec.type()<<endl;
    return 0;
};

