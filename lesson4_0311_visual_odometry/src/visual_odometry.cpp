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
        cerr << endl << "Usage: ./visual_odometry path_to_dataset path_to_settings path_to_associations" << endl;
        return 1;
    }
   
   // 加载rgb图片和depth图片的路径
    string datasetFilenames = string(argv[1]);
    string strAssociationFilename = string(argv[3]);
    vector<string> vstrRGBImageFilenames,  vstrDepthImageFilenames;
    vector<double> vTimestamps;
    LoadImages(strAssociationFilename, vstrRGBImageFilenames, vstrDepthImageFilenames, vTimestamps);
    for(auto& str:vstrRGBImageFilenames){
        str=datasetFilenames+"/"+str;
    }
    for(auto& str:vstrDepthImageFilenames){
        str=datasetFilenames+"/"+str;
    }
    int N = vstrRGBImageFilenames.size();
    cout <<"The amount of pairs(RGB+D) is " << N<< endl;
    
    // 读取参数类
    string SettingFilenames = string(argv[2]);
    CAMERA_PARAMETERS camera_p = LoadSettings(SettingFilenames);
    cout << "The camera intrisics matrix is " << endl<< camera_p.K << endl;

    cv::Ptr<cv::FeatureDetector> detector=cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor=cv::ORB::create();

    vector<pair<Sophus::SE3,double>> pose_v;
    FRAME ref_frame;
    int begin_index=0,end_index=N;

    /***********主程序开始************/
    for(int i=begin_index;i<end_index;i++){

        cout << "processing frame#" <<i<<endl;


        //初始化
        if(i==begin_index){
            ref_frame.rgb=cv::imread(vstrRGBImageFilenames[i],-1);
            ref_frame.depth=cv::imread(vstrDepthImageFilenames[i],-1);
            detector->detect(ref_frame.rgb,ref_frame.kp);
            descriptor->compute(ref_frame.rgb,ref_frame.kp,ref_frame.desp);
            ref_frame.T=Sophus::SE3(Sophus::SO3(double(0),double(0),double(0)),Eigen::Vector3d(double(0),double(0),double(0)));
            pose_v.push_back({ref_frame.T,vTimestamps[i]});
            continue;
        }
//        cv::imshow("hello",ref_frame.rgb);
//        cv::waitKey(0);

        //step1：读取当前帧
        FRAME cur_frame;
        cur_frame.rgb=cv::imread(vstrRGBImageFilenames[i],-1);
        cur_frame.depth=cv::imread(vstrDepthImageFilenames[i],-1);
        detector->detect(cur_frame.rgb,cur_frame.kp);
        descriptor->compute(cur_frame.rgb,cur_frame.kp,cur_frame.desp);

        //step2：当前帧与参考帧进行匹配
        vector<cv::DMatch> matches;
        cv::BFMatcher matcher;
        matcher.match(ref_frame.desp,cur_frame.desp,matches);

        //step3：通过最小距离筛选匹配
        double min_distance=matches[0].distance;
        for(int i=1;i<matches.size();i++){
            min_distance=min(min_distance,double(matches[i].distance));
        }
        vector<cv::DMatch> good_matches;
        for(auto match:matches){
            if(match.distance<min_distance*3) good_matches.push_back(match);
        }

        //step4：ransacPnP筛选内点
        vector<cv::Point3f> pt_ref;
        vector<cv::Point2f> pt_cur;
        for(auto good_match:good_matches){
            cv::Point2f p=ref_frame.kp[good_match.queryIdx].pt;
            ushort d=ref_frame.depth.ptr<ushort>(int(p.y))[int(p.x)];
            //if(d==0) continue;
            pt_cur.push_back(cur_frame.kp[good_match.trainIdx].pt);

            cv::Point3f pt(p.x,p.y,d);
            cv::Point3f pd=point2dTo3d(pt,camera_p);
            pt_ref.push_back(pd);
        }

        if(pt_ref.size()<=10) continue;

        cv::Mat rvec,tvec,inliers;
        cv::solvePnPRansac(pt_ref,pt_cur,camera_p.K,camera_p.DistCoef,rvec,tvec,false,100,4,0.99,inliers);

        cout<<"the amount of inliers matches is "<<inliers.rows<<endl;

        if(inliers.rows<=8) continue;

        //step5：g2o非线性优化位姿
        Sophus::SE3 T_pnp_estimated=Sophus::SE3(
                    Sophus::SO3(rvec.at<double>(0,0),rvec.at<double>(1,0),rvec.at<double>(2,0)),
                    Eigen::Vector3d(tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0))
                    );

//        cout<<"pose before nonlinear optimization is " <<T_pnp_estimated<<endl;
//        cout<<"SO3 = "<<endl<<T_pnp_estimated.rotation_matrix()<<endl<<endl;
//        cout<<"translation = "<<endl<<T_pnp_estimated.translation()<<endl<<endl;
        //SE3平移在前，旋转在后

        //g2o第一步：初始化求解器
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
        Block::LinearSolverType* linearSolver=new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solver_ptr = new Block( unique_ptr<Block::LinearSolverType>(linearSolver) );      // 矩阵块求解器
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( unique_ptr<Block>(solver_ptr) );
        g2o::SparseOptimizer optimizer;  //最后的优化器
        optimizer.setAlgorithm(solver);

        //g2o第二步：添加顶点
        g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(
             T_pnp_estimated.rotation_matrix(),
             T_pnp_estimated.translation()
        ));
        optimizer.addVertex(pose);

        //g2o第三步：添加边
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

        //g2o第四步：初始化和优化
        optimizer.initializeOptimization();
        optimizer.optimize(10); //设置迭代次数

        T_pnp_estimated = Sophus::SE3(
                    pose->estimate().rotation(),
                    pose->estimate().translation()
                    );

//        cout<<"pose after nonlinear optimization is "<<T_pnp_estimated<<endl;
//        cout<<"SO3 = "<<endl<<T_pnp_estimated.rotation_matrix()<<endl<<endl;
//        cout<<"translation = "<<endl<<T_pnp_estimated.translation()<<endl<<endl;

        //step6：更新当前帧的位姿和更新参考帧
        cur_frame.T=T_pnp_estimated*ref_frame.T;
        pose_v.push_back({cur_frame.T,vTimestamps[i]});
        ref_frame=cur_frame;

    }


    //最后，保存轨迹
    string filename="Save2FrmaeTrajectory_"+to_string(begin_index)+"to"+to_string(end_index)+".txt";
    ofstream f;
    f.open(filename.c_str());
//    for(int i=begin_index;i<end_index;i++){
//        Sophus::SE3 T_estimated=pose_v[i-begin_index];
//        Eigen::Matrix3d r=Eigen::Matrix3d(T_estimated.rotation_matrix());
//        Eigen::Quaterniond q(r);
//        Eigen::Vector3d translation=T_estimated.translation();
//        f<<setprecision(15)<<vTimestamps[i]<<setprecision(7)<<" "<<translation(0)<<" "<<translation(1)<<" "<<translation(2)\
//        <<" "<<q.x() <<" "<<q.y()<<" "<<q.z()<<" "<<q.w() <<endl;
//    }
    for(auto pair:pose_v){
        Sophus::SE3 T_estimated=pair.first;
        Eigen::Matrix3d r=Eigen::Matrix3d(T_estimated.rotation_matrix());
        Eigen::Quaterniond q(r);
        Eigen::Vector3d translation=T_estimated.translation();
        f<<setprecision(15)<<pair.second<<setprecision(7)<<" "<<translation(0)<<" "<<translation(1)<<" "<<translation(2)\
        <<" "<<q.x() <<" "<<q.y()<<" "<<q.z()<<" "<<q.w() <<endl;
    }
    f.close();


//    //第七步：求出变换矩阵
////    cv::Mat R;
////    cv::Rodrigues(rvec,R);
////    Eigen::Matrix3d r;
////    cv::Mat R;
//    Eigen::Matrix3d r=Eigen::Matrix3d(T_pnp_estimated.rotation_matrix());
////    cv::cv2eigen(R,r);

//    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
//    Eigen::AngleAxisd angle(r);
//    T=angle;
//    Eigen::Vector3d translation=T_pnp_estimated.translation();
////    T(0,3)=tvec.at<double>(0,0);
////    T(1,3)=tvec.at<double>(0,1);
////    T(2,3)=tvec.at<double>(0,2);
//    T(0,3)=translation(0);
//    T(1,3)=translation(1);
//    T(2,3)=translation(2);

//    cout<<"T = "<<endl<<T.matrix()<<endl;

//    //第八步：拼接点云
//    PointCloud::Ptr cloud1=img2PointCloud(frame1.rgb,frame1.depth,camera_p);
//    PointCloud::Ptr cloud2=img2PointCloud(frame2.rgb,frame2.depth,camera_p);
//    PointCloud::Ptr cloud_all(new PointCloud());

//    pcl::transformPointCloud(*cloud1,*cloud_all,T.matrix());

//    *cloud_all+=*cloud2;

//    pcl::visualization::CloudViewer viewer("point cloud registration");
//    viewer.showCloud(cloud_all);
//    while(!viewer.wasStopped()){}


    return 0;
};

