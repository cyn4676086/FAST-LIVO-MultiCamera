#pragma once
#include <string>
#include <Eigen/Core>
#include <vikit/abstract_camera.h>

namespace camera_manager {

struct Cameras {
    vk::AbstractCamera* cam; 
    std::string camera_ns; // 相机的ROS命名空间
    string cam_model;
    int cam_id;
    std::string img_topic;
    int width;
    int height;
    double fx;
    double fy;
    double cx;
    double cy;
    double d0;
    double d1;
    double d2;
    double d3;

        // 外参
    M3D Rcl;  // 相机旋转矩阵到雷达
    V3D Pcl;  // 相机平移向量到雷达
    M3D Rci;  // 相机到 IMU 的旋转矩阵
    V3D Pci;  // 相机到 IMU 的平移向量
    M3D Rcw;  // 相机到世界的旋转矩阵
    V3D Pcw;  // 相机到世界的平移向量

    // 雅可比矩阵
    M3D Jdphi_dR;  // 姿态变化对旋转矩阵的雅可比矩阵
    M3D Jdp_dR;    // 位置变化对旋转矩阵的雅可比矩阵
    M3D Jdp_dt;    // 位置变化对平移向量的雅可比矩阵
};

} // namespace camera_manager
