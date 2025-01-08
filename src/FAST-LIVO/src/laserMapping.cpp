
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
// #include <common_lib.h>
#include <image_transport/image_transport.h>
#include "IMU_Processing.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <fast_livo/States.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vikit/camera_loader.h>
#include"lidar_selection.h"
#include "camera_manager.h"

#ifdef USE_ikdtree
    #ifdef USE_ikdforest
    #include <ikd-Forest/ikd_Forest.h>
    #else
    #include <ikd-Tree/ikd_Tree.h>
    #endif
#else
#include <pcl/kdtree/kdtree_flann.h>
#endif

#define INIT_TIME           (0.5)
#define MAXN                (360000)
#define PUBFRAME_PERIOD     (20)

float DET_RANGE = 300.0f;
#ifdef USE_ikdforest
    const int laserCloudWidth  = 200;
    const int laserCloudHeight = 200;
    const int laserCloudDepth  = 200;
    const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;
#else
    const float MOV_THRESHOLD = 1.5f;
#endif

mutex mtx_buffer;
condition_variable sig_buffer;

// mutex mtx_buffer_pointcloud;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, config_file;;
M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);
Vector3d Lidar_offset_to_IMU(Zero3d);
M3D Lidar_rot_to_IMU(Eye3d);
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0,effct_feat_num = 0, time_log_counter = 0, publish_count = 0;
int MIN_IMG_COUNT = 0;

double res_mean_last = 0.05;
//double gyr_cov_scale, acc_cov_scale;
double gyr_cov_scale = 0, acc_cov_scale = 0;
//double last_timestamp_lidar, last_timestamp_imu = -1.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
//double filter_size_corner_min, filter_size_surf_min, filter_size_map_min, fov_deg;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
//double cube_len, HALF_FOV_COS, total_distance, lidar_end_time, first_lidar_time = 0.0;
double cube_len = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
double first_img_time=-1.0;
//double kdtree_incremental_time, kdtree_search_time;
double kdtree_incremental_time = 0, kdtree_search_time = 0, kdtree_delete_time = 0.0;
int kdtree_search_counter = 0, kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;;
//double copy_time, readd_time, fov_check_time, readd_box_time, delete_box_time;
double copy_time = 0, readd_time = 0, fov_check_time = 0, readd_box_time = 0, delete_box_time = 0;
double T1[MAXN], T2[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN];

double match_time = 0, solve_time = 0, solve_const_H_time = 0;

bool lidar_pushed, flg_reset, flg_exit = false;
bool ncc_en;
int dense_map_en = 1;
int img_en = 1;
int lidar_en = 1;
int debug = 0;
bool fast_lio_is_ready = false;
int patch_size,grid_size;
double outlier_threshold, ncc_thre;
double delta_time = 0.0;

vector<BoxPointType> cub_needrm;
vector<BoxPointType> cub_needad;
// deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
deque<PointCloudXYZI::Ptr>  lidar_buffer;
deque<double>          time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

std::vector< deque<cv::Mat> > img_buffers;
std::vector< deque<double> >   img_time_buffers;
std::vector<double> last_timestamp_imgs;    // 记录每个相机最后一帧图像的时间戳


vector<bool> point_selected_surf; 
vector<vector<int>> pointSearchInd_surf; 
vector<PointVector> Nearest_Points; 
vector<double> res_last;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> cameraextrinT(3, 0.0);
vector<double> cameraextrinR(9, 0.0);
double total_residual;
double LASER_POINT_COV, IMG_POINT_COV; 
bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
//surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr cube_points_add(new PointCloudXYZI());
PointCloudXYZI::Ptr map_cur_frame_point(new PointCloudXYZI());
PointCloudXYZI::Ptr sub_map_cur_frame_point(new PointCloudXYZI());

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

#ifdef USE_ikdtree
    #ifdef USE_ikdforest
    KD_FOREST ikdforest;
    #else
    KD_TREE ikdtree;
    #endif
#else
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
#endif

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
Eigen::Matrix3d Rcl;
Eigen::Vector3d Pcl;

//estimator inputs and output;
LidarMeasureGroup LidarMeasures;
// SparseMap sparse_map;
#ifdef USE_IKFOM
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;
#else
StatesGroup  state;
#endif

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

PointCloudXYZRGB::Ptr pcl_wait_save(new PointCloudXYZRGB());  //add save rbg map
PointCloudXYZI::Ptr pcl_wait_save_lidar(new PointCloudXYZI());  //add save xyzi map

bool pcd_save_en = true;
bool pose_output_en = true;

int pcd_save_interval = 20, pcd_index = 0;


void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    #ifdef USE_IKFOM
    //state_ikfom write_state = kf.get_x();
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);  
    #else
    V3D rot_ang(Log(state.rot_end));
    fprintf(fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state.pos_end(0), state.pos_end(1), state.pos_end(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state.vel_end(0), state.vel_end(1), state.vel_end(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state.bias_g(0), state.bias_g(1), state.bias_g(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state.bias_a(0), state.bias_a(1), state.bias_a(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state.gravity(0), state.gravity(1), state.gravity(2)); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
    #endif  
}

#ifdef USE_IKFOM
//project the lidar scan to world frame
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}
#endif

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    #ifdef USE_IKFOM
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    #else
    V3D p_global(state.rot_end * (Lidar_rot_to_IMU*p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    #ifdef USE_IKFOM
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    #else
    V3D p_global(state.rot_end * (Lidar_rot_to_IMU*p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    #ifdef USE_IKFOM
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    #else
    V3D p_global(state.rot_end * (Lidar_rot_to_IMU*p_body + Lidar_offset_to_IMU) + state.pos_end);
    #endif
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);

    int reflection_map = intensity*10000;
}

#ifndef USE_ikdforest
int points_cache_size = 0;
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    points_cache_size = points_history.size();
}
#endif

BoxPointType get_cube_point(float center_x, float center_y, float center_z)
{
    BoxPointType cube_points;
    V3F center_p(center_x, center_y, center_z);
    // cout<<"center_p: "<<center_p.transpose()<<endl;

    for(int i = 0; i < 3; i++)
    {
        cube_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
        cube_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
    }

    return cube_points;
}

BoxPointType get_cube_point(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax)
{
    BoxPointType cube_points;
    cube_points.vertex_max[0] = xmax;
    cube_points.vertex_max[1] = ymax;
    cube_points.vertex_max[2] = zmax;
    cube_points.vertex_min[0] = xmin;
    cube_points.vertex_min[1] = ymin;
    cube_points.vertex_min[2] = zmin;
    return cube_points;
}

#ifndef USE_ikdforest
BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    #ifdef USE_IKFOM
    //state_ikfom fov_state = kf.get_x();
    //V3D pos_LiD = fov_state.pos + fov_state.rot * fov_state.offset_T_L_I;
    V3D pos_LiD = pos_lid;
    #else
    V3D pos_LiD = state.pos_end;
    #endif
    if (!Localmap_Initialized){
        //if (cube_len <= 2.0 * MOV_THRESHOLD * DET_RANGE) throw std::invalid_argument("[Error]: Local Map Size is too small! Please change parameter \"cube_side_length\" to larger than %d in the launch file.\n");
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    printf("Local Map is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", LocalMap_Points.vertex_min[0],LocalMap_Points.vertex_max[0],LocalMap_Points.vertex_min[1],LocalMap_Points.vertex_max[1],LocalMap_Points.vertex_min[2],LocalMap_Points.vertex_max[2]);
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);                     
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
    // printf("Delete time: %0.6f, delete size: %d\n",kdtree_delete_time,kdtree_delete_counter);
    // printf("Delete Box: %d\n",int(cub_needrm.size()));
}
#endif


void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    // cout<<"got feature"<<endl;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    // ROS_INFO("get point cloud at time: %.6f and size: %d", msg->header.stamp.toSec() - 0.1, ptr->points.size());
    printf("[ INFO ]: get point cloud at time: %.6f and size: %d.\n", msg->header.stamp.toSec(), int(ptr->points.size()));
    lidar_buffer.push_back(ptr);
    // time_buffer.push_back(msg->header.stamp.toSec() - 0.1);
    // last_timestamp_lidar = msg->header.stamp.toSec() - 0.1;
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    printf("[ INFO ]: get point cloud at time: %.6f.\n", msg->header.stamp.toSec());
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    //cout<<"msg_in:"<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    
    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg) {
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}
void img_cbk(const sensor_msgs::ImageConstPtr& msg, 
            int cam_id, 
            lidar_selection::LidarSelectorPtr lidar_selector,
            const std::vector<camera_manager::Cameras>& cameras_info)
{
    // 检查图像处理是否启用
    if (!img_en) {
        return;
    }

    ROS_INFO("receive camera ID: %d", cam_id);

    // 计算消息的时间戳
    double msg_header_time = msg->header.stamp.toSec() + delta_time;
    ROS_INFO("timestamp image (msg_header_time): %f", msg_header_time);

    // 时间回退检查
    if (msg_header_time < last_timestamp_imgs[cam_id])
    {
        ROS_ERROR("cam %d back (now: %f, last: %f)", 
                  cam_id, msg_header_time, last_timestamp_imgs[cam_id]);
        std::lock_guard<std::mutex> lock(mtx_buffer);
        img_buffers[cam_id].clear();
        img_time_buffers[cam_id].clear();
        return;
    }

    // 转换 ROS 图像消息到 OpenCV 图像
    cv::Mat cv_image;
    try {
        cv_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (cv_image.empty()) {
        ROS_ERROR("Received empty image.");
        return;
    }
    ROS_INFO("img-cbk ok");

    // 将图像和时间戳添加到缓冲区
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        img_buffers[cam_id].push_back(cv_image);
        img_time_buffers[cam_id].push_back(msg_header_time);
        last_timestamp_imgs[cam_id] = msg_header_time;
    }
    ROS_INFO("image added in buffer %d.size: %lu", cam_id, img_buffers[cam_id].size());
    // 通知等待的线程
    sig_buffer.notify_all();
    ROS_INFO("cam %d new image", cam_id);
}


bool sync_packages(LidarMeasureGroup &meas) {
    if (lidar_buffer.empty() && img_buffers.empty()) {
        return false;
    }

    // 如果刚刚完成一个激光雷达扫描，清空测量缓存
    if (meas.is_lidar_end) {
        meas.measures.clear();
        meas.is_lidar_end = false;
    }

    // 如果不在激光雷达扫描中，初始化新的激光雷达数据
    if (!lidar_pushed) {
        if (lidar_buffer.empty()) {
            return false;
        }
        meas.lidar = lidar_buffer.front();
        if (meas.lidar->points.size() <= 1) {
            mtx_buffer.lock();
            lidar_buffer.pop_front();
            for (auto &buffer : img_buffers) {
                if (!buffer.empty()) buffer.pop_front();
            }
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            return false;
        }
        // 排序激光雷达点云并计算时间范围
        sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list);
        meas.lidar_beg_time = time_buffer.front();
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / 1000.0;
        lidar_pushed = true;
    }

    // 检查是否至少有一个相机有可用数据
    bool has_img = false;
    for (const auto &buffer : img_buffers) {
        if (!buffer.empty()) {
            has_img = true;
            break;
        }
    }

    // 没有图像数据，仅处理激光雷达和 IMU 数据
    if (!has_img) {
        if (last_timestamp_imu < lidar_end_time + 0.02) {
            return false;
        }
        struct MeasureGroup m;
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();

        mtx_buffer.lock();
        while (!imu_buffer.empty() && imu_time < lidar_end_time) {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time) break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        lidar_pushed = false;
        meas.is_lidar_end = true;
        meas.measures.push_back(m);
        return true;
    }

    // 遍历所有相机，处理图像和 IMU 数据
    struct MeasureGroup m;
    for (int cam_idx = 0; cam_idx < img_buffers.size(); ++cam_idx) {
        if (img_buffers[cam_idx].empty()) continue;

        double img_time = img_time_buffers[cam_idx].front();
        if (img_time > lidar_end_time) {
            // 当前相机图像时间晚于激光雷达结束时间，仅处理激光雷达数据
            if (last_timestamp_imu < lidar_end_time + 0.02) {
                return false;
            }
            double imu_time = imu_buffer.front()->header.stamp.toSec();
            m.imu.clear();

            mtx_buffer.lock();
            while (!imu_buffer.empty() && imu_time < lidar_end_time) {
                imu_time = imu_buffer.front()->header.stamp.toSec();
                if (imu_time > lidar_end_time) break;
                m.imu.push_back(imu_buffer.front());
                imu_buffer.pop_front();
            }
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            lidar_pushed = false;
            meas.is_lidar_end = true;
            meas.measures.push_back(m);
            return true;
        } else {
            // 当前相机图像时间在激光雷达扫描时间内，处理图像和 IMU 数据
            if (last_timestamp_imu < img_time) {
                return false;
            }
            double imu_time = imu_buffer.front()->header.stamp.toSec();
            m.imu.clear();
            m.img_offset_time = img_time - meas.lidar_beg_time;
            m.imgs.push_back(img_buffers[cam_idx].front());

            mtx_buffer.lock();
            while (!imu_buffer.empty() && imu_time < img_time) {
                imu_time = imu_buffer.front()->header.stamp.toSec();
                if (imu_time > img_time) break;
                m.imu.push_back(imu_buffer.front());
                imu_buffer.pop_front();
            }
            img_buffers[cam_idx].pop_front();
            img_time_buffers[cam_idx].pop_front();
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            meas.is_lidar_end = false;
            meas.measures.push_back(m);
            return true;
        }
    }
    return false;
}




void map_incremental()
{
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    }
#ifdef USE_ikdtree
    #ifdef USE_ikdforest
    ikdforest.Add_Points(feats_down_world->points, lidar_end_time);
    #else
    ikdtree.Add_Points(feats_down_world->points, true);
    #endif
#endif
}

// PointCloudXYZRGB::Ptr pcl_wait_pub_RGB(new PointCloudXYZRGB(500000, 1));
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
void publish_frame_world_rgb(const ros::Publisher & pubLaserCloudFullRes, 
                             lidar_selection::LidarSelectorPtr lidar_selector, 
                             const std::vector<camera_manager::Cameras>& cameras)
{
    uint size = pcl_wait_pub->points.size();
    PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));

    if(img_en)
    {
        laserCloudWorldRGB->clear();
        for (int i = 0; i < size; i++)
        {
            PointTypeRGB pointRGB;
            pointRGB.x = pcl_wait_pub->points[i].x;
            pointRGB.y = pcl_wait_pub->points[i].y;
            pointRGB.z = pcl_wait_pub->points[i].z;
            V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
            for(int cam_idx = 0; cam_idx < cameras.size(); cam_idx++){
                vk::AbstractCamera* cam = cameras[cam_idx].cam;
                V3D pf(lidar_selector->new_frame_->w2f(p_w, cam_idx));
                if (pf[2] < 0) continue;
                V2D pc(lidar_selector->new_frame_->w2c(p_w, cam_idx));

                if (cam->isInFrame(pc.cast<int>(), 0))
                {
                    cv::Mat img = lidar_selector->img_rgbs[cam_idx];
                    // 获取像素颜色 BGR->RGB
                    Eigen::Vector3f pixel;
                    pixel = lidar_selector->getpixel(img, pc);
                    pointRGB.r = pixel[2];
                    pointRGB.g = pixel[1];
                    pointRGB.b = pixel[0];
                    laserCloudWorldRGB->push_back(pointRGB);
                }
            }
        }
    }

    if(laserCloudWorldRGB->size() > 0)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
    }
    else
    {
        ROS_ERROR("publish_frame_world_rgb: No colored points to publish.");
    }
    // 保存点云到 PCD 
    if (pcd_save_en) {
        *pcl_wait_save += *laserCloudWorldRGB;
    }
}

void publish_frame_world(const ros::Publisher & pubLaserCloudFullRes)
{
    uint size = pcl_wait_pub->points.size();
    if (1)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;

        pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        
        laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mtx_buffer_pointcloud.unlock();
    if (pcd_save_en) *pcl_wait_save_lidar += *pcl_wait_pub;
}

void publish_visual_world_map(const ros::Publisher & pubVisualCloud)
{
    PointCloudXYZI::Ptr laserCloudFullRes(map_cur_frame_point);
    int size = laserCloudFullRes->points.size();
    if (size==0) return;
    // PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

    // for (int i = 0; i < size; i++)
    // {
    //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
    // }
    // mtx_buffer_pointcloud.lock();
    PointCloudXYZI::Ptr pcl_visual_wait_pub(new PointCloudXYZI());
    *pcl_visual_wait_pub = *laserCloudFullRes;
    if (1)//if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*pcl_visual_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubVisualCloud.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mtx_buffer_pointcloud.unlock();
}

void publish_visual_world_sub_map(const ros::Publisher & pubSubVisualCloud)
{
    PointCloudXYZI::Ptr laserCloudFullRes(sub_map_cur_frame_point);
    int size = laserCloudFullRes->points.size();
    if (size==0) return;
    // PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

    // for (int i = 0; i < size; i++)
    // {
    //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
    // }
    // mtx_buffer_pointcloud.lock();
    PointCloudXYZI::Ptr sub_pcl_visual_wait_pub(new PointCloudXYZI());
    *sub_pcl_visual_wait_pub = *laserCloudFullRes;
    if (1)//if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*sub_pcl_visual_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubSubVisualCloud.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mtx_buffer_pointcloud.unlock();
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time::now();
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    #ifdef USE_IKFOM
    //state_ikfom stamp_state = kf.get_x();
    out.position.x = state_point.pos(0);
    out.position.y = state_point.pos(1);
    out.position.z = state_point.pos(2);
    #else
    out.position.x = state.pos_end(0);
    out.position.y = state.pos_end(1);
    out.position.z = state.pos_end(2);
    #endif
    out.orientation.x = geoQuat.x;
    out.orientation.y = geoQuat.y;
    out.orientation.z = geoQuat.z;
    out.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = ros::Time::now();//.ros::Time()fromSec(last_timestamp_lidar);
    set_posestamp(odomAftMapped.pose.pose);
    pubOdomAftMapped.publish(odomAftMapped);
}

void publish_mavros(const ros::Publisher & mavros_pose_publisher)
{
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_odom_frame";
    set_posestamp(msg_body_pose.pose);
    mavros_pose_publisher.publish(msg_body_pose);
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
}

#ifdef USE_IKFOM
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 
        //double search_start = omp_get_wtime();
        /* transform to world frame */
        //pointBodyToWorld_ikfom(&point_body, &point_world, s);
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
    #ifdef USE_ikdtree
        auto &points_near = Nearest_Points[i];
    #else
        auto &points_near = pointSearchInd_surf[i];
    #endif
        
        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
        #ifdef USE_ikdtree
            #ifdef USE_ikdforest
                uint8_t search_flag = 0;                        
                search_flag = ikdforest.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, first_lidar_time, 5);                            
            #else
                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            #endif
        #else
            kdtreeSurfFromMap->nearestKSearch(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
        #endif

            point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;

        #ifdef USE_ikdforest
            point_selected_surf[i] = point_selected_surf[i] && (search_flag == 0);
        #endif
        }

        //kdtree_search_time += omp_get_wtime() - search_start;

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f)) //(planeValid)
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    // cout<<"pca time test: "<<pca_time1<<" "<<pca_time2<<endl;
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i] && (res_last[i] <= 2.0))
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    res_mean_last = total_residual / effct_feat_num;
    // cout << "[ mapping ]: Effective feature num: "<<effct_feat_num<<" res_mean_last "<<res_mean_last<<endl;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    //MatrixXd H(effct_feat_num, 23);
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num); // = VectorXd::Zero(effct_feat_num);
    //VectorXd meas_vec(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be +s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C); // s.rot.conjugate() * norm_vec);
        V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
        //H.row(i) = Eigen::Matrix<double, 1, 23>::Zero();
        ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        //ekfom_data.h_x.block<1, 3>(i, 6) << VEC_FROM_ARRAY(A);
        //ekfom_data.h_x.block<1, 6>(i, 17) << VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);

        /*** Measuremnt: distance to the closest surface/corner ***/
        //meas_vec(i) = - norm_p.intensity;
        ekfom_data.h(i) = -norm_p.intensity;
    }
    //ekfom_data.h_x =H;
    solve_time += omp_get_wtime() - solve_start_;
    //return meas_vec;
}
#endif         

void readParameters(ros::NodeHandle &nh, std::vector<camera_manager::Cameras> &cameras)
{
    ROS_INFO("Start reading camera parameters.");

    // 读取相机列表CamInfo初始化
    XmlRpc::XmlRpcValue cameras_param;
    if (!nh.getParam("cameras", cameras_param)) {
        ROS_ERROR("Failed to get 'cameras' parameter.");
        ros::shutdown();
    }

    int num_cameras = cameras_param.size();
    ROS_INFO("Number of cameras: %d", num_cameras);
    cameras.resize(num_cameras);

    for (int i = 0; i < num_cameras; ++i) {
        cameras[i].cam_id = static_cast<int>(cameras_param[i]["cam_id"]);
        cameras[i].img_topic = static_cast<std::string>(cameras_param[i]["img_topic"]);
        cameras[i].camera_ns = static_cast<std::string>(cameras_param[i]["camera_ns"]); // 读取相机命名空间

        ROS_INFO("Camera %d: ID=%d, Topic=%s, Namespace=%s", 
                 i, cameras[i].cam_id, cameras[i].img_topic.c_str(), cameras[i].camera_ns.c_str());

        // 读取 Rcl
        for (int j = 0; j < 9; ++j) {
            cameras[i].Rcl(j / 3, j % 3) = static_cast<double>(cameras_param[i]["Rcl"][j]);
        }

        // 读取 Pcl
        for (int j = 0; j < 3; ++j) {
            cameras[i].Pcl(j) = static_cast<double>(cameras_param[i]["Pcl"][j]);
        }
        std::cout << "Camera " << i << " Rcl:\n" << cameras[i].Rcl << std::endl;
        std::cout << "Camera " << i << " Pcl:\n" << cameras[i].Pcl.transpose() << std::endl;
        // 读取相机内参
        cameras[i].width  = static_cast<int>(cameras_param[i]["width"]);
        cameras[i].height = static_cast<int>(cameras_param[i]["height"]);
        cameras[i].fx     = static_cast<double>(cameras_param[i]["cam_fx"]);
        cameras[i].fy     = static_cast<double>(cameras_param[i]["cam_fy"]);
        cameras[i].cx     = static_cast<double>(cameras_param[i]["cam_cx"]);
        cameras[i].cy     = static_cast<double>(cameras_param[i]["cam_cy"]);
        cameras[i].d0     = static_cast<double>(cameras_param[i]["cam_d0"]);
        cameras[i].d1     = static_cast<double>(cameras_param[i]["cam_d1"]);
        cameras[i].d2     = static_cast<double>(cameras_param[i]["cam_d2"]);
        cameras[i].d3     = static_cast<double>(cameras_param[i]["cam_d3"]);


    }

    // 读取其他参数
    nh.param<int>("dense_map_enable", dense_map_en, 1);
    nh.param<int>("img_enable", img_en, 1);
    nh.param<int>("lidar_enable", lidar_en, 1);
    nh.param<int>("debug", debug, 0);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<bool>("ncc_en", ncc_en, false);
    nh.param<int>("min_img_count", MIN_IMG_COUNT, 1000);
    nh.param<double>("laser_point_cov", LASER_POINT_COV, 0.001);
    nh.param<double>("img_point_cov", IMG_POINT_COV, 10);
    nh.param<std::string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");   
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<double>("mapping/gyr_cov_scale",gyr_cov_scale,1.0);
    nh.param<double>("mapping/acc_cov_scale",acc_cov_scale,1.0);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, 0);
    nh.param<int>("grid_size", grid_size, 40);
    nh.param<int>("patch_size", patch_size, 4);
    nh.param<double>("outlier_threshold",outlier_threshold,100);
    nh.param<double>("ncc_thre", ncc_thre, 100);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<bool>("pose_output_en", pose_output_en, false);
    nh.param<double>("delta_time", delta_time, 0.0);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    // 使用 ROS_INFO_STREAM 输出所有参数
    ROS_INFO_STREAM("dense_map_enable: " << dense_map_en);
    ROS_INFO_STREAM("img_enable: " << img_en);
    ROS_INFO_STREAM("lidar_enable: " << lidar_en);
    ROS_INFO_STREAM("debug: " << debug);
    ROS_INFO_STREAM("max_iteration: " << NUM_MAX_ITERATIONS);
    ROS_INFO_STREAM("ncc_en: " << ncc_en);
    ROS_INFO_STREAM("min_img_count: " << MIN_IMG_COUNT);
    ROS_INFO_STREAM("laser_point_cov: " << LASER_POINT_COV);
    ROS_INFO_STREAM("img_point_cov: " << IMG_POINT_COV);
    ROS_INFO_STREAM("map_file_path: " << map_file_path);
    ROS_INFO_STREAM("lid_topic: " << lid_topic);
    ROS_INFO_STREAM("imu_topic: " << imu_topic);
    ROS_INFO_STREAM("filter_size_corner: " << filter_size_corner_min);
    ROS_INFO_STREAM("filter_size_surf: " << filter_size_surf_min);
    ROS_INFO_STREAM("filter_size_map: " << filter_size_map_min);
    ROS_INFO_STREAM("cube_side_length: " << cube_len);
    ROS_INFO_STREAM("gyr_cov_scale: " << gyr_cov_scale);
    ROS_INFO_STREAM("acc_cov_scale: " << acc_cov_scale);
    ROS_INFO_STREAM("preprocess/blind: " << p_pre->blind);
    ROS_INFO_STREAM("preprocess/lidar_type: " << p_pre->lidar_type);
    ROS_INFO_STREAM("preprocess/scan_line: " << p_pre->N_SCANS);
    ROS_INFO_STREAM("point_filter_num: " << p_pre->point_filter_num);
    ROS_INFO_STREAM("feature_extract_enable: " << p_pre->feature_enabled);
    ROS_INFO_STREAM("grid_size: " << grid_size);
    ROS_INFO_STREAM("patch_size: " << patch_size);
    ROS_INFO_STREAM("outlier_threshold: " << outlier_threshold);
    ROS_INFO_STREAM("ncc_thre: " << ncc_thre);
    ROS_INFO_STREAM("pcd_save_en: " << pcd_save_en);
    ROS_INFO_STREAM("pose_output_en: " << pose_output_en);
    ROS_INFO_STREAM("delta_time: " << delta_time);
    ROS_INFO("Parameters reading completed.");
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    std::vector<camera_manager::Cameras> cameras_info;
    readParameters(nh, cameras_info); // 已经将相机参数保存到 cameras_info

    // 初始化每个相机的 cam 指针
    for(int i = 0; i < cameras_info.size(); i++) {
        // 使用相机的命名空间加载相机模型
        if(!vk::camera_loader::loadFromRosNs(cameras_info[i].camera_ns, cameras_info[i].cam)) {
            std::stringstream ss;
            ss << "Camera model not correctly specified for camera " << cameras_info[i].cam_id 
               << " with namespace " << cameras_info[i].camera_ns;
            throw std::runtime_error(ss.str());
        }
        ROS_INFO("Camera %d model width %d", 
                 cameras_info[i].cam_id, cameras_info[i].cam->width());
    }

    // 初始化 LidarSelector，相机信息应该初始化到 lidar_selector
    lidar_selection::LidarSelectorPtr lidar_selector(
        new lidar_selection::LidarSelector(
            grid_size,
            new SparseMap,
            cameras_info /* 这里是 std::vector<camera_manager::Cameras> */
        )
    ); 
    lidar_selector->init();
    

    int num_cameras = cameras_info.size();
    ROS_INFO("Number of cameras: %d", num_cameras);
    img_buffers.resize(num_cameras);
    img_time_buffers.resize(num_cameras);
    last_timestamp_imgs.resize(num_cameras, -1.0);

    // 定义一个容器来存储订阅器，以保持它们的生命周期
    std::vector<ros::Subscriber> img_subs;
    img_subs.reserve(num_cameras); // 预留空间
    for(int i = 0; i < num_cameras; i++) {
        // 使用C++11 lambda表达式绑定 cam_id 和 lidar_selector
        ros::Subscriber sub_img = nh.subscribe<sensor_msgs::Image>(
            cameras_info[i].img_topic,  // topic 名
            200,                       // queue_size
            [i, &cameras_info, lidar_selector](const sensor_msgs::ImageConstPtr& msg) {
                img_cbk(msg, i, lidar_selector, cameras_info); 
            }
        );
        ROS_INFO("Subscribing to topic: %s", cameras_info[i].img_topic.c_str());
        if (cameras_info[i].img_topic.empty()) {
            ROS_ERROR("Invalid topic name for camera %d", i);
        }
        img_subs.emplace_back(sub_img);
    }
    pcl_wait_pub->clear();
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ROS_INFO("Subscribing to topic: %s", lid_topic.c_str());

    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ROS_INFO("Subscribing to topic: %s", imu_topic.c_str());

    image_transport::Publisher img_pub = it.advertise("/rgb_img", 1);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100);
    ros::Publisher pubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_visual_map", 100);
    ros::Publisher pubSubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_visual_sub_map", 100);
    ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/aft_mapped_to_init", 10);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 10);

#ifdef DEPLOY
    ros::Publisher mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
#endif
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    #ifndef USE_IKFOM
    VD(DIM_STATE) solution;
    MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
    V3D rot_add, t_add;
    StatesGroup state_propagat;
    PointType pointOri, pointSel, coeff;
    #endif
    //PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    #ifdef USE_ikdforest
        ikdforest.Set_balance_criterion_param(0.6);
        ikdforest.Set_delete_criterion_param(0.5);
        ikdforest.Set_environment(laserCloudDepth,laserCloudWidth,laserCloudHeight,cube_len);
        ikdforest.Set_downsample_param(filter_size_map_min);    
    #endif

    shared_ptr<ImuProcess> p_imu(new ImuProcess());
    Lidar_offset_to_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_rot_to_IMU<<MAT_FROM_ARRAY(extrinR);

    lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;
    lidar_selector->debug = debug;
    lidar_selector->patch_size = patch_size;
    lidar_selector->outlier_threshold = outlier_threshold;
    lidar_selector->ncc_thre = ncc_thre;
    lidar_selector->set_extrinsic(Lidar_offset_to_IMU, Lidar_rot_to_IMU);
    lidar_selector->state = &state;
    lidar_selector->state_propagat = &state_propagat;
    lidar_selector->NUM_MAX_ITERATIONS = NUM_MAX_ITERATIONS;
    lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;
    lidar_selector->img_point_cov = IMG_POINT_COV;
    lidar_selector->ncc_en = ncc_en;
    lidar_selector->init();
    
    p_imu->set_extrinsic(Lidar_offset_to_IMU, Lidar_rot_to_IMU);
    p_imu->set_gyr_cov_scale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
    p_imu->set_acc_cov_scale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
    p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
    p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));

    #ifndef USE_IKFOM
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
    #endif

    #ifdef USE_IKFOM
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
    #endif
    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_tum;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_tum.open(DEBUG_FILE_DIR("camera_pose.txt"),ios::out);

    // if (fout_pre && fout_out)
    //     cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    // else
    //     cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    #ifdef USE_ikdforest
        ikdforest.Set_balance_criterion_param(0.6);
        ikdforest.Set_delete_criterion_param(0.5);
    #endif
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(!sync_packages(LidarMeasures))
        {
            //ROS_INFO("sync_packages failed. Continuing the loop.");
            status = ros::ok();
            cv::waitKey(1);
            rate.sleep();
            continue;
        }

        /*** Packages received ***/
        if (flg_reset)
        {
            ROS_WARN("reset when rosbag play back");
            p_imu->Reset();
            flg_reset = false;
            continue;
        }

        double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

        match_time = kdtree_search_time = kdtree_search_counter = solve_time = solve_const_H_time = svd_time   = 0;
        t0 = omp_get_wtime();

        #ifdef USE_IKFOM
        p_imu->Process(LidarMeasures, kf, feats_undistort);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        #else
        p_imu->Process2(LidarMeasures, state, feats_undistort); 
        state_propagat = state;
        #endif

        if (lidar_selector->debug)
        {
            LidarMeasures.debug_show();
        }

        if (feats_undistort->empty() || (feats_undistort == nullptr))
        {
            if (!fast_lio_is_ready)
            {
                first_lidar_time = LidarMeasures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                LidarMeasures.measures.clear();
                cout<<"FAST-LIO not ready"<<endl;
                continue;
            }
        }
        else
        {
            int size = feats_undistort->points.size();
        }
        fast_lio_is_ready = true;
        flg_EKF_inited = (LidarMeasures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                        false : true;

        if (! LidarMeasures.is_lidar_end) 
        {
            cout<<"[ VIO ]: Raw feature num: "<<pcl_wait_pub->points.size() << "." << endl;

            if (first_lidar_time<10)
            {
                continue;
            }
            if (img_en) {
                ROS_INFO("Image processing enabled. Publishing image and related data.");
                for (int i = 0; i < img_buffers.size(); ++i) {
                    ROS_INFO("Camera %d: Buffer size = %lu, Last timestamp = %f",i, img_buffers[i].size(), last_timestamp_imgs[i]);
                }
                euler_cur = RotMtoEuler(state.rot_end);
                ROS_INFO("Converted rotation matrix to Euler angles: [%f, %f, %f]", euler_cur(0), euler_cur(1), euler_cur(2));
                fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose()*57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose() \
                                <<" "<<state.bias_g.transpose()<<" "<<state.bias_a.transpose()<<" "<<state.gravity.transpose()<< endl;
                lidar_selector->detect(LidarMeasures.measures.back().imgs, pcl_wait_pub);
                ROS_INFO("Completed feature detection.");
                // int size = lidar_selector->map_cur_frame_.size();
                int size_sub = lidar_selector->sub_map_cur_frame_.size();
                ROS_INFO("Number of sub-map frames detected: %d", size_sub);

                // map_cur_frame_point->clear();
                sub_map_cur_frame_point->clear();
                for(int i=0; i<size_sub; i++)
                {
                    PointType temp_map;
                    temp_map.x = lidar_selector->sub_map_cur_frame_[i]->pos_[0];
                    temp_map.y = lidar_selector->sub_map_cur_frame_[i]->pos_[1];
                    temp_map.z = lidar_selector->sub_map_cur_frame_[i]->pos_[2];
                    temp_map.intensity = 0.;
                    sub_map_cur_frame_point->push_back(temp_map);
                }
                ROS_INFO("Added %d points to sub_map_cur_frame_point.", size_sub);

               

                publish_frame_world_rgb(pubLaserCloudFullRes, lidar_selector,cameras_info);
                publish_visual_world_sub_map(pubSubVisualCloud);
                geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
                publish_odometry(pubOdomAftMapped);
                euler_cur = RotMtoEuler(state.rot_end);
                fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose()*57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose() \
                <<" "<<state.bias_g.transpose()<<" "<<state.bias_a.transpose()<<" "<<state.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
            }
            continue;
        }

        /*** Segment the map in lidar FOV ***/
        #ifndef USE_ikdforest            
            lasermap_fov_segment();
        #endif
        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
    #ifdef USE_ikdtree
        /*** initialize the map kdtree ***/
        #ifdef USE_ikdforest
        if (!ikdforest.initialized){
            if(feats_down_body->points.size() > 5){
                ikdforest.Build(feats_down_body->points, true, lidar_end_time);
            }
            continue;                
        }
        int featsFromMapNum = ikdforest.total_size;
        #else
        if(ikdtree.Root_Node == nullptr)
        {
            if(feats_down_body->points.size() > 5)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                ikdtree.Build(feats_down_body->points);
            }
            continue;
        }
        int featsFromMapNum = ikdtree.size();
        #endif
    #else
        if(featsFromMap->points.empty())
        {
            downSizeFilterMap.setInputCloud(feats_down_body);
        }
        else
        {
            downSizeFilterMap.setInputCloud(featsFromMap);
        }
        downSizeFilterMap.filter(*featsFromMap);
        int featsFromMapNum = featsFromMap->points.size();
    #endif
        feats_down_size = feats_down_body->points.size();
        cout<<"[ LIO ]: Raw feature num: "<<feats_undistort->points.size()<<" downsamp num "<<feats_down_size<<" Map num: "<<featsFromMapNum<< "." << endl;

        /*** ICP and iterated Kalman filter update ***/
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        //vector<double> res_last(feats_down_size, 1000.0); // initial //
        res_last.resize(feats_down_size, 1000.0);
        
        t1 = omp_get_wtime();
        if (lidar_en)
        {
            euler_cur = RotMtoEuler(state.rot_end);
            #ifdef USE_IKFOM
            //state_ikfom fout_state = kf.get_x();
            fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose()*57.3 << " " << state_point.pos.transpose() << " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;
            #else
            fout_pre << setw(20) << LidarMeasures.last_update_time  - first_lidar_time << " " << euler_cur.transpose()*57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose() \
            <<" "<<state.bias_g.transpose()<<" "<<state.bias_a.transpose()<<" "<<state.gravity.transpose()<< endl;
            #endif
        }

    #ifdef USE_ikdtree
        if(0)
        {
            PointVector ().swap(ikdtree.PCL_Storage);
            ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
            featsFromMap->clear();
            featsFromMap->points = ikdtree.PCL_Storage;
        }
    #else
        kdtreeSurfFromMap->setInputCloud(featsFromMap);
    #endif

        point_selected_surf.resize(feats_down_size, true);
        pointSearchInd_surf.resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        int  rematch_num = 0;
        bool nearest_search_en = true; //

        t2 = omp_get_wtime();
        
        /*** iterated state estimation ***/
        #ifdef MP_EN
        printf("[ LIO ]: Using multi-processor, used core number: %d.\n", MP_PROC_NUM);
        #endif
        double t_update_start = omp_get_wtime();
        #ifdef USE_IKFOM
        double solve_H_time = 0;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
        //state_ikfom updated_state = kf.get_x();
        state_point = kf.get_x();
        //euler_cur = RotMtoEuler(state_point.rot.toRotationMatrix());
        euler_cur = SO3ToEuler(state_point.rot);
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        // cout<<"position: "<<pos_lid.transpose()<<endl;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];
        #else

        if(img_en)
        {
            omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
            for(int i=0;i<1;i++) {}
        }

        if(lidar_en)
        {
            for (iterCount = -1; iterCount < NUM_MAX_ITERATIONS && flg_EKF_inited; iterCount++) 
            {
                match_start = omp_get_wtime();
                PointCloudXYZI ().swap(*laserCloudOri);
                PointCloudXYZI ().swap(*corr_normvect);
                // laserCloudOri->clear(); 
                // corr_normvect->clear(); 
                total_residual = 0.0; 

                /** closest surface search and residual computation **/
                #ifdef MP_EN
                    omp_set_num_threads(MP_PROC_NUM);
                    #pragma omp parallel for
                #endif
                // normvec->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    PointType &point_body  = feats_down_body->points[i];
                    PointType &point_world = feats_down_world->points[i];
                    V3D p_body(point_body.x, point_body.y, point_body.z);
                    /* transform to world frame */
                    pointBodyToWorld(&point_body, &point_world);
                    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                    #ifdef USE_ikdtree
                        auto &points_near = Nearest_Points[i];
                    #else
                        auto &points_near = pointSearchInd_surf[i];
                    #endif
                    uint8_t search_flag = 0;  
                    double search_start = omp_get_wtime();
                    if (nearest_search_en)
                    {
                        /** Find the closest surfaces in the map **/
                        #ifdef USE_ikdtree
                            #ifdef USE_ikdforest
                                search_flag = ikdforest.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, first_lidar_time, 5);
                            #else
                                ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                            #endif
                        #else
                            kdtreeSurfFromMap->nearestKSearch(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
                        #endif

                        point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;

                        #ifdef USE_ikdforest
                            point_selected_surf[i] = point_selected_surf[i] && (search_flag == 0);
                        #endif
                        kdtree_search_time += omp_get_wtime() - search_start;
                        kdtree_search_counter ++;                        
                    }


                    // if (!point_selected_surf[i]) continue;


                    // Debug
                    // if (points_near.size()<5) {
                    //     printf("\nERROR: Return Points is less than 5\n\n");
                    //     printf("Target Point is: (%0.3f,%0.3f,%0.3f)\n",point_world.x,point_world.y,point_world.z);
                    // }
                    if (!point_selected_surf[i] || points_near.size() < NUM_MATCH_POINTS) continue;

                    VF(4) pabcd;
                    point_selected_surf[i] = false;
                    if (esti_plane(pabcd, points_near, 0.1f)) //(planeValid)
                    {
                        float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                        if (s > 0.9)
                        {
                            point_selected_surf[i] = true;
                            normvec->points[i].x = pabcd(0);
                            normvec->points[i].y = pabcd(1);
                            normvec->points[i].z = pabcd(2);
                            normvec->points[i].intensity = pd2;
                            res_last[i] = abs(pd2);
                        }
                    }
                }
                // cout<<"pca time test: "<<pca_time1<<" "<<pca_time2<<endl;
                effct_feat_num = 0;
                laserCloudOri->resize(feats_down_size);
                corr_normvect->reserve(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    if (point_selected_surf[i] && (res_last[i] <= 2.0))
                    {
                        laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                        corr_normvect->points[effct_feat_num] = normvec->points[i];
                        total_residual += res_last[i];
                        effct_feat_num ++;
                    }
                }

                res_mean_last = total_residual / effct_feat_num;
                // cout << "[ mapping ]: Effective feature num: "<<effct_feat_num<<" res_mean_last "<<res_mean_last<<endl;
                match_time  += omp_get_wtime() - match_start;
                solve_start  = omp_get_wtime();
                
                /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                MatrixXd Hsub(effct_feat_num, 6);
                VectorXd meas_vec(effct_feat_num);

                for (int i = 0; i < effct_feat_num; i++)
                {
                    const PointType &laser_p  = laserCloudOri->points[i];
                    V3D point_this(laser_p.x, laser_p.y, laser_p.z);
                    point_this = Lidar_rot_to_IMU*point_this + Lidar_offset_to_IMU;
                    M3D point_crossmat;
                    point_crossmat<<SKEW_SYM_MATRX(point_this);

                    /*** get the normal vector of closest surface/corner ***/
                    const PointType &norm_p = corr_normvect->points[i];
                    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

                    /*** calculate the Measuremnt Jacobian matrix H ***/
                    V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
                    Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;

                    /*** Measuremnt: distance to the closest surface/corner ***/
                    meas_vec(i) = - norm_p.intensity;
                }
                solve_const_H_time += omp_get_wtime() - solve_start;

                MatrixXd K(DIM_STATE, effct_feat_num);

                EKF_stop_flg = false;
                flg_EKF_converged = false;
                
                /*** Iterative Kalman Filter Update ***/
                if (!flg_EKF_inited)
                {
                    cout<<"||||||||||Initiallizing LiDar||||||||||"<<endl;
                    /*** only run in initialization period ***/
                    MatrixXd H_init(MD(9, DIM_STATE)::Zero());
                    MatrixXd z_init(VD(9)::Zero());
                    H_init.block<3,3>(0,0)  = M3D::Identity();
                    H_init.block<3,3>(3,3)  = M3D::Identity();
                    H_init.block<3,3>(6,15) = M3D::Identity();
                    z_init.block<3,1>(0,0)  = - Log(state.rot_end);
                    z_init.block<3,1>(0,0)  = - state.pos_end;

                    auto H_init_T = H_init.transpose();
                    auto &&K_init = state.cov * H_init_T * (H_init * state.cov * H_init_T + \
                                    0.0001 * MD(9, 9)::Identity()).inverse();
                    solution      = K_init * z_init;

                    // solution.block<9,1>(0,0).setZero();
                    // state += solution;
                    // state.cov = (MatrixXd::Identity(DIM_STATE, DIM_STATE) - K_init * H_init) * state.cov;

                    state.resetpose();
                    EKF_stop_flg = true;
                }
                else
                {
                    auto &&Hsub_T = Hsub.transpose();
                    auto &&HTz = Hsub_T * meas_vec;
                    H_T_H.block<6,6>(0,0) = Hsub_T * Hsub;
                    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
                    MD(DIM_STATE, DIM_STATE) &&K_1 = \
                            (H_T_H + (state.cov / LASER_POINT_COV).inverse()).inverse();
                    G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
                    auto vec = state_propagat - state;
                    solution = K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);

                    int minRow, minCol;
                    if(0)//if(V.minCoeff(&minRow, &minCol) < 1.0f)
                    {
                        VD(6) V = H_T_H.block<6,6>(0,0).eigenvalues().real();
                        cout<<"!!!!!! Degeneration Happend, eigen values: "<<V.transpose()<<endl;
                        EKF_stop_flg = true;
                        solution.block<6,1>(9,0).setZero();
                    }

                    state += solution;

                    rot_add = solution.block<3,1>(0,0);
                    t_add   = solution.block<3,1>(3,0);

                    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015))
                    {
                        flg_EKF_converged = true;
                    }

                    deltaR = rot_add.norm() * 57.3;
                    deltaT = t_add.norm() * 100;
                }
                euler_cur = RotMtoEuler(state.rot_end);
                

                /*** Rematch Judgement ***/
                nearest_search_en = false;
                if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))))
                {
                    nearest_search_en = true;
                    rematch_num ++;
                }

                /*** Convergence Judgements and Covariance Update ***/
                if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1)))
                {
                    if (flg_EKF_inited)
                    {
                        /*** Covariance Update ***/
                        // G.setZero();
                        // G.block<DIM_STATE,6>(0,0) = K * Hsub;
                        state.cov = (I_STATE - G) * state.cov;
                        total_distance += (state.pos_end - position_last).norm();
                        position_last = state.pos_end;
                        geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                    (euler_cur(0), euler_cur(1), euler_cur(2));

                        VD(DIM_STATE) K_sum  = K.rowwise().sum();
                        VD(DIM_STATE) P_diag = state.cov.diagonal();
                        // cout<<"K: "<<K_sum.transpose()<<endl;
                        // cout<<"P: "<<P_diag.transpose()<<endl;
                        // cout<<"position: "<<state.pos_end.transpose()<<" total distance: "<<total_distance<<endl;
                    }
                    EKF_stop_flg = true;
                }
                solve_time += omp_get_wtime() - solve_start;

                if (EKF_stop_flg)   break;
            }
        }
        
        // cout<<"[ mapping ]: iteration count: "<<iterCount+1<<endl;
        #endif

        if(pose_output_en)
        {
            // 遍历所有相机
            for(int i = 0; i < cameras_info.size(); i++) 
            {
                int cam_id = cameras_info[i].cam_id; // 获取相机ID
                SE3 T_cam_world;

                try {
                    // 获取当前相机的 T_f_w_ 变换
                    T_cam_world = lidar_selector->new_frame_->T_f_w_[cam_id];
                }
                catch (const std::out_of_range& e) {
                    ROS_ERROR("Transform for camera ID %d not found: %s", cam_id, e.what());
                    continue; // 如果找不到对应的变换，跳过该相机
                }

                Eigen::Vector3d t = T_cam_world.translation();  
                Eigen::Quaterniond q(T_cam_world.rotation_matrix()); 

                // 输出格式：时间戳 相机ID tx ty tz qx qy qz qw
                fout_tum << std::fixed << std::setprecision(6)
                        << LidarMeasures.lidar_beg_time << " "
                        << cam_id << " "
                        << t.x() << " " << t.y() << " " << t.z() << " "
                        << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                        << std::endl;
            }
        }

        // SaveTrajTUM(LidarMeasures.lidar_beg_time, state.rot_end, state.pos_end);
        double t_update_end = omp_get_wtime();
        /******* Publish odometry *******/
        euler_cur = RotMtoEuler(state.rot_end);
        geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
        publish_odometry(pubOdomAftMapped);

        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();
        map_incremental();
        t5 = omp_get_wtime();
        kdtree_incremental_time = t5 - t3 + readd_time;
        /******* Publish points *******/

        PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);          
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_pub = *laserCloudWorld;

        if(!img_en) publish_frame_world(pubLaserCloudFullRes);
        // publish_visual_world_map(pubVisualCloud);
        publish_effect_world(pubLaserCloudEffect);
        // publish_map(pubLaserCloudMap);
        publish_path(pubPath);
        #ifdef DEPLOY
        publish_mavros(mavros_pose_publisher);
        #endif

        /*** Debug variables ***/
        frame_num ++;
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
        aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
        #ifdef USE_IKFOM
        aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
        #else
        aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time)/frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_const_H_time / frame_num;
        //cout << "construct H:" << aver_time_const_H_time << std::endl;
        #endif
        // aver_time_consu = aver_time_consu * 0.9 + (t5 - t0) * 0.1;
        T1[time_log_counter] = LidarMeasures.lidar_beg_time;
        s_plot[time_log_counter] = aver_time_consu;
        s_plot2[time_log_counter] = kdtree_incremental_time;
        s_plot3[time_log_counter] = kdtree_search_time/kdtree_search_counter;
        s_plot4[time_log_counter] = featsFromMapNum;
        s_plot5[time_log_counter] = t5 - t0;
        time_log_counter ++;
        // cout<<"[ mapping ]: time: fov_check "<< fov_check_time <<" fov_check and readd: "<<t1-t0<<" match "<<aver_time_match<<" solve "<<aver_time_solve<<" ICP "<<t3-t1<<" map incre "<<t5-t3<<" total "<<aver_time_consu << "icp:" << aver_time_icp << "construct H:" << aver_time_const_H_time <<endl;
        printf("[ LIO ]: time: fov_check: %0.6f fov_check and readd: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f icp: %0.6f construct H: %0.6f.\n",fov_check_time,t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
        if (lidar_en)
        {
            euler_cur = RotMtoEuler(state.rot_end);
            #ifdef USE_IKFOM
            fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose()*57.3 << " " << state_point.pos.transpose() << " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
            #else
            fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose()*57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose() \
            <<" "<<state.bias_g.transpose()<<" "<<state.bias_a.transpose()<<" "<<state.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
            #endif
        }
        // dump_lio_state_to_log(fp);
    }
    
         /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("rgb_scan_all.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current rgb scan saved" << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (pcl_wait_save_lidar->size() > 0 && pcd_save_en)
    {
        string file_name = string("intensity_sacn_all.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current intensity scan saved" << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_lidar);
    }

    fout_out.close();
    fout_pre.close();
    fout_tum.close();

    return 0;
}
