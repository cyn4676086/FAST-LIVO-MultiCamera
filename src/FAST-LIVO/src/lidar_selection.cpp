#include "lidar_selection.h"
#include "camera_manager.h"
#include <iterator>
#include <vector>
namespace lidar_selection {

LidarSelector::LidarSelector(int gridsize, SparseMap* sparsemap, std::vector<camera_manager::Cameras> &cameras_info)
    : grid_size(gridsize), sparse_map(sparsemap)
{
    downSizeFilter.setLeafSize(0.2f, 0.2f, 0.2f);
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    Rli = M3D::Identity();
    Pli = V3D::Zero();
    //grid_size = gridsize;
    // 多相机初始化
    for(const auto& info : cameras_info) {
        camera_manager::Cameras cam_info;
        cam_info.cam = info.cam;
        cam_info.cam_id = info.cam_id;

        // 内参
        cam_info.fx = cam_info.cam->errorMultiplier2();
        cam_info.fy = cam_info.cam->errorMultiplier() / (4. * cam_info.fx);
        
        cam_info.cx = info.cx;
        cam_info.cy = info.cy;

        // 图像尺寸
        cam_info.width = info.width;
        cam_info.height = info.height;

        // 外参
        cam_info.Rcl = info.Rcl;
        cam_info.Pcl = info.Pcl;

        cam_info.Rci = cam_info.Rcl * Rli;
        cam_info.Pci = cam_info.Rcl * Pli + cam_info.Pcl;
        //初始化为0 在状态更新中计算
        cam_info.Rcw = M3D::Identity();
        cam_info.Pcw = V3D::Zero();

        // 雅可比矩阵
        cam_info.Jdphi_dR = cam_info.Rci;
        cam_info.Jdp_dR = -cam_info.Rci * (SKEW_SYM_MATRX(-cam_info.Rci.transpose() * cam_info.Pci));
        cam_info.Jdp_dt = M3D::Identity();
        // 添加到相机列表
        cameras.push_back(cam_info);
    }
}
LidarSelector::~LidarSelector() 
{
    delete[] align_flag;
    delete[] grid_num;
    delete[] map_index;
    delete[] map_value;

    delete sparse_map;
    delete sub_sparse_map;

    unordered_map<int, Warp*>().swap(Warp_map);
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<VOXEL_KEY, VOXEL_POINTS*>().swap(feat_map);  
}
void LidarSelector::init()
{
    //图像尺寸一致 建图参数一致
    width = cameras[0].cam->width();
    height = cameras[0].cam->height();
    sub_sparse_map = new SubSparseMap;
    grid_n_width = static_cast<int>(width/grid_size);
    grid_n_height = static_cast<int>(height/grid_size);
    length = grid_n_width * grid_n_height;
    grid_num = new int[length];
    map_index = new int[length];
    map_value = new float[length];
    map_dist = new float[length];

    

    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    memset(map_value, 0, sizeof(float)*length);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size/2);
    patch_cache.resize(patch_size_total);
    stage_ = STAGE_FIRST_FRAME;
    pg_down.reset(new PointCloudXYZI());

    // Initialize weight functions
    weight_scale_ = 10.0f;
    weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
    // weight_function_.reset(new vk::robust_cost::TukeyWeightFunction());
    scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
    // scale_estimator_.reset(new vk::robust_cost::MADScaleEstimator());
}


//雷达和IMU外参 保持不变
void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void LidarSelector::reset_grid()
{
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    fill_n(map_dist, length, 10000);
    std::vector<PointPtr>(length).swap(voxel_points_);
    std::vector<V3D>(length).swap(add_voxel_points_);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
}

void LidarSelector::dpi(const V3D& p, MD(2,3)& J, double fx, double fy) {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1.0 / p[2];
    const double z_inv_2 = z_inv * z_inv;
    
    J(0,0) = fx * z_inv;
    J(0,1) = 0.0;
    J(0,2) = -fx * x * z_inv_2;
    
    J(1,0) = 0.0;
    J(1,1) = fy * z_inv;
    J(1,2) = -fy * y * z_inv_2;
}


void LidarSelector::getpatch(const cv::Mat& img, const Eigen::Vector2d& pc, float* patch_tmp, int level, int cam_id) 
{
    camera_manager::Cameras& cam_info = cameras[cam_id];
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale =  (1 << level);
    const int u_ref_i = floorf(u_ref / scale) * scale; 
    const int v_ref_i = floorf(v_ref / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0f - subpix_v_ref);
    const float w_ref_bl = (1.0f - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    for (int x = 0; x < patch_size; x++) 
    {
        int row = v_ref_i - patch_size_half * scale + x * scale;
        if(row < 0 || row >= cam_info.height - scale) continue;
        uint8_t* img_ptr = img.data + row * cam_info.width + (u_ref_i - patch_size_half * scale);
        for (int y = 0; y < patch_size; y++, img_ptr += scale)
        {
            int col = u_ref_i - patch_size_half * scale + y * scale;
            if(col < 0 || col >= cam_info.width - scale) 
            {
                patch_tmp[patch_size_total * level + x * patch_size + y] = 0.0f;
                continue;
            }
            patch_tmp[patch_size_total * level + x * patch_size + y] = 
                w_ref_tl * img_ptr[0] + 
                w_ref_tr * img_ptr[scale] + 
                w_ref_bl * img_ptr[scale * cam_info.width] + 
                w_ref_br * img_ptr[scale * cam_info.width + scale];
        }
    }
}


/**
 * @brief 将新的激光点加入全局地图 (覆盖对所有相机的投影)
 * @param imgs   多相机的灰度图像数组(必须与 cameras.size() 相同)
 * @param pg     当前帧激光雷达点云
 */
void LidarSelector::addSparseMap( std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg)
{
    // 1) 重置栅格
    reset_grid();

    // 调试统计量
    int num_projected_points = 0;   // 多少个点落在图像范围内(任意相机)
    int num_map_updated      = 0;   // 多少次把 map_value[index] 更新
    int num_final_added      = 0;   // 最终真正插入 feat_map 的点数量

    // （3）对每个激光点，对所有相机投影
    for(int i = 0; i < (int)pg->size(); i++)
    {
        Eigen::Vector3d pt_w(pg->points[i].x, pg->points[i].y, pg->points[i].z);

        bool projected_any_cam = false; // 标记：此点是否在某相机落入有效范围

        for(int cam_idx = 0; cam_idx < (int)cameras.size(); cam_idx++)
        {
            Eigen::Vector2d pc = new_frame_->w2c(pt_w, cam_idx);
            

            // 判断是否在图像内
            if(cameras[cam_idx].cam->isInFrame(pc.cast<int>(), (patch_size_half + 1)*8))
            {
                

                // 属于有效投影
                projected_any_cam = true;
                int gx = (int)(pc[0] / grid_size);
                int gy = (int)(pc[1] / grid_size);
                

                int index = gx*grid_n_height + gy;

                float cur_value = vk::shiTomasiScore(imgs[cam_idx], pc[0], pc[1]);
                

                // 对比当前栅格最大得分
                if(cur_value > map_value[index])
                {
                    map_value[index] = cur_value;
                    add_voxel_points_[index] = pt_w;
                    grid_num[index] = TYPE_POINTCLOUD;
                    num_map_updated++;
                    
                }
            }
        } // end for cam_idx

        if(projected_any_cam)
        {
            num_projected_points++;
        }
    } // end for i in pg->size()


    // （4）第二次循环: 把 grid_num[] == TYPE_POINTCLOUD 的点插入 feat_map
    int add_count = 0;
    for(int i = 0; i < length; i++)
    {
        if(grid_num[i] == TYPE_POINTCLOUD)
        {
            Eigen::Vector3d pt = add_voxel_points_[i];
            float score        = map_value[i];

            // 这里随便用相机0做 feature，也可记录 best_cam_idx ...
            Eigen::Vector2d pc = new_frame_->w2c(pt, 0);

            PointPtr pt_new(new Point(pt));
            Eigen::Vector3d f = cameras[0].cam->cam2world(pc);

            FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_[0], score, 0));
            ftr_new->img = imgs[0]; 
            ftr_new->id_ = new_frame_->id_;

            // 调试：显示一下我们要插入的点坐标和得分
            //ROS_WARN("[addSparseMap] => AddPoint: idx=%d, xyz=[%.2f, %.2f, %.2f], score=%.2f", i, pt.x(), pt.y(), pt.z(), score);

            pt_new->addFrameRef(ftr_new);
            pt_new->value = map_value[i];
            AddPoint(pt_new);
            add_count++;
        }
    }
    num_final_added = add_count;

}




void LidarSelector::AddPoint(PointPtr pt_new)
{
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pt_w[j] / voxel_size;
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_KEY position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->voxel_points.push_back(pt_new);
      iter->second->count++;
    }
    else
    {
      VOXEL_POINTS *ot = new VOXEL_POINTS(0);
      ot->voxel_points.push_back(pt_new);
      feat_map[position] = ot;
    }
}

void LidarSelector::getWarpMatrixAffine(
    const vk::AbstractCamera& cam,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,    // the corresponding pyrimid level of px_ref
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref*depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)*(1<<pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)*(1<<pyramid_level)));
//   Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
//   Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref*(xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref*(xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

void LidarSelector::warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int pyramid_level,
    const int halfpatch_size,
    float* patch)
{
  const int patch_size = halfpatch_size*2 ;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if(isnan(A_ref_cur(0,0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }
//   Perform the warp on a larger patch.
//   float* patch_ptr = patch;
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref) / (1<<pyramid_level);
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  for (int y=0; y<patch_size; ++y)
  {
    for (int x=0; x<patch_size; ++x)//, ++patch_ptr)
    {
      // P[patch_size_total*level + x*patch_size+y]
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
      px_patch *= (1<<search_level);
      px_patch *= (1<<pyramid_level);
      const Vector2f px(A_ref_cur*px_patch + px_ref.cast<float>());
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        patch[patch_size_total*pyramid_level + y*patch_size+x] = 0;
        // *patch_ptr = 0;
      else
        patch[patch_size_total*pyramid_level + y*patch_size+x] = (float) vk::interpolateMat_8u(img_ref, px[0], px[1]);
        // *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

double LidarSelector::NCC(float* ref_patch, float* cur_patch, int patch_size)
{    
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref =  sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr =  sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) 
    {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

int LidarSelector::getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();

  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

#ifdef FeatureAlign
void LidarSelector::createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref)
{
  float* ref_patch_ptr = patch_ref;
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)
  {
    float* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
#endif



void LidarSelector::addFromSparseMap(std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg)
{
    // 1. 检查 feat_map 是否为空
    if(feat_map.empty()) {
        ROS_WARN("feat_map is empty.");
        return;
    } else {
        ROS_INFO("feat_map size = %lu", feat_map.size());
    }

    // 2. 下采样点云
    pg_down->reserve(feat_map.size());
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);

    // 3. 重置栅格和相关数据结构
    reset_grid();
    memset(map_value, 0, sizeof(float) * length);

    sub_sparse_map->reset();
    std::deque<PointPtr>().swap(sub_map_cur_frame_);

    float voxel_size = 0.5f;

    // 4. 重置临时特征图和Warp映射
    sub_feat_map.clear();
    Warp_map.clear();

    // 5. 为每个相机创建深度图
    std::vector<cv::Mat> depth_imgs;
    depth_imgs.reserve(cameras.size());
    for(size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
        depth_imgs.emplace_back(cv::Mat::zeros(cameras[cam_idx].height, cameras[cam_idx].width, CV_32FC1));
    }

    // 获取每个深度图的数据指针
    std::vector<float*> depth_iters;
    depth_iters.reserve(cameras.size());
    for(size_t cam_idx = 0; cam_idx < depth_imgs.size(); ++cam_idx) {
        depth_iters.emplace_back(reinterpret_cast<float*>(depth_imgs[cam_idx].data));
    }

    int loc_xyz[3];

    // 6. 遍历下采样后的点云，投影到每个相机并填充深度图
    for(size_t i = 0; i < pg_down->size(); ++i)
    {
        // 将点转换为世界坐标
        Eigen::Vector3d pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);

        // 计算对应的voxel key
        for(int j = 0; j < 3; ++j)
        {
            loc_xyz[j] = static_cast<int>(floor(pt_w[j] / voxel_size));
        }
        VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        // 更新sub_feat_map
        if(sub_feat_map.find(position) == sub_feat_map.end()) {
            sub_feat_map[position] = 1.0f;
        }
        
        // 遍历所有相机，进行投影
        for(size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx)
        {
            // 将点转换到当前相机坐标系
            Eigen::Vector3d pt_cam = new_frame_->w2f(pt_w, cam_idx);

            Eigen::Vector2d px;
            if(pt_cam[2] > 0)
            {
                // 计算像素坐标
                px[0] = cameras[cam_idx].fx * pt_cam[0] / pt_cam[2] + cameras[cam_idx].cx;
                px[1] = cameras[cam_idx].fy * pt_cam[1] / pt_cam[2] + cameras[cam_idx].cy;

                // 检查点是否在图像内（带有一定的边缘填充）
                if(cameras[cam_idx].cam->isInFrame(px.cast<int>(), (patch_size_half + 1) * 8))
                {
                    int col = static_cast<int>(px[0]);
                    int row = static_cast<int>(px[1]);

                    // 确保像素坐标在图像范围内
                    if(col >= 0 && col < cameras[cam_idx].width && row >=0 && row < cameras[cam_idx].height)
                    {
                        float depth = static_cast<float>(pt_cam[2]);
                        depth_iters[cam_idx][cameras[cam_idx].width * row + col] = depth;
                    }
                }
            }
        }
    }
    

    // 7. 遍历sub_feat_map，处理每个voxel中的点
    for(auto& iter : sub_feat_map)
    {   
        VOXEL_KEY position = iter.first;
        auto corre_voxel = feat_map.find(position);

        if(corre_voxel != feat_map.end())
        {
            std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
            int voxel_num = voxel_points.size();
            for(auto& pt : voxel_points)
            {
                if(pt == nullptr) {
                    ROS_ERROR("pt null");
                    continue;
                }

                // 寻找最佳相机（最近的深度）
                int best_cam_idx = -1;
                float min_depth = std::numeric_limits<float>::max();
                Eigen::Vector3d best_pt_cam;
                Eigen::Vector2d best_px;

                for(size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx)
                {
                    Eigen::Vector3d pt_cam = new_frame_->w2f(pt->pos_, cam_idx);
                    

                    if(pt_cam[2] <= 0) {
                        continue;
                    }

                    Eigen::Vector2d px;
                    px[0] = cameras[cam_idx].fx * pt_cam[0] / pt_cam[2] + cameras[cam_idx].cx;
                    px[1] = cameras[cam_idx].fy * pt_cam[1] / pt_cam[2] + cameras[cam_idx].cy;


                    if(cameras[cam_idx].cam->isInFrame(px.cast<int>(), (patch_size_half + 1) * 8))
                    {
                        float depth = static_cast<float>(pt_cam[2]);
                        if(depth < min_depth)
                        {
                            min_depth = depth;
                            best_cam_idx = cam_idx;
                            best_pt_cam = pt_cam;
                            best_px = px;
                        }
                    }
                }
                if(best_cam_idx!= -1)
                {
                    // 计算栅格索引
                    int grid_x = static_cast<int>(best_px[0] / grid_size);
                    int grid_y = static_cast<int>(best_px[1] / grid_size);
                    int index = grid_x * grid_n_height + grid_y;

                    grid_num[index] = TYPE_MAP;
                    Vector3d obs_vec(new_frame_->pos(best_cam_idx) - pt->pos_); // 传入cam_id

                    float cur_dist = static_cast<float>(obs_vec.norm());
                    float cur_value = pt->value;

                    if(cur_dist <= map_dist[index])
                    {
                        map_dist[index] = cur_dist;
                        voxel_points_[index] = pt;
                    }

                    if(cur_value >= map_value[index])
                    {
                        map_value[index] = cur_value;
                    }
                }
            }
        } 
    }

    // 8. 遍历所有栅格，进行深度连续性检查和特征匹配
    float total_error = 0.0f; // 用于累计总错误
    int selected_points = 0;   // 记录选中的点数量

    for(int i = 0; i < length; i++) 
    { 
        if(grid_num[i] == TYPE_MAP)
        {
            PointPtr pt = voxel_points_[i];
            if(pt == nullptr) continue;

            // 寻找该点对应的最佳相机
            int best_cam_idx = -1;
            float min_depth = std::numeric_limits<float>::max();
            Eigen::Vector3d best_pt_cam;
            Eigen::Vector2d best_px;

            for(size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx)
            {
                Eigen::Vector3d pt_cam = new_frame_->w2f(pt->pos_, cam_idx);
                if(pt_cam[2] <= 0) continue;

                Eigen::Vector2d px;
                px[0] = cameras[cam_idx].fx * pt_cam[0] / pt_cam[2] + cameras[cam_idx].cx;
                px[1] = cameras[cam_idx].fy * pt_cam[1] / pt_cam[2] + cameras[cam_idx].cy;

                if(cameras[cam_idx].cam->isInFrame(px.cast<int>(), (patch_size_half + 1) * 8))
                {
                    float depth = static_cast<float>(pt_cam[2]);
                    if(depth < min_depth)
                    {
                        min_depth = depth;
                        best_cam_idx = cam_idx;
                        best_pt_cam = pt_cam;
                        best_px = px;
                    }
                }
            }

            if(best_cam_idx == -1)
                continue;

            // 获取最佳相机的深度图
            cv::Mat& depth_img = depth_imgs[best_cam_idx];
            int px_col = static_cast<int>(best_px[0]);
            int px_row = static_cast<int>(best_px[1]);
            
            // 确保像素坐标在深度图范围内
            if(px_col < 0 || px_col >= cameras[best_cam_idx].width || px_row < 0 || px_row >= cameras[best_cam_idx].height)
                continue;

            float depth = depth_img.at<float>(px_row, px_col);
            

            double delta_dist = abs(best_pt_cam[2] - depth);
            bool depth_continuous = false;
            // 检查邻域的深度连续性
            for(int u = -patch_size_half; u <= patch_size_half; u++)
            {
                for(int v = -patch_size_half; v <= patch_size_half; v++)
                {
                    if(u == 0 && v == 0) continue;

                    int neighbor_col = px_col + u;
                    int neighbor_row = px_row + v;

                    // 边界检查
                    if(neighbor_col < 0 || neighbor_col >= cameras[best_cam_idx].width || 
                       neighbor_row < 0 || neighbor_row >= cameras[best_cam_idx].height)
                        continue;

                    float neighbor_depth = depth_img.at<float>(neighbor_row, neighbor_col);
                    if(neighbor_depth == 0.0f)
                        continue;

                    double delta = abs(best_pt_cam[2] - neighbor_depth);
                    if(delta > 1.5)
                    {
                        depth_continuous = true;
                        break;
                    }
                }
                if(depth_continuous) break;
            }
            if(depth_continuous) continue;
            // cv::imshow("depth",depth_img);
            // cv::waitKey(0);


            // 特征匹配和错误检查
            FeaturePtr ref_ftr;

            if(!pt->getCloseViewObs(new_frame_->pos(best_cam_idx), ref_ftr, best_px, best_cam_idx)) continue;

            std::vector<float> patch_wrap(patch_size_total * 3);

            // 获取仿射变换矩阵
            int search_level ;
            Eigen::Matrix2d A_cur_ref_zero;

            auto iter_warp = Warp_map.find(ref_ftr->id_);
            if(iter_warp != Warp_map.end())
            {
                search_level = iter_warp->second->search_level;
                A_cur_ref_zero = iter_warp->second->A_cur_ref;
            }
            else
            {
                getWarpMatrixAffine(*cameras[best_cam_idx].cam, ref_ftr->px, ref_ftr->f, 
                                    (ref_ftr->pos() - pt->pos_).norm(), 
                                    new_frame_->T_f_w_[best_cam_idx] * ref_ftr->T_f_w_.inverse(), 
                                    0, 0, patch_size_half, A_cur_ref_zero);
                
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);

                Warp* ot = new Warp(search_level, A_cur_ref_zero);
                Warp_map[ref_ftr->id_] = ot;
            }
            // cv::imshow("ref_ftr",ref_ftr->img);
            // cv::waitKey(0);
            // 仿射变换生成patch_wrap
            for(int pyramid_level = 0; pyramid_level <= 2; pyramid_level++)
            {                
                warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, 
                           search_level, pyramid_level, patch_size_half, patch_wrap.data());
            }

            // 获取当前帧的patch
            getpatch(imgs[best_cam_idx], best_px, patch_cache.data(), 0, best_cam_idx);

            // NCC检查
            if(ncc_en)
            {
                double ncc = NCC(patch_wrap.data(), patch_cache.data(), patch_size_total);
                if(ncc < ncc_thre) continue;
            }

            // SSE检查
            float error = 0.0f;
            for(int ind = 0; ind < patch_size_total; ind++) 
            {
                error += (patch_wrap[ind] - patch_cache[ind]) * (patch_wrap[ind] - patch_cache[ind]);
            }
            if(error > outlier_threshold * patch_size_total) {
                continue;
            }

            // 将通过检查的点添加到sub_sparse_map
            sub_map_cur_frame_.push_back(pt);
            sub_sparse_map->propa_errors.push_back(error);
            sub_sparse_map->search_levels.push_back(search_level);
            sub_sparse_map->errors.push_back(error);
            sub_sparse_map->index.push_back(i);  
            sub_sparse_map->voxel_points.push_back(pt);
            sub_sparse_map->patch.push_back(std::move(patch_wrap));

           
        }
    }
    ROS_INFO ("[ VIO ]: choose %d points from sub_sparse_map.\n", int(sub_sparse_map->index.size()));
}



#ifdef FeatureAlign
bool LidarSelector::align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    int index)
{
#ifdef __ARM_NEON__
  if(!no_simd)
    return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
#endif

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged=false;

  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H; H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_+2;
  float* it_dx = ref_patch_dx;
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y) 
  {
    float* it = ref_patch_with_border + (y+1)*ref_step + 1; 
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
    {
      Vector3f J;
      J[0] = 0.5 * (it[1] - it[-1]); 
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); 
      J[2] = 1; 
      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose(); 
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;//0.03*0.03
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  chi2 = sub_sparse_map->propa_errors[index];
  Vector3f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = floor(u);
    int v_r = floor(v);
    if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    float new_chi2 = 0.0;
    Vector3f Jres; Jres.setZero();
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_; 
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
        float res = search_pixel - *it_ref + mean_diff;
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);
        Jres[2] -= res;
        new_chi2 += res*res;
      }
    }

    if(iter > 0 && new_chi2 > chi2)
    {
    //   cout << "error increased." << endl;
      u -= update[0];
      v -= update[1];
      break;
    }
    chi2 = new_chi2;

    sub_sparse_map->align_errors[index] = new_chi2;

    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

void LidarSelector::FeatureAlignment(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    memset(align_flag, 0, length);
    int FeatureAlignmentNum = 0;
       
    for (int i=0; i<total_points; i++) 
    {
        bool res;
        int search_level = sub_sparse_map->search_levels[i];
        Vector2d px_scaled(sub_sparse_map->px_cur[i]/(1<<search_level));
        res = align2D(new_frame_->img[search_level], sub_sparse_map->patch_with_border[i], sub_sparse_map->patch[i],
                        20, px_scaled, i);
        sub_sparse_map->px_cur[i] = px_scaled * (1<<search_level);
        if(res)
        {
            align_flag[i] = 1;
            FeatureAlignmentNum++;
        }
    }
}
#endif


float LidarSelector::UpdateState(const std::vector<cv::Mat>& imgs, float total_residual, int level) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points == 0) 
    {
        ROS_WARN("No points in sub_sparse_map, returning 0.");
        return 0.0f;
    }

    StatesGroup old_state = (*state);
    bool EKF_end = false;

    float error = 0.0f;
    float last_error = total_residual;
    n_meas_ = 0;

    // 多相机时的残差向量/雅可比大小
    const int H_DIM = total_points * patch_size_total * imgs.size();

    // 分配 z, H_sub
    VectorXd z(H_DIM);
    z.setZero();
    H_sub.resize(H_DIM, 6);
    H_sub.setZero();

    for (int iteration = 0; iteration < NUM_MAX_ITERATIONS; iteration++) 
    {
        ROS_INFO("Iteration %d, Total Error: %f", iteration, last_error);

        // 每次迭代初始化
        H_sub.setZero();
        z.setZero();
        n_meas_ = 0;
        error = 0.0f;

        // 从当前状态获取姿态
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);

        // ------- 遍历所有相机 -------
        for (size_t cam_idx = 0; cam_idx < imgs.size(); cam_idx++)
        {
            camera_manager::Cameras& cam = cameras[cam_idx];
            const cv::Mat& img = imgs[cam_idx];
            if (img.empty())
            {
                ROS_WARN("Empty image at cam_idx=%lu, skip this camera.", cam_idx);
                continue;
            }

            // 相机到世界的变换
            M3D Rcw = cam.Rci * Rwi.transpose();
            V3D Pcw = -cam.Rci * Rwi.transpose() * Pwi + cam.Pci;
            M3D Jdp_dt = cam.Rci * Rwi.transpose();

            // 给该相机预留行段偏移
            // 避免多相机写同一块空间
            const int row_offset_cam = cam_idx * (total_points * patch_size_total);

            // ------- 遍历每个点 -------
            for (int i = 0; i < total_points; i++) 
            {
                PointPtr pt = sub_sparse_map->voxel_points[i];
                if (!pt) continue;

                // 每个点先置 patch_error=0
                float patch_error = 0.0f;

                // 世界->相机坐标
                V3D pf = Rcw * pt->pos_ + Pcw;
                // 若深度 <=0，直接跳过
                if (pf[2] <= 0.0)
                {
                    continue;
                }
                // 若深度过大/过小，也可跳过(可选)
                if (pf[2] < 1e-4 || pf[2] > 1e5) 
                {
                    // 防止数值极端
                    continue;
                }

                // 投影到像素坐标
                Eigen::Vector2d pc = cam.cam->world2cam(pf);
                // 若像素坐标爆炸(>1e5)，跳过
                if (std::fabs(pc[0])>1e5 || std::fabs(pc[1])>1e5) 
                {
                    continue;
                }

                // 计算 dpi
                MD(2,3) Jdpi;
                dpi(pf, Jdpi, cam.fx, cam.fy);

                // p_hat
                M3D p_hat;
                p_hat << SKEW_SYM_MATRX(pf);

                // scale=1
                const int scale = 1;
                int u_ref_i = static_cast<int>(floorf(pc[0] / scale) * scale);
                int v_ref_i = static_cast<int>(floorf(pc[1] / scale) * scale);

                // 基本边界检查(含patch_size_half范围)
                if (u_ref_i - patch_size_half < 0 || u_ref_i + patch_size_half >= img.cols ||
                    v_ref_i - patch_size_half < 0 || v_ref_i + patch_size_half >= img.rows )
                {
                    continue;
                }

                // 取参考patch
                std::vector<float>& P = sub_sparse_map->patch[i];

                // 计算双线性插值系数
                float subpix_u_ref = (pc[0] - (float)u_ref_i)/scale;
                float subpix_v_ref = (pc[1] - (float)v_ref_i)/scale;
                float w_ref_tl = (1.f - subpix_u_ref)*(1.f - subpix_v_ref);
                float w_ref_tr = subpix_u_ref*(1.f - subpix_v_ref);
                float w_ref_bl = (1.f - subpix_u_ref)*subpix_v_ref;
                float w_ref_br = subpix_u_ref*subpix_v_ref;

                //ROS_INFO("Processing point %d (cam_idx=%lu): (u_ref, v_ref)=(%.3f,%.3f), w=[%.3f, %.3f, %.3f, %.3f]",i, cam_idx, pc[0], pc[1], w_ref_tl, w_ref_tr, w_ref_bl, w_ref_br);

                // ------- 遍历patch x方向 -------
                for (int x = 0; x < patch_size; x++)
                {
                    int row_img = (v_ref_i - patch_size_half) + x*scale;
                    // 行偏移(针对当前相机+当前点)
                    int row_offset_pt = row_offset_cam + i*patch_size_total + x*patch_size;

                    // 行首指针
                    uint8_t* row_ptr = (uint8_t*)img.data + row_img*img.cols + (u_ref_i - patch_size_half);

                    // ------- 遍历patch y方向 -------
                    for (int y = 0; y < patch_size; y++)
                    {
                        int col_img = (u_ref_i - patch_size_half) + y*scale;
                        int row_idx = row_offset_pt + y;

                        // 检查 row_idx
                        if (row_idx<0 || row_idx>=H_DIM)
                        {
                            ROS_ERROR("row_idx out of range! row_idx=%d, H_DIM=%d", row_idx, H_DIM);
                            continue;
                        }

                        // 当前像素指针
                        uint8_t* img_ptr = row_ptr + y*scale;

                        // 再次检查周边像素
                        if (col_img-1<0 || col_img+1>=img.cols || row_img-1<0 || row_img+1>=img.rows)
                        {
                            // 无法算du,dv，跳过
                            continue;
                        }

                        // 计算 du
                        float du = 0.5f * (
                            ( w_ref_tl * img_ptr[ scale ]
                              + w_ref_tr * img_ptr[ scale*2 ]
                              + w_ref_bl * img_ptr[ scale*img.cols + scale ]
                              + w_ref_br * img_ptr[ scale*img.cols + scale*2] )
                            - ( w_ref_tl * img_ptr[ -scale ]
                              + w_ref_tr * img_ptr[ 0 ]
                              + w_ref_bl * img_ptr[ scale*img.cols - scale ]
                              + w_ref_br * img_ptr[ scale*img.cols ] )
                        );

                        // 计算 dv
                        float dv = 0.5f * (
                            ( w_ref_tl * img_ptr[ scale*img.cols ]
                              + w_ref_tr * img_ptr[ scale + scale*img.cols ]
                              + w_ref_bl * img_ptr[ img.cols*scale*2 ]
                              + w_ref_br * img_ptr[ img.cols*scale*2 + scale] )
                            - ( w_ref_tl * img_ptr[ -scale*img.cols ]
                              + w_ref_tr * img_ptr[ -scale*img.cols + scale ]
                              + w_ref_bl * img_ptr[ 0 ]
                              + w_ref_br * img_ptr[ scale ] )
                        );

                        MD(1,2) Jimg;
                        Jimg << du, dv;
                        Jimg *= (1.0f / (float)scale);

                        // Jdphi, Jdp, ...
                        MD(1,3) Jdphi = Jimg * Jdpi * p_hat;
                        MD(1,3) Jdp   = -Jimg * Jdpi;
                        MD(1,3) JdR   = Jdphi*cam.Jdphi_dR + Jdp*cam.Jdp_dR;
                        MD(1,3) Jdt   = Jdp * Jdp_dt;

                        // 计算光度残差 current - P[..]
                        float current_val =
                                w_ref_tl*img_ptr[0]
                              + w_ref_tr*img_ptr[scale]
                              + w_ref_bl*img_ptr[scale*img.cols]
                              + w_ref_br*img_ptr[scale*img.cols + scale];

                        double ref_val = (double)P[patch_size_total*level + x*patch_size + y];
                        double res = (double)current_val - ref_val;

                        // 写入 z, H_sub
                        z(row_idx) = res;
                        H_sub.block<1,6>(row_idx, 0) << JdR, Jdt;

                        patch_error += (float)(res*res);
                        n_meas_++;
                    } // y
                } // x

                // 将该点误差累加到全局
                sub_sparse_map->errors[i] = patch_error;
                error += patch_error;
            } // end for i
        } // end for cam_idx

        // 若无测量
        if (n_meas_ == 0)
        {
            //ROS_WARN("No valid measurements, break from iteration.");
            break;
        }

        // 均方误差
        error /= (float)n_meas_;
        //ROS_INFO("Total Error after this iteration: %f", error);

        // -------- EKF/GN 更新 --------
        if (error <= last_error)
        {
            old_state = (*state);
            last_error = error;

            // 构造法方程
            auto H_sub_T = H_sub.transpose();
            H_T_H.setZero();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;

            MD(DIM_STATE, DIM_STATE) K_1 = 
                (H_T_H + (state->cov / img_point_cov).inverse()).inverse();

            auto HTz = H_sub_T * z;

            // 如果存在先验: 
            auto vec = (*state_propagat) - (*state);

            G.block<DIM_STATE,6>(0,0) = 
                K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);

            auto solution = 
                -K_1.block<DIM_STATE,6>(0,0)*HTz 
                + vec 
                - G.block<DIM_STATE,6>(0,0)*vec.block<6,1>(0,0);

            (*state) += solution;

            auto rot_add = solution.block<3,1>(0,0);
            auto t_add   = solution.block<3,1>(3,0);
            //ROS_INFO("State updated. dRot=%f deg, dTrans=%f", rot_add.norm()*57.3, t_add.norm());

            // 收敛检查
            if ((rot_add.norm()*57.3f < 0.001f) && 
                (t_add.norm()*100.0f < 0.001f))
            {
                EKF_end = true;
            }
        }
        else
        {
            // 回退
            (*state) = old_state;
            EKF_end = true;
        }

        if (iteration == NUM_MAX_ITERATIONS || EKF_end)
        {
            break;
        }
    }

    ROS_INFO("UpdateState finished with last error: %f", last_error);
    return last_error;
}




void LidarSelector::updateFrameState(StatesGroup state)
{
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);
    
    // 遍历每个相机，更新其位姿
    for (size_t cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
        // 获取当前相机的旋转和平移到参考坐标系的外参
        M3D Rci = cameras[cam_idx].Rci;
        V3D Pci = cameras[cam_idx].Pci;

        // 计算当前帧到世界坐标系的旋转矩阵 Rcw
        M3D Rcw = Rci * Rwi.transpose();

        // 计算当前帧到世界坐标系的平移向量 Pcw
        V3D Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

        // 更新新帧中对应相机的位姿
        new_frame_->T_f_w_[cam_idx] = SE3(Rcw, Pcw);
    }
}


void LidarSelector::addObservation(const std::vector<cv::Mat>& imgs)
{
    // 前置检查
    if (imgs.size() < cameras.size()) {
        ROS_ERROR("[addObservation] Number of images (%lu) is less than number of cameras (%lu)", imgs.size(), cameras.size());
        return;
    }

    if (new_frame_ == nullptr) {
        ROS_ERROR("[addObservation] new_frame_ is nullptr");
        return;
    }

    int total_points = sub_sparse_map->index.size();
    if (total_points == 0){
        ROS_WARN("[addObservation] total_points is 0");
        return;
    }

    // 检查 search_levels 大小
    if (sub_sparse_map->search_levels.size() < total_points) {
        ROS_ERROR("[addObservation] search_levels size (%lu) is less than total_points (%d)", sub_sparse_map->search_levels.size(), total_points);
        return;
    }

    for(int i = 0; i < total_points; i++) 
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        if(pt == nullptr) continue;

        for(int cam_idx = 0; cam_idx < cameras.size(); cam_idx++)
        {

            Eigen::Vector2d pc = new_frame_->w2c(pt->pos_, cam_idx);
            SE3 pose_cur = new_frame_->T_f_w_[cam_idx];
            bool add_flag = false;

            // 步骤1: 时间条件
            if(pt->obs_.empty()){
                continue; // 没有观测则跳过
            }
            FeaturePtr last_feature = pt->obs_.back();
            
            // 步骤2: 姿态变化条件
            SE3 pose_ref = last_feature->T_f_w_;

            SE3 delta_pose = pose_ref * pose_cur.inverse();

            double delta_p = delta_pose.translation().norm();

            double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));  
          
            if(delta_p > 0.5 || delta_theta > 10.0) {
                add_flag = true;
            }
            // 步骤3: 像素距离条件
            Eigen::Vector2d last_px = last_feature->px;
            double pixel_dist = (pc - last_px).norm();
            if(pixel_dist > 40.0) {
                add_flag = true;
            }

            // 保持每个点的观测特征数量
            if(pt->obs_.size() >= 20)
            {
                FeaturePtr ref_ftr;
                pt->getFurthestViewObs(new_frame_->pos(cam_idx), ref_ftr);
                pt->deleteFeatureRef(ref_ftr);
                //ROS_WARN("ref_ftr->id_ is %d", ref_ftr->id_);
            }
            if(add_flag)
            {
                // 检查 pc 是否在图像范围内
                if (pc[0] < 0 || pc[0] >= imgs[cam_idx].cols || pc[1] < 0 || pc[1] >= imgs[cam_idx].rows) {
                    //ROS_WARN("[addObservation] Pixel coordinate (%lf, %lf) is out of image bounds for camera %d", pc[0], pc[1], cam_idx);
                    continue;
                }

                float val = vk::shiTomasiScore(imgs[cam_idx], pc[0], pc[1]);
                Eigen::Vector3d f = cameras[cam_idx].cam->cam2world(pc);
                FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_[cam_idx], val, sub_sparse_map->search_levels[i], cam_idx)); 
                ftr_new->img = imgs[cam_idx];
                ftr_new->id_ = new_frame_->id_;

                pt->addFrameRef(ftr_new);      
            }
        }
    }
}



void LidarSelector::ComputeJ(const std::vector<cv::Mat>& imgs) 
{
    int total_points = sub_sparse_map->index.size();

    if (total_points == 0) 
    {
        return;
    }

    float error = 1e10f;
    float now_error = error;

    for (int level = 2; level >= 0; level--) 
    {
        now_error = UpdateState(imgs, error, level);
    }

    if (now_error < error) 
    {
        state->cov -= G * state->cov;
    } 

    updateFrameState(*state);
}



// void LidarSelector::display_keypatch(double time)
// {
//     int total_points = sub_sparse_map->index.size();
//     if (total_points==0) return;
//     for(int i=0; i<total_points; i++)
//     {
//         PointPtr pt = sub_sparse_map->voxel_points[i];
//         V2D pc(new_frame_->w2c(pt->pos_,0));//默认为主相机
//         cv::Point2f pf;
//         pf = cv::Point2f(pc[0], pc[1]); 
//         if (sub_sparse_map->errors[i]<8000) // 5.5
//             cv::circle(img_cp, pf, 6, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
//         else
//             cv::circle(img_cp, pf, 6, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
//     }   s
//     std::string text = std::to_string(int(1/time))+" HZ";
//     cv::Point2f origin;
//     origin.x = 20;
//     origin.y = 20;
//     cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
// }

V3F LidarSelector::getpixel(const cv::Mat & img, const Eigen::Vector2d & pc) 

{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]); 
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref-u_ref_i);
    const float subpix_v_ref = (v_ref-v_ref_i);
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t* img_ptr = (uint8_t*) img.data + ((v_ref_i)*width + (u_ref_i))*3;
    float B = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[0+3] + w_ref_bl*img_ptr[width*3] + w_ref_br*img_ptr[width*3+0+3];
    float G = w_ref_tl*img_ptr[1] + w_ref_tr*img_ptr[1+3] + w_ref_bl*img_ptr[1+width*3] + w_ref_br*img_ptr[width*3+1+3];
    float R = w_ref_tl*img_ptr[2] + w_ref_tr*img_ptr[2+3] + w_ref_bl*img_ptr[2+width*3] + w_ref_br*img_ptr[width*3+2+3];
    V3F pixel(B,G,R);
    return pixel;
}


void LidarSelector::detect(std::vector<cv::Mat>& imgs, PointCloudXYZI::Ptr pg) 
{
    //保存副本
    img_rgbs=imgs;
    std::vector<vk::AbstractCamera*> cam_ptrs;
    cam_ptrs.reserve(cameras.size());
    for(size_t i = 0; i < cameras.size(); i++){
        cam_ptrs.push_back(cameras[i].cam);
    }

    std::vector<cv::Mat> img_grays(cameras.size());
    for(size_t cam_idx = 0; cam_idx < cameras.size(); cam_idx++)
    {
        if(imgs[cam_idx].channels() > 1)
        {
            cv::cvtColor(imgs[cam_idx], img_grays[cam_idx], cv::COLOR_BGR2GRAY);
        }
        else
        {
            ROS_ERROR("Please input rgb images");
        }
    }


    // 使用灰度图像创建 Frame 实例
    new_frame_.reset(new Frame(cam_ptrs, img_grays));
    updateFrameState(*state);

    // 3) 初始状态：如果是第一帧且雷达点云够多
    if(stage_ == STAGE_FIRST_FRAME && pg->size() > 10)
    {
        new_frame_->setKeyframe();
        stage_ = STAGE_DEFAULT_FRAME;

    }

    double t1 = omp_get_wtime();

    // 4) 从全局地图中，挑选与当前帧可匹配的点 -> 填充 sub_sparse_map
    addFromSparseMap(img_grays, pg);

    double t2 = omp_get_wtime();

    // 5) 将新激光点也加入到全局地图 feat_map
    //    这里对每个相机都调一次 addSparseMap，可将同一个雷达点投影到不同相机
    
    addSparseMap(img_grays, pg);
    
    double t3 = omp_get_wtime();

    // 6) 优化/更新状态
    ComputeJ(img_grays);

    double t4 = omp_get_wtime();

    // 7) 补充观测
    addObservation(img_grays);
    double t5 = omp_get_wtime();

    // 可视化等
    // display_keypatch(t5 - t1); // 只显示主相机

    // 打印耗时
    double total_time = t5 - t1;
    printf("[ VIO multi-cam ]: addFromSparseMap=%.6f, addSparseMap=%.6f, "
           "ComputeJ=%.6f, addObs=%.6f, total=%.6f.\n",
           (t2 - t1), (t3 - t2), (t4 - t3), (t5 - t4), total_time);
}

} // namespace lidar_selection