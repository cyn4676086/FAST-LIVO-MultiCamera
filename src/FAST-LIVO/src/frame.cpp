// frame.cpp
#include <stdexcept>
#include <frame.h>
#include <feature.h>
#include <point.h>
#include <boost/bind.hpp>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/performance_monitor.h>
// #include <fast/fast.h>
#include <algorithm>
#include <iostream>

namespace lidar_selection {

// 初始化静态成员变量
int Frame::frame_counter_ = 0; 

Frame::Frame(const std::vector<vk::AbstractCamera*>& cams, const std::vector<cv::Mat>& imgs) :
    id_(frame_counter_++), 
    cams_(cams), 
    key_pts_(cams.size() * 5, nullptr), // 每个相机预留5个关键点
    is_keyframe_(false),
    T_f_w_(cams.size(),SE3()) // 初始化为单位变换
{
    if(cams_.size() != imgs.size()) {
        throw std::runtime_error("Frame: Number of cameras and images must be equal.");
    }
    initFrame(imgs);
}

Frame::~Frame()
{
    // 清理特征点
    std::for_each(fts_.begin(), fts_.end(), [&](std::shared_ptr<Feature> i){ i.reset(); });
}

void Frame::initFrame(const std::vector<cv::Mat>& imgs)
{
    // 检查图像
    for(size_t cam_id = 0; cam_id < cams_.size(); ++cam_id) {
        const cv::Mat& img = imgs[cam_id];
        if(img.empty() || img.type() != CV_8UC1 || img.cols != cams_[cam_id]->width() || img.rows != cams_[cam_id]->height()) {
            throw std::runtime_error("Frame: as the camera model or image is not BGR.");
        }
    }
    std::for_each(key_pts_.begin(), key_pts_.end(), [&](FeaturePtr ftr){ ftr=nullptr; });
}

void Frame::setKeyframe()
{
    is_keyframe_ = true;
    setKeyPoints();
}

void Frame::addFeature(std::shared_ptr<Feature> ftr)
{
    fts_.push_back(ftr);
}

std::vector<std::shared_ptr<Feature>> Frame::getKeyPointsForCam(int cam_id) const
{
    std::vector<std::shared_ptr<Feature>> cam_keypts;
    size_t start_idx = cam_id * 5;
    size_t end_idx = start_idx + 5;
    for(size_t i = start_idx; i < end_idx && i < key_pts_.size(); ++i) {
        if(key_pts_[i] != nullptr) {
            cam_keypts.push_back(key_pts_[i]);
        }
    }
    return cam_keypts;
}

void Frame::setKeyPoints()
{
    // 首先将无效的关键点设为nullptr
    for(auto& kp : key_pts_) {
        if(kp != nullptr && kp->point == nullptr) {
            kp = nullptr;
        }
    }

    // 遍历所有特征点，更新关键点
    for(auto& ftr : fts_) {
        if(ftr->point != nullptr) {
            checkKeyPoints(ftr);
        }
    }
}

void Frame::checkKeyPoints(std::shared_ptr<Feature> ftr)
{
    // 根据 camera_id 确定对应的关键点索引
    int cam_id = ftr->camera_id;
    if(cam_id < 0 || static_cast<size_t>(cam_id) >= cams_.size()) {
        throw std::runtime_error("Frame::checkKeyPoints: Invalid camera_id.");
    }

    // 每个相机预留5个关键点，索引范围 [cam_id*5, cam_id*5 +4]
    size_t base_idx = cam_id * 5;

    const int cu = cams_[cam_id]->width() / 2;
    const int cv = cams_[cam_id]->height() / 2;

    // 中心像素
    if(key_pts_[base_idx] == nullptr) {
        key_pts_[base_idx] = ftr;
    }
    else {
        double current_dist = std::max(std::fabs(ftr->px[0] - cu), std::fabs(ftr->px[1] - cv));
        double existing_dist = std::max(std::fabs(key_pts_[base_idx]->px[0] - cu), std::fabs(key_pts_[base_idx]->px[1] - cv));
        if(current_dist < existing_dist) {
            key_pts_[base_idx] = ftr;
        }
    }

    // 第1象限
    if(ftr->px[0] >= cu && ftr->px[1] >= cv) {
        size_t idx = base_idx + 1;
        if(idx >= key_pts_.size()) return;
        if(key_pts_[idx] == nullptr) {
            key_pts_[idx] = ftr;
        }
        else {
            double current_product = (ftr->px[0] - cu) * (ftr->px[1] - cv);
            double existing_product = (key_pts_[idx]->px[0] - cu) * (key_pts_[idx]->px[1] - cv);
            if(current_product > existing_product) {
                key_pts_[idx] = ftr;
            }
        }
    }

    // 第4象限
    if(ftr->px[0] >= cu && ftr->px[1] < cv) {
        size_t idx = base_idx + 2;
        if(idx >= key_pts_.size()) return;
        if(key_pts_[idx] == nullptr) {
            key_pts_[idx] = ftr;
        }
        else {
            double current_product = (ftr->px[0] - cu) * (cv - ftr->px[1]);
            double existing_product = (key_pts_[idx]->px[0] - cu) * (cv - key_pts_[idx]->px[1]);
            if(current_product > existing_product) {
                key_pts_[idx] = ftr;
            }
        }
    }

    // 第3象限
    if(ftr->px[0] < cu && ftr->px[1] < cv) {
        size_t idx = base_idx + 3;
        if(idx >= key_pts_.size()) return;
        if(key_pts_[idx] == nullptr) {
            key_pts_[idx] = ftr;
        }
        else {
            double current_product = (ftr->px[0] - cu) * (ftr->px[1] - cv);
            double existing_product = (key_pts_[idx]->px[0] - cu) * (key_pts_[idx]->px[1] - cv);
            if(current_product > existing_product) {
                key_pts_[idx] = ftr;
            }
        }
    }

    // 第2象限
    if(ftr->px[0] < cu && ftr->px[1] >= cv) {
        size_t idx = base_idx + 4;
        if(idx >= key_pts_.size()) return;
        if(key_pts_[idx] == nullptr) {
            key_pts_[idx] = ftr;
        }
        else {
            double current_metric = (cu - ftr->px[0]) * (ftr->px[1] - cv);
            double existing_metric = (cu - key_pts_[idx]->px[0]) * (key_pts_[idx]->px[1] - cv);
            if(current_metric > existing_metric) {
                key_pts_[idx] = ftr;
            }
        }
    }
}

void Frame::removeKeyPoint(std::shared_ptr<Feature> ftr)
{
    bool found = false;
    for(auto& kp : key_pts_) {
        if(kp == ftr) {
            kp = nullptr;
            found = true;
        }
    }
    if(found) {
        setKeyPoints();
    }
}

bool Frame::isVisible(const Eigen::Vector3d& xyz_w) const
{
    for(size_t cam_id = 0; cam_id < cams_.size(); ++cam_id) {
        if(isVisibleInCam(xyz_w, cam_id)) return true;
    }
    return false;
}


bool Frame::isVisibleInCam(const Eigen::Vector3d& xyz_w, int cam_id) const
{
    if(cam_id < 0 || static_cast<size_t>(cam_id) >= cams_.size())
        return false;

    Eigen::Vector3d xyz_f = T_f_w_[cam_id] * xyz_w;

    if(xyz_f.z() < 0.0)
        return false; // 点在相机后方

    Eigen::Vector2d px = f2c(xyz_f, cam_id); // 传递 cam_id

    return (px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cams_[cam_id]->width() && px[1] < cams_[cam_id]->height());
}

} // namespace lidar_selection