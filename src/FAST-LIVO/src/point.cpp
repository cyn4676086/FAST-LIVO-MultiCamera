// point.cpp
#include <stdexcept>
#include <vikit/math_utils.h>
#include <point.h>
#include <mutex> // 添加此行

namespace lidar_selection {

int Point::point_counter_ = 0;

Point::Point(const Vector3d& pos) :
    id_(point_counter_++),
    pos_(pos),
    normal_set_(false),
    n_obs_(0),
    last_published_ts_(0),
    last_projected_kf_id_(-1),
    // type_(TYPE_UNKNOWN),
    n_failed_reproj_(0),
    n_succeeded_reproj_(0),
    last_structure_optim_(0),
    have_scaled(false)
{}

Point::Point(const Vector3d& pos, FeaturePtr ftr) :
    id_(point_counter_++),
    pos_(pos),
    normal_set_(false),
    n_obs_(1),
    last_published_ts_(0),
    last_projected_kf_id_(-1),
    // type_(TYPE_UNKNOWN),
    n_failed_reproj_(0),
    n_succeeded_reproj_(0),
    last_structure_optim_(0),
    have_scaled(false)
{
    std::lock_guard<std::mutex> lock(mutex_);
    obs_.push_front(ftr);
}

Point::~Point()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::for_each(obs_.begin(), obs_.end(), [&](FeaturePtr i){i.reset();});
}

void Point::addFrameRef(FeaturePtr ftr)
{
    std::lock_guard<std::mutex> lock(mutex_);
    obs_.push_front(ftr);
    ++n_obs_;
}

FeaturePtr Point::findFrameRef(Frame* frame)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for(auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
        if((*it)->frame == frame)
            return *it;
    return nullptr;    // no keyframe found
}

bool Point::deleteFrameRef(Frame* frame)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for(auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
    {
        if((*it)->frame == frame)
        {
            obs_.erase(it);
            return true;
        }
    }
    return false;
}

void Point::deleteFeatureRef(FeaturePtr ftr)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for(auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
    {
        if((*it) == ftr)
        {
            obs_.erase(it);
            return;
        }
    }
}

void Point::initNormal()
{
    std::lock_guard<std::mutex> lock(mutex_);
    assert(!obs_.empty());
    const FeaturePtr ftr = obs_.back();
    assert(ftr->frame != nullptr);
    normal_ = ftr->frame->T_f_w_[ftr->camera_id].rotation_matrix().transpose() * (-ftr->f);
    normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos(ftr->camera_id)).norm(),2), 1.0, 1.0);
    normal_set_ = true;
}

bool Point::getClosePose(const FramePtr& new_frame, FeaturePtr& ftr) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(obs_.size() <= 0) return false;

    auto min_it=obs_.begin();
    double min_cos_angle = 3.14;
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    {
        int cam_id = (*it)->camera_id;
        SE3 delta_pose = (*it)->frame->T_f_w_[cam_id] * new_frame->T_f_w_[cam_id].inverse();
        double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));            
        double delta_p = delta_pose.translation().norm();
        double p_in_ref = ((*it)->frame->T_f_w_[cam_id] * pos_).norm();
        if(delta_p > p_in_ref*0.8) continue;
        if(delta_theta < min_cos_angle)
        {
            min_cos_angle = delta_theta;
            min_it = it;
        }
    }
    ftr = *min_it;
    
    if(min_cos_angle > 2.0) // assume that observations larger than 60° are useless 0.5
    {
        // ROS_ERROR("The observed angle is larger than 60°.");
        return false;
    }

    return true;
}

bool Point::getCloseViewObs(const Vector3d& framepos, FeaturePtr& ftr, const Vector2d& cur_px, int cam_id) const
{
    if(obs_.size() <= 0) {
        ROS_ERROR("getCloseViewObs obs_size is 0");
    }

    Vector3d obs_dir = (framepos - pos_).normalized();
    auto min_it = obs_.begin();
    double min_cos_angle = 0;

    for(auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
    {

        Vector3d dir = ((*it)->T_f_w_.inverse().translation() - pos_).normalized();

        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
            min_cos_angle = cos_angle;
            min_it = it;
        }
    }
    ftr = *min_it;


    if(min_cos_angle < 0.5) // assume that observations larger than 60° are useless
    {
        ROS_ERROR("[getCloseViewObs] The observed angle is larger than 60°.");
        return false;
    }

    return true;
}

bool Point::getCloseViewObs_test(const Vector3d& framepos, FeaturePtr& ftr, const Vector2d& cur_px, double& min_cos_angle) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(obs_.size() <= 0) return false;

    Vector3d obs_dir = (framepos - pos_).normalized();
    auto min_it = obs_.begin();
    min_cos_angle = 0;

    for(auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
    {
        Vector3d dir = ((*it)->T_f_w_.inverse().translation() - pos_).normalized();
        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
            min_cos_angle = cos_angle;
            min_it = it;
        }
    }
    ftr = *min_it;

    if(min_cos_angle < 0.5) // assume that observations larger than 60° are useless
    {
        // ROS_ERROR("The observed angle is larger than 60°.");
        return false;
    }

    return true;
}

void Point::getFurthestViewObs(const Vector3d& framepos, FeaturePtr& ftr) const
{
  
  Vector3d obs_dir(framepos - pos_); obs_dir.normalize();
  auto max_it=obs_.begin();
  double maxdist = 0.0;
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    double dist= ((*it)->T_f_w_.inverse().translation() - framepos).norm();
    if(dist > maxdist)
    {
      maxdist = dist;
      max_it = it;
    }
  }
  ftr = *max_it;
}

void Point::optimize(const size_t n_iter)
{
    Vector3d old_point = pos_;
    double chi2 = 0.0;
    Matrix3d A;
    Vector3d b;

    for(size_t i = 0; i < n_iter; i++)
    {
        A.setZero();
        b.setZero();
        double new_chi2 = 0.0;

        // 计算残差
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for(auto it = obs_.begin(); it != obs_.end(); ++it)
            {
                Matrix23d J;
                int cam_id = (*it)->camera_id;
                const Vector3d p_in_f = (*it)->frame->T_f_w_[cam_id] * pos_;
                Point::jacobian_xyz2uv(p_in_f, (*it)->frame->T_f_w_[cam_id].rotation_matrix(), J);
                const Vector2d e = vk::project2d((*it)->f) - vk::project2d(p_in_f);

                if((*it)->type == Feature::EDGELET)
                {
                    float err_edge = (*it)->grad.transpose() * e;
                    new_chi2 += err_edge * err_edge;
                    A.noalias() += J.transpose() * (*it)->grad * (*it)->grad.transpose() * J;
                    b.noalias() -= J.transpose() * (*it)->grad * err_edge;
                }
                else
                {
                    new_chi2 += e.squaredNorm();
                    A.noalias() += J.transpose() * J;
                    b.noalias() -= J.transpose() * e;
                }
            }
        }

        // 解线性系统
        const Vector3d dp(A.ldlt().solve(b));

        // 检查误差是否增加
        if((i > 0 && new_chi2 > chi2) || std::isnan(dp[0]))
        {
            pos_ = old_point; // 回滚
            break;
        }

        // 更新模型
        Vector3d new_point = pos_ + dp;
        old_point = pos_;
        pos_ = new_point;
        chi2 = new_chi2;

        // 收敛判断
        if(vk::norm_max(dp) <= 1e-10)
            break;
    }
}

} // namespace lidar_selection
