// feature.h
#ifndef SVO_FEATURE_H_
#define SVO_FEATURE_H_

#include <frame.h>
#include <point.h>
#include <common_lib.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace lidar_selection {

// A salient image region that is tracked across frames.
struct Feature
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum FeatureType {
        CORNER,
        EDGELET
    };
    int id_;
    FeatureType type;     //!< Type can be corner or edgelet.
    Frame* frame;         //!< Pointer to frame in which the feature was detected.
    cv::Mat img;
    std::vector<cv::Mat> ImgPyr;
    Eigen::Vector2d px;          //!< Coordinates in pixels on pyramid level 0.
    Eigen::Vector3d f;           //!< Unit-bearing vector of the feature.
    int level;            //!< Image pyramid level where feature was extracted.
    PointPtr point;         //!< Pointer to 3D point which corresponds to the feature.
    Eigen::Vector2d grad;        //!< Dominant gradient direction for edgelets, normalized.
    float score;
    float error;
    SE3 T_f_w_; //这里是单个相机的tfw
    
    int camera_id;        //!< 新增

    Feature(const Eigen::Vector2d& _px, 
            const Eigen::Vector3d& _f, 
            const SE3& _T_f_w, 
            const float &_score, 
            int _level,
            int _camera_id = 0)  // 默认camera_id=0，如果后续需要可在实例化时指定
        : type(CORNER),
          px(_px),
          f(_f),
          level(_level),
          score(_score),
          T_f_w_(_T_f_w),
          camera_id(_camera_id)
    {
    }

    inline Eigen::Vector3d pos() const { return T_f_w_.inverse().translation(); }

    ~Feature()
    {
        // printf("The feature %d has been destructed.", id_);
    }
};

} // namespace lidar_selection

#endif // SVO_FEATURE_H_
