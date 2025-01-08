// frame.h
#ifndef SVO_FRAME_H_
#define SVO_FRAME_H_

#include <common_lib.h>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <boost/noncopyable.hpp>
#include <vector>
#include <memory>
#include <list>
#include <opencv2/opencv.hpp>

namespace lidar_selection {

struct Feature;
typedef std::shared_ptr<Feature>  FeaturePtr;

class Point;
typedef std::shared_ptr<Point> PointPtr;

typedef std::list<FeaturePtr> Features;
typedef std::vector<cv::Mat> ImgPyr;

/// A frame saves the images, the associated features, and the estimated pose.
class Frame : boost::noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static int                    frame_counter_;         //!< Counts the number of created frames. Used to set the unique id.
    int                           id_;                    //!< Unique id of the frame.
    // double                        timestamp_;             //!< Timestamp of when the image was recorded.
    std::vector<vk::AbstractCamera*> cams_;              //!< Vector of camera models for multi-camera support.
    std::vector<SE3> T_f_w_; //!< Transforms (f)rame from (w)orld for each camera.
    Matrix<double, 6, 6>          Cov_;                   //!< Covariance.
    std::vector<std::vector<cv::Mat>>  img_pyr_;
    Features                      fts_;                   //!< List of features in the images.
    std::vector<FeaturePtr>        key_pts_;               //!< 关键帧 重叠视野检查 5个位置快速检查 camid*5偏移量
    bool                          is_keyframe_;           //!< Was this frame selected as keyframe
    

    /// Constructor accepting multiple cameras and their corresponding images
    Frame(const std::vector<vk::AbstractCamera*>& cams, const std::vector<cv::Mat>& imgs);
    
    ~Frame();

    /// Initialize new frame and create image pyramids for each camera.
    void initFrame(const std::vector<cv::Mat>& imgs);

    /// Select this frame as keyframe.
    void setKeyframe();

    /// Add a feature to the images (handled internally for each camera)
    void addFeature(std::shared_ptr<Feature> ftr);

    /// Get key points for a specific camera
    std::vector<FeaturePtr> getKeyPointsForCam(int cam_id) const;

    /// Set key points for all cameras
    void setKeyPoints();

    /// Check and update key points based on a new feature
    void checkKeyPoints(std::shared_ptr<Feature> ftr);

    /// Remove a feature from key points if it's deleted
    void removeKeyPoint(std::shared_ptr<Feature> ftr);

    /// Return number of point observations.
    inline size_t nObs() const { return fts_.size(); }

    /// Check if a point in (w)orld coordinate frame is visible in any camera's image.
    bool isVisible(const Vector3d& xyz_w) const;

    /// Check if a point in (w)orld coordinate frame is visible in a specific camera's image.
    bool isVisibleInCam(const Vector3d& xyz_w, int cam_id) const;

    /// Full resolution image stored in the frame for a specific camera.
    inline const cv::Mat& img(int cam_id) const { return img_pyr_[cam_id][0]; }

    /// Was this frame selected as keyframe?
    inline bool isKeyframe() const { return is_keyframe_; }

    inline int getId() const {
        return id_;
    }

    inline Vector2d w2c(const Vector3d& xyz_w, size_t cam_id) const { 
        return cams_[cam_id]->world2cam(T_f_w_[cam_id] * xyz_w); 
    }

    /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f) for a specific camera.
    inline Vector3d c2f(const Vector2d& px, size_t cam_id) const { return cams_[cam_id]->cam2world(px[0], px[1]); }

    /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f) for a specific camera.
    inline Vector3d c2f(const double x, const double y, size_t cam_id) const { return cams_[cam_id]->cam2world(x, y); }
    inline Vector3d w2f(const Vector3d& xyz_w, size_t cam_id) const { 
        return T_f_w_[cam_id] * xyz_w; 
    }
    inline Vector3d f2w(const Vector3d& f, size_t cam_id) const { 
        return T_f_w_[cam_id].inverse() * f; 
    }
    inline Vector2d f2c(const Vector3d& f, size_t cam_id) const { 
        return cams_[cam_id]->world2cam(f); 
    }
    
    inline Vector3d pos(size_t cam_id) const { 
        return T_f_w_[cam_id].inverse().translation(); 
    }
    /// Return the list of features (const reference)
    const Features& getFeatures() const { return fts_; }

    /// Frame jacobian for projection of 3D point in (f)rame coordinate to
    /// unit plane coordinates uv (focal length = 1) for a specific camera.
    inline static void jacobian_xyz2uv_change(
        const Vector3d& xyz_in_world,
        const Vector3d& xyz_in_f,
        Matrix<double,2,6>& J,
        SE3& Tbc,
        SE3& T_ref_w,
        double fx)
    {
        // Implementation remains the same
        const double x = xyz_in_f[0];
        const double y = xyz_in_f[1];
        const double z_inv = 1./xyz_in_f[2];
        const double z_inv_2 = z_inv*z_inv;

        const double x_in_world = xyz_in_world[0];
        const double y_in_world = xyz_in_world[1];
        const double z_in_world = xyz_in_world[2];

        Matrix<double,2,3> J1;
        Matrix<double,3,6> J2;

        J1(0,0) = -fx * z_inv;              
        J1(0,1) = 0.0;              
        J1(0,2) = fx * x * z_inv_2;           

        J1(1,0) = 0.0;           
        J1(1,1) = -fx * z_inv;           
        J1(1,2) = fx * y * z_inv_2;         

        J2(0,0) = 1.0;             
        J2(0,1) = 0.0;                 
        J2(0,2) = 0.0;           
        J2(0,3) = 0.0;            
        J2(0,4) = z_in_world;   
        J2(0,5) = -y_in_world;        

        J2(1,0) = 0.0;               
        J2(1,1) = 1.0;            
        J2(1,2) = 0.0;          
        J2(1,3) = -z_in_world;     
        J2(1,4) = 0.0;             
        J2(1,5) = x_in_world;          

        J2(2,0) = 0.0;      
        J2(2,1) = 0.0;       
        J2(2,2) = 1.0;        
        J2(2,3) = y_in_world;  
        J2(2,4) = -x_in_world;       
        J2(2,5) = 0.0;   
        
        J = J1 * T_ref_w.rotation_matrix() * J2;  
    }

    /// Frame jacobian for projection of 3D point in (f)rame coordinate to
    /// unit plane coordinates uv (focal length = 1) for a specific camera.
    inline static void jacobian_xyz2uv(
      const Vector3d& xyz_in_f,
      Matrix<double,2,6>& J)
    {
        const double x = xyz_in_f[0];
        const double y = xyz_in_f[1];
        const double z_inv = 1./xyz_in_f[2];
        const double z_inv_2 = z_inv*z_inv;

        J(0,0) = -z_inv;              // -1/z
        J(0,1) = 0.0;                 // 0
        J(0,2) = x*z_inv_2;           // x/z^2
        J(0,3) = y*J(0,2);            // x*y/z^2
        J(0,4) = -(1.0 + x*J(0,2));  // -(1.0 + x^2/z^2)
        J(0,5) = y*z_inv;             // y/z

        J(1,0) = 0.0;                 // 0
        J(1,1) = -z_inv;              // -1/z
        J(1,2) = y*z_inv_2;           // y/z^2
        J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
        J(1,4) = -J(0,3);             // -x*y/z^2
        J(1,5) = -x*z_inv;            // -x/z
    }
};

typedef std::shared_ptr<Frame> FramePtr;

} // namespace lidar_selection

#endif // SVO_FRAME_H_
