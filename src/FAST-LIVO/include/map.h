// map.h

#ifndef SVO_MAP_H_
#define SVO_MAP_H_

#include <queue>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <common_lib.h>
#include <frame.h>
#include <camera_manager.h>


namespace lidar_selection {

class Seed;

/// Container for converged 3D points that are not already assigned to two keyframes.
class MapPointCandidates
{
public:
    typedef std::pair<PointPtr, FeaturePtr> PointCandidate;
    typedef std::list<PointCandidate> PointCandidateList;

    /// The depth-filter is running in a parallel thread and fills the candidate list.
    /// This mutex controls concurrent access to point_candidates.
    boost::mutex mut_;

    /// Candidate points are created from converged seeds.
    /// Until the next keyframe, these points can be used for reprojection and pose optimization.
    PointCandidateList candidates_;
    std::list<PointPtr> trash_points_;

    MapPointCandidates();
    ~MapPointCandidates();

    PointCandidateList GetAllMapPoints() const { return candidates_; }

    /// Add a candidate point.
    void newCandidatePoint(PointPtr point, double depth_sigma2);

    /// Adds the feature to the frame and deletes candidate from list.
    void addCandidatePointToFrame(FramePtr frame);

    /// Remove a candidate point from the list of candidates.
    bool deleteCandidatePoint(PointPtr point);

    /// Remove all candidates that belong to a frame.
    void removeFrameCandidates(FramePtr frame);

    /// Reset the candidate list, remove and delete all points.
    void reset();

    void deleteCandidate(PointCandidate& c);

    void emptyTrash();
};

/// Map object which saves all keyframes which are in a map.
class Map : boost::noncopyable
{
public:
    std::list<FramePtr> keyframes_;          //!< List of keyframes in the map.
    std::deque<PointPtr> map_points_;        //!< Deque of all map points.
    std::deque<float> values_;               //!< The score of every Point in cur frame.
    std::list<PointPtr> trash_points_;       //!< A deleted point is moved to the trash bin.
    MapPointCandidates point_candidates_;    //!< Candidate points container.
    int MaxKFid;                             //!< Maximum Keyframe ID.

    Map();
    ~Map();

    std::list<FramePtr> getAllKeyframe() const { return keyframes_; }

    void delete_points(int size);
    void clear();

    void addPoint(std::shared_ptr<Point> point);

    /// Reset the map. Delete all keyframes and reset the frame and point counters.
    void reset();

    /// Delete a point in the map and remove all references in keyframes to it.
    void safeDeletePoint(std::shared_ptr<Point> point );

    /// Moves the point to the trash queue which is cleaned now and then.
    void deletePoint(std::shared_ptr<Point> point);

    /// Moves the frame to the trash queue which is cleaned now and then.
    bool safeDeleteFrame(FramePtr frame);

    /// Remove the references between a point and a frame.
    void removePtFrameRef(Frame* frame, std::shared_ptr<Feature> ftr);
    
    /// Add a new keyframe to the map.
    void addKeyframe(FramePtr new_keyframe);

    /// Given a frame, return all keyframes which have an overlapping field of view.
    void getCloseKeyframes(const FramePtr& frame, std::list< std::pair<FramePtr, double> >& close_kfs) const;

    /// Return the keyframe which is spatially closest and has overlapping field of view.
    FramePtr getClosestKeyframe(const FramePtr& frame) const;

    /// Return the keyframe which is furthest apart from pos.
    FramePtr getFurthestKeyframe(const Vector3d& pos) const;

    bool getKeyframeById(const int id, FramePtr& frame) const;


    /// Empty trash bin of deleted keyframes and map points. We don't delete the
    /// points immediately to ensure proper cleanup and to provide the visualizer
    /// a list of objects which must be removed.
    void emptyTrash();

    /// Return the keyframe which was last inserted in the map.
    inline FramePtr lastKeyframe() { return keyframes_.empty() ? nullptr : keyframes_.back(); }

    /// Return the number of keyframes in the map
    inline size_t size() const { return keyframes_.size(); }

private:
    mutable std::mutex map_mutex_;           //!< Mutex for thread safety.
    std::list<FramePtr> deleted_keyframes_; //!< List of keyframes to be deleted.
    std::deque<PointPtr> deleted_map_points_; //!< List of map points to be deleted.
};

/// A collection of debug functions to check the data consistency.
namespace map_debug {

void mapStatistics(Map* map);
void mapValidation(Map* map, int id);
void frameValidation(Frame* frame, int id);
void pointValidation(PointPtr point, int id);

} // namespace map_debug

} // namespace lidar_selection

#endif // SVO_MAP_H_
