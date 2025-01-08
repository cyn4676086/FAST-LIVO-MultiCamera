// map.cpp

#include <set>
#include <map.h>
#include <point.h>
#include <frame.h>
#include <feature.h>
#include <boost/bind.hpp>
#include <iostream>

namespace lidar_selection {

Map::Map() : MaxKFid(0) {}

Map::~Map()
{
    reset();
    std::cout << "Map destructed" << std::endl;
}

void Map::reset()
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    keyframes_.clear();
    map_points_.clear();
    values_.clear();
    trash_points_.clear();
    deleted_keyframes_.clear();
    deleted_map_points_.clear();
    point_candidates_.reset();
}

bool Map::safeDeleteFrame(FramePtr frame)
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    bool found = false;
    for(auto it = keyframes_.begin(); it != keyframes_.end(); ++it)
    {
        if(*it == frame)
        {
            // 移除所有特征点与地图点的关联
            std::for_each((*it)->getFeatures().begin(), (*it)->getFeatures().end(), [&](std::shared_ptr<Feature> ftr){
                removePtFrameRef(it->get(), ftr);
            });
            keyframes_.erase(it);
            found = true;
            break;
        }
    }

    // point_candidates_.removeFrameCandidates(frame); // 根据需要取消注释

    if(found)
        return true;

    std::cout << "Tried to delete Keyframe in map which was not there." << std::endl;
    return false;
}

void Map::removePtFrameRef(Frame* frame, std::shared_ptr<Feature> ftr)
{
    if(ftr->point == nullptr)
        return; // 地图点可能已被删除
    std::shared_ptr<Point> pt = ftr->point;
    if(!pt) return; // 确保点是 Point 类型
    ftr->point = nullptr;

    // 检查地图点的观测数量
    if(pt->nRefs() <= 2)
    {
        // 如果观测数量小于等于2，删除地图点
        safeDeletePoint(pt);
        return;
    }

    // 移除地图点与关键帧的关联
    pt->deleteFrameRef(frame);
    frame->removeKeyPoint(ftr);
}

void Map::delete_points(int size)
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    if(map_points_.size() <= size) return;

    std::cout << "\t map_points_.size(): " << map_points_.size()
              << "\t size: " << size
              << std::endl;

    // 删除前 size - 10 个地图点
    for(int i = 0; i < size - 10; ++i)
    {
        if(map_points_.empty()) break;
        std::shared_ptr<Point> pt = map_points_.front();
        map_points_.pop_front();
        if(pt)
            safeDeletePoint(pt);
    }

    // 同时清除对应的值
    for(int i = 0; i < size - 10 && !values_.empty(); ++i)
    {
        values_.pop_front();
    }
}

void Map::clear()
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    // 清除所有地图点
    for(auto& pt : map_points_)
    {
        std::for_each(pt->obs_.begin(), pt->obs_.end(), [&](std::shared_ptr<Feature> ftr){
            if(ftr) {
                ftr->point = nullptr;
                ftr->frame->removeKeyPoint(ftr);
            }
        });
        pt.reset();
    }
    map_points_.clear();
    values_.clear();

    // 清除垃圾点
    trash_points_.clear();
}

void Map::safeDeletePoint(std::shared_ptr<Point> pt)
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    // 删除所有观测中的引用
    std::for_each(pt->obs_.begin(), pt->obs_.end(), [&](std::shared_ptr<Feature> ftr){
        if(ftr) {
            ftr->point = nullptr;
            ftr->frame->removeKeyPoint(ftr);
        }
    });
    pt->obs_.clear();

    // 将地图点移动到垃圾点列表
    deletePoint(pt);
}

void Map::deletePoint(std::shared_ptr<Point> pt)
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    trash_points_.push_back(pt);
}

void Map::addKeyframe(FramePtr new_keyframe)
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    keyframes_.push_back(new_keyframe);
}

void Map::addPoint(std::shared_ptr<Point> point)
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    map_points_.push_back(point);
}

void Map::getCloseKeyframes(
    const FramePtr& frame,
    std::list< std::pair<FramePtr, double> >& close_kfs) const
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    for (auto& kf : keyframes_)
    {
        bool has_overlap = false;

        for(int cam_id = 0; cam_id < frame->cams_.size(); cam_id++)
        {
            // 获取当前关键帧中该相机的所有关键点
            const auto& kf_keypts = kf->getKeyPointsForCam(cam_id);

            for (auto& keypoint : kf_keypts)
            {
                if (keypoint == nullptr || keypoint->camera_id != static_cast<int>(cam_id))
                    continue;

                // 检查关键点是否在当前帧的相机视野内
                if (frame->isVisibleInCam(keypoint->point->pos_, cam_id))
                {
                    double dist = (frame->T_f_w_[cam_id].translation() - kf->T_f_w_[cam_id].translation()).norm();
                    close_kfs.emplace_back(kf, dist);
                    has_overlap = true;
                    break; // 当前相机已有重叠，检查下一个关键帧。
                }
            }
            if(has_overlap) break; // 关键帧中任意一个相机视场有重叠即可。
        }
    }
}


FramePtr Map::getClosestKeyframe(const FramePtr& frame) const
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    std::list< std::pair<FramePtr, double> > close_kfs;
    getCloseKeyframes(frame, close_kfs);
    if(close_kfs.empty())
    {
        return nullptr;
    }
    // 按距离排序
    close_kfs.sort([](const std::pair<FramePtr, double>& a, const std::pair<FramePtr, double>& b) {
        return a.second < b.second;
    });

    if(close_kfs.front().first != frame)
        return close_kfs.front().first;
    close_kfs.pop_front();
    return close_kfs.empty() ? nullptr : close_kfs.front().first;
}


bool Map::getKeyframeById(const int id, FramePtr& frame) const
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    bool found = false;
    for(auto it = keyframes_.begin(); it != keyframes_.end(); ++it)
    {
        if((*it)->getId() == id) {
            frame = *it;
            found = true;
            break;
        }
    }
    return found;
}



void Map::emptyTrash()
{
    std::lock_guard<std::mutex> lock(map_mutex_);
    // 清除垃圾点
    for(auto& pt : trash_points_)
    {
        pt.reset();
    }
    trash_points_.clear();

    // 清除 MapPointCandidates 的垃圾点
    point_candidates_.emptyTrash();
}

MapPointCandidates::MapPointCandidates()
{}

MapPointCandidates::~MapPointCandidates()
{
    reset();
}

void MapPointCandidates::newCandidatePoint(PointPtr point, double depth_sigma2)
{
    // point->type_ = Point::TYPE_CANDIDATE;
    boost::unique_lock<boost::mutex> lock(mut_);
    candidates_.emplace_back(point, point->obs_.front());
}

void MapPointCandidates::addCandidatePointToFrame(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    auto it = candidates_.begin();
    while(it != candidates_.end())
    {
        if(it->first->obs_.front()->frame == frame.get())
        {
            // 插入特征到帧中
            // it->first->type_ = Point::TYPE_UNKNOWN;
            it->first->n_failed_reproj_ = 0;
            it->second->frame->addFeature(it->second);
            it = candidates_.erase(it);
        }
        else
            ++it;
    }
}

bool MapPointCandidates::deleteCandidatePoint(PointPtr point)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    for(auto it = candidates_.begin(); it != candidates_.end(); ++it)
    {
        if(it->first == point)
        {
            deleteCandidate(*it);
            candidates_.erase(it);
            return true;
        }
    }
    return false;
}

void MapPointCandidates::removeFrameCandidates(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock(mut_);
    auto it = candidates_.begin();
    while(it != candidates_.end())
    {
        if(it->second->frame == frame.get())
        {
            deleteCandidate(*it);
            it = candidates_.erase(it);
        }
        else
            ++it;
    }
}

void MapPointCandidates::reset()
{
    boost::unique_lock<boost::mutex> lock(mut_);
    std::for_each(candidates_.begin(), candidates_.end(), [&](PointCandidate& c){
        // delete c.first;
        c.first.reset();
        c.second.reset();
    });
    candidates_.clear();
}

void MapPointCandidates::deleteCandidate(PointCandidate& c)
{
    // camera-rig: another frame might still be pointing to the candidate point
    // therefore, we can't delete it right now.
    c.second.reset(); c.second = nullptr;
    // c.first->type_ = Point::TYPE_DELETED;
    trash_points_.push_back(c.first);
}

void MapPointCandidates::emptyTrash()
{
    std::for_each(trash_points_.begin(), trash_points_.end(), [&](PointPtr& p){
        // delete p; p=nullptr;
        p.reset();
    });
    trash_points_.clear();
}

namespace map_debug {

void mapValidation(Map* map, int id)
{
    for(auto it = map->keyframes_.begin(); it != map->keyframes_.end(); ++it)
        frameValidation(it->get(), id);
}

void frameValidation(Frame* frame, int id)
{
    for(auto it = frame->fts_.begin(); it != frame->fts_.end(); ++it)
    {
        if((*it)->point == nullptr)
            continue;

        // if((*it)->point->type_ == Point::TYPE_DELETED)
        //   printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);

        if(!(*it)->point->findFrameRef(frame))
            printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);

        pointValidation((*it)->point, id);
    }
    for(auto it = frame->key_pts_.begin(); it != frame->key_pts_.end(); ++it)
        if(*it != nullptr)
            if((*it)->point == nullptr)
                printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

void pointValidation(PointPtr point, int id)
{
    for(auto it = point->obs_.begin(); it != point->obs_.end(); ++it)
    {
        bool found = false;
        for(auto it_ftr = (*it)->frame->fts_.begin(); it_ftr != (*it)->frame->fts_.end(); ++it_ftr)
            if((*it_ftr)->point == point) {
                found = true; break;
            }
        // if(!found)
        //   printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
    }
}

void mapStatistics(Map* map)
{
    // compute average number of features which each frame observes
    size_t n_pt_obs(0);
    for(auto it = map->keyframes_.begin(); it != map->keyframes_.end(); ++it)
        n_pt_obs += (*it)->nObs();
    printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs / map->size());

    // compute average number of observations that each point has
    size_t n_frame_obs(0);
    size_t n_pts(0);
    std::set<PointPtr> points;
    for(auto it = map->keyframes_.begin(); it != map->keyframes_.end(); ++it)
    {
        for(auto ftr = (*it)->fts_.begin(); ftr != (*it)->fts_.end(); ++ftr)
        {
            if((*ftr)->point == nullptr)
                continue;
            if(points.insert((*ftr)->point).second) {
                ++n_pts;
                n_frame_obs += (*ftr)->point->nRefs();
            }
        }
    }
    printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs / n_pts);
}

} // namespace map_debug

} // namespace lidar_selection
