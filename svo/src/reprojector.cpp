#include <svo/reprojector.h>

#include <set>
#include <random> // std::mt19937
#include <algorithm>
#include <stdexcept>
#include <future>

#include <svo/common/camera.h>
#include <svo/common/point.h>
#include <svo/common/frame.h>
#include <svo/common/occupancy_grid_2d.h>
#include <svo/direct/matcher.h>
#include <svo/direct/depth_filter.h>
#include <svo/direct/feature_detection_utils.h>
#include <vikit/math_utils.h>
#include <vikit/timer.h>
#define SEGMENT_ENABLE_
#define NAME_VALUE_LOG(x) std::cout << #x << ": " << (x) << std::endl;
#define CHECK_EIGEN_HAVE_NAN(x,str) if((x.array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any()) std::cout<<str<< #x <<":\n"<< (x).transpose()<<std::endl;


namespace svo {

Reprojector::Reprojector(
    const ReprojectorOptions& options,
    size_t camera_index)
  : options_(options)
  , camera_index_(camera_index)
{}

void Reprojector::reprojectFrames(//reproject feature and segment rules should be same like when it being detected
    const FramePtr& cur_frame,
    const std::vector<FramePtr>& visible_kfs,
    std::vector<PointPtr>& trash_points
    )//reproject is likely to link the frame
{
  const size_t max_n_fixed_lm =
      (options_.use_kfs_from_global_map? options_.max_fixed_landmarks : 0u);
  const size_t max_total_n_features =
      options_.max_n_features_per_frame + max_n_fixed_lm;
  CHECK_GT(options_.max_n_features_per_frame, 0u);
  cur_frame->resizeFeatureStorage(max_total_n_features);//let current frame has history point

  // Initialize grid
  if(!grid_)
  {
    grid_.reset(
          new OccupandyGrid2D(
            options_.cell_size,
            OccupandyGrid2D::getNCell(cur_frame->cam()->imageWidth(),
                                      options_.cell_size),
            OccupandyGrid2D::getNCell(cur_frame->cam()->imageHeight(),
                                      options_.cell_size)));
  }//reset the grid volume
  grid_->reset();
  stats_.reset();

#ifdef SVO_GLOBAL_MAP
  size_t n_fixed_landmarks = 0;
  fixed_lm_stats_.reset();
  if (options_.use_kfs_from_global_map && options_.max_n_global_kfs > 0)
  {
    if (!fixed_landmark_grid_)
    {
      fixed_landmark_grid_.reset(
            new OccupandyGrid2D(
              options_.fixed_lm_grid_size,
              OccupandyGrid2D::getNCell(cur_frame->cam()->imageWidth(),
                                        options_.fixed_lm_grid_size),
              OccupandyGrid2D::getNCell(cur_frame->cam()->imageHeight(),
                                        options_.fixed_lm_grid_size)));
    }
    fixed_landmark_grid_->reset();
    candidates_.clear();
    for(const FramePtr& ref_frame : visible_kfs)
    {
      for(size_t i = 0; i < ref_frame->num_features_; ++i)
      {
        if(!isFixedLandmark(ref_frame->type_vec_[i]))
        {
          continue;
        }
        const PointPtr& point = ref_frame->landmark_vec_[i];
        CHECK(point);

        if (point->n_failed_reproj_ >= 3 &&
            point->n_succeeded_reproj_ < 3 * point->n_failed_reproj_)
        {
          continue;
        }

        if(point->last_projected_kf_id_.at(camera_index_) == cur_frame->id_)
        {
          continue;
        }
        point->last_projected_kf_id_[camera_index_] = cur_frame->id_;

        Candidate candidate;
        if(reprojector_utils::getCandidate(cur_frame, ref_frame, i, candidate))
        {
          candidates_.push_back(candidate);
        }
      }
    }
    VLOG(10) << "Landmark from global map candidates num: "
             << candidates_.size() << std::endl;
    using SucFailN = std::array<int, 2>;
    std::vector<std::pair<PointPtr, SucFailN>> old_proj_info;
    for (const auto& c : candidates_)
    {
      const PointPtr& pt = c.ref_frame->landmark_vec_[c.ref_index];
      old_proj_info.emplace_back(
            std::make_pair(pt,
                           SucFailN {pt->n_succeeded_reproj_, pt->n_failed_reproj_}));
    }
    reprojector_utils::sortCandidatesByNumObs(candidates_);
    reprojector_utils::matchCandidates(
          cur_frame, options_.max_fixed_landmarks,
          options_.affine_est_offset, options_.affine_est_gain,
          candidates_, *fixed_landmark_grid_, fixed_lm_stats_,
          options_.seed_sigma2_thresh);
    VLOG(10) << "Reproject landmarks from global map in cam-" << camera_index_
            << ": trials = " << fixed_lm_stats_.n_trials
            << ", matches = " << fixed_lm_stats_.n_matches << std::endl;
    if (fixed_lm_stats_.n_matches == 0)
    {
      for (const auto& s_f_info : old_proj_info)
      {
        s_f_info.first->n_succeeded_reproj_ = s_f_info.second[0];
        s_f_info.first->n_failed_reproj_ = s_f_info.second[1];
      }
    }
    n_fixed_landmarks = cur_frame->numFixedLandmarks();
  }
#endif

  // Reproject all map points of the closest N kfs with overlap. We only store
  // in which grid cell the points fall. this map mean the local active map but globle map
  candidates_.clear();
  for(const FramePtr& ref_frame : visible_kfs)
  {
    // Try to reproject each map point that the other KF observes
    for(size_t i = 0; i < ref_frame->num_features_; ++i)
    {
      const FeatureType& type = ref_frame->type_vec_[i];//get the feature type in the frame
      if(ref_frame->landmark_vec_[i] == nullptr
      // the landmark vector record the 3d point information and related frame infomation,when initialize would be setup landmark
         || type == FeatureType::kOutlier
         || isMapPoint(ref_frame->type_vec_[i])
         || isFixedLandmark(type))
        continue;

      const PointPtr& point = ref_frame->landmark_vec_[i];

      // first check if the point is valid.
      if(point->n_failed_reproj_ > 10)
      {
        trash_points.push_back(point);
        continue;
      }

      // make sure we project a point only once
      if(point->last_projected_kf_id_.at(camera_index_) == cur_frame->id_)
        continue;
      point->last_projected_kf_id_[camera_index_] = cur_frame->id_;

      // make sure the point has at least two observations
      // ignore for the first 5 frames because in the benchmark node we may have
      // initialized some 3d points from the ground-truth.
      if(point->obs_.size() < 2 && options_.remove_unconstrained_points)
      {
        trash_points.push_back(point);
        continue;
      }

      // finally, reproject point
      Candidate candidate;
      if(reprojector_utils::getCandidate(cur_frame, ref_frame, i, candidate))
        candidates_.push_back(candidate);//candidate define the ref_frame 's ith feature(seed or landmark)
    }
  }

  Statistics lm_stats;
  VLOG(10) << "Landmark candidates num: " << candidates_.size() << std::endl;
  reprojector_utils::sortCandidatesByReprojStats(candidates_);//compare with score reproject_num and feature type
  reprojector_utils::matchCandidates(
        cur_frame, max_total_n_features,
        options_.affine_est_offset, options_.affine_est_gain,
        candidates_, *grid_, lm_stats, options_.seed_sigma2_thresh);
  VLOG(10) << "After projection: remaining landmark candidates num: "
           << candidates_.size() << std::endl;
  VLOG(5) << "Reproject landmarks in cam-" << camera_index_
          << ": trials = " << lm_stats.n_trials
          << ", matches = " << lm_stats.n_matches;
  stats_.add(lm_stats);

  // if we have enough landmarks, we still need to set the grid occupancy
  // to avoid extracting too many new landmarks from the rest candidates
  if (doesFrameHaveEnoughFeatures(cur_frame))
  {
    reprojector_utils::setGridCellsOccupied(candidates_, *grid_);
    // We don't return here because we want to
    // check and set all the other grid cells occupied from seeds
  }

  // ---------------------------------------------------------------------------
  // if we don't have enough features, we reproject the converged seeds
  candidates_.clear();
  for(const FramePtr& ref_frame : visible_kfs)
  {
    for(size_t i = 0; i < ref_frame->num_features_; ++i)
    {
      if(isConvergedCornerEdgeletSeed(ref_frame->type_vec_[i]))
      {
        Candidate candidate;
        if(reprojector_utils::getCandidate(cur_frame, ref_frame, i, candidate))
          candidates_.push_back(candidate);
      }
    }
  }
  if (doesFrameHaveEnoughFeatures(cur_frame))
  {
    // In this case we already have enough features but we just want to set the
    // occupancy such that the depth filter doesn't extract too many new features.
    // TODO(cfo): Actually, this should only be done if a new keyframe is selected.
    reprojector_utils::setGridCellsOccupied(candidates_, *grid_);
    candidates_.clear();
    return;
  }
  Statistics sd_stats;
  VLOG(10) << "Converged seed candidates num " << candidates_.size()
           << std::endl;
  reprojector_utils::sortCandidatesByReprojStats(candidates_);
  reprojector_utils::matchCandidates(
        cur_frame, max_total_n_features,
        options_.affine_est_offset, options_.affine_est_gain,
        candidates_, *grid_, sd_stats, options_.seed_sigma2_thresh);
  VLOG(10) << "After projection:"
           << " remaining converged seed candidates num " << candidates_.size()
           << std::endl;
  VLOG(5) << "Reproject converged seeds in cam-" << camera_index_
          << ": trials = " << sd_stats.n_trials
          << ", matches = " << sd_stats.n_matches;
  stats_.add(sd_stats);

  if (doesFrameHaveEnoughFeatures(cur_frame) ||
      !options_.reproject_unconverged_seeds)
  {
    reprojector_utils::setGridCellsOccupied(candidates_, *grid_);
    candidates_.clear();
    return;
  }

  // ---------------------------------------------------------------------------
  // if we still don't have enough features, reproject the unconverged seeds
  candidates_.clear();
  for(const FramePtr& ref_frame : visible_kfs)
  {
    for(size_t i = 0; i < ref_frame->num_features_; ++i)
    {
      if(isUnconvergedCornerEdgeletSeed(ref_frame->type_vec_[i]))
      {
        Candidate candidate;
        if(reprojector_utils::getCandidate(cur_frame, ref_frame, i, candidate))
          candidates_.push_back(candidate);
      }
    }
  }
  Statistics un_sd_stats;
  VLOG(10) << "Unconverged seeds candidates num " << candidates_.size()
           << std::endl;
  size_t max_allowed_total = max_total_n_features;
  if (options_.max_unconverged_seeds_ratio > 0)
  {
    const double min_lm_seeds_ratio = 1 - options_.max_unconverged_seeds_ratio;
    size_t max_allowed_alternative =
        static_cast<size_t>(
          (cur_frame->numTrackedFeatures() + cur_frame->numFixedLandmarks())
          / min_lm_seeds_ratio);
    if (max_allowed_total > max_allowed_alternative)
    {
      max_allowed_total = max_allowed_alternative;
    }
  }
  if (max_allowed_total < options_.min_required_features)
  {
    max_allowed_total = options_.min_required_features;
  }
  VLOG(10) << "Maximum allowed unconverged seeds are " << max_allowed_total;
  reprojector_utils::sortCandidatesByReprojStats(candidates_);
  reprojector_utils::matchCandidates(
        cur_frame, max_allowed_total,
        options_.affine_est_offset, options_.affine_est_gain,
        candidates_, *grid_, un_sd_stats, options_.seed_sigma2_thresh);
  VLOG(10) << "After projection: "
              "remaining unconverged seeds candidates num " << candidates_.size()
           << std::endl;
  VLOG(5) << "Update unconverged seeds in cam-" << camera_index_
          << ": trials = " << un_sd_stats.n_trials
          << ", success = " << un_sd_stats.n_matches;
  stats_.add(un_sd_stats);

  if (doesFrameHaveEnoughFeatures(cur_frame))
  {
    reprojector_utils::setGridCellsOccupied(candidates_, *grid_);
  }
  const double un_sd_ratio = (1.0 * un_sd_stats.n_matches) / stats_.n_matches;
  if (un_sd_ratio > 0.2)
  {
    LOG(WARNING) << "More than 20% matches are unconverged seeds: "
                 << un_sd_ratio * 100 << "%";
  }
}

#ifdef SEGMENT_ENABLE
void Reprojector::reprojectFramesSegment(// just for segment projection
void Reprojector::reprojectFramesSegment(// just for segment projection
    const FramePtr& cur_frame,
    const std::vector<FramePtr>& visible_kfs,
    std::vector<LinePtr>& trash_segments)//reproject is likely to link the frame
{

  cur_frame->resizeSegmentStorage(options_.max_n_segments_per_frame);
  // Initialize grid
  if(!grid_)
  {
    grid_.reset(
          new OccupandyGrid2D(
            options_.cell_size,
            OccupandyGrid2D::getNCell(cur_frame->cam()->imageWidth(),
                                      options_.cell_size),
            OccupandyGrid2D::getNCell(cur_frame->cam()->imageHeight(),
                                      options_.cell_size)));
  }//reset the grid volume
  grid_->reset();
  stats_.reset();



  // Reproject all map points of the closest N kfs with overlap. We only store
  // in which grid cell the points fall. this map mean the local active map but globle map
  candidates_.clear();
  for(auto & ref_frame:visible_kfs)
  {
    
    
    for(size_t i =0; i< ref_frame->num_segments_;++i)
    {
      const FeatureType& type = ref_frame->seg_type_vec_[i];//get the feature type in the frame
      if(ref_frame->seg_landmark_vec_[i] == nullptr
      // the landmark vector record the 3d point information and related frame infomation,when initialize would be setup landmark
         || type == FeatureType::KSegmentOutlier
         || isFixedSegmentLandmark(type))
        continue;

      const LinePtr& Line3d = ref_frame->seg_landmark_vec_[i];

      // first check if the point is valid.
      if(Line3d->n_failed_reproj_ > 10)
      {
        trash_segments.push_back(Line3d);
        continue;
      }

      // make sure we project a point only once
      if(Line3d->last_projected_kf_id_.at(camera_index_) == cur_frame->id_)
        continue;
      Line3d->last_projected_kf_id_[camera_index_] = cur_frame->id_;

      // make sure the point has at least two observations
      // ignore for the first 5 frames because in the benchmark node we may have
      // initialized some 3d points from the ground-truth.
      if(Line3d->obs_.size() < 2 && options_.remove_unconstrained_points)
      {
        trash_segments.push_back(Line3d);
        continue;
      }

      // finally, reproject point
      Candidate candidate;
      if(reprojector_utils::getCandidateSegment(cur_frame, ref_frame, i, candidate))
        candidates_.push_back(candidate);
      // NAME_VALUE_LOG(ref_frame->seg_vec_[i]);
      // NAME_VALUE_LOG(ref_frame->seg_f_vec_[i*2]);
      // NAME_VALUE_LOG(ref_frame->seg_f_vec_[i*2+1]);
      // NAME_VALUE_LOG(ref_frame->seg_vec_[i]);
      // NAME_VALUE_LOG(ref_frame->seg_f_vec_[i*2]);
      // NAME_VALUE_LOG(ref_frame->seg_f_vec_[i*2+1]);
    }
  }
  if (doesFrameHaveEnoughSegments(cur_frame))
  {
    // In this case we already have enough features but we just want to set the
    // occupancy such that the depth filter doesn't extract too many new features.
    // TODO(cfo): Actually, this should only be done if a new keyframe is selected.
    // reprojector_utils::setGridCellsOccupiedSegment(candidates_, *grid_);
    candidates_.clear();
    return;
  }
  if (doesFrameHaveEnoughSegments(cur_frame))
  {
    // In this case we already have enough features but we just want to set the
    // occupancy such that the depth filter doesn't extract too many new features.
    // TODO(cfo): Actually, this should only be done if a new keyframe is selected.
    // reprojector_utils::setGridCellsOccupiedSegment(candidates_, *grid_);
    candidates_.clear();
    return;
  }
  Statistics lm_stats;
  VLOG(10) << "seg ment Landmark candidates num: " << candidates_.size() << std::endl;
  reprojector_utils::sortCandidatesByReprojStats(candidates_);//compare with score reproject_num and feature type
  reprojector_utils::matchSegmentCandidates(
        cur_frame, options_.max_n_segments_per_frame,// notice
        cur_frame, options_.max_n_segments_per_frame,// notice
        options_.affine_est_offset, options_.affine_est_gain,
        candidates_, *grid_, lm_stats, options_.seed_sigma2_thresh);
  VLOG(10) << "After projection: remaining segment landmark candidates num: "
           << candidates_.size() << std::endl;
  VLOG(5) << "Reproject  segment landmarks in cam-" << camera_index_
          << ": trials = " << lm_stats.n_trials
          << ", matches = " << lm_stats.n_matches;
  stats_.add(lm_stats);

  // if we have enough landmarks, we still need to set the grid occupancy
  // to avoid extracting too many new landmarks from the rest candidates
  if (doesFrameHaveEnoughSegments(cur_frame))
  if (doesFrameHaveEnoughSegments(cur_frame))
  {
    reprojector_utils::setGridCellsOccupiedSegment(candidates_, *grid_);
    // We don't return here because we want to
    // check and set all the other grid cells occupied from seeds
  }

  // ---------------------------------------------------------------------------
  // if we don't have enough features, we reproject the converged seeds
  candidates_.clear();
  for(const FramePtr& ref_frame : visible_kfs)
  {
    for(size_t i = 0; i < ref_frame->num_segments_; ++i)
    {
      // the segment would be sign concergedSegment when update segment seed and two endpoint in this segment are both converged
      if(isConvergedSegmentSeed(ref_frame->seg_type_vec_[i]))
      {
        Candidate candidate;
        if(reprojector_utils::getCandidateSegment(cur_frame, ref_frame, i, candidate))
          candidates_.push_back(candidate);
      }
    }
  }
  if (doesFrameHaveEnoughSegments(cur_frame))
  if (doesFrameHaveEnoughSegments(cur_frame))
  {
    // In this case we already have enough features but we just want to set the
    // occupancy such that the depth filter doesn't extract too many new features.
    // TODO(cfo): Actually, this should only be done if a new keyframe is selected.
    reprojector_utils::setGridCellsOccupiedSegment(candidates_, *grid_);
    candidates_.clear();
    return;
  }
  Statistics sd_stats;
  VLOG(10) << "Converged seed candidates num " << candidates_.size()
           << std::endl;
  reprojector_utils::sortCandidatesByReprojStats(candidates_);
  reprojector_utils::matchSegmentCandidates(
        cur_frame, options_.max_n_segments_per_frame,
        cur_frame, options_.max_n_segments_per_frame,
        options_.affine_est_offset, options_.affine_est_gain,
        candidates_, *grid_, sd_stats, options_.seed_sigma2_thresh);
  VLOG(10) << "After projection:"
           << " remaining converged seed candidates num " << candidates_.size()
           << std::endl;
  VLOG(5) << "Reproject converged seeds in cam-" << camera_index_
          << ": trials = " << sd_stats.n_trials
          << ", matches = " << sd_stats.n_matches;
  stats_.add(sd_stats);

  if (doesFrameHaveEnoughSegments(cur_frame) ||
  if (doesFrameHaveEnoughSegments(cur_frame) ||
      !options_.reproject_unconverged_seeds)
  {
    reprojector_utils::setGridCellsOccupiedSegment(candidates_, *grid_);
    candidates_.clear();
    return;
  }

  // ---------------------------------------------------------------------------
  // if we still don't have enough features, reproject the unconverged seeds
  candidates_.clear();
  for(const FramePtr& ref_frame : visible_kfs)
  {
    for(size_t i = 0; i < ref_frame->num_segments_; ++i)
    {
      if(isUnconvergedSegmentSeed(ref_frame->seg_type_vec_[i]))
      {
        Candidate candidate;
        if(reprojector_utils::getCandidateSegment(cur_frame, ref_frame, i, candidate))
          candidates_.push_back(candidate);
      }
    }
  }
  Statistics un_sd_stats;
  VLOG(10) << "Unconverged seeds candidates num " << candidates_.size()
           << std::endl;
  size_t max_allowed_total = options_.max_n_segments_per_frame;
  size_t max_allowed_total = options_.max_n_segments_per_frame;
  if (options_.max_unconverged_seeds_ratio > 0)
  {
    const double min_lm_seeds_ratio = 1 - options_.max_unconverged_seeds_ratio;
    size_t max_allowed_alternative =
        static_cast<size_t>(
          (cur_frame->numTrackedSegment() + cur_frame->numFixedLandmarks())
          / min_lm_seeds_ratio);
    if (max_allowed_total > max_allowed_alternative)
    {
      max_allowed_total = max_allowed_alternative;
    }
  }
  if (max_allowed_total < options_.min_required_features)
  {
    max_allowed_total = options_.min_required_features;
  }
  VLOG(10) << "Maximum allowed unconverged segment seeds are " << max_allowed_total;
  reprojector_utils::sortCandidatesByReprojStats(candidates_);
  reprojector_utils::matchSegmentCandidates(
        cur_frame, max_allowed_total,
        options_.affine_est_offset, options_.affine_est_gain,
        candidates_, *grid_, un_sd_stats, options_.seed_sigma2_thresh);
  VLOG(10) << "After projection: "
              "remaining unconverged segment seeds candidates num " << candidates_.size()
           << std::endl;
  VLOG(5) << "Update unconverged segment seeds in cam-" << camera_index_
          << ": trials = " << un_sd_stats.n_trials
          << ", success = " << un_sd_stats.n_matches;
  stats_.add(un_sd_stats);

  if (doesFrameHaveEnoughSegments(cur_frame))
  if (doesFrameHaveEnoughSegments(cur_frame))
  {
    reprojector_utils::setGridCellsOccupiedSegment(candidates_, *grid_);
  }
  const double un_sd_ratio = (1.0 * un_sd_stats.n_matches) / stats_.n_matches;
  if (un_sd_ratio > 0.2)
  {
    LOG(WARNING) << "More than 20% matches are unconverged  segment seeds: "
                 << un_sd_ratio * 100 << "%";
  }
}

#endif

namespace reprojector_utils {

void sortCandidatesByReprojStats(
    Reprojector::Candidates& candidates)
{
  std::sort(candidates.begin(), candidates.end(),
            [](const Reprojector::Candidate& lhs, const Reprojector::Candidate& rhs)
  {
    if(lhs.type > rhs.type
       || (lhs.type == rhs.type && lhs.n_reproj > rhs.n_reproj)
       || (lhs.type == rhs.type && lhs.n_reproj == rhs.n_reproj && lhs.score > rhs.score)) // TODO(check)
      return true;
    return false;
  });
}

void sortCandidatesByNumObs(
    Reprojector::Candidates& candidates)
{
  std::sort(candidates.begin(), candidates.end(),
            [](const Reprojector::Candidate& lhs, const Reprojector::Candidate& rhs)
  {
    if(lhs.n_obs > rhs.n_obs
       || (lhs.n_obs == rhs.n_obs && lhs.n_reproj > rhs.n_reproj)
       || (lhs.n_obs == rhs.n_obs && lhs.n_reproj == rhs.n_reproj &&
           lhs.score > rhs.score))
      return true;
    return false;
  });

}

void matchCandidates(
    const FramePtr& frame,
    const size_t max_n_features_per_frame,
    const bool affine_est_offset,
    const bool affine_est_gain,
    Reprojector::Candidates& candidates,
    OccupandyGrid2D& grid,
    Reprojector::Statistics& stats,
    const double seed_sigma2_thresh)
{
  Matcher matcher;
  matcher.options_.affine_est_offset_ = affine_est_offset;
  matcher.options_.affine_est_gain_ = affine_est_gain;
  int i = 0;
  for(Reprojector::Candidate& candidate : candidates)
  {
    ++i;
    size_t grid_index =
        grid.getCellIndex(candidate.cur_px.x(),candidate.cur_px.y(), 1);
    if(max_n_features_per_frame > 0 && grid.isOccupied(grid_index))
    {
      continue;
    }

    ++stats.n_trials;
    FeatureWrapper feature_wrapper = frame->getEmptyFeatureWrapper();
    if(matchCandidate(frame, candidate, matcher, feature_wrapper,
                      seed_sigma2_thresh))
    {
      ++stats.n_matches;
      ++frame->num_features_;
      grid.setOccupied(grid_index);
      if(max_n_features_per_frame > 0
         && frame->num_features_ >= max_n_features_per_frame)
      {
        break;
      }
    }
  }
  candidates.erase(candidates.begin(), candidates.begin()+i);
}


#ifdef SEGMENT_ENABLE

#ifdef SEGMENT_ENABLE
void matchSegmentCandidates(
    const FramePtr &frame,
    const size_t max_n_segments_per_frame,
    const bool affine_est_offset,
    const bool affine_est_gain,
    Reprojector::Candidates &candidates,
    OccupandyGrid2D &grid,
    Reprojector::Statistics &stats,
    const double seed_sigma2_thresh)
{
  Matcher matcher;
  matcher.options_.affine_est_offset_ = affine_est_offset;
  matcher.options_.affine_est_gain_ = affine_est_gain;
  int i = 0;
  for(Reprojector::Candidate& candidate : candidates)
  {
    ++i;
    size_t grid_index_s =
        grid.getCellIndex(candidate.cur_px.x(),candidate.cur_px.y(), 1);
    size_t grid_index_e =
        grid.getCellIndex(candidate.cur_px_end.x(),candidate.cur_px_end.y(), 1);
    if(max_n_segments_per_frame > 0 && grid.isOccupied(grid_index_s) && grid.isOccupied(grid_index_e) )
    {
      continue;
    }
      if (frame->num_segments_>=static_cast<svo::size_t>(frame->seg_vec_.cols()))
  {
    break;
  }
      if (frame->num_segments_>=static_cast<svo::size_t>(frame->seg_vec_.cols()))
  {
    break;
  }
    ++stats.n_trials;
    SegmentWrapper segment_wrapper = frame->getEmptySegmentWrapper();//change the current frame s_f_vec_ e_f_vec_ 
    
    SegmentWrapper segment_wrapper = frame->getEmptySegmentWrapper();//change the current frame s_f_vec_ e_f_vec_ 
    
    if(matchSegmentCandidate(frame, candidate, matcher, segment_wrapper,
                      seed_sigma2_thresh))
    {
      ++stats.n_matches;
      ++frame->num_segments_;
      grid.setOccupied(grid_index_s);
      grid.setOccupied(grid_index_e);
      if(max_n_segments_per_frame > 0
         && frame->num_segments_ >= max_n_segments_per_frame)
      {
        break;
      }
      if(segment_wrapper.line_landmark!=nullptr)CHECK(static_cast<int>(segment_wrapper.type)>8);
      // segment_wrapper.e_f=candidate.ref_frame->seg_f_vec_.col(candidate.ref_index*2+1);
      // NAME_VALUE_LOG(candidate.ref_frame->seg_f_vec_.col(candidate.ref_index*2));
      // NAME_VALUE_LOG(candidate.ref_frame->seg_f_vec_.col(candidate.ref_index*2+1));
      // NAME_VALUE_LOG(segment_wrapper.e_f);
      // NAME_VALUE_LOG(segment_wrapper.s_f);
      // NAME_VALUE_LOG(frame->seg_f_vec_.col((frame->num_segments_-1)*2));
      // NAME_VALUE_LOG(frame->seg_f_vec_.col((frame->num_segments_-1)*2+1));
      if(segment_wrapper.line_landmark!=nullptr)CHECK(static_cast<int>(segment_wrapper.type)>8);
      // segment_wrapper.e_f=candidate.ref_frame->seg_f_vec_.col(candidate.ref_index*2+1);
      // NAME_VALUE_LOG(candidate.ref_frame->seg_f_vec_.col(candidate.ref_index*2));
      // NAME_VALUE_LOG(candidate.ref_frame->seg_f_vec_.col(candidate.ref_index*2+1));
      // NAME_VALUE_LOG(segment_wrapper.e_f);
      // NAME_VALUE_LOG(segment_wrapper.s_f);
      // NAME_VALUE_LOG(frame->seg_f_vec_.col((frame->num_segments_-1)*2));
      // NAME_VALUE_LOG(frame->seg_f_vec_.col((frame->num_segments_-1)*2+1));
    }


  }
  candidates.erase(candidates.begin(), candidates.begin()+i);
}

  std::ostream& operator<<(std::ostream& os, const SegmentWrapper& wrapper)
{
    os << "Type: " << static_cast<int>(wrapper.type) << "\n";
    os << "Segment: " << wrapper.segment.transpose() << "\n"; // Assuming Segment is a matrix or vector
    os << "Start BearingVector: " << wrapper.s_f.transpose() << "\n"; // Assuming BearingVector is a matrix or vector
    os << "End BearingVector: " << wrapper.e_f.transpose() << "\n"; // Assuming BearingVector is a matrix or vector
    os << "Gradient Vector: " << wrapper.grad.transpose() << "\n"; // Assuming GradientVector is a matrix or vector
    os << "Score: " << wrapper.score << "\n";
    os << "Level: " << wrapper.level << "\n";
    // os << "Line Landmark: " << wrapper.line_landmark->spos_ <<" "<<wrapper.line_landmark->epos_ <<"\n"; // Assuming LinePtr is a pointer to some type that has an operator<<
    os << "Seed Reference: " << wrapper.seed_ref.seed_id << "\n";
    os << "Track ID: " << wrapper.track_id << "\n";

    return os;
}
  std::ostream& operator<<(std::ostream& os, const SegmentWrapper& wrapper)
{
    os << "Type: " << static_cast<int>(wrapper.type) << "\n";
    os << "Segment: " << wrapper.segment.transpose() << "\n"; // Assuming Segment is a matrix or vector
    os << "Start BearingVector: " << wrapper.s_f.transpose() << "\n"; // Assuming BearingVector is a matrix or vector
    os << "End BearingVector: " << wrapper.e_f.transpose() << "\n"; // Assuming BearingVector is a matrix or vector
    os << "Gradient Vector: " << wrapper.grad.transpose() << "\n"; // Assuming GradientVector is a matrix or vector
    os << "Score: " << wrapper.score << "\n";
    os << "Level: " << wrapper.level << "\n";
    // os << "Line Landmark: " << wrapper.line_landmark->spos_ <<" "<<wrapper.line_landmark->epos_ <<"\n"; // Assuming LinePtr is a pointer to some type that has an operator<<
    os << "Seed Reference: " << wrapper.seed_ref.seed_id << "\n";
    os << "Track ID: " << wrapper.track_id << "\n";

    return os;
}
bool matchSegmentCandidate(
    const FramePtr& frame,
    Reprojector::Candidate& c,
    Matcher& matcher,
    SegmentWrapper& segment,
    const double seed_sigma2_thresh)// from map project to this frame 
{
  GradientVector grad_ref;

  // direct matching
  CHECK_NOTNULL(c.ref_frame.get());
  CHECK_LT(c.ref_index, c.ref_frame->num_segments_);
  int track_id = -1;
  BearingVector s_f_cur,e_f_cur;
  if(c.ref_frame->seg_landmark_vec_[c.ref_index]!=nullptr)
    CHECK(static_cast<int>(c.ref_frame->seg_type_vec_[c.ref_index])>8);
  if(c.ref_frame->seg_landmark_vec_[c.ref_index]!=nullptr)
    CHECK(static_cast<int>(c.ref_frame->seg_type_vec_[c.ref_index])>8);
  if(c.ref_frame->seg_landmark_vec_.at(c.ref_index) == nullptr)
  {
    SegmentWrapper ref_ftr = c.ref_frame->getSegmentWrapper(c.ref_index);
    
    
    if(isConvergedSegmentSeed(c.type) )
    {
      const Eigen::Matrix<FloatType ,2,1> ref_depth = c.ref_frame->getSegmentSeedDepth(c.ref_index);
      // NAME_VALUE_LOG(isConvergedSegmentSeed(c.type) );
      // NAME_VALUE_LOG(c.ref_frame->getSegmentSeedDepth(c.ref_index));//test the inv_depth
      // NAME_VALUE_LOG(c.ref_frame->seg_f_vec_(c.ref_index*2));//test the inv_depth
      // NAME_VALUE_LOG(c.ref_frame->seg_f_vec_(c.ref_index*2+1));//test the inv_depth
      CHECK_EIGEN_HAVE_NAN(ref_depth,"whrn matchcandidata")
      CHECK_EIGEN_HAVE_NAN(ref_depth,"whrn matchcandidata")


      // NAME_VALUE_LOG(isConvergedSegmentSeed(c.type) );
      // NAME_VALUE_LOG(c.ref_frame->getSegmentSeedDepth(c.ref_index));//test the inv_depth
      // NAME_VALUE_LOG(c.ref_frame->seg_f_vec_(c.ref_index*2));//test the inv_depth
      // NAME_VALUE_LOG(c.ref_frame->seg_f_vec_(c.ref_index*2+1));//test the inv_depth
      CHECK_EIGEN_HAVE_NAN(ref_depth,"whrn matchcandidata")
      CHECK_EIGEN_HAVE_NAN(ref_depth,"whrn matchcandidata")


      std::vector<Matcher::MatchResult>  res =
          matcher.findMatchDirectSegment(*c.ref_frame, *frame, ref_ftr, ref_depth(0),ref_depth(1),
                                  c.cur_px,c.cur_px_end,s_f_cur,e_f_cur);
      
      if(res[0] != Matcher::MatchResult::kSuccess|| res[1]!=Matcher::MatchResult::kSuccess)
                                  c.cur_px,c.cur_px_end,s_f_cur,e_f_cur);
      
      if(res[0] != Matcher::MatchResult::kSuccess|| res[1]!=Matcher::MatchResult::kSuccess)
        return false;


    }
    else if(isUnconvergedSegmentSeed(c.type) )
    {
      Segment seg_;
      Segment seg_;
      if(!depth_filter_utils::updateSegmentSeed(
           *frame, *c.ref_frame, c.ref_index, matcher, seed_sigma2_thresh,s_f_cur,e_f_cur,seg_,
           *frame, *c.ref_frame, c.ref_index, matcher, seed_sigma2_thresh,s_f_cur,e_f_cur,seg_,
           false, false))
      {
        return false;
      }
        c.cur_px=seg_.head<2>();
        c.cur_px_end=seg_.tail<2>();
        // NAME_VALUE_LOG(static_cast<int>(c.type));
        c.cur_px=seg_.head<2>();
        c.cur_px_end=seg_.tail<2>();
        // NAME_VALUE_LOG(static_cast<int>(c.type));
    }
    else
    {
      CHECK(false) << "Seed type unknown";
    }

    grad_ref = ref_ftr.grad;
    track_id = ref_ftr.track_id;// just relate a point3d or line3d
    segment.seed_ref.keyframe = c.ref_frame;// change the segment would change the seedref vector in frame
    segment.seed_ref.seed_id = c.ref_index;
    // segment.s_f=s_f_cur;
    // segment.e_f=e_f_cur;
    // segment.s_f=s_f_cur;
    // segment.e_f=e_f_cur;
  }
  else// is the seg_landmar
  else// is the seg_landmar
  {
    const LinePtr& line3d = c.ref_frame->seg_landmark_vec_[c.ref_index];//the landmark and seed is different
    FramePtr ref_frame;
    size_t ref_segment_index;
    if(!line3d->getCloseViewObs(frame->pos(), ref_frame, ref_segment_index))// the 2 ray from different frame position to point position,get the min angle frame  
    {
      return false;
    }
    SegmentWrapper ref_ftr = ref_frame->getSegmentWrapper(ref_segment_index);
    CHECK(static_cast<int>(ref_ftr.type)>8);
    CHECK(static_cast<int>(ref_ftr.type)>8);
    CHECK_NOTNULL(ref_ftr.line_landmark.get()); // debug
    const FloatType ref_depth_s = (ref_frame->pos() - ref_ftr.line_landmark->spos_).norm();
    const FloatType ref_depth_e = (ref_frame->pos() - ref_ftr.line_landmark->epos_).norm();
    CHECK(!std::isnan(ref_depth_e)&&!std::isnan(ref_depth_e));
    CHECK(!std::isnan(ref_depth_e)&&!std::isnan(ref_depth_e));
    std::vector<Matcher::MatchResult>  res =
          matcher.findMatchDirectSegment(*ref_frame, *frame, ref_ftr, ref_depth_s,ref_depth_e,
                                  c.cur_px,c.cur_px_end,s_f_cur,e_f_cur);

          matcher.findMatchDirectSegment(*ref_frame, *frame, ref_ftr, ref_depth_s,ref_depth_e,
                                  c.cur_px,c.cur_px_end,s_f_cur,e_f_cur);

    if(res[0] != Matcher::MatchResult::kSuccess||res[1] != Matcher::MatchResult::kSuccess)
    {
      line3d->n_failed_reproj_++;
      return false; // TODO(cfo): We should return match result and write in statistics.
    }
    line3d->n_succeeded_reproj_ += 1;
    grad_ref = ref_ftr.grad;
    segment.line_landmark = line3d;
    CHECK(static_cast<int>(segment.type)>8);
    CHECK(static_cast<int>(segment.type)>8);
    track_id = line3d->id();
  }

  // Set edgelet direction, check if is consistent.
  if(isEdgelet(c.type)||isSegment(c.type))
  if(isEdgelet(c.type)||isSegment(c.type))
  {
    GradientVector g_predicted = (matcher.A_cur_ref_ * grad_ref).normalized();
    segment.grad = g_predicted;
    /*
    double angle = feature_detection_utils::getAngleAtPixelUsingHistogram(
          frame->img_pyr_[matcher.search_level_],
          (matcher.px_cur_ / (1<<matcher.search_level_)).cast<int>(), 4);
    feature.grad = GradientVector(std::cos(angle), std::sin(angle));

    constexpr double thresh = std::cos(30.0/180.0*3.81);
    if(std::abs(feature.grad.dot(g_predicted)) > thresh)
      return false;
    */
  }
  // NAME_VALUE_LOG(s_f_cur);
  // NAME_VALUE_LOG(e_f_cur);
  // NAME_VALUE_LOG(s_f_cur);
  // NAME_VALUE_LOG(e_f_cur);
  // Here we add a reference in the segment and frame to the 3D point, the other way
  // round is only done if this frame is selected as keyframe.
  segment.type = c.type;
  segment.segment.head<2>() = c.cur_px;
  segment.segment.tail<2>() = c.cur_px_end;
  segment.s_f = s_f_cur;
  segment.e_f = e_f_cur;
  segment.s_f = s_f_cur;
  segment.e_f = e_f_cur;
  segment.level = matcher.search_level_;
  segment.track_id = track_id;
  segment.score = c.score;
  // TODO: ideally the following should be done via FeatureWrapper
  // This assumes that the feature points to the first free slot
  // i.e., the wrapper is got from getEmptyFeatureWrapper function
  frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2) =
      c.ref_frame->seg_invmu_sigma2_a_b_vec_.col(c.ref_index*2);
    frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2+1) =
      c.ref_frame->seg_invmu_sigma2_a_b_vec_.col(c.ref_index*2+1);
  // NAME_VALUE_LOG(c.ref_index);
  // NAME_VALUE_LOG(c.ref_frame);
  // NAME_VALUE_LOG(static_cast<int>(c.type));
    // if(isSeed(segment.type)){
    // CHECK(!(frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())<<static_cast<int>(segment.type);
    // CHECK(!(frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2+1).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())<<static_cast<int>(segment.type);
    // }
  // CHECK_EIGEN_HAVE_NAN(c.ref_frame->seg_invmu_sigma2_a_b_vec_,"after project");
  
  // CHECK_EIGEN_HAVE_NAN(c.ref_frame->seg_invmu_sigma2_a_b_vec_,"after project");
  frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2) =
      c.ref_frame->seg_invmu_sigma2_a_b_vec_.col(c.ref_index*2);
    frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2+1) =
      c.ref_frame->seg_invmu_sigma2_a_b_vec_.col(c.ref_index*2+1);
  // NAME_VALUE_LOG(c.ref_index);
  // NAME_VALUE_LOG(c.ref_frame);
  // NAME_VALUE_LOG(static_cast<int>(c.type));
    // if(isSeed(segment.type)){
    // CHECK(!(frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())<<static_cast<int>(segment.type);
    // CHECK(!(frame->seg_invmu_sigma2_a_b_vec_.col(frame->numSegments()*2+1).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())<<static_cast<int>(segment.type);
    // }
  // CHECK_EIGEN_HAVE_NAN(c.ref_frame->seg_invmu_sigma2_a_b_vec_,"after project");
  
  // CHECK_EIGEN_HAVE_NAN(c.ref_frame->seg_invmu_sigma2_a_b_vec_,"after project");
  frame->seg_in_ba_graph_vec_[frame->numSegments()] =
      c.ref_frame->seg_in_ba_graph_vec_[c.ref_index];
  return true;
}

#endif

#endif

bool matchCandidate(
    const FramePtr& frame,
    Reprojector::Candidate& c,
    Matcher& matcher,
    FeatureWrapper& feature,
    const double seed_sigma2_thresh)// from map project to this frame 
{
  GradientVector grad_ref;

  // direct matching
  CHECK_NOTNULL(c.ref_frame.get());
  CHECK_LT(c.ref_index, c.ref_frame->num_features_);
  int track_id = -1;
  if(c.ref_frame->landmark_vec_.at(c.ref_index) == nullptr)
  {
    FeatureWrapper ref_ftr = c.ref_frame->getFeatureWrapper(c.ref_index);
    if(isConvergedCornerEdgeletSeed(c.type) || isConvergedMapPointSeed(c.type))
    {
      const FloatType ref_depth = c.ref_frame->getSeedDepth(c.ref_index);
      Matcher::MatchResult res =
          matcher.findMatchDirect(*c.ref_frame, *frame, ref_ftr, ref_depth,
                                  c.cur_px);
      if(res != Matcher::MatchResult::kSuccess)
        return false;
    }
    else if(isUnconvergedCornerEdgeletSeed(c.type) ||
            isUnconvergedMapPointSeed(c.type))
    {
      if(!depth_filter_utils::updateSeed(
           *frame, *c.ref_frame, c.ref_index, matcher, seed_sigma2_thresh,
           false, false))
      {
        return false;
      }
    }
    else
    {
      CHECK(false) << "Seed type unknown";
    }

    grad_ref = ref_ftr.grad;
    track_id = ref_ftr.track_id;
    feature.seed_ref.keyframe = c.ref_frame;
    feature.seed_ref.seed_id = c.ref_index;
  }
  else
  {
    const PointPtr& point = c.ref_frame->landmark_vec_[c.ref_index];//the landmark and seed is different
    FramePtr ref_frame;
    size_t ref_feature_index;
    if(!point->getCloseViewObs(frame->pos(), ref_frame, ref_feature_index))// the 2 ray from different frame position to point position,get the min angle frame  
    {
      return false;
    }
    FeatureWrapper ref_ftr = ref_frame->getFeatureWrapper(ref_feature_index);
    CHECK_NOTNULL(ref_ftr.landmark.get()); // debug
    const FloatType ref_depth = (ref_frame->pos() - ref_ftr.landmark->pos()).norm();
    Matcher::MatchResult res = matcher.findMatchDirect(
          *ref_frame, *frame, ref_ftr, ref_depth, c.cur_px);// matcher get the ref_frame and ref_feature
    if(res != Matcher::MatchResult::kSuccess)
    {
      point->n_failed_reproj_++;
      return false; // TODO(cfo): We should return match result and write in statistics.
    }
    point->n_succeeded_reproj_ += 1;
    grad_ref = ref_ftr.grad;
    feature.landmark = point;
    track_id = point->id();
  }

  // Set edgelet direction, check if is consistent.
  if(isEdgelet(c.type))
  {
    GradientVector g_predicted = (matcher.A_cur_ref_ * grad_ref).normalized();
    feature.grad = g_predicted;
    /*
    double angle = feature_detection_utils::getAngleAtPixelUsingHistogram(
          frame->img_pyr_[matcher.search_level_],
          (matcher.px_cur_ / (1<<matcher.search_level_)).cast<int>(), 4);
    feature.grad = GradientVector(std::cos(angle), std::sin(angle));

    constexpr double thresh = std::cos(30.0/180.0*3.81);
    if(std::abs(feature.grad.dot(g_predicted)) > thresh)
      return false;
    */
  }

  // Here we add a reference in the feature and frame to the 3D point, the other way
  // round is only done if this frame is selected as keyframe.
  feature.type = c.type;
  feature.px = matcher.px_cur_;
  feature.f = matcher.f_cur_;
  feature.level = matcher.search_level_;
  feature.track_id = track_id;
  feature.score = c.score;
  // TODO: ideally the following should be done via FeatureWrapper
  // This assumes that the feature points to the first free slot
  // i.e., the wrapper is got from getEmptyFeatureWrapper function
  frame->invmu_sigma2_a_b_vec_.col(frame->numFeatures()) =
      c.ref_frame->invmu_sigma2_a_b_vec_.col(c.ref_index);
  frame->in_ba_graph_vec_[frame->numFeatures()] =
      c.ref_frame->in_ba_graph_vec_[c.ref_index];
  return true;
}

bool getCandidate(
    const FramePtr& cur_frame,
    const FramePtr& ref_frame,
    const size_t& ref_index,
    Reprojector::Candidate& candidate)
{
  Position xyz_world = Position::Zero();
  int n_reproj = 0;
  if(ref_frame->landmark_vec_[ref_index] != nullptr)
  {
    const PointPtr& point = ref_frame->landmark_vec_[ref_index];
    xyz_world = point->pos();
    n_reproj =
        point->n_succeeded_reproj_ - point->n_failed_reproj_;
  }
  else
  {
    xyz_world = ref_frame->T_world_cam() *
        ref_frame->getSeedPosInFrame(ref_index);
  }

  Keypoint px;
  if(!projectPointAndCheckVisibility(cur_frame, xyz_world, &px))
  {
    return false;
  }

  candidate = Reprojector::Candidate(
        ref_frame, ref_index, px, n_reproj, ref_frame->score_vec_[ref_index],
        ref_frame->type_vec_[ref_index],
        ref_frame->landmark_vec_[ref_index]?
          ref_frame->landmark_vec_[ref_index]->obs_.size() : 0u);

  return true;
}



bool projectPointAndCheckVisibility(
    const FramePtr& frame,
    const Eigen::Vector3d& xyz,
    Eigen::Vector2d* px)
{
  CHECK_NOTNULL(px);

  // compute where the point projects and check visibility
  // TODO: insert reasonable clipping distance
  if (!frame->isVisible(xyz, px))
  {
    return false;
  }
  const Eigen::Vector2i pxi = px->cast<int>();
  constexpr int kPatchSize = 8;
  if(!frame->cam()->isKeypointVisibleWithMargin(pxi, kPatchSize))
    return false;
  return true;
}

void setGridCellsOccupied(
    const Reprojector::Candidates& candidates,
    OccupandyGrid2D& grid)
{
  for(auto candidate : candidates)
  {
    size_t grid_index =
        grid.getCellIndex(candidate.cur_px.x(), candidate.cur_px.y(), 1);
    grid.setOccupied(grid_index);
    if( isSegment(candidate.type))
    {
            grid_index =grid.getCellIndex(candidate.cur_px_end.x(), candidate.cur_px_end.y(), 1);
            grid_index =grid.getCellIndex(candidate.cur_px_end.x(), candidate.cur_px_end.y(), 1);
    grid.setOccupied(grid_index);
    }

  }
}

#ifdef SEGMENT_ENABLE
#ifdef SEGMENT_ENABLE
void setGridCellsOccupiedSegment(
    const Reprojector::Candidates& candidates,
    OccupandyGrid2D& grid)
{
  for(auto candidate : candidates)
  {
    size_t grid_index =
        grid.getCellIndex(candidate.cur_px.x(), candidate.cur_px.y(), 1);
    grid.setOccupied(grid_index);
      grid_index =grid.getCellIndex(candidate.cur_px_end.x(), candidate.cur_px_end.y(), 1);
      grid_index =grid.getCellIndex(candidate.cur_px_end.x(), candidate.cur_px_end.y(), 1);
    grid.setOccupied(grid_index);
  }
}

bool getCandidateSegment(
    const FramePtr& cur_frame,
    const FramePtr& ref_frame,
    const size_t& seg_ref_index,
    Reprojector::Candidate& candidate
    )
{
  Position xyz_world_s = Position::Zero();
  Position xyz_world_e = Position::Zero();
  Eigen::Matrix<FloatType, 3, 2> xyz_world_seg;
  int n_reproj = 0;
  if(ref_frame->seg_landmark_vec_[seg_ref_index] != nullptr)
  {
    const LinePtr& line3d = ref_frame->seg_landmark_vec_[seg_ref_index];
    xyz_world_s = line3d->spos_;
    xyz_world_e = line3d->epos_;
    n_reproj =
        line3d->n_succeeded_reproj_ - line3d->n_failed_reproj_;
  }
  else 
  {
    xyz_world_seg = 
        ref_frame->getSegmentSeedPosInFrame(seg_ref_index);
    xyz_world_s = ref_frame->T_world_cam() *xyz_world_seg.col(0);
    xyz_world_e = ref_frame->T_world_cam() *xyz_world_seg.col(1);
  //   std::cout<<"getCandidateSegment\n";
  //     NAME_VALUE_LOG(ref_frame->T_world_cam() *xyz_world_seg.col(0));
  // NAME_VALUE_LOG(ref_frame->T_world_cam() *xyz_world_seg.col(1));
  // CHECK_EIGEN_HAVE_NAN(xyz_world_seg,"getCandidateSegment");
// if((xyz_world_seg.array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
// {
//   // std::cout<<"frame id"<<ref_frame->bundle_id_<<std::endl;
//   for(size_t a=0;a<ref_frame->num_segments_;a++)
//   {
//     std::cout<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(a*2).transpose()<<"\n type "<< static_cast<int>(ref_frame->seg_type_vec_[a])<<"\nf_vec"
//     <<"\n landmark"<<ref_frame->seg_landmark_vec_[a]<<"\n"<<std::endl;    
//     // std::cout<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(a*2).transpose()<<'\n'<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(a*2+1).transpose()
//     // <<"\n type "<< static_cast<int>(ref_frame->seg_type_vec_[a])<<"\nf_vec"<<ref_frame->seg_f_vec_.col(a*2).transpose()<<'\n'<<ref_frame->seg_f_vec_.col(a*2+1).transpose()
//     // <<"\n landmark"<<ref_frame->landmark_vec_[a]<<"\n"<<std::endl;
//   }
//   CHECK(false)<<"candidate's data"<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(seg_ref_index*2)<<
//   "\n"<<seg_ref_index<<"\nref_type"<<static_cast<int>(ref_frame->seg_type_vec_[seg_ref_index])<<"\n landmark"<<ref_frame->seg_landmark_vec_[seg_ref_index]<<"\n"<<std::endl;    
//   return false;
// }
  }
  Keypoint px_s;
  Keypoint px_e;
  if(!projectPointAndCheckVisibility(cur_frame, xyz_world_s, &px_s)|| !projectPointAndCheckVisibility(cur_frame, xyz_world_e, &px_e))
  {
    return false;
  }

  candidate = Reprojector::Candidate(
        ref_frame, seg_ref_index, px_s, px_e,n_reproj, ref_frame->seg_score_vec_[seg_ref_index],
        ref_frame->seg_type_vec_[seg_ref_index],
        ref_frame->seg_landmark_vec_[seg_ref_index]?
          ref_frame->seg_landmark_vec_[seg_ref_index]->obs_.size() : 0u);
  return true;
}

#endif

bool getCandidateSegment(
    const FramePtr& cur_frame,
    const FramePtr& ref_frame,
    const size_t& seg_ref_index,
    Reprojector::Candidate& candidate
    )
{
  Position xyz_world_s = Position::Zero();
  Position xyz_world_e = Position::Zero();
  Eigen::Matrix<FloatType, 3, 2> xyz_world_seg;
  int n_reproj = 0;
  if(ref_frame->seg_landmark_vec_[seg_ref_index] != nullptr)
  {
    const LinePtr& line3d = ref_frame->seg_landmark_vec_[seg_ref_index];
    xyz_world_s = line3d->spos_;
    xyz_world_e = line3d->epos_;
    n_reproj =
        line3d->n_succeeded_reproj_ - line3d->n_failed_reproj_;
  }
  else 
  {
    xyz_world_seg = 
        ref_frame->getSegmentSeedPosInFrame(seg_ref_index);
    xyz_world_s = ref_frame->T_world_cam() *xyz_world_seg.col(0);
    xyz_world_e = ref_frame->T_world_cam() *xyz_world_seg.col(1);
  //   std::cout<<"getCandidateSegment\n";
  //     NAME_VALUE_LOG(ref_frame->T_world_cam() *xyz_world_seg.col(0));
  // NAME_VALUE_LOG(ref_frame->T_world_cam() *xyz_world_seg.col(1));
  // CHECK_EIGEN_HAVE_NAN(xyz_world_seg,"getCandidateSegment");
// if((xyz_world_seg.array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
// {
//   // std::cout<<"frame id"<<ref_frame->bundle_id_<<std::endl;
//   for(size_t a=0;a<ref_frame->num_segments_;a++)
//   {
//     std::cout<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(a*2).transpose()<<"\n type "<< static_cast<int>(ref_frame->seg_type_vec_[a])<<"\nf_vec"
//     <<"\n landmark"<<ref_frame->seg_landmark_vec_[a]<<"\n"<<std::endl;    
//     // std::cout<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(a*2).transpose()<<'\n'<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(a*2+1).transpose()
//     // <<"\n type "<< static_cast<int>(ref_frame->seg_type_vec_[a])<<"\nf_vec"<<ref_frame->seg_f_vec_.col(a*2).transpose()<<'\n'<<ref_frame->seg_f_vec_.col(a*2+1).transpose()
//     // <<"\n landmark"<<ref_frame->landmark_vec_[a]<<"\n"<<std::endl;
//   }
//   CHECK(false)<<"candidate's data"<<ref_frame->seg_invmu_sigma2_a_b_vec_.col(seg_ref_index*2)<<
//   "\n"<<seg_ref_index<<"\nref_type"<<static_cast<int>(ref_frame->seg_type_vec_[seg_ref_index])<<"\n landmark"<<ref_frame->seg_landmark_vec_[seg_ref_index]<<"\n"<<std::endl;    
//   return false;
// }
  }
  Keypoint px_s;
  Keypoint px_e;
  if(!projectPointAndCheckVisibility(cur_frame, xyz_world_s, &px_s)|| !projectPointAndCheckVisibility(cur_frame, xyz_world_e, &px_e))
  {
    return false;
  }

  candidate = Reprojector::Candidate(
        ref_frame, seg_ref_index, px_s, px_e,n_reproj, ref_frame->seg_score_vec_[seg_ref_index],
        ref_frame->seg_type_vec_[seg_ref_index],
        ref_frame->seg_landmark_vec_[seg_ref_index]?
          ref_frame->seg_landmark_vec_[seg_ref_index]->obs_.size() : 0u);
  return true;
}

#endif

void reprojectMapPoints(const FramePtr& frame,
                        const std::vector<FramePtr>& overlap_kfs,
                        const ReprojectorOptions& options,
                        OccupandyGrid2D* grid)
{
  frame->resizeFeatureStorage(frame->numFeatures()+grid->occupancy_.size());
  std::vector<Reprojector::Candidate> candidates;
  grid->reset();
  Reprojector::Statistics stats;
  size_t n_converge_map_point_seed = 0;
  size_t n_map_point = 0;
  size_t n_map_point_seed = 0;
  for(const FramePtr& ref_frame : overlap_kfs)
  {
    // Try to reproject each map point that the other KF observes
    for(size_t i = 0; i < ref_frame->num_features_; ++i)
    {
      if (ref_frame->type_vec_[i] == FeatureType::kMapPointSeedConverged)
      {
        n_converge_map_point_seed++;
      }
      else if (ref_frame->type_vec_[i] == FeatureType::kMapPointSeed)
      {
        n_map_point_seed++;
      }
      else if (ref_frame->type_vec_[i] == FeatureType::kMapPoint)
      {
        n_map_point++;
      }

      if(ref_frame->type_vec_[i] == FeatureType::kMapPoint ||
         ref_frame->type_vec_[i] == FeatureType::kMapPointSeedConverged)
      {
        if (ref_frame->landmark_vec_[i] != nullptr &&
            ref_frame->landmark_vec_[i]->n_failed_reproj_ > 10)
        {
          continue;
        }
        Reprojector::Candidate candidate;
        if(reprojector_utils::getCandidate(frame, ref_frame, i, candidate))
        {
          candidates.push_back(candidate);
        }
      }
    }
  }

  VLOG(5) << "Map points count: " << std::endl
            << " - map point: " << n_map_point << std::endl
            << " - converged map point seed: " << n_converge_map_point_seed << std::endl
            << " - map point seed: " << n_map_point_seed;

  reprojector_utils::sortCandidatesByReprojStats(candidates);
  reprojector_utils::matchCandidates(
        frame,
        options.max_n_features_per_frame + options.max_map_features_per_frame,
        options.affine_est_offset, options.affine_est_gain,
        candidates, *grid, stats);

  VLOG(5) << "Reproject landmarks "
          << ": trials = " << stats.n_trials
          << ", matches = " << stats.n_matches;
}

} // namespace reprojector_utils
} // namespace svo
