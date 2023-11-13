// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#include <numeric>
#include <svo/direct/matcher.h>
#include <svo/common/point.h>
#include <svo/common/frame.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/tracker/feature_tracker.h>

#define SEGMENT_ENABLE
#define NAME_VALUE_LOG(x) std::cout << #x << ": \n" << (x) << std::endl;

namespace svo
{

  StereoTriangulation::StereoTriangulation(
      const StereoTriangulationOptions &options,
      const AbstractDetector::Ptr &feature_detector)
      : options_(options), feature_detector_(feature_detector)
  {
    ;
  }

  StereoTriangulation::StereoTriangulation(
      const StereoTriangulationOptions &options,
      const AbstractDetector::Ptr &feature_detector,
      const SegmentAbstractDetector::Ptr &segment_detector)
      : options_(options), feature_detector_(feature_detector), segment_detector_(segment_detector)
  {
    ;
  }

  void StereoTriangulation::compute(const FramePtr &frame0,
                                    const FramePtr &frame1)
  {
    // Check if there is something to do
    if (frame0->numLandmarks() >= options_.triangulate_n_features)
    {
      VLOG(5) << "Calling stereo triangulation with sufficient number of features"
              << " has no effect.";
      return;
    }

    // Detect new features.
    Keypoints new_px;
    Levels new_levels;
    Scores new_scores;
    Gradients new_grads;
    FeatureTypes new_types;
    const size_t max_n_features = feature_detector_->grid_.size();
    feature_detector_->detect(
        frame0->img_pyr_, frame0->getMask(), max_n_features, new_px,
        new_scores, new_levels, new_grads, new_types);

    if (new_px.cols() == 0)
    {
      SVO_ERROR_STREAM("Stereo Triangulation: No features detected.");
      return;
    }

    // Compute and normalize all bearing vectors.
    // bearing vector corresponing the pixel place in a special camera distort model
    Bearings new_f;
    frame_utils::computeNormalizedBearingVectors(new_px, *frame0->cam(), &new_f);

#ifdef SEGMENT_ENABLE
    // dgz todo

      Segments new_seg;
      Scores new_seg_score;
      FeatureTypes new_seg_types;
      Levels new_seg_levels;
      Gradients new_seg_grads;

      segment_detector_->detect(
          frame0->img_pyr_, frame0->getMask(), segment_detector_->options_.max_segment_num, new_seg,
          new_seg_score, new_seg_levels,new_seg_grads,new_seg_types);
      Bearings new_seg_f;
      frame_utils::computeSegmentNormalizedBearingVectors(new_seg, *frame0->cam(), &new_seg_f);

      const long n_seg_old = static_cast<long>(frame0->numSegments());
      const long n_seg_new = static_cast<long>(new_seg.cols());
      frame0->resizeSegmentStorage(n_seg_old + static_cast<size_t>(n_seg_new));
      // frame0->check_segment_infolist_vaild();
      frame0->seg_vec_.middleCols(n_seg_old, n_seg_new) = new_seg;
      frame0->seg_f_vec_.middleCols(n_seg_old * 2, n_seg_new * 2) = new_seg_f; // note add 2 time segment num
      
      // NAME_VALUE_LOG(frame0->seg_f_vec_.middleCols(n_seg_old * 2, n_seg_new * 2));//pass
      // NAME_VALUE_LOG(new_seg_f.cols());
      // NAME_VALUE_LOG(new_seg.cols());
      if (new_seg.cols() == 0)
      {
        SVO_ERROR_STREAM("Stereo Triangulation: No segment detected.");
      }
      frame0->seg_level_vec_.segment(n_seg_old, n_seg_new) = new_seg_levels;
      frame0->seg_score_vec_.segment(n_seg_old, n_seg_new) = new_seg_score;
      frame0->seg_grad_vec_.middleCols(n_seg_old , n_seg_new ) = new_seg_grads;
      frame0->num_segments_ += static_cast<size_t>(n_seg_new);
      frame0->seg_type_vec_.insert(
          frame0->seg_type_vec_.begin() + n_seg_old, new_seg_types.cbegin(), new_seg_types.cend());


      std::vector<size_t> seg_indices(static_cast<size_t>(n_seg_new));
      std::iota(seg_indices.begin(), seg_indices.end(), n_seg_old);
      long n_segments = std::count_if(
          new_seg_types.begin(), new_seg_types.end(),
          [](const FeatureType &t)
          { return t == FeatureType::kSegment; });

      // shuffle twice before we prefer corners!
      std::random_shuffle(seg_indices.begin(), seg_indices.begin() + n_segments);
      // std::random_shuffle(seg_indices.begin() + n_segments, seg_indices.end());
    
#endif

    // Add features to first frame.
    const long n_old = static_cast<long>(frame0->numFeatures());
    const long n_new = new_px.cols();
    frame0->resizeFeatureStorage(
        frame0->num_features_ + static_cast<size_t>(n_new));
    
    frame0->px_vec_.middleCols(n_old, n_new) = new_px;
    frame0->f_vec_.middleCols(n_old, n_new) = new_f;
    frame0->grad_vec_.middleCols(n_old, n_new) = new_grads;
    frame0->score_vec_.segment(n_old, n_new) = new_scores;
    frame0->level_vec_.segment(n_old, n_new) = new_levels;
    frame0->num_features_ += static_cast<size_t>(n_new);
    frame0->type_vec_.insert(
        frame0->type_vec_.begin() + n_old, new_types.cbegin(), new_types.cend());

    // We only want a limited number of features. Therefore, we create a random
    // vector of indices that we will process.
    std::vector<size_t> indices(static_cast<size_t>(n_new));
    std::iota(indices.begin(), indices.end(), n_old);
    long n_corners = std::count_if(
        new_types.begin(), new_types.end(),
        [](const FeatureType &t)
        { return t == FeatureType::kCorner; });

    // shuffle twice before we prefer corners!
    std::random_shuffle(indices.begin(), indices.begin() + n_corners);
    std::random_shuffle(indices.begin() + n_corners, indices.end());

    // now for all maximum corners, initialize a new seed
    size_t n_succeded = 0, n_failed = 0;

    const size_t n_desired =
        options_.triangulate_n_features - frame0->numLandmarks();
    // need the minimum num feature to stereo triangulate
    // note: we checked already at start that n_desired will be larger than 0

#ifdef SEGMENT_ENABLE
    // dgztodo
    const size_t n_desired_segment =
        options_.triangulate_n_segment - frame0->numSegmentLandmarks(); // segmentlandmark corresponding the fixed line
    if (frame1->num_segments_ + n_desired_segment > frame1->seg_landmark_vec_.size())
    {
      frame1->resizeSegmentStorage(frame1->num_segments_ + n_desired_segment);
      // frame1->check_segment_infolist_vaild();
    }

#endif

    // reserve space for features in second frame
    if (frame1->num_features_ + n_desired > frame1->landmark_vec_.size())
    {
      frame1->resizeFeatureStorage(frame1->num_features_ + n_desired);
    }

    Matcher matcher;
    matcher.options_.max_epi_search_steps = 500;
    matcher.options_.subpix_refinement = true;
    const Transformation T_f1f0 = frame1->T_cam_body_ * frame0->T_body_cam_;
    for (const size_t &i_ref : indices)
    {
      matcher.options_.align_1d = isEdgelet(frame0->type_vec_[i_ref]); // TODO(cfo): check effect
      FloatType depth = 0.0;
      FeatureWrapper ref_ftr = frame0->getFeatureWrapper(i_ref);
      Matcher::MatchResult res =
          matcher.findEpipolarMatchDirect(
              *frame0, *frame1, T_f1f0, ref_ftr, options_.mean_depth_inv,
              options_.min_depth_inv, options_.max_depth_inv, depth);

      if (res == Matcher::MatchResult::kSuccess)
      {
        const Position xyz_world = frame0->T_world_cam() * (frame0->f_vec_.col(static_cast<int>(i_ref)) * depth);
        PointPtr new_point(new Point(xyz_world));
        frame0->landmark_vec_[i_ref] = new_point;
        frame0->track_id_vec_(static_cast<int>(i_ref)) = new_point->id(); // track id is
        new_point->addObservation(frame0, i_ref);

        const int i_cur = static_cast<int>(frame1->num_features_);
        frame1->type_vec_[static_cast<size_t>(i_cur)] = ref_ftr.type;
        frame1->level_vec_[i_cur] = ref_ftr.level;
        frame1->px_vec_.col(i_cur) = matcher.px_cur_;
        frame1->f_vec_.col(i_cur) = matcher.f_cur_;
        frame1->score_vec_[i_cur] = ref_ftr.score;
        GradientVector g = matcher.A_cur_ref_ * ref_ftr.grad;
        frame1->grad_vec_.col(i_cur) = g.normalized();
        frame1->landmark_vec_[static_cast<size_t>(i_cur)] = new_point;
        frame1->track_id_vec_(i_cur) = new_point->id();
        new_point->addObservation(frame1, static_cast<size_t>(i_cur));
        frame1->num_features_++;
        ++n_succeded;
      }
      else
      {
        ++n_failed;
      }
      if (n_succeded >= n_desired)
        break;
    }

#ifdef SEGMENT_ENABLE
    size_t n_segment_succeded = 0;
    size_t n_segment_failed = 0;
    FloatType depth_s = 0.0;
    FloatType depth_e = 0.0;
    Segment segment_cur;

    // SegmentWrapper ref_ftr_s;
    std::vector<Matcher::MatchResult> res_s;
    for (const size_t &i_ref :seg_indices)
    {
      // matcher.options_.align_1d = isEdgelet(frame0->type_vec_[i_ref]); // TODO(cfo): check effect
      depth_s = 0.0;
      depth_e = 0.0;
      SegmentWrapper ref_ftr_s = frame0->getSegmentWrapper(i_ref);
      // NAME_VALUE_LOG(ref_ftr_s.s_f);
      // NAME_VALUE_LOG(ref_ftr_s.segment);
      // NAME_VALUE_LOG(ref_ftr_s.e_f);

      BearingVector s_f_cur;
      BearingVector e_f_cur;
      res_s =
          matcher.findEpipolarMatchDirectSegment(
              *frame0, *frame1, ref_ftr_s, options_.mean_depth_inv,
              options_.min_depth_inv, options_.max_depth_inv, depth_s,
              options_.mean_depth_inv,
              options_.min_depth_inv, options_.max_depth_inv, depth_e, segment_cur, s_f_cur,e_f_cur);

      if (res_s[0] == Matcher::MatchResult::kSuccess && res_s[1] == Matcher::MatchResult::kSuccess)
      {

//           std::cout<<"after stereo find epipolarmatch : and now frame1's idx"<< frame1->num_segments_<<std::endl;
//   std::cout<<"s_f_cur"<<s_f_cur<<"\ne_f_cur"<<e_f_cur<<"\nseg_cur.tail<2>()"<<segment_cur.tail<2>()<<"\nseg_cur.head<2>()"<<segment_cur.head<2>()<<std::endl;
//   // std::cout<<<<std::endl;
//  std::cout<<"s_matchresult"<<(res_s[0]== Matcher::MatchResult::kSuccess)<<"e_matchresult"<<(res_s[1]== Matcher::MatchResult::kSuccess)<<std::endl;
        const Position xyz_world_s = frame0->T_world_cam() * (frame0->seg_f_vec_.col(static_cast<int>(i_ref * 2)) * depth_s);
        const Position xyz_world_e = frame0->T_world_cam() * (frame0->seg_f_vec_.col(static_cast<int>(i_ref * 2 + 1)) * depth_e);

        // PointPtr new_point_s(new Point(xyz_world_s));
        LinePtr new_seg = std::make_shared<Line>(xyz_world_s, xyz_world_e);

        frame0->seg_landmark_vec_[i_ref] = new_seg;
        // CHECK(new_seg->id()>-1)<<"seg_track_id_vec_ is not set";
        frame0->seg_track_id_vec_(static_cast<int>(i_ref)) = new_seg->id(); // track id is
        new_seg->addObservation(frame0, i_ref);

        const size_t i_cur = frame1->num_segments_;
        frame1->seg_type_vec_[i_cur] = ref_ftr_s.type;
        frame1->seg_level_vec_[i_cur] = ref_ftr_s.level;
        frame1->seg_vec_.col(i_cur) = segment_cur;
        frame1->seg_f_vec_.col(i_cur * 2) = s_f_cur;
        frame1->seg_f_vec_.col(i_cur * 2 + 1) = e_f_cur;
        frame1->seg_score_vec_[i_cur] = ref_ftr_s.score;
        GradientVector g = matcher.A_cur_ref_ * ref_ftr_s.grad;
        frame1->seg_grad_vec_.col(i_cur) = g.normalized();

        frame1->seg_landmark_vec_[i_cur] = new_seg;
        frame1->seg_track_id_vec_(i_cur) = new_seg->id();
        new_seg->addObservation(frame1, i_cur);
        frame1->num_segments_++;
        ++n_segment_succeded;
        CHECK(static_cast<int>(frame1->seg_type_vec_[i_cur])>8);
        CHECK(static_cast<int>(frame0->seg_type_vec_[i_ref])>8);
      }
      else
      {
        ++n_segment_failed;
      }
      if (n_segment_succeded >= n_desired_segment)
        break;
    }
    // std::cout<<"after stereo triangulation two frame data status : frame 1 detect  feauture , segment :"<< frame1->num_features_<<","<<frame1->num_segments_<<
    // "frame 0 detect feature, segment"<< frame0->num_features_<< ","<< frame0->num_segments_<<std::endl;
    
    // std::cout<<"after stereo triangulation two frame data status :"<<std::endl;
    // NAME_VALUE_LOG(frame1->num_segments_)
    // NAME_VALUE_LOG(frame1->seg_f_vec_);
    // NAME_VALUE_LOG(frame0->num_segments_)

    // NAME_VALUE_LOG(frame0->seg_f_vec_);
    VLOG(20) << "Stereo: Triangulated " << n_segment_succeded << " features,"
             << n_segment_failed << " failed.";
#endif
  }

} // namespace svo
