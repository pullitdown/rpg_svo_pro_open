// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
#include <random>

#include <numeric>
#include <svo/direct/matcher.h>
#include <svo/common/point.h>
#include <svo/common/frame.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/tracker/feature_tracker.h>
#include <svo/direct/feature_detection_utils.h>


#define SEGMENT_ENABLE_stereo_triangulation
#define NAME_VALUE_LOG(x) std::cout << #x << ": \n" << (x) << std::endl;

namespace svo
{

void nonmax_line(const std::vector<Eigen::Vector2i> &corners, const std::vector<int> &scores,
                    const size_t nonmax_list_range,const size_t nonmax_space_range,std::vector<int> &nonmax_corners)
{
  CHECK((nonmax_list_range&1)==1);
  CHECK((nonmax_space_range&1)==1);
  size_t half_nonmax_range= nonmax_list_range/2;
  size_t half_nonmax_space_range =nonmax_space_range/2;
  size_t square_half_nonmax_range=half_nonmax_range*half_nonmax_range;
  if(scores.size()<half_nonmax_range)return;
  for(size_t idx=half_nonmax_range;idx<(corners.size()-half_nonmax_range);++idx)
  { 
    
    for(size_t in_idx=idx-half_nonmax_range;in_idx<idx+half_nonmax_range;++in_idx)
    {
      if(idx==in_idx)continue;
      size_t distance=(corners[in_idx].x()-corners[idx].x())*(corners[in_idx].x()-corners[idx].x())+(corners[in_idx].y()-corners[idx].y())*(corners[in_idx].y()-corners[idx].y());
      if(scores[in_idx]>scores[idx] &&distance<=square_half_nonmax_range )//this point could be continue
      {
        goto nonmax;
      }
    }
    nonmax_corners.push_back(idx);
    nonmax:;
  }
}
void project_all_points_onto_line(const Eigen::Vector3d& line_direction, const Eigen::Vector3d& point_on_line, std::vector<Position>& points) {
    // 确保方向向量是单位向量
    Eigen::Vector3d unit_line_direction = line_direction.normalized();

    for (auto& point_vector : points) {
        // 计算点到过直线的点的向量
        Eigen::Vector3d point_to_line_vector = point_vector - point_on_line;
        // 计算该向量在直线方向向量上的投影（标量投影）
        double scalar_projection = point_to_line_vector.dot(unit_line_direction);
        // 将标量投影乘以直线的单位方向向量，得到向量投影
        Eigen::Vector3d vector_projection = scalar_projection * unit_line_direction;
        // 得到该点在直线上的投影坐标
        Eigen::Vector3d projected_point_vector = point_on_line + vector_projection;
        // 直接修改点的坐标
        point_vector = projected_point_vector;
    }
}
void project_points_onto_line(const Eigen::Vector3d& line_direction, const Eigen::Vector3d& point_on_line, std::vector<Position>& points, const std::vector<size_t>& indices) {
    // 确保方向向量是单位向量
    Eigen::Vector3d unit_line_direction = line_direction.normalized();

    for (size_t index : indices) {
        // 检查索引是否有效
        if (index >= 0 && index < points.size()) {
            Eigen::Vector3d point_vector=points[index];
            // 计算点到过直线的点的向量
            Eigen::Vector3d point_to_line_vector = point_vector - point_on_line;
            // 计算该向量在直线方向向量上的投影（标量投影）
            double scalar_projection = point_to_line_vector.dot(unit_line_direction);
            // 将标量投影乘以直线的单位方向向量，得到向量投影
            Eigen::Vector3d vector_projection = scalar_projection * unit_line_direction;
            // 得到该点在直线上的投影坐标
            Eigen::Vector3d projected_point_vector = point_on_line + vector_projection;
            // 直接修改点的坐标
            points[index] = projected_point_vector;
        
        }
    }
}
// best_fit_line函数，通过引用返回最佳拟合线的方向和质心
void best_fit_line(const std::vector<Position>& points, const std::vector<size_t>& indices, Eigen::Vector3d& direction, Eigen::Vector3d& centroid) {
    Eigen::MatrixXd data(indices.size(), 3);
    for (size_t i = 0; i < indices.size(); ++i) {
        const Position& p = points[indices[i]];
        data(i, 0) = p.x();
        data(i, 1) = p.y();
        data(i, 2) = p.z();
    }

    centroid = data.colwise().mean();
    data.rowwise() -= centroid.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(data, Eigen::ComputeFullU | Eigen::ComputeFullV);
    direction = svd.matrixV().col(2);
}

// ransac_line_fit函数，通过引用返回最佳拟合线的方向和质心
void ransac_line_fit(const std::vector<Position>& points, int max_iterations, double distance_threshold, Eigen::Vector3d& best_direction, Eigen::Vector3d& best_centroid,std::vector<size_t>& best_inliers_indices) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> uni(0, points.size() - 1);

    size_t best_inliers_count = 0;
    ;

    for (int i = 0; i < max_iterations; ++i) {
        size_t idx1 = uni(rng);
        size_t idx2;
        do {
            idx2 = uni(rng);
        } while (idx2 == idx1);

        Eigen::Vector3d pt1(points[idx1]);
        Eigen::Vector3d pt2(points[idx2]);
        Eigen::Vector3d line_direction = (pt2 - pt1).normalized();

        size_t inliers_count = 0;
        std::vector<size_t> inliers_indices;
        for (size_t j = 0; j < points.size(); ++j) {
            const Position& point_vector = points[j];
            double distance = (point_vector - pt1).cross(point_vector - pt2).norm() / line_direction.norm();
            if (distance < distance_threshold) {
                inliers_count++;
                inliers_indices.push_back(j);
            }
        }

        if (inliers_count > best_inliers_count) {
            best_inliers_count = inliers_count;
            best_inliers_indices = inliers_indices;

            // 使用最佳内点集合的下标更新最佳方向和质心
            best_fit_line(points, best_inliers_indices, best_direction, best_centroid);
        }
    }
}


bool detectorLine(const FramePtr frame,
                  const std::vector<Eigen::Vector2i> &linePoints,
                  const double threshold,
                  std::vector<Score> &scores_,
                  Keypoints &corners_)
{
  // std::cout<<"linePoints.size()"<<linePoints.size()<<std::endl;
  std::vector<Eigen::Vector2i> shitomasi_corners;

  std::vector<int> nm_corners,scores;

  for (size_t idx=0;idx<static_cast<size_t>(linePoints.size());++idx)
  {
    double score;
    if (feature_detection_utils::getShiTomasiScore(frame->img_pyr_[0], linePoints[idx], &score))
    {
      if (score > threshold)
      {
        shitomasi_corners.push_back(linePoints[idx]);
        scores.push_back(score);
      }
    }
  }
  // for(auto a:shitomasi_corners)std::cout<<a<<std::endl;
  // std::cout<<"scores.size()"<<scores.size()<<shitomasi_corners.size()<<std::endl;
  if(scores.size()==0)return false;
  nonmax_line(shitomasi_corners, scores,7,9, nm_corners);
  // nm_corners.push_back(0);
  if(nm_corners.size()==0)return false;
  // std::cout<<"nm_corners.size()"<<nm_corners.size()<<std::endl;
   

  if (nm_corners.size() < 4)
  {
    for (size_t idx = 0; idx < nm_corners.size(); ++idx)
    {
      scores_[idx]=(scores[nm_corners[idx]]);
      corners_.col(idx) = shitomasi_corners[nm_corners[idx]].cast<svo::FloatType>();

    }

    return true;
  }

  std::nth_element(nm_corners.begin(), nm_corners.begin() + 3, nm_corners.end(), [&scores](int i1, int i2)
                   { return scores[i1] > scores[i2]; });

  for (size_t i = 0; i < 4; ++i)
  {
      scores_[i]=(scores[nm_corners[i]]); 
      corners_.col(i) = shitomasi_corners[nm_corners[i]].cast<svo::FloatType>();
  }
  // std::cout<<"corners_"<<corners_<<std::endl;
  return true;
}

void bresenhamLine(const Segments &line, std::vector<Eigen::Vector2i> &linePoints)
{
  
  int x0 = static_cast<int>(line(0));
  int y0 = static_cast<int>(line(1));
  int x1 = static_cast<int>(line(2));
  int y1 = static_cast<int>(line(3));
  // 判断线段斜率是否大于1
  bool steep = abs(y1 - y0) > abs(x1 - x0);

  // 如果斜率大于1，则交换x和y的值（对坐标轴进行旋转）
  if (steep)
  {
    std::swap(x0, y0);
    std::swap(x1, y1);
  }

  // 如果起点在终点的右边，则交换起点和终点
  if (x0 > x1)
  {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }

  // 计算直线参数
  int dx = x1 - x0;
  int dy = abs(y1 - y0);
  int error = dx / 2;
  int ystep = (y0 < y1) ? 1 : -1;
  int y = y0;

  // Bresenham's line algorithm核心实现
  for (int x = x0; x <= x1; x++)
  {
    if (steep)
    {
      // 如果之前对坐标轴进行了旋转，现在需要旋转回来
      linePoints.push_back({y, x});
    }
    else
    {
      linePoints.push_back({x, y});
    }
    error -= dy;
    if (error < 0)
    {
      y += ystep;
      error += dx;
    }
  }
  
}


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
        if (frame0->numLandmarks() >= 120)
    {
      VLOG(5) << "Calling stereo triangulation with sufficient number of features"
              << " has no effect.";
      return ;
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
      return ;
    }

    // Compute and normalize all bearing vectors.
    // bearing vector corresponing the pixel place in a special camera distort model
    Bearings new_f;
    frame_utils::computeNormalizedBearingVectors(new_px, *frame0->cam(), &new_f);
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
#ifndef SEGMENT_ENABLE_stereo_triangulation
    std::vector<size_t> indices(static_cast<size_t>(n_new));
    std::iota(indices.begin(), indices.end(), n_old);
    long n_corners = std::count_if(
        new_types.begin(), new_types.end(),
        [](const FeatureType &t)
        { return t == FeatureType::kCorner; });

    // shuffle twice before we prefer corners!
    std::random_shuffle(indices.begin(), indices.begin() + n_corners);
    std::random_shuffle(indices.begin() + n_corners, indices.end());
#endif


#ifdef SEGMENT_ENABLE_stereo_triangulation
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
        // SVO_ERROR_STREAM("Stereo Triangulation: No segment detected.");
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
      // std::random_shuffle(seg_indices.begin() + n_segments, seg_indices.end());


//----------------------------------------------------------------------------------

      // std::cout<<"start line point process"<<std::endl;
      Keypoints new_line_px;
      Levels new_line_levels;
      Scores new_line_scores;
      Gradients new_line_grads;
      FeatureTypes new_line_types;
      Bearings new_line_f;

      new_line_px.conservativeResize(Eigen::NoChange, new_seg.cols() * 4);
      new_line_grads.conservativeResize(Eigen::NoChange, new_seg.cols() * 4);
      new_line_scores.conservativeResize(new_seg.cols() * 4);
      new_line_levels.conservativeResize(new_seg.cols() * 4);
      new_line_types.resize(new_seg.cols() * 4, FeatureType::kOutlier);


      // 访问数组越界元素
      size_t total_linepoint_line = 0;
      for (size_t i = 0; i < static_cast<size_t>(new_seg.cols()); i++)
      {
        Keypoints corners_;
        corners_.conservativeResize(Eigen::NoChange,4);
        std::vector<Score> scores_(4,-1);
        std::vector<Eigen::Vector2i> onelinepoints;
        bresenhamLine(new_seg.col(i), onelinepoints);
        if(!detectorLine(frame0, onelinepoints, 30, scores_, corners_))
        {
          continue;
        }
        for (size_t j = 0; j < scores_.size(); ++j)
        {
          size_t idx = i * 4 + j;
          new_line_px.col(idx) = corners_.col(j);
          new_line_scores(idx) = scores_[j];
          new_line_levels(idx) = new_seg_levels[i];
          new_line_types[idx] = FeatureType::kLinePoint;
          ++total_linepoint_line;
        }
        // std::cout<<"finish detectorline"<<std::endl;
      }
      // std::cout<<"end of line point process"<<std::endl;
      frame_utils::computeNormalizedBearingVectors(new_line_px, *frame0->cam(), &new_line_f);


      frame0->resizeFeatureStorage(
          frame0->num_features_ + total_linepoint_line);
      size_t now_idx = frame0->num_features_ ;
      // frame0->line_point_idx_vec_.reserve(frame0->line_point_idx_vec_.size() + total_linepoint_line);
      size_t seg_idx = n_seg_old;
      for (size_t linepoint_idx = 0; linepoint_idx < new_line_types.size(); ++linepoint_idx)
      {
        if (new_line_types[linepoint_idx] == FeatureType::kOutlier)
        {
          ++seg_idx;
          continue;
        }


        frame0->px_vec_.col(now_idx) = new_line_px.col(linepoint_idx);
        frame0->f_vec_.col(now_idx) = new_line_f.col(linepoint_idx);
        frame0->score_vec_[now_idx] = new_line_scores(linepoint_idx);
        frame0->level_vec_[now_idx] = new_line_levels(linepoint_idx);
        frame0->type_vec_[now_idx]=FeatureType::kLinePoint;
        frame0->line_point_idx_vec_.insert({seg_idx,now_idx});
        ++now_idx;
      }
      frame0->num_features_ += total_linepoint_line;
      // std::cout<<"frame_feature_num"<<frame0->num_features_<<std::endl;



    std::vector<size_t> indices(static_cast<size_t>(n_new+total_linepoint_line));
    std::iota(indices.begin(), indices.end(), n_old);
    long n_corners = std::count_if(
        new_types.begin(), new_types.end(),
        [](const FeatureType &t)
        { return t == FeatureType::kCorner ; });

    // shuffle twice before we prefer corners!
    std::random_shuffle(indices.begin(), indices.begin() + n_corners);
    std::random_shuffle(indices.begin() + n_corners, indices.begin()+n_new);

#endif

    // now for all maximum corners, initialize a new seed
    size_t n_succeded = 0, n_failed = 0;

    const size_t n_desired =
        options_.triangulate_n_features - frame0->numLandmarks();
    // need the minimum num feature to stereo triangulate
    // note: we checked already at start that n_desired will be larger than 0

#ifdef SEGMENT_ENABLE_stereo_triangulation
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

#ifdef SEGMENT_ENABLE_stereo_triangulation
    size_t n_segment_succeded = 0;
    size_t n_segment_failed = 0;
    FloatType depth_s = 0.0;
    FloatType depth_e = 0.0;
    Segment segment_cur;


    // SegmentWrapper ref_ftr_s;
    std::vector<Matcher::MatchResult> res_s;
    for (const size_t &i_ref :seg_indices)
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

        auto range = frame0->line_point_idx_vec_.equal_range(i_ref);
        std::vector<Position> all_point;
        
        for (auto it = range.first; it != range.second; ++it) 
        {
          if(frame0->landmark_vec_[it->second]!=nullptr)
          {
            all_point.push_back(frame0->landmark_vec_[it->second]->pos_);
          }
        }
        if(all_point.size()==0)goto out;
        all_point.push_back(xyz_world_s);
        all_point.push_back(xyz_world_e);
        Position direction,centoid;
        std::vector<size_t> best_inlier;
        ransac_line_fit(all_point,100,2,direction,centoid,best_inlier);
        // project_points_onto_line(direction,centoid,all_point,best_inlier);
        project_all_points_onto_line(direction,centoid,all_point);
        frame0->seg_landmark_vec_[i_ref]->spos_ = all_point[all_point.size()-2];
        frame0->seg_landmark_vec_[i_ref]->epos_ = all_point[all_point.size()-1];
      }
      else
      {
        ++n_segment_failed;
      }
      out:
      if (n_segment_succeded >= n_desired_segment)
        break;
    }

    VLOG(20) << "Stereo: Triangulated " << n_segment_succeded << " features,"
             << n_segment_failed << " failed.";
#endif
  }

} // namespace svo
