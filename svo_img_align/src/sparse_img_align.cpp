// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/img_align/sparse_img_align.h>

#include <algorithm>
#include <random> // std::mt19937

#include <opencv2/highgui/highgui.hpp>

#include <vikit/vision.h>
#include <vikit/math_utils.h>

#include <svo/common/logging.h>
#include <svo/common/point.h>
#include <svo/common/camera.h>
#include <svo/common/seed.h>
#define NAME_VALUE_LOG(x) std::cout << #x << ": \n" << (x) << std::endl;

namespace svo
{

  SparseImgAlign::SparseImgAlign(
      SolverOptions optimization_options,
      SparseImgAlignOptions options)
      : SparseImgAlignBase(optimization_options, options) // use point and pose to bundle adjustment
  {
    setPatchSize<SparseImgAlign>(4);
  }

  size_t SparseImgAlign::run(
      const FrameBundle::Ptr &ref_frames,
      const FrameBundle::Ptr &cur_frames)
  {
    CHECK(!ref_frames->empty());
    CHECK_EQ(ref_frames->size(), cur_frames->size());

    // Select all visible features and subsample if required.
    fts_vec_.clear();
    size_t n_fts_to_track = 0;

    for (auto frame : ref_frames->frames_)
    {
      std::vector<size_t> fts;
      sparse_img_align_utils::extractFeaturesSubset(
          *frame, options_.max_level, patch_size_with_border_, fts);
      n_fts_to_track += fts.size();
      fts_vec_.push_back(fts); // shape=ncamera*feature
    }
#ifdef SEGMENT_ENABLE
    segs_vec_.clear();
    size_t n_segs_to_track = 0;
    int idx=0;
    for (auto frame : ref_frames->frames_)
    {
      
      std::vector<size_t> segs;
      sparse_img_align_utils::extractSegmentsSubset(*frame, options_.max_level, patch_size_with_border_, segs);
      n_segs_to_track += segs.size();
      segs_vec_.push_back(segs);
      // std::cout<<idx<<" seg subset info,now extra segment len"<<segs.size() <<std::endl;
      // for(auto a:segs){std::cout<<a<<" frame->seg_f_vec_"<<frame->seg_f_vec_.col(a*2)<<" "<<frame->seg_f_vec_.col(a*2+1)<<std::endl;}
      // std::cout<<std::endl;
      idx++;
    }

    
    SVO_DEBUG_STREAM("Img Align: Tracking "  << n_segs_to_track << " segments.");
    if (n_segs_to_track == 0)
    {
      SVO_ERROR_STREAM("SparseImgAlign: no segments to track! use feature for vio only!");
      // return 0;
    }
#endif
    SVO_DEBUG_STREAM("Img Align: Tracking " << n_fts_to_track << " features, ");
    
    if (n_fts_to_track == 0)
    {
      SVO_ERROR_STREAM("SparseImgAlign: no features to track!");
      return 0;
    }
    // set member variables
    ref_frames_ = ref_frames;
    cur_frames_ = cur_frames;
    T_iref_world_ = ref_frames->at(0)->T_imu_world();

    // prepare caches
#ifndef SEGMENT_ENABLE
    uv_cache_.resize(Eigen::NoChange, n_fts_to_track);
    xyz_ref_cache_.resize(Eigen::NoChange, n_fts_to_track);
    jacobian_proj_cache_.resize(Eigen::NoChange, n_fts_to_track * 2);
    jacobian_cache_.resize(Eigen::NoChange, n_fts_to_track * patch_area_);
    residual_cache_.resize(patch_area_, n_fts_to_track);
    visibility_mask_.resize(n_fts_to_track, Eigen::NoChange);
    ref_patch_cache_.resize(patch_area_, n_fts_to_track);
#endif

#ifdef SEGMENT_ENABLE
    uv_cache_.resize(Eigen::NoChange, n_fts_to_track + n_segs_to_track * 2);
    xyz_ref_cache_.resize(Eigen::NoChange, n_fts_to_track + n_segs_to_track * 2);
    jacobian_proj_cache_.resize(Eigen::NoChange, n_fts_to_track * 2 + n_segs_to_track * 4);
    jacobian_cache_.resize(Eigen::NoChange, n_fts_to_track * patch_area_ + n_segs_to_track * 2*patch_area_);
    residual_cache_.resize(patch_area_, n_fts_to_track + n_segs_to_track * 2);
    visibility_mask_.resize(n_fts_to_track + n_segs_to_track * 2, Eigen::NoChange);
    ref_patch_cache_.resize(patch_area_, n_fts_to_track + n_segs_to_track * 2);
#endif

    // the variable to be optimized is the imu-pose of the current frame
    Transformation T_icur_iref =
        cur_frames_->at(0)->T_imu_world() * T_iref_world_.inverse();

    SparseImgAlignState state;
    state.T_icur_iref = T_icur_iref;
    state.alpha = alpha_init_;
    state.beta = beta_init_;

    // precompute values common to all pyramid levels
    size_t feature_counter = 0;

    for (size_t i = 0; i < ref_frames_->size(); ++i)
    {
      sparse_img_align_utils::precomputeBaseCaches(
          *ref_frames_->at(i), fts_vec_.at(i),
          options_.use_distortion_jacobian,
          feature_counter, uv_cache_,
          xyz_ref_cache_, jacobian_proj_cache_);

    }

#ifdef SEGMENT_ENABLE
    size_t segment_counter = 0;
    for (size_t i = 0; i < ref_frames_->size(); ++i)
    {
      sparse_img_align_utils::precomputeBaseCachesAddSegment(
          *ref_frames_->at(i), segs_vec_.at(i),
          options_.use_distortion_jacobian,
          feature_counter, segment_counter, uv_cache_,
          xyz_ref_cache_, jacobian_proj_cache_);
    // std::cout<<"frame id------------------- no"<<i<<std::endl;          
    // NAME_VALUE_LOG(ref_frames_->at(i)->num_segments_);
    // for(auto single_idx:segs_vec_.at(i))
    // NAME_VALUE_LOG(single_idx);
    // NAME_VALUE_LOG(segs_vec_.at(i).size());
    }
    // NAME_VALUE_LOG(feature_counter*2+segment_counter*4);
    // NAME_VALUE_LOG(jacobian_proj_cache_);
    // NAME_VALUE_LOG(xyz_ref_cache_);
    // CHECK_EQ(static_cast<long unsigned int>(jacobian_proj_cache_.cols()),feature_counter*2+segment_counter*4);
    // CHECK_EQ(static_cast<long unsigned int>(xyz_ref_cache_.cols()),feature_counter+segment_counter*2);
    // std::cout<<"jacobian_proj_cache_ "<<jacobian_proj_cache_<<"\nlen:"<<jacobian_proj_cache_.cols()<<std::endl;
    // std::cout<<"xyz_ref_cache_ "<<xyz_ref_cache_<<"\nlen:"<<xyz_ref_cache_.cols()<<std::endl;
#endif


    for (level_ = options_.max_level; level_ >= options_.min_level; --level_)
    {
      mu_ = 0.1;
      have_cache_ = false; // at every level, recompute the jacobians
      if (solver_options_.verbose)
        printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
      optimize(state);
    }

    // finished, we save the pose in the frame
    for (auto f : cur_frames->frames_)
    {
      f->T_f_w_ = f->T_cam_imu() * state.T_icur_iref * T_iref_world_;
    }

    // reset initial values of illumination estimation TODO: make reset function
    alpha_init_ = 0.0;
    beta_init_ = 0.0;

    return n_fts_to_track;
  }

  double SparseImgAlign::evaluateError_(
      const SparseImgAlignState &state,
      HessianMatrix *H,
      GradientVector *g)
  {
    if (!have_cache_) // is reset at every new level.
    {
      size_t feature_counter = 0;

      for (size_t i = 0; i < ref_frames_->size(); ++i) // for bundle's frame such
      {
        sparse_img_align_utils::precomputeJacobiansAndRefPatches(
            ref_frames_->at(i), uv_cache_,
            jacobian_proj_cache_, level_, patch_size_,
            fts_vec_.at(i).size(),
            options_.estimate_illumination_gain,
            options_.estimate_illumination_offset,
            feature_counter,
            jacobian_cache_, ref_patch_cache_);
      }
      have_cache_ = true;
    }

    size_t feature_counter = 0; // used to compute the cache index
    for (size_t i = 0; i < ref_frames_->size(); ++i)
    {
      const Transformation T_cur_ref =
          cur_frames_->at(i)->T_cam_imu() * state.T_icur_iref * ref_frames_->at(i)->T_imu_cam();
      std::vector<Vector2d> *match_pxs = nullptr;
      sparse_img_align_utils::computeResidualsOfFrame(
          cur_frames_->at(i), level_,
          patch_size_, fts_vec_.at(i).size(), T_cur_ref,
          state.alpha, state.beta,
          ref_patch_cache_, xyz_ref_cache_,
          feature_counter, match_pxs, residual_cache_, visibility_mask_);
    } // compute the pixel luminance error

    float chi2 = sparse_img_align_utils::computeHessianAndGradient(
        jacobian_cache_, residual_cache_,
        visibility_mask_, weight_scale_, weight_function_, H, g);
    return chi2;
  }

  double SparseImgAlign::evaluateError(
      const SparseImgAlignState &state,
      HessianMatrix *H,
      GradientVector *g)
  {
    if (!have_cache_) // is reset at every new level.
    {
      size_t feature_counter = 0;
      for (size_t i = 0; i < ref_frames_->size(); ++i) // for bundle's frame such
      {
        sparse_img_align_utils::precomputeJacobiansAndRefPatches(
            ref_frames_->at(i), uv_cache_,
            jacobian_proj_cache_, level_, patch_size_,
            fts_vec_.at(i).size(),
            options_.estimate_illumination_gain,
            options_.estimate_illumination_offset,
            feature_counter,
            jacobian_cache_, ref_patch_cache_);

      }
#ifdef SEGMENT_ENABLE
      size_t segment_counter = 0;

      for (size_t i = 0; i < ref_frames_->size(); ++i) // for bundle's frame such
      {
        sparse_img_align_utils::precomputeJacobiansAndRefPatchesAddSegment(
            ref_frames_->at(i), uv_cache_,
            jacobian_proj_cache_, level_, patch_size_,
            segs_vec_.at(i).size(),
            feature_counter,
            segment_counter,
            options_.estimate_illumination_gain,
            options_.estimate_illumination_offset,
            jacobian_cache_, ref_patch_cache_);
      }
      // CHECK(!(jacobian_cache_.array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any());
#endif
      have_cache_ = true;
    }

    size_t feature_counter = 0; // used to compute the cache index
    size_t segment_counter = 0;

    for (size_t i = 0; i < ref_frames_->size(); ++i)
    {
      const Transformation T_cur_ref =
          cur_frames_->at(i)->T_cam_imu() * state.T_icur_iref * ref_frames_->at(i)->T_imu_cam();
      std::vector<Vector2d> *match_pxs = nullptr;
      sparse_img_align_utils::computeResidualsOfFrame(
          cur_frames_->at(i), level_,
          patch_size_, fts_vec_.at(i).size(), T_cur_ref,
          state.alpha, state.beta,
          ref_patch_cache_, xyz_ref_cache_,
          feature_counter, match_pxs, residual_cache_, visibility_mask_);
    }
#ifdef SEGMENT_ENABLE
    for (size_t i = 0; i < ref_frames_->size(); ++i)
    {
      const Transformation T_cur_ref =
          cur_frames_->at(i)->T_cam_imu() * state.T_icur_iref * ref_frames_->at(i)->T_imu_cam();
      std::vector<Vector2d> *match_pxs = nullptr;
    sparse_img_align_utils::computeResidualsOfFrameAddSegment(
       cur_frames_->at(i), level_,
          patch_size_, segs_vec_.at(i).size(), T_cur_ref,
          state.alpha, state.beta,
          ref_patch_cache_, xyz_ref_cache_,
          feature_counter,segment_counter,match_pxs, residual_cache_, visibility_mask_);// get the residuals error in each pixel
    }
#endif
     // compute the pixel luminance error

    CHECK_EQ(segment_counter*2+feature_counter,static_cast<long unsigned int>(residual_cache_.cols()));
    CHECK_EQ(segment_counter*2+feature_counter,static_cast<long unsigned int>(visibility_mask_.size()));
    CHECK_EQ(static_cast<long unsigned int>(jacobian_cache_.cols()), (feature_counter + segment_counter * 2) * patch_size_ * patch_size_);
    CHECK_EQ(static_cast<long unsigned int>(ref_patch_cache_.cols()), (feature_counter + segment_counter * 2) );
    // std::cout<<"evaluate track segment"<<segment_counter<<"; trace feature"<<feature_counter<<std::endl;
    float chi2 = sparse_img_align_utils::computeHessianAndGradient(
        jacobian_cache_, residual_cache_,
        visibility_mask_, weight_scale_, weight_function_, H, g);
    // std::cout<<"-----------chi2"<<chi2<<std::endl;
    return chi2;
  }

  void SparseImgAlign::finishIteration()
  {
    if (false)
    {
      const size_t cam_index = 0;
      cv::Mat residual_image(
          cur_frames_->at(cam_index)->img_pyr_.at(level_).size(),
          CV_32FC1, cv::Scalar(0.0));
      const size_t nr_features = fts_vec_.at(cam_index).size();
      const FloatType scale = 1.0f / (1 << level_);
      const int patch_size_wb = patch_size_ + 2 * border_size_; // patch size with border
      const FloatType patch_center_wb = (patch_size_wb - 1) / 2.0f;

      for (size_t i = 0; i < nr_features; ++i)
      {
        if (!visibility_mask_(i))
          continue;

        // compute top left coordinate of patch to be interpolated
        const FloatType u_tl = uv_cache_(0, i) * scale - patch_center_wb;
        const FloatType v_tl = uv_cache_(1, i) * scale - patch_center_wb;

        size_t pixel_counter = 0; // is used to compute the index of the cached residual
        for (int y = 0; y < patch_size_; ++y)
        {
          for (int x = 0; x < patch_size_; ++x, ++pixel_counter)
          {
            residual_image.at<float>(v_tl + y, u_tl + x) =
                (residual_cache_(pixel_counter, i) + 50) / 255.0;
          }
        }
      }

      double minval, maxval;
      cv::minMaxLoc(residual_image, &minval, &maxval);
      residual_image = (residual_image - minval) / (maxval - minval);

      std::vector<cv::Mat> residual_image_rgb_vec({cv::Mat(residual_image.size(), CV_32FC1, cv::Scalar(1.0)),
                                                   residual_image,
                                                   cv::Mat(residual_image.size(), CV_32FC1, cv::Scalar(0.0))});
      cv::Mat residual_image_rgb;
      cv::merge(residual_image_rgb_vec, residual_image_rgb);

      cv::imshow("residual_image", residual_image_rgb);
      cv::waitKey(0);
    }
  }

  namespace sparse_img_align_utils
  {

    void extractFeaturesSubset(const Frame &ref_frame,  // last frame for data reference
                               const int max_level,     // get the max pyramid
                               const int patch_size_wb, // patch_size + border (usually 2 for gradient),
                               std::vector<size_t> &fts)
    {
      // TODO(cfo): add seeds
      //  if(fts.size() < 100)
      //  {
      //    std::unique_lock<decltype(ref_frame->seeds_mut_)> lock(ref_frame->seeds_mut_);
      //    size_t n = 0;
      //    for(const SeedPtr& seed : ref_frame->seeds_)
      //    {
      //      if(seed->is_converged_)
      //      {
      //        fts.push_back(seed->ftr_);
      //        ++n;
      //      }
      //    }
      //    SVO_DEBUG_STREAM("SparseImgAlign: add " << n << " additional seeds.");
      //  }

      // ignore any feature point, which does not project fully in the image
      // checking on highest level is sufficient.
      const FloatType scale = 1.0f / (1 << max_level);
      const cv::Mat &ref_img = ref_frame.img_pyr_.at(max_level);
      const int rows_minus_two = ref_img.rows - 2;
      const int cols_minus_two = ref_img.cols - 2;
      const FloatType patch_center_wb = (patch_size_wb - 1) / 2.0f;

      // check if reference with patch size is within image
      fts.reserve(ref_frame.num_features_);
      for (size_t i = 0; i < ref_frame.num_features_; ++i)
      {
        if (ref_frame.landmark_vec_[i] == nullptr &&
            ref_frame.seed_ref_vec_[i].keyframe == nullptr)
          continue;
        if (isMapPoint(ref_frame.type_vec_[i]))
        {
          continue;
        }
        const FloatType u_tl = ref_frame.px_vec_(0, i) * scale - patch_center_wb;
        const FloatType v_tl = ref_frame.px_vec_(1, i) * scale - patch_center_wb;
        const int u_tl_i = std::floor(u_tl);
        const int v_tl_i = std::floor(v_tl);
        if (!(u_tl_i < 0 || v_tl_i < 0 || u_tl_i + patch_size_wb >= cols_minus_two || v_tl_i + patch_size_wb >= rows_minus_two))
          fts.emplace_back(i);
      }
      SVO_DEBUG_STREAM("Img Align: Maximum Number of Features = " << ref_frame.num_features_);
    }

    void extractSegmentsSubset(const Frame &ref_frame,  // last frame for data reference
                               const int max_level,     // get the max pyramid
                               const int patch_size_wb, // patch_size + border (usually 2 for gradient),
                               std::vector<size_t> &fts)
    {
      
      const FloatType scale = 1.0f / (1 << max_level);
      const cv::Mat &ref_img = ref_frame.img_pyr_.at(max_level);
      const int rows_minus_two = ref_img.rows - 2;
      const int cols_minus_two = ref_img.cols - 2;
      const FloatType patch_center_wb = (patch_size_wb - 1) / 2.0f;

      // check if reference with patch size is within image
      fts.reserve(ref_frame.num_segments_);
      for (size_t i = 0; i < ref_frame.num_segments_; ++i)
      {
        if (ref_frame.seg_landmark_vec_[i] == nullptr &&
            ref_frame.seg_seed_ref_vec_[i].keyframe == nullptr)
          continue;
        const FloatType u_tl_s = ref_frame.seg_vec_(0, i) * scale - patch_center_wb;
        const FloatType v_tl_s = ref_frame.seg_vec_(1, i) * scale - patch_center_wb;
        const FloatType u_tl_e = ref_frame.seg_vec_(2, i) * scale - patch_center_wb;
        const FloatType v_tl_e = ref_frame.seg_vec_(3, i) * scale - patch_center_wb;
        const int u_tl_i_s = std::floor(u_tl_s);
        const int v_tl_i_s = std::floor(v_tl_s);
        const int u_tl_i_e = std::floor(u_tl_e);
        const int v_tl_i_e = std::floor(v_tl_e);
        if (!(u_tl_i_s < 0 || v_tl_i_s < 0 || u_tl_i_s + patch_size_wb >= cols_minus_two || v_tl_i_s + patch_size_wb >= rows_minus_two) &&
            !(u_tl_i_e < 0 || v_tl_i_e < 0 || u_tl_i_e + patch_size_wb >= cols_minus_two || v_tl_i_e + patch_size_wb >= rows_minus_two))
        {
          fts.emplace_back(i);
          
          const SeedRef &seed_ref = ref_frame.seg_seed_ref_vec_[i];                                                                                    // must kown where to add the

          if(!isSegmentSeed(ref_frame.seg_type_vec_[i])||seed_ref.keyframe==nullptr)continue;
          // std::cout<<"in sparese_img_align"<<std::endl;
          CHECK(seed_ref.keyframe!=nullptr)<<seed_ref.keyframe;
          CHECK(seed_ref.seed_id>=0&& static_cast<size_t>(seed_ref.seed_id)<seed_ref.keyframe->num_segments_);
          
          const Eigen::Matrix<FloatType, 3, 1> pos_s = seed_ref.keyframe->T_world_cam() * 
          seed_ref.keyframe->getSegmentSeedPosInFrame(seed_ref.seed_id).col(0); // seed ref normally in match step,update step
          const Eigen::Matrix<FloatType, 3, 1> pos_e = seed_ref.keyframe->T_world_cam() * 
          seed_ref.keyframe->getSegmentSeedPosInFrame(seed_ref.seed_id).col(1); // seed ref normally in match step,update step
          // CHECK_EIGEN_HAVE_NAN(seed_ref.keyframe->seg_f_vec_.col(seed_ref.seed_id*2),"in precomputeBaseCachesAddSegment");
          // CHECK_EIGEN_HAVE_NAN(seed_ref.keyframe->seg_f_vec_.col(seed_ref.seed_id*2+1),"in precomputeBaseCachesAddSegment");
          //   if((pos_e.array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
          //   {
          //     std::cout<<"extractSegmentsSubset"<<std::endl;
          //                 for(auto s:ref_frame.seg_type_vec_)
          // std::cout<<static_cast<int>(s)<<" ";
          //   fts.pop_back();
          //   }
          //  CHECK(!(pos_e.array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())<<"cur segment idx"<<i<<" "
          //  <<"current keyframe" << ref_frame.bundle_id_<<"  seed_ref id keyframe:\n"<<seed_ref.seed_id<<" "<<seed_ref.keyframe->bundle_id_
          //   <<"\npos e\n"<<pos_e.transpose()<<"getSegmentSeedDepth(idx)\n"<<seed_ref.keyframe->getSegmentSeedDepth(seed_ref.seed_id)<<
          //   " \nseg_f_vec:\n"<<seed_ref.keyframe->seg_f_vec_.col(seed_ref.seed_id*2)<<ref_frame.seg_landmark_vec_[i]
          //  <<"ref  type:\n"<<static_cast<int>(seed_ref.keyframe->seg_type_vec_[seed_ref.seed_id])<< " cur type "<<static_cast<int>(ref_frame.seg_type_vec_[i])
          //   <<" seed_ref.seed_id\n"<<seed_ref.seed_id<<"cur seg_invmu_sigma2_a_b_vec_ :\n"<<
          //   ref_frame.seg_invmu_sigma2_a_b_vec_.transpose()<<"ref seg_invmu_sigma2_a_b_vec_ :\n"<<seed_ref.keyframe->seg_invmu_sigma2_a_b_vec_.transpose()
          //   ;


        }

      }
      SVO_DEBUG_STREAM("Img Align: Maximum Number of segment = " << ref_frame.num_segments_);
    }

    void precomputeBaseCaches(const Frame &ref_frame,
                              const std::vector<size_t> &fts,
                              const bool use_distortion_jac,
                              size_t &feature_counter,
                              UvCache &uv_cache,
                              XyzRefCache &xyz_ref_cache,
                              JacobianProjCache &jacobian_proj_cache)
    {
      const double focal_length = ref_frame.getErrorMultiplier();
      const Vector3d ref_pos = ref_frame.pos();
      const Transformation &T_imu_cam = ref_frame.T_imu_cam();
      const Transformation &T_cam_imu = ref_frame.T_cam_imu();

      for (const size_t i : fts) // directly adding the line endpoint jacobian to jacobianprojcache
      {
        uv_cache.col(feature_counter) = ref_frame.px_vec_.col(i).cast<FloatType>(); // cause px_vec could add the not downsample pixel

        // evaluate jacobian. cannot just take the 3d points coordinate because of
        // the reprojection errors in the reference image!!!

        FloatType depth = 0;
        if (ref_frame.landmark_vec_[i])
        {
          depth = ((ref_frame.landmark_vec_[i]->pos_ - ref_pos).norm());
        }
        else
        {
          const SeedRef &seed_ref = ref_frame.seed_ref_vec_[i];
          const Position pos = seed_ref.keyframe->T_world_cam() * seed_ref.keyframe->getSeedPosInFrame(seed_ref.seed_id);
          depth = (pos - ref_pos).norm();
        }

        const Vector3d xyz_ref(ref_frame.f_vec_.col(i) * depth); // get the 3d point in
        xyz_ref_cache.col(feature_counter) = xyz_ref.cast<FloatType>();
        const Vector3d xyz_in_imu(T_imu_cam * xyz_ref);

        Matrix<double, 2, 6> frame_jac;
        if (!use_distortion_jac &&
            ref_frame.cam()->getType() == Camera::Type::kPinhole)
        {                                                               // only allow ignoring jacobian for pinhole projection
          Frame::jacobian_xyz2uv_imu(T_cam_imu, xyz_in_imu, frame_jac); // derivative of pixel offset to rotation vector
          frame_jac *= focal_length;
        }
        else
        {
          Frame::jacobian_xyz2image_imu(*ref_frame.cam(), T_cam_imu, xyz_in_imu, frame_jac); // derivative of pixel offset to rotation vector
          frame_jac *= (-1.0);
        }

        size_t col_index = 2 * feature_counter;
        jacobian_proj_cache.col(col_index) = frame_jac.row(0).cast<FloatType>();
        jacobian_proj_cache.col(col_index + 1) = frame_jac.row(1).cast<FloatType>();
        ++feature_counter;
      }
    }

    void precomputeBaseCachesAddSegment(const Frame &ref_frame,
                                        const std::vector<size_t> &segs, // notice it mentioned the landmark and seedreference
                                        const bool use_distortion_jac,
                                        const size_t &feature_counter,
                                        size_t &segment_counter,
                                        UvCache &uv_cache,
                                        XyzRefCache &xyz_ref_cache,
                                        JacobianProjCache &jacobian_proj_cache)
    {
      const double focal_length = ref_frame.getErrorMultiplier();
      const Vector3d ref_pos = ref_frame.pos();
      const Transformation &T_imu_cam = ref_frame.T_imu_cam();
      const Transformation &T_cam_imu = ref_frame.T_cam_imu();

      for (const size_t i : segs) // directly adding the line endpoint jacobian to jacobianprojcache
      {
        uv_cache.col(feature_counter + segment_counter * 2) = ref_frame.seg_vec_.col(i).head<2>().cast<FloatType>();
        uv_cache.col(feature_counter + segment_counter * 2 + 1) = ref_frame.seg_vec_.col(i).tail<2>().cast<FloatType>();

        // evaluate jacobian. cannot just take the 3d points coordinate because of
        // the reprojection errors in the reference image!!!

        FloatType depth_s = 0, depth_e = 0;
        if (ref_frame.seg_landmark_vec_[i]!= nullptr)
        {
          depth_s = ((ref_frame.seg_landmark_vec_[i]->spos_ - ref_pos).norm());
          depth_e = ((ref_frame.seg_landmark_vec_[i]->epos_ - ref_pos).norm());
        // CHECK(depth_s!=0)<<ref_frame.seg_landmark_vec_[i]->spos_<<" "<<depth_s<<" "<<depth_s<<" "<<ref_pos.transpose();

        }// else if(isSegmentSeed(ref_frame.seg_type_vec_[i])&& ref_frame.seg_seed_ref_vec_[i].keyframe!=nullptr) becasue the add set
        else
        {
          const SeedRef &seed_ref = ref_frame.seg_seed_ref_vec_[i];                                                                                    // must kown where to add the
          const Eigen::Matrix<FloatType, 3, 1> pos_s = seed_ref.keyframe->T_world_cam() * 
          seed_ref.keyframe->getSegmentSeedPosInFrame(seed_ref.seed_id).col(0); // seed ref normally in match step,update step
          const Eigen::Matrix<FloatType, 3, 1> pos_e = seed_ref.keyframe->T_world_cam() * 
          seed_ref.keyframe->getSegmentSeedPosInFrame(seed_ref.seed_id).col(1); // seed ref normally in match step,update step
          depth_s = (pos_s - ref_pos).norm();
          depth_e = (pos_e - ref_pos).norm();
          // CHECK_EIGEN_HAVE_NAN(seed_ref.keyframe->seg_f_vec_.col(seed_ref.seed_id*2),"in precomputeBaseCachesAddSegment");
          // CHECK_EIGEN_HAVE_NAN(seed_ref.keyframe->seg_f_vec_.col(seed_ref.seed_id*2+1),"in precomputeBaseCachesAddSegment");
          //  CHECK(!(pos_e.array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
          //   <<"\n"<<pos_e.transpose()<<"getSegmentSeedDepth(idx)"<<seed_ref.keyframe->getSegmentSeedDepth(seed_ref.seed_id)<<
          //   " seg_f_vec"<<seed_ref.keyframe->seg_f_vec_.col(seed_ref.seed_id*2)<<"\n"<<seed_ref.keyframe->seg_f_vec_.col(seed_ref.seed_id*2+1)<<ref_frame.seg_landmark_vec_[i]
          //  <<"ref  type:\n"<<static_cast<int>(seed_ref.keyframe->seg_type_vec_[seed_ref.seed_id])<< " cur "
          //   <<" seed_ref.seed_id\n"<<seed_ref.seed_id<<"ref_farme seed_id :\n"<<ref_frame.seg_invmu_sigma2_a_b_vec_.col(i*2);

        // CHECK(depth_s!=0)<<ref_frame.seg_f_vec_.col(i * 2).transpose()<<" "<<depth_s<<" "<<depth_s<<" "<<ref_pos.transpose();

        //   NAME_VALUE_LOG(seed_ref.keyframe->T_world_cam().asVector());
        // NAME_VALUE_LOG( seed_ref.keyframe->getSegmentSeedPosInFrame(seed_ref.seed_id).col(1));
        // NAME_VALUE_LOG( seed_ref.keyframe->getSegmentSeedPosInFrame(seed_ref.seed_id).col(0));
        // NAME_VALUE_LOG(pos_s);
        // NAME_VALUE_LOG(pos_e);
        // NAME_VALUE_LOG(ref_frame.seg_f_vec_.col(i * 2));
        // NAME_VALUE_LOG(ref_frame.seg_f_vec_.col(i * 2 + 1));
        // NAME_VALUE_LOG(depth_s);
        // NAME_VALUE_LOG(depth_e);
        }

        const Vector3d xyz_ref_s(ref_frame.seg_f_vec_.col(i * 2) * depth_s);     // get the 3d point in
        const Vector3d xyz_ref_e(ref_frame.seg_f_vec_.col(i * 2 + 1) * depth_e); // get the 3d point in
        xyz_ref_cache.col(feature_counter + segment_counter * 2) = xyz_ref_s.cast<FloatType>();
        xyz_ref_cache.col(feature_counter + segment_counter * 2 + 1) = xyz_ref_e.cast<FloatType>();
        #define NAME_VALUE_LOG(x) std::cout << #x << ": \n" << (x) << std::endl;

        const Vector3d xyz_in_imu_s(T_imu_cam * xyz_ref_s);
        const Vector3d xyz_in_imu_e(T_imu_cam * xyz_ref_e);
        // CHECK(!(xyz_ref_s.unaryExpr([](double v) { return std::isinf(v); })).any())
        //     <<"\n"<<xyz_ref_s.transpose()<<"\n seg_f_vec"<<ref_frame.seg_f_vec_.col(i * 2)<<"\n"<<depth_s<<"\n"<<i*2;
        // CHECK(!(xyz_ref_e.unaryExpr([](double v) { return std::isinf(v); })).any())
        //     <<"\n"<<xyz_ref_e.transpose()<<"\n seg_f_vec"<<ref_frame.seg_f_vec_.col(i * 2+1)<<"\n"<<depth_e<<"\n"<<i*2;
        Matrix<double, 2, 6> frame_jac_s;
        Matrix<double, 2, 6> frame_jac_e;
        if (!use_distortion_jac &&
            ref_frame.cam()->getType() == Camera::Type::kPinhole)
        {                                                                   // only allow ignoring jacobian for pinhole projection
          Frame::jacobian_xyz2uv_imu(T_cam_imu, xyz_in_imu_s, frame_jac_s); // derivative of pixel offset to rotation vector
          frame_jac_s *= focal_length;
          Frame::jacobian_xyz2uv_imu(T_cam_imu, xyz_in_imu_e, frame_jac_e); // derivative of pixel offset to rotation vector
          frame_jac_e *= focal_length;
        }
        else
        {
          Frame::jacobian_xyz2image_imu(*ref_frame.cam(), T_cam_imu, xyz_in_imu_s, frame_jac_s); // derivative of pixel offset to rotation vector
          frame_jac_s *= (-1.0);
          Frame::jacobian_xyz2image_imu(*ref_frame.cam(), T_cam_imu, xyz_in_imu_e, frame_jac_e); // derivative of pixel offset to rotation vector
          frame_jac_e *= (-1.0);
        }

        size_t col_index = 2 * feature_counter + 4 * segment_counter;
        jacobian_proj_cache.col(col_index) = frame_jac_s.row(0).cast<FloatType>();
        jacobian_proj_cache.col(col_index + 1) = frame_jac_s.row(1).cast<FloatType>();
        jacobian_proj_cache.col(col_index + 2) = frame_jac_e.row(0).cast<FloatType>();
        jacobian_proj_cache.col(col_index + 3) = frame_jac_e.row(1).cast<FloatType>();

          
            // CHECK(!(jacobian_proj_cache.col(col_index).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
            // <<"\n"<<jacobian_proj_cache.col(col_index).transpose()<<"\n pjc"<<col_index<<"\n"<<feature_counter<<" "<<
            // segment_counter<<"\n"<<xyz_ref_s<<"\n"<<frame_jac_s<<"\n seg_f_vec"<<ref_frame.seg_f_vec_.col(i * 2)<<"\n"<<depth_s<<"\n"<<i*2;
            // CHECK(!(jacobian_proj_cache.col(col_index+1).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
            // <<"\n"<<jacobian_proj_cache.col(col_index+1).transpose()<<"\n pjc"<<col_index+1<<"\n"<<feature_counter<<
            // " "<<segment_counter<<"\n"<<xyz_ref_s<<"\n"<<frame_jac_s<<"\n seg_f_vec"<<ref_frame.seg_f_vec_.col(i * 2)<<"\n"<<depth_s<<"\n"<<i*2;         
            //  CHECK(!(jacobian_proj_cache.col(col_index+2).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
            // <<"\n"<<jacobian_proj_cache.col(col_index+2).transpose()<<"\n pjc"<<col_index+2<<"\n"<<feature_counter<<
            // " "<<segment_counter<<"\n"<<xyz_ref_e<<"\n"<<frame_jac_e<<"\n seg_f_vec"<<ref_frame.seg_f_vec_.col(i * 2+1)<<"\n"<<depth_e<<"\n"<<i*2+1;         
            //  CHECK(!(jacobian_proj_cache.col(col_index+3).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
            // <<"\n"<<jacobian_proj_cache.col(col_index+3).transpose()<<"\n pjc"<<col_index+3<<"\n"<<feature_counter<<
            // " "<<segment_counter<<"\n"<<xyz_ref_e<<"\n"<<frame_jac_e<<"\n seg_f_vec"<<ref_frame.seg_f_vec_.col(i * 2+1)<<"\n"<<depth_e<<"\n"<<i*2+1; 

        ++segment_counter;
      }
    }

    void precomputeJacobiansAndRefPatches // compute the different level Pyramid image pixel jacobian from luminance error to rotate vector
        (
            const FramePtr &ref_frame,
            const UvCache &uv_cache, // uv chche no the pixel index but the normal index in camera plane
            const JacobianProjCache &jacobian_proj_cache,
            const size_t level,
            const int patch_size,
            const size_t nr_features,
            bool estimate_alpha,
            bool estimate_beta,
            size_t &feature_counter, // feature counter is compute the cache
            JacobianCache &jacobian_cache,
            RefPatchCache &ref_patch_cache)
    {
      const cv::Mat &ref_img = ref_frame->img_pyr_.at(level);
      const int stride = ref_img.step; // must be real stride
      const FloatType scale = 1.0f / (1 << level);
      const int patch_area = patch_size * patch_size;
      const int border = 1;
      const int patch_size_wb = patch_size + 2 * border; // patch size with border
      const int patch_area_wb = patch_size_wb * patch_size_wb;
      const FloatType patch_center_wb = (patch_size_wb - 1) / 2.0f;

      // interpolate patch + border (filled in row major format)
      FloatType interp_patch_array[patch_area_wb];

      for (size_t i = 0; i < nr_features; ++i, ++feature_counter)
      {
        // compute top left coordinate of patch to be interpolated
        const FloatType u_tl = uv_cache(0, feature_counter) * scale - patch_center_wb;
        const FloatType v_tl = uv_cache(1, feature_counter) * scale - patch_center_wb;

        const int u_tl_i = std::floor(u_tl);
        const int v_tl_i = std::floor(v_tl);

        // compute bilateral interpolation weights for reference image
        const FloatType subpix_u_tl = u_tl - u_tl_i;
        const FloatType subpix_v_tl = v_tl - v_tl_i;
        const FloatType wtl = (1.0 - subpix_u_tl) * (1.0 - subpix_v_tl);
        const FloatType wtr = subpix_u_tl * (1.0 - subpix_v_tl);
        const FloatType wbl = (1.0 - subpix_u_tl) * subpix_v_tl;
        const FloatType wbr = subpix_u_tl * subpix_v_tl;
        const int jacobian_proj_col = 2 * feature_counter;

        // interpolate patch with border
        size_t pixel_counter = 0;
        for (int y = 0; y < patch_size_wb; ++y)
        {
          // reference image pointer (openCv stores data in row major format)
          uint8_t *r =
              static_cast<uint8_t *>(ref_img.data) + (v_tl_i + y) * stride + u_tl_i; // reference patch
          for (int x = 0; x < patch_size_wb; ++x, ++r, ++pixel_counter)
          {
            // precompute interpolated reference patch color
            interp_patch_array[pixel_counter] = wtl * r[0] + wtr * r[1] + wbl * r[stride] + wbr * r[stride + 1]; // bilinear interpolate
          }
        }

        // fill ref_patch_cache and jacobian_cache
        pixel_counter = 0;
        for (int y = 0; y < patch_size; ++y)
        {
          for (int x = 0; x < patch_size; ++x, ++pixel_counter)
          {
            int offset_center = (x + border) + patch_size_wb * (y + border);
            ref_patch_cache(pixel_counter, feature_counter) = interp_patch_array[offset_center]; // compute the reference pixel cache

            // we use the inverse compositional: thereby we can take the gradient
            // always at the same position.
            const FloatType dx = 0.5f * (interp_patch_array[offset_center + 1] - interp_patch_array[offset_center - 1]);
            const FloatType dy = 0.5f * (interp_patch_array[offset_center + patch_size_wb] - interp_patch_array[offset_center - patch_size_wb]); // compute the pixel luminance gradients

            // cache the jacobian
            int jacobian_col = feature_counter * patch_area + pixel_counter;
            jacobian_cache.block<6, 1>(0, jacobian_col) =
                (dx * jacobian_proj_cache.col(jacobian_proj_col) + dy * jacobian_proj_cache.col(jacobian_proj_col + 1)) * scale;
            jacobian_cache(6, jacobian_col) = estimate_alpha ? -(interp_patch_array[offset_center]) : 0.0;
            jacobian_cache(7, jacobian_col) = estimate_beta ? -1.0 : 0.0;
          }
        }
      }
    }

    void precomputeJacobiansAndRefPatchesAddSegment(
        const FramePtr &ref_frame,
        const UvCache &uv_cache, // uv chche no the pixel index but the normal index in camera plane
        const JacobianProjCache &jacobian_proj_cache,
        const size_t level,
        const int patch_size,
        const size_t nr_segment,
        const size_t &feature_counter, // feature counter is compute the cache
        size_t &segment_counter,
        bool estimate_alpha,
        bool estimate_beta,
        JacobianCache &jacobian_cache,
        RefPatchCache &ref_patch_cache)
    {
      const cv::Mat &ref_img = ref_frame->img_pyr_.at(level);
      const int stride = ref_img.step; // must be real stride
      const FloatType scale = 1.0f / (1 << level);
      const int patch_area = patch_size * patch_size;
      const int border = 1;
      const int patch_size_wb = patch_size + 2 * border; // patch size with border
      const int patch_area_wb = patch_size_wb * patch_size_wb;
      const FloatType patch_center_wb = (patch_size_wb - 1) / 2.0f;

      // interpolate patch + border (filled in row major format)
      FloatType interp_patch_array[patch_area_wb];
      // std::cout<<"nr_segment"<<nr_segment<<std::endl;
      for (size_t iiii = 0; iiii < nr_segment; ++iiii, ++segment_counter)
      {
        for (size_t line_endpoint_idx = 0; line_endpoint_idx < 2; ++line_endpoint_idx)
        {
          
          size_t now_index = feature_counter + segment_counter * 2 + line_endpoint_idx;
          // compute top left coordinate of patch to be interpolated
          const FloatType u_tl = uv_cache(0, now_index) * scale - patch_center_wb;
          const FloatType v_tl = uv_cache(1, now_index) * scale - patch_center_wb;

          const int u_tl_i = std::floor(u_tl);
          const int v_tl_i = std::floor(v_tl);

          // compute bilateral interpolation weights for reference image
          const FloatType subpix_u_tl = u_tl - u_tl_i;
          const FloatType subpix_v_tl = v_tl - v_tl_i;
          const FloatType wtl = (1.0 - subpix_u_tl) * (1.0 - subpix_v_tl);
          const FloatType wtr = subpix_u_tl * (1.0 - subpix_v_tl);
          const FloatType wbl = (1.0 - subpix_u_tl) * subpix_v_tl;
          const FloatType wbr = subpix_u_tl * subpix_v_tl;
          const int jacobian_proj_col = 2 * now_index;

          // interpolate patch with border
          size_t pixel_counter = 0;
          for (int y = 0; y < patch_size_wb; ++y)
          {
            // reference image pointer (openCv stores data in row major format)
            uint8_t *r =
                static_cast<uint8_t *>(ref_img.data) + (v_tl_i + y) * stride + u_tl_i; // reference patch
            for (int x = 0; x < patch_size_wb; ++x, ++r, ++pixel_counter)
            {
              // precompute interpolated reference patch color
              interp_patch_array[pixel_counter] = wtl * r[0] + wtr * r[1] + wbl * r[stride] + wbr * r[stride + 1]; // bilinear interpolate
            }
          }

          // fill ref_patch_cache and jacobian_cache
          pixel_counter = 0;
          for (int y = 0; y < patch_size; ++y)
          {
            for (int x = 0; x < patch_size; ++x, ++pixel_counter)
            {
              int offset_center = (x + border) + patch_size_wb * (y + border);
              ref_patch_cache(pixel_counter, now_index) = interp_patch_array[offset_center]; // compute the reference pixel cache

              // we use the inverse compositional: thereby we can take the gradient
              // always at the same position.
              const FloatType dx = 0.5f * (interp_patch_array[offset_center + 1] - interp_patch_array[offset_center - 1]);
              const FloatType dy = 0.5f * (interp_patch_array[offset_center + patch_size_wb] - interp_patch_array[offset_center - patch_size_wb]); // compute the pixel luminance gradients

              // cache the jacobian
              int jacobian_col = now_index * patch_area + pixel_counter;
            //   CHECK(!(jacobian_proj_cache.col(jacobian_proj_col).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
            // <<"\n"<<jacobian_proj_cache.col(jacobian_proj_col).transpose()<<"\n pjc"<<jacobian_proj_col<<"\n jc"<<jacobian_col<<" ";
            // CHECK(!(jacobian_proj_cache.col(jacobian_proj_col+1).array().unaryExpr([](double v) { return std::isinf(v)||std::isnan(v); })).any())
            // <<"\n"<<jacobian_proj_cache.col(jacobian_proj_col).transpose()<<"\n pjc"<<jacobian_proj_col<<"\n jc"<<jacobian_col<<" ";

              jacobian_cache.block<6, 1>(0, jacobian_col) =
                  (dx * jacobian_proj_cache.col(jacobian_proj_col) + dy * jacobian_proj_cache.col(jacobian_proj_col + 1)) * scale;
              jacobian_cache(6, jacobian_col) = estimate_alpha ? -(interp_patch_array[offset_center]) : 0.0;
              jacobian_cache(7, jacobian_col) = estimate_beta ? -1.0 : 0.0;
              // std::cout<<"now comp cols "<<jacobian_col<<std::endl;
            }
          }
        }
      }
    }

    // void precomputeJacobiansAndRefPatches_(
    //     const FramePtr &ref_frame,
    //     const UvCache &uv_cache,
    //     const JacobianProjCache &jacobian_proj_cache,
    //     const size_t level,
    //     const int patch_size,
    //     const size_t nr_features,
    //     bool estimate_alpha,
    //     bool estimate_beta,
    //     size_t &feature_counter,
    //     JacobianCache &jacobian_cache,
    //     RefPatchCache &ref_patch_cache)
    // {
    //   const cv::Mat &ref_img = ref_frame->img_pyr_.at(level);
    //   const int stride = ref_img.step; // must be real stride
    //   const FloatType scale = 1.0f / (1 << level);
    //   const int patch_area = patch_size * patch_size;
    //   const int border = 1;
    //   const int patch_size_wb = patch_size + 2 * border; // patch size with border
    //   const int patch_area_wb = patch_size_wb * patch_size_wb;
    //   const FloatType patch_center_wb = (patch_size_wb - 1) / 2.0f;

    //   // interpolate patch + border (filled in row major format)
    //   FloatType interp_patch_array[patch_area_wb];

    //   for (size_t i = 0; i < nr_features; ++i, ++feature_counter)
    //   {
    //     // compute top left coordinate of patch to be interpolated
    //     const FloatType u_tl = uv_cache(0, feature_counter) * scale - patch_center_wb;
    //     const FloatType v_tl = uv_cache(1, feature_counter) * scale - patch_center_wb;

    //     const int u_tl_i = std::floor(u_tl);
    //     const int v_tl_i = std::floor(v_tl);

    //     // compute bilateral interpolation weights for reference image
    //     const FloatType subpix_u_tl = u_tl - u_tl_i;
    //     const FloatType subpix_v_tl = v_tl - v_tl_i;
    //     const FloatType wtl = (1.0 - subpix_u_tl) * (1.0 - subpix_v_tl);
    //     const FloatType wtr = subpix_u_tl * (1.0 - subpix_v_tl);
    //     const FloatType wbl = (1.0 - subpix_u_tl) * subpix_v_tl;
    //     const FloatType wbr = subpix_u_tl * subpix_v_tl;
    //     const int jacobian_proj_col = 2 * feature_counter;

    //     // interpolate patch with border
    //     size_t pixel_counter = 0;
    //     for (int y = 0; y < patch_size_wb; ++y)
    //     {
    //       // reference image pointer (openCv stores data in row major format)
    //       uint8_t *r =
    //           static_cast<uint8_t *>(ref_img.data) + (v_tl_i + y) * stride + u_tl_i;
    //       for (int x = 0; x < patch_size_wb; ++x, ++r, ++pixel_counter)
    //       {
    //         // precompute interpolated reference patch color
    //         interp_patch_array[pixel_counter] = wtl * r[0] + wtr * r[1] + wbl * r[stride] + wbr * r[stride + 1];
    //       }
    //     }

    //     // fill ref_patch_cache and jacobian_cache
    //     pixel_counter = 0;
    //     for (int y = 0; y < patch_size; ++y)
    //     {
    //       for (int x = 0; x < patch_size; ++x, ++pixel_counter)
    //       {
    //         int offset_center = (x + border) + patch_size_wb * (y + border);
    //         ref_patch_cache(pixel_counter, feature_counter) = interp_patch_array[offset_center];

    //         // we use the inverse compositional: thereby we can take the gradient
    //         // always at the same position.
    //         const FloatType dx = 0.5f * (interp_patch_array[offset_center + 1] - interp_patch_array[offset_center - 1]);
    //         const FloatType dy = 0.5f * (interp_patch_array[offset_center + patch_size_wb] - interp_patch_array[offset_center - patch_size_wb]);

    //         // cache the jacobian
    //         int jacobian_col = feature_counter * patch_area + pixel_counter;
    //         jacobian_cache.block<6, 1>(0, jacobian_col) =
    //             (dx * jacobian_proj_cache.col(jacobian_proj_col) + dy * jacobian_proj_cache.col(jacobian_proj_col + 1)) * scale;
    //         jacobian_cache(6, jacobian_col) = estimate_alpha ? -(interp_patch_array[offset_center]) : 0.0;
    //         jacobian_cache(7, jacobian_col) = estimate_beta ? -1.0 : 0.0;
    //       }
    //     }
    //   }
    // }

    void computeResidualsOfFrame(
        const FramePtr &cur_frame,
        const size_t level,
        const int patch_size,
        const size_t nr_features,
        const Transformation &T_cur_ref,
        const float alpha,
        const float beta,
        const RefPatchCache &ref_patch_cache,
        const XyzRefCache &xyz_ref_cache,
        size_t &feature_counter,
        std::vector<Vector2d> *match_pxs,
        ResidualCache &residual_cache,
        VisibilityMask &visibility_mask)
    {
      const cv::Mat &cur_img = cur_frame->img_pyr_.at(level);
      const int stride = cur_img.step;
      const FloatType scale = 1.0f / (1 << level);
      const int patch_area = patch_size * patch_size;
      const FloatType patch_center = (patch_size - 1) / 2.0f;

      FloatType total_intensity = 0.0;

      for (size_t i = 0; i < nr_features; ++i, ++feature_counter)
      {
        Vector3ft xyz_ref = xyz_ref_cache.col(feature_counter);
        const Vector3d xyz_cur(T_cur_ref * xyz_ref.cast<double>());
        if (cur_frame->cam()->getType() ==
                vk::cameras::CameraGeometryBase::Type::kPinhole &&
            xyz_cur.z() < 0.0)
        {
          visibility_mask(feature_counter) = false;
          continue;
        }

        Eigen::Vector2d uv_cur;
        cur_frame->cam()->project3(xyz_cur, &uv_cur);
        const Vector2ft uv_cur_pyr = uv_cur.cast<FloatType>() * scale;

        // compute top left coordinate of patch to be interpolated
        const FloatType u_tl = uv_cur_pyr[0] - patch_center;
        const FloatType v_tl = uv_cur_pyr[1] - patch_center;

        // check if projection is within the image
        if (u_tl < 0.0 || v_tl < 0.0 || u_tl + patch_size + 2.0 >= cur_img.cols || v_tl + patch_size + 2.0 >= cur_img.rows)
        {
          visibility_mask(feature_counter) = false;
          continue;
        }
        else
        {
          visibility_mask(feature_counter) = true;
        }

        const int u_tl_i = std::floor(u_tl);
        const int v_tl_i = std::floor(v_tl);

        // compute bilateral interpolation weights for the current image
        const FloatType subpix_u_tl = u_tl - u_tl_i;
        const FloatType subpix_v_tl = v_tl - v_tl_i;
        const FloatType wtl = (1.0 - subpix_u_tl) * (1.0 - subpix_v_tl);
        const FloatType wtr = subpix_u_tl * (1.0 - subpix_v_tl);
        const FloatType wbl = (1.0 - subpix_u_tl) * subpix_v_tl;
        const FloatType wbr = subpix_u_tl * subpix_v_tl;

        size_t pixel_counter = 0; // is used to compute the index of the cached residual
        float total_res = 0.0;
        for (int y = 0; y < patch_size; ++y)
        {
          uint8_t *cur_img_ptr =
              static_cast<uint8_t *>(cur_img.data) + (v_tl_i + y) * stride + u_tl_i;

          for (int x = 0; x < patch_size; ++x, ++pixel_counter, ++cur_img_ptr)
          {
            // compute residual, with bilinear interpolation
            FloatType intensity_cur =
                wtl * cur_img_ptr[0] + wtr * cur_img_ptr[1] + wbl * cur_img_ptr[stride] + wbr * cur_img_ptr[stride + 1];
            FloatType res = static_cast<FloatType>(intensity_cur * (1.0 + alpha) + beta) - ref_patch_cache(pixel_counter, feature_counter);
            residual_cache(pixel_counter, feature_counter) = res;

            // for camera control:
            total_res += intensity_cur - static_cast<float>(ref_patch_cache(pixel_counter, feature_counter));
            total_intensity += intensity_cur;
          }
        }
      }
    }

    void computeResidualsOfFrameAddSegment(
        const FramePtr &cur_frame,
        const size_t level,
        const int patch_size,
        const size_t nr_segment,
        const Transformation &T_cur_ref,
        const float alpha,
        const float beta,
        const RefPatchCache &ref_patch_cache,
        const XyzRefCache &xyz_ref_cache,
        const size_t &feature_counter,
        size_t &segment_counter,
        std::vector<Vector2d> *match_pxs,
        ResidualCache &residual_cache,
        VisibilityMask &visibility_mask)
    {
      // NAME_VALUE_LOG(ref_patch_cache);
      // NAME_VALUE_LOG(xyz_ref_cache);
      // NAME_VALUE_LOG(ref_patch_cache.cols());
      // NAME_VALUE_LOG(xyz_ref_cache.cols());
      // NAME_VALUE_LOG(residual_cache.cols());
      // NAME_VALUE_LOG(visibility_mask.cols());

      // NAME_VALUE_LOG(feature_counter);
      const cv::Mat &cur_img = cur_frame->img_pyr_.at(level);
      const int stride = cur_img.step;
      const FloatType scale = 1.0f / (1 << level);
      const int patch_area = patch_size * patch_size;
      const FloatType patch_center = (patch_size - 1) / 2.0f;

      FloatType total_intensity = 0.0;

      for (size_t i = 0; i < nr_segment; ++i, ++segment_counter)
      {
        for (size_t idx = 0; idx < 2; ++idx)
        {
          size_t now_idx=feature_counter+segment_counter*2+idx;
          // NAME_VALUE_LOG(now_idx);
          Vector3ft xyz_ref = xyz_ref_cache.col(now_idx);
          const Vector3d xyz_cur(T_cur_ref * xyz_ref.cast<double>());
          if (cur_frame->cam()->getType() ==
                  vk::cameras::CameraGeometryBase::Type::kPinhole &&
              xyz_cur.z() < 0.0)
          {
            visibility_mask(now_idx) = false;
            continue;
          }

          Eigen::Vector2d uv_cur;
          cur_frame->cam()->project3(xyz_cur, &uv_cur);
          const Vector2ft uv_cur_pyr = uv_cur.cast<FloatType>() * scale;

          // compute top left coordinate of patch to be interpolated
          const FloatType u_tl = uv_cur_pyr[0] - patch_center;
          const FloatType v_tl = uv_cur_pyr[1] - patch_center;

          // check if projection is within the image
          if (u_tl < 0.0 || v_tl < 0.0 || u_tl + patch_size + 2.0 >= cur_img.cols || v_tl + patch_size + 2.0 >= cur_img.rows)
          {
            visibility_mask(now_idx) = false;
            continue;
          }
          else
          {
            visibility_mask(now_idx) = true;
          }

          const int u_tl_i = std::floor(u_tl);
          const int v_tl_i = std::floor(v_tl);

          // compute bilateral interpolation weights for the current image
          const FloatType subpix_u_tl = u_tl - u_tl_i;
          const FloatType subpix_v_tl = v_tl - v_tl_i;
          const FloatType wtl = (1.0 - subpix_u_tl) * (1.0 - subpix_v_tl);
          const FloatType wtr = subpix_u_tl * (1.0 - subpix_v_tl);
          const FloatType wbl = (1.0 - subpix_u_tl) * subpix_v_tl;
          const FloatType wbr = subpix_u_tl * subpix_v_tl;

          size_t pixel_counter = 0; // is used to compute the index of the cached residual
          float total_res = 0.0;
          for (int y = 0; y < patch_size; ++y)
          {
            uint8_t *cur_img_ptr =
                static_cast<uint8_t *>(cur_img.data) + (v_tl_i + y) * stride + u_tl_i;

            for (int x = 0; x < patch_size; ++x, ++pixel_counter, ++cur_img_ptr)
            {
              // compute residual, with bilinear interpolation
              FloatType intensity_cur =
                  wtl * cur_img_ptr[0] + wtr * cur_img_ptr[1] + wbl * cur_img_ptr[stride] + wbr * cur_img_ptr[stride + 1];
              FloatType res = static_cast<FloatType>(intensity_cur * (1.0 + alpha) + beta) - ref_patch_cache(pixel_counter, now_idx);
              // CHECK(static_cast<FloatType>(intensity_cur * (1.0 + alpha) + beta)<256);
              // CHECK(ref_patch_cache(pixel_counter, now_idx)<256);
              // CHECK(res<256);
              residual_cache(pixel_counter, now_idx) = res;

              // for camera control:
              total_res += intensity_cur - static_cast<float>(ref_patch_cache(pixel_counter, now_idx));
              total_intensity += intensity_cur;
            }
          }
        }
      }
      // NAME_VALUE_LOG(segment_counter);

    }

    FloatType computeHessianAndGradient(
        const JacobianCache &jacobian_cache,
        const ResidualCache &residual_cache,
        const VisibilityMask &visibility_mask,
        const float weight_scale,
        const vk::solver::WeightFunctionPtr &weight_function,
        SparseImgAlign::HessianMatrix *H,
        SparseImgAlign::GradientVector *g)
    {
      float chi2 = 0.0;
      size_t n_meas = 0;
      const size_t patch_area = residual_cache.rows();
      const size_t mask_size = visibility_mask.size();

      for (size_t i = 0; i < mask_size; ++i)
      {
        if (visibility_mask(i) == true)
        {

          size_t patch_offset = i * patch_area;
          
           
          for (size_t j = 0; j < patch_area; ++j)
          {
            FloatType res = residual_cache(j, i);
            // std::cout<<res<<" "<<std::endl;
            // Robustification.
            float weight = 1.0;
            if (weight_function)
              weight = weight_function->weight(res / weight_scale);

            chi2 += res * res * weight;
            ++n_meas;

            // Compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error).
            const Vector8ft J_ft = jacobian_cache.col(patch_offset + j);
            const Vector8d J_d = J_ft.cast<double>();
            Eigen::VectorXd J_d_copy = J_d;
            // CHECK(!(J_d_copy.array().unaryExpr([](double v) { return std::isinf(v); })).any())
            // <<"\n"<<J_d_copy.transpose()<<"\n"<<i<<" "<<j<<" "<< patch_offset + j<<" "<<jacobian_cache.cols();

            // CHECK(res<500)<<res<<" "<<i<<" "<<j;
            H->noalias() += J_d * J_d.transpose() * weight;
            g->noalias() -= J_d * res * weight;
          }
          // std::cout<<i<<"cols \n"<<std::endl;
        }
      }

      return chi2 / n_meas;
    }

    float computeResidualHessianGradient(const FramePtr &cur_frame,
                                         const size_t level,
                                         const int patch_size,
                                         const size_t nr_features,
                                         const Transformation &T_cur_ref,
                                         const float alpha,
                                         const float beta,
                                         const RefPatchCache &ref_patch_cache,
                                         const XyzRefCache &xyz_ref_cache,
                                         const JacobianCache &jacobian_cache,
                                         const float weight_scale,
                                         const vk::solver::WeightFunctionPtr &weight_function,
                                         SparseImgAlign::HessianMatrix *H,
                                         SparseImgAlign::GradientVector *g,
                                         size_t &feature_counter)
    {
      const cv::Mat &cur_img = cur_frame->img_pyr_.at(level);
      const int stride = cur_img.step;
      const float scale = 1.0f / (1 << level);
      const int patch_area = patch_size * patch_size;
      const FloatType patch_center = (patch_size - 1) / 2.0f;

      float total_intensity = 0.0;
      float chi2 = 0.0;
      size_t n_meas = 0;

      if (!(H && g))
      {
        return chi2;
      }

      for (size_t i = 0; i < nr_features; ++i, ++feature_counter)
      {
        Vector3ft xyz_ref = xyz_ref_cache.col(feature_counter);
        const Vector3d xyz_cur(T_cur_ref * xyz_ref.cast<double>());
        if (cur_frame->cam()->getType() ==
                vk::cameras::CameraGeometryBase::Type::kPinhole &&
            xyz_cur.z() < 0.0)
        {
          continue;
        }
        Eigen::Vector2d uv_cur;
        cur_frame->cam()->project3(xyz_cur, &uv_cur);
        const Vector2ft uv_cur_pyr = uv_cur.cast<FloatType>() * scale;

        // compute top left coordinate of patch to be interpolated
        const FloatType u_tl = uv_cur_pyr[0] - patch_center;
        const FloatType v_tl = uv_cur_pyr[1] - patch_center;

        // check if projection is within the image
        if (u_tl < 0.0 || v_tl < 0.0 || u_tl + patch_size + 2.0 >= cur_img.cols || v_tl + patch_size + 2.0 >= cur_img.rows)
        {
          continue;
        }
        const int u_tl_i = std::floor(u_tl);
        const int v_tl_i = std::floor(v_tl);

        // compute bilateral interpolation weights for the current image
        const FloatType subpix_u_tl = u_tl - u_tl_i;
        const FloatType subpix_v_tl = v_tl - v_tl_i;
        const FloatType wtl = (1.0 - subpix_u_tl) * (1.0 - subpix_v_tl);
        const FloatType wtr = subpix_u_tl * (1.0 - subpix_v_tl);
        const FloatType wbl = (1.0 - subpix_u_tl) * subpix_v_tl;
        const FloatType wbr = subpix_u_tl * subpix_v_tl;

        size_t pixel_counter = 0; // is used to compute the index of the cached residual
        float total_res = 0.0;
        size_t patch_offset = feature_counter * patch_area;
        for (int y = 0; y < patch_size; ++y)
        {
          uint8_t *cur_img_ptr = (uint8_t *)cur_img.data + (v_tl_i + y) * stride + u_tl_i;

          for (int x = 0; x < patch_size; ++x, ++pixel_counter, ++cur_img_ptr)
          {
            // compute residual, with bilinear interpolation
            // TODO: check what if we just do nearest neighbour?
            FloatType intensity_cur =
                wtl * cur_img_ptr[0] + wtr * cur_img_ptr[1] + wbl * cur_img_ptr[stride] + wbr * cur_img_ptr[stride + 1];
            FloatType res = static_cast<FloatType>(intensity_cur * (1.0 + alpha) + beta) - ref_patch_cache(pixel_counter, feature_counter);

            // robustification
            float weight = 1.0;
            if (weight_function)
              weight = weight_function->weight(res / weight_scale);

            chi2 += res * res * weight;
            ++n_meas;

            // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
            const Vector8ft J_ft = jacobian_cache.col(patch_offset + pixel_counter);
            const Vector8d J_d = J_ft.cast<double>();
            H->noalias() += J_d * J_d.transpose() * weight;
            g->noalias() -= J_d * res * weight;

            // for camera control:
            total_res += intensity_cur - static_cast<float>(ref_patch_cache(pixel_counter, feature_counter));
            total_intensity += intensity_cur;
          }
        }
      }

      return chi2 / n_meas;
    }
  } // namespace sparse_img_align_utils
} // namespace svo
