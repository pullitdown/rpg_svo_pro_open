// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/occupancy_grid_2d.h>
#include <svo/direct/feature_detection_types.h>

#ifdef CUDAFAST_ENABLE
#include <opencv2/cudafeatures2d.hpp>
#endif
#include "svo/direct/ELSED.h"
namespace svo {

//------------------------------------------------------------------------------
/// All detectors should derive from this abstract class.
class AbstractDetector
{
public:
  typedef std::shared_ptr<AbstractDetector> Ptr;

  DetectorOptions options_;

  /// Default constructor.
  AbstractDetector(
      const DetectorOptions& options,
      const CameraPtr& cam);

  /// Default destructor.
  virtual ~AbstractDetector() = default;

  // no copy
  AbstractDetector& operator=(const AbstractDetector&) = delete;
  AbstractDetector(const AbstractDetector&) = delete;

  void detect(const FramePtr &frame);

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) = 0;

  inline void resetGrid()
  {
    grid_.reset();
    closeness_check_grid_.reset();
  }

  // occupancy for the current feature type
  OccupandyGrid2D grid_;
  // this is to additionally check whether detected features are near some exiting ones
  // useful for having mutliple detectors
  OccupandyGrid2D closeness_check_grid_;
};

//------------------------------------------------------------------------------
/// FAST detector by Edward Rosten.
class FastDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~FastDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};

//------------------------------------------------------------------------------
/// Detect pixels that have a high gradient magnitude over multiple pyramid levels.
/// These gradient pixels are good for camera tracking.
class GradientDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~GradientDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};


//------------------------------------------------------------------------------
/// Detect pixels that have a high gradient magnitude over multiple pyramid levels.
/// These gradient pixels are good for camera tracking.
class GradientDetectorGrid : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~GradientDetectorGrid() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};

//------------------------------------------------------------------------------
/// @todo
class FastGradDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~FastGradDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};


// //------------------------------------------------------------------------------
// /// @todo
// class FastGradLineDetector : public AbstractDetector
// {
// public:
//   using AbstractDetector::AbstractDetector; // default constructor
//   virtual ~FastGradLineDetector() = default;

//   virtual void detect(
//       const ImgPyr& img_pyr,
//       const cv::Mat& mask,
//       const size_t max_n_features,
//       Keypoints& px_vec,
//       Scores& score_vec,
//       Levels& level_vec,
//       Gr adients& grad_vec,
//       FeatureTypes& types_vec) override;
// };


#ifdef CUDAFAST_ENABLE
//------------------------------------------------------------------------------
/// @brief the cuda fast detector with gradient detector
class CudaFastGradDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  CudaFastGradDetector(const DetectorOptions& options,
      const CameraPtr& cam):AbstractDetector(options,cam)
      {
      
    cuda_fast_detector=cv::cuda::FastFeatureDetector::create(options_.threshold_primary,false);
  }
  virtual ~CudaFastGradDetector() = default;
  cv::Ptr<cv::cuda::FastFeatureDetector> cuda_fast_detector;
  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};
#endif
//------------------------------------------------------------------------------
/// shitomasi detector with gradient detector
class ShiTomasiGradDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~ShiTomasiGradDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};

//------------------------------------------------------------------------------
/// shitomasi detector: This should be called after other detectors. The intention
/// of this detector is to get some extra features for loop closing. It detects shitomasi
/// features and keeps the ones which are a minimum distance away from existing corners.
class ShiTomasiDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~ShiTomasiDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};

//------------------------------------------------------------------------------
/// Dummy detector that selects all pixels
class AllPixelsDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~AllPixelsDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};

//------------------------------------------------------------------------------
/// Detect pixels that have strong gradients according to the paper
/// Huang, J. and Mumford, D. (1999). Statistics of natural images and models. (CVPR)
class GradientHuangMumfordDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~GradientHuangMumfordDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};

class CannyDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~CannyDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};

class SobelDetector : public AbstractDetector
{
public:
  using AbstractDetector::AbstractDetector; // default constructor
  virtual ~SobelDetector() = default;

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Keypoints& px_vec,
      Scores& score_vec,
      Levels& level_vec,
      Gradients& grad_vec,
      FeatureTypes& types_vec) override;
};




class SegmentAbstractDetector
{
public:
  typedef std::shared_ptr<SegmentAbstractDetector> Ptr;

  SegmentDetectorOptions options_;

  /// Default constructor.
  SegmentAbstractDetector(
      const SegmentDetectorOptions& options,
      const CameraPtr& cam);

  /// Default destructor.
  virtual ~SegmentAbstractDetector() = default;

  // no copy
  SegmentAbstractDetector& operator=(const SegmentAbstractDetector&) = delete;
  SegmentAbstractDetector(const SegmentAbstractDetector&) = delete;

  void detect(const FramePtr &frame);

  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Segments& seg_vec,
      Scores& len_vec,
      Levels& level_vec,
      Gradients &grad_vec,

      FeatureTypes& types_vec) = 0;

  inline void resetGrid()
  {
    grid_.reset();
  }

  OccupandyGrid2D grid_;

};


class ElsedDetector : public SegmentAbstractDetector
{
public:
  ElsedDetector( const SegmentDetectorOptions& options,
      const CameraPtr& cam): SegmentAbstractDetector(options, cam){
    elsed_.reset(new upm::ELSED(options_));
  }
  virtual ~ElsedDetector() = default;
  
  virtual void detect(
      const ImgPyr& img_pyr,
      const cv::Mat& mask,
      const size_t max_n_features,
      Segments& seg_vec,
      Scores& len_vec,
      Levels& level_vec,
      Gradients &grad_vec,

      FeatureTypes& types_vec) override;
  
  private:
    std::unique_ptr<upm::ELSED> elsed_;
};



} // namespace svo
