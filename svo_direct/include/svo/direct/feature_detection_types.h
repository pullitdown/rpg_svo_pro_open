// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/common/types.h>

namespace svo
{

  //------------------------------------------------------------------------------
  /// Temporary container used for corner detection. Features are initialized from these.
  struct Corner
  {
    int x;       ///< x-coordinate of corner in the image.
    int y;       ///< y-coordinate of corner in the image.
    int level;   ///< pyramid level of the corner.
    float score; ///< shi-tomasi score of the corner.
    float angle; ///< for gradient-features: dominant gradient angle.

    Corner(int _x, int _y, float _score, int _level, float _angle)
        : x(_x), y(_y), level(_level), score(_score), angle(_angle)
    {
    }
  };

  struct ScoreSegment
  {
    double x0;
    double y0;
    double x1;
    double y1;
    double score_;
    int level_;
    int seq_id_;
    ScoreSegment() = default;
    ScoreSegment(const cv::Mat &segment_cv, double score,int level,int seq_id=-1) : x0(segment_cv.at<float>(0, 0)),
    y0(segment_cv.at<float>(1, 0)),x1(segment_cv.at<float>(2, 0)),y1(segment_cv.at<float>(3, 0)), score_(score),level_(level),seq_id_(seq_id) {}
    ScoreSegment(double x0, double y0, double x1, double y1, double score, int level,int seq_id=-1) : x0(x0), y0(y0), x1(x1), y1(y1), score_(score), level_(level) ,seq_id_(seq_id) {}
    inline bool operator<(const ScoreSegment &rhs) const
    {
      if (score_ == rhs.score_)
      {
        float dx1 = x0 - x1;
        float dx2 = rhs.x0 - rhs.x1;
        float dy1 = y0 - y1;
        float dy2 = rhs.y0 - rhs.y1;
        return std::sqrt(dx1 * dx1 + dy1 * dy1) > std::sqrt(dx2 * dx2 + dy2 * dy2);
      }
      else
      {
        return score_ > rhs.score_;
      }
    }
  };

  using Corners = std::vector<Corner>;
  using ScoreSegments = std::vector<ScoreSegment>;
  //------------------------------------------------------------------------------
  /// Available Feature Detector Types
  enum class DetectorType
  {
    kFast,             ///< Fast Corner Detector by Edward Rosten
    kGrad,             ///< Gradient detector for edgelets
    kFastGrad,         ///< Combined Fast and Gradient detector
    kShiTomasi,        ///< Shi-Tomasi detector
    kShiTomasiGrad,    ///< Combined Shi-Tomasi, fast and Gradient detector
    kGridGrad,         ///< Gradient detector with feature grid.
    kAll,              ///< Every pixel is a feature!
    kGradHuangMumford, ///< 'Natural' edges (see Huang CVPR'99)
    kCanny,            ///< Canny edge detector
    kSobel,            ///< Sobel edge detector
    kCudaFastGrad,     ///< cuda Combined Fast and Gradient detector
    kELSEDSegment      /// < combined fast , gradien and elsed segment detector
  };

  //------------------------------------------------------------------------------
  /// Common options of all feature detectors.
  struct DetectorOptions
  {
    /// Maximum one feature per bucked with cell_size width and height
    size_t cell_size = 30;

    /// Extract features on pyramid
    int max_level = 2;

    /// minimum pyramid level at which features should be selected
    int min_level = 0;

    /// no feature should be within border.
    int border = 8;

    /// Choose between {FAST, FAST_GRAD}, FAST_GRAD will use Edgelets.
    DetectorType detector_type = DetectorType::kFast;

    /// Primary detector threshold
    double threshold_primary = 10.0;

    /// Secondary detector threshold.
    /// Used if the detector uses two different detectors. E.g. in the case of
    /// FAST_GRAD, it is the gradient detector threshold.
    double threshold_secondary = 100.0;

    /// Level where features are initialized.
    /// Only for detectors supporting specific feature levels like the AllPixelDetector.
    int sampling_level = 0;
    int level = 0;

    /// fineness level of the secondary grid (used for extra shi tomasi features when loop closing is enabled)
    size_t sec_grid_fineness = 1;

    /// Corner Strength Thrshold for shitomasi features (used only when loop closing is enabled)
    double threshold_shitomasi = 100.0;
  };

  struct SegmentDetectorOptions
  {

    size_t best_n_segment = 100;
    /// Maximum one feature per bucked with cell_size width and height
    size_t cell_size = 16;

    /// Extract features on pyramid
    int max_level = 2;

    /// minimum pyramid level at which features should be selected
    int min_level = 0;

    /// no feature should be within border.
    int border = 8;

    /// Choose between {FAST, FAST_GRAD}, FAST_GRAD will use Edgelets.
    DetectorType detector_type = DetectorType::kELSEDSegment;

    /// Primary detector threshold
    double threshold_primary = 10.0;

    double segment_socre_threshold = 10.0;

    size_t max_segment_num = 100;

    //-----------------------elsed param---------------------------
    int ksize = 5;
    // Sigma of the gaussian kernel
    float sigma = 1;
    // The threshold of pixel gradient magnitude.
    // Only those pixels whose gradient magnitude are larger than
    // this threshold will be taken as possible edge points.
    float gradientThreshold = 30;
    // If the pixel's gradient value is bigger than both of its neighbors by a
    // certain anchorThreshold, the pixel is marked to be an anchor.
    uint8_t anchorThreshold = 8;
    // Anchor testing can be performed at different scan intervals, i.e.,
    // every row/column, every second row/column
    unsigned int scanIntervals = 2;

    // Minimum line segment length
    int minLineLen = 20;
    // Threshold used to check if a list of edge points for a line segment
    double lineFitErrThreshold = 0.2;
    // Threshold used to check if a new pixel is part of an already fit line segment
    double pxToSegmentDistTh = 1.5;
    // Threshold used to validate the junction jump region. The first eigenvalue of the gradient
    // auto-correlation matrix should be at least junctionEigenvalsTh times bigger than the second eigenvalue
    double junctionEigenvalsTh = 10;
    // the difference between the perpendicular segment direction and the direction of the gradient
    // in the region to be validated must be less than junctionAngleTh radians
    double junctionAngleTh = 10 * (M_PI / 180.0);
    // The threshold over the validation criteria. For ELSED, it is the gradient angular error in pixels.
    double validationTh = 0.15;

    // Whether to validate or not the generated segments
    bool validate = true;
    // Whether to jump over junctions
    bool treatJunctions = true;
    // List of junction size that will be tested (in pixels)
    std::vector<int> listJunctionSizes = {5, 7, 9};
    //---------------elsed param-------------------------------------
  };

} // namespace svo
