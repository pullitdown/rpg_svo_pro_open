#pragma once

#include <memory>
#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

namespace svo {

//------------------------------------------------------------------------------
// Scalars and fp precision.
using size_t = std::size_t;
using uint8_t = std::uint8_t;
using uint64_t = std::uint64_t;
using FloatType = double;

//------------------------------------------------------------------------------
// Feature containers.
using Keypoint = Eigen::Matrix<FloatType, 2, 1>;
using Segment =Eigen::Matrix<FloatType, 4, 1>;
using BearingVector = Eigen::Matrix<FloatType, 3, 1>;
using Position = Eigen::Matrix<FloatType, 3, 1>;
using GradientVector = Eigen::Matrix<FloatType, 2, 1>;
using SeedState = Eigen::Matrix<FloatType, 4, 1>;
using Level = int;
using Score = FloatType;
using Keypoints = Eigen::Matrix<FloatType, 2, Eigen::Dynamic, Eigen::ColMajor>;
using Linepoints =Eigen::Matrix<FloatType,2,4,Eigen::ColMajor>;
using Bearings = Eigen::Matrix<FloatType, 3, Eigen::Dynamic, Eigen::ColMajor>;
using Positions = Eigen::Matrix<FloatType, 3, Eigen::Dynamic, Eigen::ColMajor>;
using Gradients = Eigen::Matrix<FloatType, 2, Eigen::Dynamic, Eigen::ColMajor>;
using Scores = Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>;
using Levels = Eigen::Matrix<Level, Eigen::Dynamic, 1, Eigen::ColMajor>;
using InlierMask = Eigen::Matrix<bool, Eigen::Dynamic, 1, Eigen::ColMajor>;
using SeedStates = Eigen::Matrix<FloatType, 4, Eigen::Dynamic, Eigen::ColMajor>;
using TrackIds = Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::ColMajor>;
using Segments =Eigen::Matrix<FloatType,4, Eigen::Dynamic,  Eigen::ColMajor>;
using Lengths = Eigen::Matrix<FloatType,1, Eigen::Dynamic,  Eigen::ColMajor>;

//------------------------------------------------------------------------------
// Forward declarations and common types for simplicity.
struct Feature;
using FeaturePtr = std::shared_ptr<Feature>;
class Frame;
using FramePtr = std::shared_ptr<Frame>;
using FrameWeakPtr = std::weak_ptr<Frame>;
class FrameBundle;
using FrameBundlePtr = std::shared_ptr<FrameBundle>;
using BundleId = int;
using ImgPyr = std::vector<cv::Mat>;

//------------------------------------------------------------------------------
// Point matches
struct MatchedPointsInfo
{
  int lc_kf_id_ = -1;
  int cur_kf_id_ = -1;
  std::map<int, int> pt_id_matches_ {};
};

//------------------------------------------------------------------------------
// Feature Type.
enum class FeatureType : uint8_t
{
  kEdgeletSeed = 0,
  kCornerSeed = 1,
  kMapPointSeed = 2,
  kLinePointSeed =3,
  kSegmentSeed=4,
  kEdgeletSeedConverged = 5,
  kCornerSeedConverged = 6,
  kMapPointSeedConverged = 7,
  kLinePointSeedConverged =8,
  kSegmentSeedConverged = 9,
  kEdgelet = 10,
  kCorner = 11,
  kMapPoint = 12,
  kLinePoint = 13,
  kSegment=14,
  kFixedLandmark = 15,
  kFixedSegmentLandmark = 16,
  kOutlier = 17,
  KSegmentOutlier=18
  
};

using FeatureTypes = std::vector<FeatureType>;


inline bool isSeed(const FeatureType& t)
{
  return static_cast<uint8_t>(t) < 10;
}

inline bool isCornerEdgeletSeed(const FeatureType& t)
{
  return (t == FeatureType::kEdgeletSeedConverged
          || t == FeatureType::kCornerSeedConverged
          || t == FeatureType::kEdgeletSeed
          || t == FeatureType::kCornerSeed);
}

inline bool isSegmentSeed(const FeatureType& t)
{
  return (t == FeatureType::kSegmentSeedConverged
          || t == FeatureType::kSegmentSeed);
}


inline bool isConvergedCornerEdgeletSeed(const FeatureType& t)
{
  return (t == FeatureType::kEdgeletSeedConverged
          || t == FeatureType::kCornerSeedConverged);
}

inline bool isConvergedSegmentSeed(const FeatureType& t)
{
  return (t == FeatureType::kSegmentSeedConverged);
}


inline bool isConvergedMapPointSeed(const FeatureType& t)
{
  return (t == FeatureType::kMapPointSeedConverged);
}

inline bool isUnconvergedCornerEdgeletSeed(const FeatureType& t)
{
  return (t == FeatureType::kEdgeletSeed
          || t == FeatureType::kCornerSeed);
}

inline bool isUnconvergedSegmentSeed(const FeatureType& t)
{
  return (t == FeatureType::kSegmentSeed);
}
inline bool isUnconvergedMapPointSeed(const FeatureType& t)
{
  return (t == FeatureType::kMapPointSeed);
}

inline bool isEdgelet(const FeatureType& t)
{
  return (t == FeatureType::kEdgelet
          || t == FeatureType::kEdgeletSeed
          || t == FeatureType::kEdgeletSeedConverged);
}

inline bool isCorner(const FeatureType& t)
{
  return (t == FeatureType::kCorner
          || t == FeatureType::kCornerSeed
          || t == FeatureType::kCornerSeedConverged);
}

inline bool isMapPoint(const FeatureType& t)
{
  return (t == FeatureType::kMapPoint
          || t == FeatureType::kMapPointSeed
          || t == FeatureType::kMapPointSeedConverged);
}

inline bool isMapPointSeed(const FeatureType& t)
{
  return (t == FeatureType::kMapPointSeed
          || t == FeatureType::kMapPointSeedConverged);
}

inline bool isUnconvergedSeed(const FeatureType& t)
{
  return (t == FeatureType::kCornerSeed
          || t == FeatureType::kMapPointSeed
          || t == FeatureType::kEdgeletSeed
          || t== FeatureType::kSegmentSeed);
}

inline bool isFixedLandmark(const FeatureType& t)
{
  return t == FeatureType::kFixedLandmark;
}

inline bool isFixedSegmentLandmark(const FeatureType& t)
{
  return t == FeatureType::kFixedSegmentLandmark;
}

inline bool isSegment(const FeatureType& t)
{
  return t== FeatureType::kSegment|| t==FeatureType::kSegmentSeed || t==FeatureType::kSegmentSeedConverged ;
}

inline std::string str(const FeatureType& type)
{
  if (type == FeatureType::kEdgeletSeed)
  {
    return std::string("EdgeletSeed");
  }
  else if (type == FeatureType::kCornerSeed)
  {
    return std::string("CornerSeed");
  }
  else if (type == FeatureType::kMapPointSeed)
  {
    return std::string("MapPointSeed");
  }
  else if (type == FeatureType::kEdgeletSeedConverged)
  {
    return std::string("EdgeletSeedConverged");
  }
  else if (type == FeatureType::kCornerSeedConverged)
  {
    return std::string("CornerSeedConverged");
  }
  else if (type == FeatureType::kMapPointSeedConverged)
  {
    return std::string("MapPointSeedConverged");
  }
  else if (type == FeatureType::kEdgelet)
  {
    return std::string("Edgelet");
  }
  else if (type == FeatureType::kCorner)
  {
    return std::string("Corner");
  }
  else if (type == FeatureType::kMapPoint)
  {
    return std::string("MapPoint");
  }
  else if (type == FeatureType::kFixedLandmark)
  {
    return std::string("FixedLandmark");
  }
  else if (type == FeatureType::kOutlier)
  {
    return std::string("Outlier");
  }
  else
  {
    return std::string("Unknown");
  }
}

struct EnumClassHash
{
  template <typename T>
  std::size_t operator()(T t) const
  {
    return static_cast<std::size_t>(t);
  }
};

} // namespace svo
