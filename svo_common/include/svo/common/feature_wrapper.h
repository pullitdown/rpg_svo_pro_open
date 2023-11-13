#pragma once

#include <memory>
#include <svo/common/types.h>

namespace svo {

// forward declaration
class Point;
using PointPtr = std::shared_ptr<Point>;

class Line;
using LinePtr = std::shared_ptr<Line>;

class Frame;
using FramePtr = std::shared_ptr<Frame>;

struct SeedRef
{
  FramePtr keyframe;
  int seed_id = -1;
  SeedRef(const FramePtr& _keyframe, const int _seed_id)
    : keyframe(_keyframe)
    , seed_id(_seed_id)
  { ; }
  SeedRef() = default;
  ~SeedRef() = default;
};



/** @todo (MWE)
 */

struct SegmentWrapper{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatureType& type;               

  Eigen::Ref<Segment> segment;
  Eigen::Ref<BearingVector> s_f;
  Eigen::Ref<BearingVector> e_f;
  Eigen::Ref<GradientVector> grad;
  Score& score;
  Level& level;
  LinePtr& line_landmark;
  SeedRef& seed_ref;
  // SeedRef& e_seed_ref;
  int& track_id;
    SegmentWrapper(
      FeatureType& _type,
      Eigen::Ref<Segment> _seg,
      Eigen::Ref<BearingVector> s_f,
      Eigen::Ref<BearingVector> e_f,
      Score& _score,
      Level& _pyramid_level,
      GradientVector grad,
      LinePtr& line_landmark,
      SeedRef& seed_ref,
      // SeedRef& e_seed_ref,
      int& _track_id)
    : type(_type)
    , segment(_seg)
    , s_f(s_f)
    , e_f(e_f)
    , grad(grad)
    , score(_score)
    , level(_pyramid_level)
    , line_landmark(line_landmark)
    , seed_ref(seed_ref)
    // , seed_ref(e_seed_ref)
    , track_id(_track_id)
  { ; }



  SegmentWrapper() = delete;
  ~SegmentWrapper() = default;

  //! @todo (MWE) do copy and copy-asignment operators make sense?
  SegmentWrapper(const SegmentWrapper& other) = default;
  SegmentWrapper& operator=(const SegmentWrapper& other) = default;


};


struct FeatureWrapper
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FeatureType& type;                //!< Type can be corner or edgelet.
  Eigen::Ref<Keypoint> px;          //!< Coordinates in pixels on pyramid level 0.
  Eigen::Ref<BearingVector> f;      //!< Unit-bearing vector of the feature.
  Eigen::Ref<GradientVector> grad;  //!< Dominant gradient direction for edglets, normalized.
  Score& score;                     //!< 
  Level& level;                     //!< Image pyramid level where feature was extracted.
  PointPtr& landmark;
  SeedRef& seed_ref;
  int& track_id;

  FeatureWrapper(
      FeatureType& _type,
      Eigen::Ref<Keypoint> _px,
      Eigen::Ref<BearingVector> _f,
      Eigen::Ref<GradientVector> _grad,
      Score& _score,
      Level& _pyramid_level,
      PointPtr& _landmark,
      SeedRef& _seed_ref,
      int& _track_id)
    : type(_type)
    , px(_px)
    , f(_f)
    , grad(_grad)
    , score(_score)
    , level(_pyramid_level)
    , landmark(_landmark)
    , seed_ref(_seed_ref)
    , track_id(_track_id)
  { ; }

  FeatureWrapper() = delete;
  ~FeatureWrapper() = default;

  //! @todo (MWE) do copy and copy-asignment operators make sense?
  FeatureWrapper(const FeatureWrapper& other) = default;
  FeatureWrapper& operator=(const FeatureWrapper& other) = default;
};

} // namespace svo
