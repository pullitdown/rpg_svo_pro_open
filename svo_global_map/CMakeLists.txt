project(svo_global_map)
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(catkin_simple REQUIRED)
catkin_simple()

find_package(GTSAMCMakeTools REQUIRED)
find_package(GTSAM REQUIRED)
message(STATUS ${GTSAM_INCLUDE_DIR})
message(STATUS ${GTSAM_LIBRARIES})

include(SvoSetup)
include_directories(${GTSAM_INCLUDE_DIR})
include_directories(/home/sunteng/catkin_ws/src/gtsam-4.2.0/gtsam_unstable)
include_directories(/home/sunteng/catkin_ws/src/gtsam-4.2.0/gtsam)


# get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(${GTSAM_INCLUDE_DIR})   


set(HEADERS
  # gtsam interface
  include/svo/gtsam/gtsam_optimizer.h
  include/svo/gtsam/graph_manager.h
  include/svo/gtsam/smart_factors_fwd.h
  include/svo/gtsam/camera_bearing_factor.h
  include/svo/gtsam/camera_bearing_extrinsics_factor.h
  include/svo/global_map.h
  include/svo/null_stream.h
  )

set(SOURCES
  # gtsam interface
  src/gtsam/gtsam_optimizer.cpp
  src/gtsam/graph_manager.cpp
  src/global_map.cpp
  src/null_stream.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} gtsam)

cs_install()
cs_export()
