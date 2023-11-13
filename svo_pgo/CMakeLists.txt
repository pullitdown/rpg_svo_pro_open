project(svo_pgo)
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(catkin_simple REQUIRED)
catkin_simple()

include(SvoSetup)

set(HEADERS
  include/svo/pgo/pgo.h
  )

set(SOURCES
  src/pgo.cpp
  )

#add_executable(svo_online_loopclosing ${SOURCES} ${HEADERS})
cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

cs_add_executable(test_pgo test/test.cpp)
target_link_libraries(test_pgo ${PROJECT_NAME})

cs_install()
cs_export()
