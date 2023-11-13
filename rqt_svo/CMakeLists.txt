cmake_minimum_required(VERSION 2.8.3)
project(rqt_svo)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
IF(NOT DEFINED ENV{ARM_ARCHITECTURE})
  find_package(catkin REQUIRED COMPONENTS
    rospy
    rqt_gui
    rqt_gui_py
  )
  catkin_package()
  catkin_python_setup()

  install(FILES plugin.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

  install(DIRECTORY resource
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

  install(PROGRAMS scripts/rqt_svo
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
ENDIF()
