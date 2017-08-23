cmake_minimum_required(VERSION 2.8.6)
project(GOT)

# Build Options
option(SHOW_PATH "Show path of libs and headers" OFF)
#set(LOCAL_PREFIX "/home/osep/local/" CACHE PATH "Path to local installations e.g. USER/local/")
#set(RPATH "/home/osep/local/lib/" CACHE PATH "Path to local libraries e.g. USER/local/lib/")
set(OpenCV_DIR "/home/osep/local/share/OpenCV" CACHE PATH "Path to local OpenCV e.g. USER/local/share/OpenCV")
set(PCL_DIR "/path/to/PCLConfig.cmake" CACHE PATH "Path to local PCL e.g. USER/local/share/pcl-1.8")
set(EIGEN_INCLUDE_DIRS "/path/to/include/eige3" CACHE PATH "Path to local eigen/include e.g. USER/local/include/eigen3")

# Compiler Flags
set(CMAKE_CXX_COMPILER_FLAGS "${CMAKE_CXX_COMPILER_FLAGS} -Wall") # Enables all compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")  # Generate position independent code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")  # Enable c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

#set(Boost_LIBS
#        "${LOCAL_PREFIX}/lib/libboost_program_options.so"
#        "${LOCAL_PREFIX}/lib/libboost_thread.so"
#        "${LOCAL_PREFIX}/lib/libboost_system.so"
#        "${LOCAL_PREFIX}/lib/libboost_signals.so"
#        "${LOCAL_PREFIX}/lib/libboost_log.so"
#        )

# Here we set *_LIBS, *_INCLUDE_DIRS, etc
#find_package(OpenCV 3.1.0 REQUIRED COMPONENTS core highgui imgproc calib3d video)
find_package(Boost COMPONENTS program_options thread system signals) 
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc calib3d video)
find_package(PCL 1.8.0 REQUIRED COMPONENTS common io visualization filters sample_consensus segmentation kdtree)

# Include
include_directories(include ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Project src
file(GLOB PROJ_SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src)

# External code
file(GLOB EXTERNAL_SRC_FILES ${CMAKE_SOURCE_DIR}/src/external/*.cpp)
set(EXTERNAL_SRC_DIR ${CMAKE_SOURCE_DIR}/src/external)
include_directories(${EXTERNAL_SRC_DIR})

# Utils
file(GLOB SUN_UTILS_SRC_FILES ${CMAKE_SOURCE_DIR}/src/sun_utils/*.cpp)
set(SUN_UTILS_SRC_DIR ${CMAKE_SOURCE_DIR}/src/sun_utils)
include_directories(${SUN_UTILS_SRC_DIR})

# CIWT core
file(GLOB CIWT_SRC_FILES ${CMAKE_SOURCE_DIR}/src/CIWT/*.cpp)
set(CIWT_SRC_DIR ${CMAKE_SOURCE_DIR}/src/CIWT)
include_directories(${CIWT_SRC_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Use OpenMP")
endif()

# Libraries
add_library (connected_components
        src/connected_components/CC.cpp
        src/connected_components/CPoint.cpp

        include/connected_components/CC.h
        include/connected_components/CPoint.h
        )

add_library (viso2
        src/libviso2/matrix.cpp
        src/libviso2/viso.cpp
        src/libviso2/viso_mono.cpp
        src/libviso2/viso_stereo.cpp
        src/libviso2/matcher.cpp
        src/libviso2/reconstruction.cpp
        src/libviso2/triangle.cpp
        src/libviso2/filter.cpp

        include/libviso2/viso.h
        include/libviso2/viso_mono.h
        include/libviso2/viso_stereo.h
        include/libviso2/matcher.h
        include/libviso2/reconstruction.h
        include/libviso2/triangle.h
        include/libviso2/filter.h
        )

add_library (elas
        src/libelas/matrix.cpp
        src/libelas/triangle.cpp
        src/libelas/filter.cpp
        src/libelas/elas.cpp
        src/libelas/descriptor.cpp

        include/libelas/elas.h
        include/libelas/image.h
        include/libelas/timer.h
        include/libelas/matrix.h
        include/libelas/triangle.h
        include/libelas/filter.h
        include/libelas/descriptor.h
        )

add_library (tracking
        src/tracking/resource_manager.cpp
        src/tracking/observation.cpp
        src/tracking/multi_object_tracker_base.cpp
        src/tracking/hypothesis.cpp
        src/tracking/kalman_filter.cpp
        src/tracking/extended_kalman_filter.cpp
        src/tracking/visualization.cpp
        src/tracking/qpbo.cpp
        src/tracking/utils_tracking.cpp
        src/tracking/category_filter.cpp

        include/tracking/resource_manager.h
        include/tracking/observation.h
        include/tracking/multi_object_tracker_base.h
        include/tracking/hypothesis.h
        include/tracking/kalman_filter.h
        include/tracking/extended_kalman_filter.h
        include/tracking/visualization.h
        include/tracking/qpbo.h
        include/tracking/utils_tracking.h
        include/tracking/dynamics_handler.h
        src/tracking/category_filter.cpp include/tracking/category_filter.h
        )

add_library (scene_segmentation
        src/scene_segmentation/ground_histogram.cpp
        src/scene_segmentation/object_proposal.cpp
        src/scene_segmentation/scene_segmentation.cpp
        src/scene_segmentation/utils_segmentation.cpp
        src/scene_segmentation/multi_scale_quickshift.cpp

        include/scene_segmentation/ground_histogram.h
        include/scene_segmentation/object_proposal.h
        include/scene_segmentation/scene_segmentation.h
        include/scene_segmentation/utils_segmentation.h
        include/scene_segmentation/multi_scale_quickshift.h
        include/scene_segmentation/parameters_gop3D.h
        )

target_link_libraries(connected_components)
target_link_libraries(elas)
target_link_libraries(viso2)
target_link_libraries(scene_segmentation)
target_link_libraries(tracking viso2)

# Subdirectories
# Add subfolder that also contain CMakeLists.txt
add_subdirectory(apps)
add_subdirectory(experimental)

message(STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES})
#message(STATUS "Boost libs: " ${Boost_LIBS})
message(STATUS "PCL libs: " ${PCL_LIBRARIES})

# Debug info
if(SHOW_PATH)
    message(STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES})
    #message(STATUS "Boost libs: " ${Boost_LIBS})
    message(STATUS ${OpenCV_LIBS})
    message(STATUS "Project Source Dirs: " ${PROJECT_SOURCE_DIR})
    message(STATUS "OpenCV Include Dirs: " ${OpenCV_INCLUDE_DIRS})
    message(STATUS "PCL Include Dirs: " ${PCL_INCLUDE_DIRS})
    message(STATUS "")
endif()
