CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(OpenPose_ZED)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

SET(EXECUTABLE_OUTPUT_PATH ".")

set(OpenCV_DIR "D:/opencv/opencv/build")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0")

find_package(OpenCV REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# include file of openpose
include_directories("D:/Code/OpenPose/openpose/include"
                    "D:/Code/OpenPose/openpose/3rdparty/windows/opencv/include"
					"D:/Code/OpenPose/openpose/3rdparty/windows/caffe/include"
					"D:/Code/OpenPose/openpose/3rdparty/windows/caffe/include2"
					"D:/Code/OpenPose/openpose/3rdparty/windows/caffe3rdparty/include"
					"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include"
)

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.h)
ADD_EXECUTABLE(${PROJECT_NAME} ${HDR_FILES} ${SRC_FILES})
add_definitions(-std=c++14 -O3)

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
                      ${ZED_LIBS}
					  ${OpenCV_LIBRARIES}
					  D:/Code/OpenPose/openpose/build/src/openpose/Release/openpose.lib
	                  D:/Code/OpenPose/openpose/3rdparty/windows/opencv/x64/vc15/lib/opencv_world411.lib
                      D:/Code/OpenPose/openpose/3rdparty/windows/caffe3rdparty/lib/gflags.lib
					  D:/Code/OpenPose/openpose/3rdparty/windows/caffe3rdparty/lib/glog.lib
					  D:/Code/OpenPose/openpose/3rdparty/windows/caffe/lib/caffe.lib
					  D:/Code/OpenPose/openpose/3rdparty/windows/caffe/lib/caffeproto.lib
					  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/cudart_static.lib"
)
