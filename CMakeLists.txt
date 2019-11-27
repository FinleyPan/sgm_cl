cmake_minimum_required(VERSION 2.8)

project(sgm_cl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCL REQUIRED)
find_package(OpenCV 3 REQUIRED)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sgm_source_path.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/sgm_source_path.h)

include_directories(${OpenCL_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sgm_cl main.cpp sgm_cl.cc)
target_link_libraries(sgm_cl ${OpenCL_LIBRARIES} opencv_core opencv_highgui opencv_imgproc)
