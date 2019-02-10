cmake_minimum_required(VERSION 3.1)
project("Ceres-SLAM")
SET(CMAKE_CXX_STANDARD 11)

find_package(Ceres 2.0 REQUIRED)
find_package(Eigen3 3.2 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

message(STATUS ${OpenCV_INCLUDE_DIRS})

include_directories(include)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB source_files src/*.cpp)
add_executable(${PROJECT_NAME} ${source_files})

target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})