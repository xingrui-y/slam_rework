project(slam_app)

add_library(${PROJECT_NAME} SHARED "")

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
    PRIVATE
        src/tum_dataset_wrapper.cpp
)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.2 REQUIRED)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})