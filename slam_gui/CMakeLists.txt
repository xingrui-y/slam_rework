project(slam_gui)

add_library(${PROJECT_NAME} SHARED "")

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(${PROJECT_NAME}
PRIVATE
    src/opengl_display.cpp
)

target_link_libraries(${PROJECT_NAME} 
    Eigen3::Eigen
    ${OpenGL_LIBRARIES}
    ${Pangolin_LIBRARIES}
)

# set_target_properties(${PROJECT_NAME} 
# PROPERTIES 
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_STANDARD 11
# )

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)

file (GLOB_RECURSE resources "glsl/*.*")
foreach(resource ${resources})
    get_filename_component(filename ${resource} NAME)

    set(output "${CMAKE_BINARY_DIR}/glsl/${filename}")
 
    add_custom_command(
        COMMENT "Moving updated resource-file '${filename}'"
        OUTPUT ${output}
        DEPENDS ${resource}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${resource}
        ${output}
    )
 
    add_custom_target(${filename} ALL DEPENDS ${resource} ${output})
 
endforeach()