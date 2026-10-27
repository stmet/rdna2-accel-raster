cmake_minimum_required(VERSION 3.20)
project(rdna2_accel_raster LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

# Optional local user include directory
if(DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/.local/include")
    include_directories("$ENV{HOME}/.local/include")
endif()

# Find Vulkan SDK / Loader & GLM
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

# Shader Compilation Tool
find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslc HINTS ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} ${Vulkan_GLSLC_EXECUTABLE})
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "Neither glslangValidator nor glslc was found!")
endif()
message(STATUS "Found shader compiler: ${GLSLANG_VALIDATOR}")

set(SHADER_RQ_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/verify_rayquery.comp")
set(SHADER_RQ_BIN "${CMAKE_BINARY_DIR}/shaders/verify_rayquery.spv")
set(SHADER_RQ_DST "${CMAKE_CURRENT_SOURCE_DIR}/shaders/verify_rayquery.spv")

set(SHADER_BVH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_bvh_builder.comp")
set(SHADER_BVH_BIN "${CMAKE_BINARY_DIR}/shaders/fast_bvh_builder.spv")
set(SHADER_BVH_DST "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast_bvh_builder.spv")

add_custom_command(
    OUTPUT ${SHADER_RQ_BIN} ${SHADER_RQ_DST}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 "${SHADER_RQ_SRC}" -o "${SHADER_RQ_BIN}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SHADER_RQ_BIN}" "${SHADER_RQ_DST}"
    DEPENDS ${SHADER_RQ_SRC}
    COMMENT "Compiling SPIR-V shader: verify_rayquery.comp -> verify_rayquery.spv"
)

add_custom_command(
    OUTPUT ${SHADER_BVH_BIN} ${SHADER_BVH_DST}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 "${SHADER_BVH_SRC}" -o "${SHADER_BVH_BIN}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SHADER_BVH_BIN}" "${SHADER_BVH_DST}"
    DEPENDS ${SHADER_BVH_SRC}
    COMMENT "Compiling SPIR-V shader: fast_bvh_builder.comp -> fast_bvh_builder.spv"
)

add_custom_target(compile_shaders ALL DEPENDS
    ${SHADER_RQ_BIN} ${SHADER_RQ_DST}
    ${SHADER_BVH_BIN} ${SHADER_BVH_DST}
)

# ------------------------------------------------------------------------------
# Common Support Library
# ------------------------------------------------------------------------------
add_library(common STATIC
    common/vk_context.cpp
    common/vk_buffer.cpp
    common/vk_bvh.cpp
    common/vk_timer.cpp
    common/vk_shader.cpp
)

target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(common PUBLIC
    Vulkan::Vulkan
    ${CMAKE_DL_LIBS}
    pthread
)

# ------------------------------------------------------------------------------
# Target 1: Microbenchmark & Hardware Unit Test (rdna2_bvh_test)
# ------------------------------------------------------------------------------
add_executable(rdna2_bvh_test
    src/main.cpp
    external/volk/volk.c
)
add_dependencies(rdna2_bvh_test compile_shaders)

target_include_directories(rdna2_bvh_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/volk
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(rdna2_bvh_test PRIVATE
    glm::glm
    ${CMAKE_DL_LIBS}
    pthread
)

# ------------------------------------------------------------------------------
# Target 2: Integrated Full-Frame Deferred Rendering Pipeline (fullframe_benchmark)
# ------------------------------------------------------------------------------
add_executable(fullframe_benchmark
    src/fullframe_benchmark.cpp
)
add_dependencies(fullframe_benchmark compile_shaders)

target_include_directories(fullframe_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(fullframe_benchmark PRIVATE
    common
    glm::glm
)

# Copy shaders directory to build directory for runtime lookup
add_custom_command(
    TARGET fullframe_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:fullframe_benchmark>/shaders"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:fullframe_benchmark>/shaders/deferred"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SHADER_RQ_DST}" "$<TARGET_FILE_DIR:fullframe_benchmark>/shaders/verify_rayquery.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SHADER_BVH_DST}" "$<TARGET_FILE_DIR:fullframe_benchmark>/shaders/fast_bvh_builder.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/shaders/deferred" "$<TARGET_FILE_DIR:fullframe_benchmark>/shaders/deferred"
)

# ------------------------------------------------------------------------------
# Optional Benchmark Subdirectories
# ------------------------------------------------------------------------------
option(BUILD_BENCHMARKS "Build modular research experiment suites" ON)
if(BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/expA_async")
        add_subdirectory(benchmarks/expA_async)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/expB_shadows")
        add_subdirectory(benchmarks/expB_shadows)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/expC_reflections")
        add_subdirectory(benchmarks/expC_reflections)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/expD_screenspace")
        add_subdirectory(benchmarks/expD_screenspace)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/expE_fullframe")
        add_subdirectory(benchmarks/expE_fullframe)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/exp0_probe")
        add_subdirectory(benchmarks/exp0_probe)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/exp1_visbuf")
        add_subdirectory(benchmarks/exp1_visbuf)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/exp2_occlusion")
        add_subdirectory(benchmarks/exp2_occlusion)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/exp3_spatial")
        add_subdirectory(benchmarks/exp3_spatial)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/exp4_integration")
        add_subdirectory(benchmarks/exp4_integration)
    endif()
endif()
