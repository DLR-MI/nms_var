cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(PYTORCH_NMS_VARIANCE CXX CUDA)

find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED PATHS $ENV{HOME}/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(nms_variance_cu STATIC src/cuda/nms_var_kernel.cu)
target_include_directories(nms_variance_cu PRIVATE "${TORCH_INCLUDE_DIRS}" ${PYTHON_INCLUDE_DIRS})
target_link_libraries(nms_variance_cu ${PYTHON_LIBRARIES})
set_property(TARGET nms_variance_cu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET nms_variance_cu PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(extern/pybind11)

pybind11_add_module(pytorch_nms_variance src/nms.cpp)
target_link_libraries(pytorch_nms_variance PRIVATE "${TORCH_LIBRARIES}" nms_variance_cu)
set_property(TARGET pytorch_nms_variance PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET pytorch_nms_variance
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:pytorch_nms_variance>)
endif (MSVC)