cmake_minimum_required(VERSION 2.8)
project( precTest )

# Make flags
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g" )
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -gencode arch=compute_86,code=sm_86 )

# Enable CUDA
find_package( CUDA REQUIRED )
include_directories( ${CUDA_INCLUDE_DIRS} )

# Add excutable
cuda_compile(CUDA_OBJECT precTest.cu)
cuda_add_executable( ${PROJECT_NAME} ${CUDA_OBJECT})

# Linking
target_link_libraries( ${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} -lcublas)