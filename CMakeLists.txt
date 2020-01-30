cmake_minimum_required( VERSION 3.14 FATAL_ERROR )
project( dvcxx VERSION 0.0.0 LANGUAGES C CXX CUDA )

add_subdirectory( src )

add_library( dvcxx INTERFACE IMPORTED )
target_link_libraries( dvcxx INTERFACE dvcxx_memory_resource dvcxx_cuda_backend )

add_subdirectory( examples )
