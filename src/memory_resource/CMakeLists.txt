add_library( dvcxx_memory_resource
  device_memory_resource.cxx
  device_malloc_free_memory_resource.cxx
  unified_malloc_free_memory_resource.cxx
  pinned_malloc_free_memory_resource.cxx
  segregated_memory_resource.cxx
)
target_include_directories( dvcxx_memory_resource PUBLIC ${PROJECT_SOURCE_DIR}/include )
target_compile_features(    dvcxx_memory_resource PUBLIC cxx_std_17                    )
