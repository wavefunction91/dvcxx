add_library( dvcxx_cuda_backend api.cu types_pimpl.cu types.cu )

target_include_directories( dvcxx_cuda_backend PUBLIC ${PROJECT_SOURCE_DIR}/include )
target_compile_features(    dvcxx_cuda_backend PRIVATE cxx_std_14                   )
