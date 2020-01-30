#include <dvcxx/backend/cuda/api.hpp>
#include <dvcxx/backend/cuda/exceptions.hpp>
#include <iostream>

namespace dvcxx    {
namespace cuda     {
namespace wrappers {

void memcpy( void* dest, const void* src, size_t len ) {

  CUDA_THROW( cudaMemcpy( dest, src, len, cudaMemcpyDefault ) );

}





void memcpy2d( void* dest, size_t dpitch, const void* src, size_t spitch,
  size_t width, size_t height ) {

  CUDA_THROW( cudaMemcpy2D( dest, dpitch, src, spitch, width, height,
    cudaMemcpyDefault ) );

}





void* malloc( size_t len ) {

  void* ptr;
  CUDA_THROW( cudaMalloc( &ptr, len ) );
  std::cout << "CUDA MALLOC " << len << ", " << ptr << std::endl;
  return ptr;

}





void* malloc_pinned( size_t len ) {

  void* ptr;
  CUDA_THROW( cudaMallocHost( &ptr, len ) );
  //std::cout << "CUDA MALLOC HOST" << len << ", " << ptr << std::endl;
  return ptr;

}





void* malloc_unified( size_t len ) {

  void* ptr;
  CUDA_THROW( cudaMallocManaged( &ptr, len ) );
  std::cout << "CUDA MALLOC UNIFIED " << len << ", " << ptr << std::endl;
  return ptr;

}





void free( void* ptr ) {
  std::cout << "CUDA FREE " << ptr << std::endl;
  CUDA_THROW( cudaFree( ptr ) );
}





void  free_pinned( void* ptr ) {
  //std::cout << "CUDA FREE Host" << ptr << std::endl;
  CUDA_THROW( cudaFreeHost( ptr ) );
}





void memset( void* data, int val, size_t len ) {
  CUDA_THROW( cudaMemset( data, val, len ) );
}





void device_sync() {
  CUDA_THROW( cudaDeviceSynchronize() );
}

} // namespace wrappers
} // namespace cuda
} // namespace dvcxx
