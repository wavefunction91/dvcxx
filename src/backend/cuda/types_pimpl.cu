#include "types_pimpl.hpp"
#include <dvcxx/backend/cuda/exceptions.hpp>

namespace dvcxx  {
namespace cuda   {
namespace detail {


managed_event_pimpl::managed_event_pimpl() {
  CUDA_THROW( cudaEventCreate( &this->event ) );
}

managed_event_pimpl::~managed_event_pimpl() noexcept {
  CUDA_ASSERT( cudaEventDestroy( this->event ) );
}





managed_stream_pimpl::managed_stream_pimpl() {
  CUDA_THROW( cudaStreamCreate( &this->stream ) );
}

managed_stream_pimpl::~managed_stream_pimpl() noexcept {
  CUDA_ASSERT( cudaStreamDestroy( this->stream ) );
}





pointer_attributes_pimpl::pointer_attributes_pimpl( const void* p ) : ptr( p ) {

  auto ret = cudaPointerGetAttributes( &attributes, ptr );
  if( ret != cudaSuccess and ret != cudaErrorInvalidValue )
    CUDA_THROW( ret )
  else if( ret == cudaErrorInvalidValue )
    attributes.type = cudaMemoryTypeUnregistered;

}




} // namespace detail
} // namespace cuda
} // namespace dvcxx
