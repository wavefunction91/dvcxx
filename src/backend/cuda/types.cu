#include <iostream>
#include "types_pimpl.hpp"
#include <dvcxx/backend/cuda/exceptions.hpp>

namespace dvcxx {
namespace cuda  {

event::event() :
  pimpl_( std::make_shared<detail::managed_event_pimpl>() ) { }

event::event( const std::shared_ptr<detail::event_pimpl>& p ) :
  pimpl_( p ) { }

event::event( std::shared_ptr<detail::event_pimpl>&& p ) noexcept :
  pimpl_( std::move(p) ) { }

event::~event()              noexcept = default;
event::event( const event& )          = default;
event::event( event&&      ) noexcept = default;





void event::record( const stream& stream ) {
  CUDA_THROW( cudaEventRecord( pimpl_->event, stream.pimpl_->stream ) );
}

void event::record() {
  CUDA_THROW( cudaEventRecord( pimpl_->event ) );
}

void event::synchronize() const {
  CUDA_THROW( cudaEventSynchronize( pimpl_->event ) );
}

float event::elapsed_time( const event& first, const event& second ) {
  float time;
  CUDA_THROW( cudaEventElapsedTime( &time, first.pimpl_->event, 
    second.pimpl_->event ) );
  return time;
}





stream::stream() :
  pimpl_( std::make_shared<detail::managed_stream_pimpl>() ) { }

stream::stream( const std::shared_ptr<detail::stream_pimpl>& p ) :
  pimpl_( p ) { }

stream::stream( std::shared_ptr<detail::stream_pimpl>&& p ) noexcept :
  pimpl_( std::move(p) ) { }

stream::~stream()              noexcept = default;
stream::stream( const stream& )          = default;
stream::stream( stream&&      ) noexcept = default;





void stream::synchronize() const {
  CUDA_THROW( cudaStreamSynchronize( pimpl_->stream ) );
}






pointer_attributes::pointer_attributes( const void* ptr ) :
  pimpl_( std::make_shared<detail::pointer_attributes_pimpl>( ptr ) ){ }

pointer_attributes::~pointer_attributes( ) = default;



bool pointer_attributes::is_device() const {
  return pimpl_->attributes.type == cudaMemoryTypeDevice or 
         pimpl_->attributes.type == cudaMemoryTypeManaged;
}

bool pointer_attributes::is_host() const {
  return pimpl_->attributes.type == cudaMemoryTypeHost or 
         pimpl_->attributes.type == cudaMemoryTypeUnregistered or 
         pimpl_->attributes.type == cudaMemoryTypeManaged;
}

int pointer_attributes::device_id() const {
  if( is_device() ) return pimpl_->attributes.device;
  else              return -1;
}

} // namespace cuda
} // namespace dvcxx
