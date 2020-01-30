#pragma once
#include <iostream>
#include <dvcxx/backend/cuda/types.hpp>

namespace dvcxx  {
namespace cuda   {
namespace detail {

  struct event_pimpl {
    
    cudaEvent_t event;

  };

  struct managed_event_pimpl : public event_pimpl {

    managed_event_pimpl();
    ~managed_event_pimpl() noexcept;

  };





  struct stream_pimpl {

    cudaStream_t stream;

  };

  struct managed_stream_pimpl : public stream_pimpl {

    managed_stream_pimpl();
    ~managed_stream_pimpl() noexcept;

  };





  struct pointer_attributes_pimpl {

    cudaPointerAttributes attributes;
    const void*           ptr;

    pointer_attributes_pimpl( const void* p );
    ~pointer_attributes_pimpl() noexcept = default;

  };

} // namespace detail
} // namespace cuda
} // namespace dvcxx
