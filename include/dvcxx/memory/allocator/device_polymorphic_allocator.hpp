#pragma once
#include <dvcxx/memory/memory_resource/device_memory_resource.hpp>
#include "no_construct_allocator.hpp"

namespace dvcxx::pmr {

template <typename T = std::byte>
class device_polymorphic_allocator : public no_construct_allocator_base<T> {

  using dmr = device_memory_resource;

  dmr* resource_ = nullptr;

public:

  using value_type = T;

  device_polymorphic_allocator( ) noexcept = delete;
  device_polymorphic_allocator( dmr* r ) : resource_(r) { }

  device_polymorphic_allocator( const device_polymorphic_allocator& )
    noexcept = default;
  device_polymorphic_allocator( device_polymorphic_allocator&& )
    noexcept = default;
  ~device_polymorphic_allocator( ) noexcept = default;


  T* allocate( std::size_t n ) { 
    return static_cast<T*>(resource_->allocate( n * sizeof(T) ));
  }

  void deallocate( T* ptr, std::size_t n ) { 
    resource_->deallocate( (void*)ptr, n * sizeof(T) ); 
  }

  dmr* resource() const noexcept { return resource_; }

};

}
