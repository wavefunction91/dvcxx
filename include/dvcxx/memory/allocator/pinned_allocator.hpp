#pragma once
#include "no_construct_allocator.hpp"
#include <dvcxx/env.hpp>

namespace dvcxx {

/**
 *  \brief Allocator for page-locked host memory.
 */
template <typename T>
struct pinned_allocator {

  using value_type = T;

  T* allocate( size_t n ) {
    return device_backend::pinned_malloc<T>( n );
  }
  void deallocate( T* ptr, size_t n ) {
    device_backend::pinned_free( ptr );
  }

  // Construct and Destroy get handled by std::allocator_traits

  // Stateless type -> trivial construct / copy / move
  pinned_allocator( )                         noexcept = default;
  pinned_allocator( const pinned_allocator& ) noexcept = default;
  pinned_allocator( pinned_allocator&& )      noexcept = default;
  ~pinned_allocator( )                        noexcept = default;

}; // struct pinned_allocator

} // namespace dvcxx
