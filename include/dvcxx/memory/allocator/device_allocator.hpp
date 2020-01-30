#pragma once
#include "no_construct_allocator.hpp"
#include <dvcxx/env.hpp>

namespace dvcxx {

/**
 *  \brief Allocator for device accessable memory.
 *
 *  Stateless type which satisfies the Allocator concept for device accessable
 *  memory. Utilizes the device_backend namespace alias to deduce malloc/free 
 *  functions.
 *
 *  As device memory is often not accessable (dereferenceable) from the host,
 *  this type inherits from no_construct_allocator_base.
 *
 *  @tparam T Type to be allocated. See no_construct_allocator_base for
 *  for requirements on T.
 */
template <typename T>
struct device_allocator : public no_construct_allocator_base<T> {

  using value_type = typename no_construct_allocator_base<T>::value_type;

  /**
   *  \brief Allocate space for a specified number of T objects on
   *  the device.
   *
   *  @param[in] n Number of objects for which to allocate space.
   *  @returns Device pointer to allocated memory.
   */
  T* allocate( size_t n ) {
    return device_backend::device_malloc<T>( n );
  }

  /**
   *  \brief Deallocate a previously allocated device memory block.
   *
   *  @param[in] ptr Device pointer of memory block to deallocate.
   *  @param[in] n   Length of this memory block in terms of sizeof(T). [unused].
   */
  void deallocate( T* ptr, size_t n ) {
    device_backend::device_free( ptr );
  }

  // Stateless type -> trivial construct / copy / move
  device_allocator( )                         noexcept = default;
  device_allocator( const device_allocator& ) noexcept = default;
  device_allocator( device_allocator&& )      noexcept = default;
  ~device_allocator( )                        noexcept = default;

}; // struct device_allocator

} // namespace dvcxx
