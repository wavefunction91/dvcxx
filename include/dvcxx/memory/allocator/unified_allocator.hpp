#pragma once
#include "no_construct_allocator.hpp"
#include <dvcxx/env.hpp>

namespace dvcxx {

/**
 *  \brief Allocator for memory accessable from both host and device (unified/unified).
 *
 *  Stateless type which satisfies the Allocator concept for memory which
 *  is accessable from both host and device. Utilizes the device_backend 
 *  namespace alias to deduce malloc/free functions.
 *
 *  @tparam T Type to be allocated. See no_construct_allocator_base for
 *  for requirements on T. TODO: See if this is actually a requirement
 */
template <typename T>
struct unified_allocator : public no_construct_allocator_base<T> {

  using value_type = typename no_construct_allocator_base<T>::value_type;

  /**
   *  \brief Allocate space for a specified number of T objects.
   *
   *  @param[in] n Number of objects for which to allocate space.
   *  @returns Managed pointer to allocated memory.
   */
  T* allocate( size_t n ) {
    return device_backend::unified_malloc<T>( n );
  }

  /**
   *  \brief Deallocate a previously allocated unified memory block.
   *
   *  @param[in] ptr Device pointer of memory block to deallocate.
   *  @param[in] n   Length of this memory block in terms of sizeof(T). [unused].
   */
  void deallocate( T* ptr, size_t n ) {
    device_backend::unified_free( ptr );
  }


  // Stateless type -> trivial construct / copy / move
  unified_allocator( )                          noexcept = default;
  unified_allocator( const unified_allocator& ) noexcept = default;
  unified_allocator( unified_allocator&& )      noexcept = default;
  ~unified_allocator( )                         noexcept = default;

}; // struct unified_allocator

} // namespace dvcxx
