#pragma once
#include <dvcxx/util/sfinae.hpp>

namespace dvcxx {

/**
 *  \brief Base type for allocators which allocate typed raw memory 
 *  without construction/destruction
 *
 *  Provides a prototype for allocators which must allocate typed raw
 *  memory without construction / destruction (e.g. on device memory).
 *  no_constuct_allocator_base does not, in and of itself, satisfy the
 *  Allocator concept.
 *
 *  To satisfy the Allocator concept, the derived class must define
 *  the proper allocate / deallocate semantics oulined in the standard.
 *
 *  @tparam Type of the data to be allocated. Only valid for trivial
 *  types.
 */
template <typename T, typename = detail::enable_if_trivial_t<T> >
struct no_construct_allocator_base {

  using value_type = T;

  /**
   *  \brief Construction given an allocated pointer is a null-op.
   *
   *  @tparam Args Parameter pack which handles all possible calling
   *  signatures of construct outlined in the Allocator concept.
   */
  template <typename... Args>
  constexpr void construct( T* ptr, Args&&... args ){ };

  /**
   *  \brief Destruction of an object in allocated memory is a null-op
   */
  constexpr void destroy( T* ptr ){ }


  // Stateless type -> trivial construct / copy / move
  no_construct_allocator_base( )                                             noexcept = default;
  explicit no_construct_allocator_base( const no_construct_allocator_base& ) noexcept = default;
  explicit no_construct_allocator_base( no_construct_allocator_base&& )      noexcept = default;
  ~no_construct_allocator_base( )                                            noexcept = default;

}; // struct no_construct_allocator_base

} // namespace dvcxx
