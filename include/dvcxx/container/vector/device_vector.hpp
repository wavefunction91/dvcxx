#pragma once
#include <vector>
#include "host_vector.hpp"

#include <dvcxx/env.hpp>

namespace dvcxx {

template <typename T, typename Alloc = device_allocator<T>,
  typename = detail::enable_if_device_allocator_t<Alloc> >
class device_vector : public std::vector<T, Alloc> {

  using base = std::vector<T, Alloc>;

public:

  using size_type      = typename base::size_type;
  using allocator_type = typename base::allocator_type;

  // Inherit iterator from std::vector
  struct iterator : public base::iterator { 
    template <typename... Args>
    iterator( Args&&... args ) : base::iterator( std::forward<Args>(args)... ){};
  };

  using const_iterator = const iterator;

  // Inherit basic construction semantics from std::vector

  device_vector() : base() { }
  explicit device_vector( const allocator_type& alloc ) noexcept : base(alloc) { }

  explicit device_vector( size_type n, const T& val, 
    const allocator_type& alloc = allocator_type() ) : base( n, val, alloc ) { }

  explicit device_vector( size_type n, 
    const allocator_type& alloc = allocator_type() ) : base( n, alloc ) { }

  explicit device_vector( device_vector&& other ) noexcept : base( std::move(other) ) { }


  // Copy semantics are special

  explicit device_vector( const device_vector& other ) : 
    device_vector( other.size() ) /* Allocate device memory */ {

    if( this->size() )
      device_backend::memcpy( this->data(), other.data(), this->size() );

  }

  template <typename OtherAlloc>
  explicit device_vector( const host_vector<T, OtherAlloc>& other ) :
    device_vector(other.size()) /* Allocate device memory */ {

    // Copy to device
    if( this->size() )
      device_backend::memcpy( this->data(), other.data(), this->size() ); 

  }



  device_vector& operator=( const device_vector& other ) {

    this->resize( other.size() ); // Allocate device memory if need be

    // Copy on device
    if( this->size() )
      device_backend::memcpy( this->data(), other.data(), this->size() );

    return *this;

  }

  template <typename OtherAlloc>
  device_vector& operator=( const host_vector<T, OtherAlloc>& other ) {

    this->resize( other.size() ); // Allocate device memory if need be

    // Copy on device
    if( this->size() )
      device_backend::memcpy( this->data(), other.data(), this->size() );

    return *this;

  }



  iterator begin(){ return iterator( this->data() ); }
  iterator end()  { return iterator( this->data() + this->size() ); }

};

namespace pmr {
  template <typename T>
  using device_vector = dvcxx::device_vector<T,device_polymorphic_allocator<T> >;
}

} // namespace dvcxx
