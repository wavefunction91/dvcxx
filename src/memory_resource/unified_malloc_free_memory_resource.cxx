#include <dvcxx/memory/memory_resource/unified_malloc_free_memory_resource.hpp>
#include <dvcxx/env.hpp>


namespace dvcxx::pmr {

using umfmr = unified_malloc_free_memory_resource;

void* umfmr::do_allocate( std::size_t bytes, std::size_t alignment ) {

  return (void*)device_backend::unified_malloc<std::byte>( bytes );

}

void umfmr::do_deallocate( void* ptr, std::size_t bytes, std::size_t alignment ) {

  device_backend::unified_free( ptr );

}

bool umfmr::do_is_equal( const device_memory_resource& other ) const noexcept {

  bool is_eq = false;
  try {
    const umfmr& temp = dynamic_cast< const umfmr& >(other);
    is_eq = true;
  } catch( const std::bad_cast& ){
    is_eq = false;
  }

  return is_eq;

}



bool umfmr::do_is_equal( const host_memory_resource& other ) const noexcept {

  bool is_eq = false;
  try {
    const umfmr& temp = dynamic_cast< const umfmr& >(other);
    is_eq = true;
  } catch( const std::bad_cast& ){
    is_eq = false;
  }

  return is_eq;

}






device_memory_resource& umfmr::as_device_memory_resource() noexcept {

  return static_cast<device_memory_resource&>(*this);

}

host_memory_resource& umfmr::as_host_memory_resource() noexcept {

  return static_cast<host_memory_resource&>(*this);

}

}
