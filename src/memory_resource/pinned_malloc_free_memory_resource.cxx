#include <dvcxx/memory/memory_resource/pinned_malloc_free_memory_resource.hpp>
#include <dvcxx/env.hpp>

namespace dvcxx::pmr {

using pmfmr = pinned_malloc_free_memory_resource;

void* pmfmr::do_allocate( std::size_t bytes, std::size_t alignment ) {

  return (void*)device_backend::pinned_malloc<std::byte>( bytes );

}

void pmfmr::do_deallocate( void* ptr, std::size_t bytes, std::size_t alignment ) {

  device_backend::pinned_free( ptr );

}

bool pmfmr::do_is_equal( const host_memory_resource& other ) const noexcept {

  bool is_eq = false;
  try {
    const pmfmr& temp = dynamic_cast< const pmfmr& >(other);
    is_eq = true;
  } catch( const std::bad_cast& ){
    is_eq = false;
  }

  return is_eq;

}

}
