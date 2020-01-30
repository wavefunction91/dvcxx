#include <dvcxx/memory/memory_resource/device_malloc_free_memory_resource.hpp>
#include <dvcxx/env.hpp>

namespace dvcxx::pmr {

using dmfmr = device_malloc_free_memory_resource;

void* dmfmr::do_allocate( std::size_t bytes,
  std::size_t alignment ) {

  return (void*)device_backend::device_malloc<std::byte>( bytes );

}

void dmfmr::do_deallocate( void* ptr, std::size_t bytes, std::size_t alignment ) {

  device_backend::device_free( ptr );

}

bool dmfmr::do_is_equal( const device_memory_resource& other ) const noexcept {

  bool is_eq = false;
  try {
    const dmfmr& temp = dynamic_cast< const dmfmr& >(other);
    is_eq = true;
  } catch( const std::bad_cast& ){
    is_eq = false;
  }

  return is_eq;

}





device_memory_resource* device_malloc_free_resource() {
  return static_cast<device_memory_resource*>(device_malloc_free_memory_resource::get_instance());
}



}
