#include <dvcxx/memory/memory_resource/device_memory_resource.hpp>

namespace dvcxx::pmr {


void* device_memory_resource::allocate( std::size_t bytes, std::size_t alignment ) {
  return do_allocate( bytes, alignment );
}
void  device_memory_resource::deallocate( void* ptr, std::size_t bytes, std::size_t alignment ) {
  do_deallocate( ptr, bytes, alignment );
}
bool  device_memory_resource::is_equal( const device_memory_resource& other ) const noexcept {
  return do_is_equal( other );
}

}

