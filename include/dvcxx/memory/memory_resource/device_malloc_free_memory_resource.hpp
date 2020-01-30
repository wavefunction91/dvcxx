#pragma once
#include "device_memory_resource.hpp"

namespace dvcxx::pmr {

class device_malloc_free_memory_resource : public device_memory_resource {

  void* do_allocate( std::size_t bytes, std::size_t alignment );
  void  do_deallocate( void* ptr, std::size_t bytes, std::size_t alignment );
  bool  do_is_equal( const device_memory_resource& other ) const noexcept;

};

}
