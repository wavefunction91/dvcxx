#pragma once
#include "host_memory_resource.hpp"

namespace dvcxx::pmr {

class pinned_malloc_free_memory_resource : public host_memory_resource {

  void* do_allocate( std::size_t bytes, std::size_t alignment );
  void  do_deallocate( void* ptr, std::size_t bytes, std::size_t alignment );
  bool  do_is_equal( const host_memory_resource& other ) const noexcept;

  pinned_malloc_free_memory_resource() noexcept = default;

public:

  pinned_malloc_free_memory_resource( const pinned_malloc_free_memory_resource& )         = delete;
  pinned_malloc_free_memory_resource( pinned_malloc_free_memory_resource&&     ) noexcept = delete;

  void operator=( const pinned_malloc_free_memory_resource& )         = delete;
  void operator=( pinned_malloc_free_memory_resource&&     ) noexcept = delete;

  static pinned_malloc_free_memory_resource* get_instance() noexcept {
    static pinned_malloc_free_memory_resource r;
    return &r;
  }

};

host_memory_resource* pinned_malloc_free_resource();

}

