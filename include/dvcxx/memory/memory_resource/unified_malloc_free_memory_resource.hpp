#pragma once
#include "device_memory_resource.hpp"
#include "host_memory_resource.hpp"

namespace dvcxx::pmr {

class unified_malloc_free_memory_resource : 
  public device_memory_resource,
  public host_memory_resource {

  void* do_allocate( std::size_t bytes, std::size_t alignment );
  void  do_deallocate( void* ptr, std::size_t bytes, std::size_t alignment );
  bool  do_is_equal( const device_memory_resource& other ) const noexcept;
  bool  do_is_equal( const host_memory_resource& other ) const noexcept;

  unified_malloc_free_memory_resource() noexcept = default;

public:

  //device_memory_resource& as_device_memory_resource() noexcept; 
  //const device_memory_resource& as_device_memory_resource() const noexcept;

  //host_memory_resource& as_host_memory_resource() noexcept;
  //const host_memory_resource& as_host_memory_resource() const noexcept;

  unified_malloc_free_memory_resource( const unified_malloc_free_memory_resource& )         = delete;
  unified_malloc_free_memory_resource( unified_malloc_free_memory_resource&&     ) noexcept = delete;

  void operator=( const unified_malloc_free_memory_resource& )         = delete;
  void operator=( unified_malloc_free_memory_resource&&     ) noexcept = delete;

  static unified_malloc_free_memory_resource* get_instance() noexcept;
};

unified_malloc_free_memory_resource* unified_malloc_free_resource();

}

