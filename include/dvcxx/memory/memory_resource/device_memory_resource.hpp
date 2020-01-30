#pragma once
#if __has_include(<memory_resource>)
  #include <memory_resource>
#else
  #include <experimental/memory_resource>
#endif

namespace dvcxx::pmr {

class device_memory_resource {

protected:

  virtual void* do_allocate( std::size_t bytes, std::size_t alignment ) = 0;
  virtual void  do_deallocate( void* ptr, std::size_t bytes, std::size_t alignment ) = 0;
  virtual bool  do_is_equal( const device_memory_resource& other ) const noexcept = 0;

public:

  void* allocate( std::size_t bytes, std::size_t alignment = alignof(std::max_align_t) );
  void deallocate( void* ptr, std::size_t bytes, std::size_t alignment = alignof( std::max_align_t ) );
  bool is_equal( const device_memory_resource& other ) const noexcept;

};

} // namespace dvcxx::pmr

