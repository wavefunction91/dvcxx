#pragma once
#include <cstddef>
#include <list>

namespace dvcxx {

namespace detail {

  struct memory_block {
    void* top;
    std::size_t nbytes;
    std::size_t block_size;
    std::size_t nblocks;
  };

  struct memory_chunk {
    std::list< memory_block >::iterator parent_block;
    void* ptr;
  };

} // namespace detail





template <class UpstreamResource>
class segregated_memory_resource : public UpstreamResource {

  using memory_block = detail::memory_block;
  using memory_chunk = detail::memory_chunk;

  UpstreamResource* upstream_ = nullptr;

  std::list< memory_block > mem_blocks_;
  std::list< memory_chunk > free_list_;
  std::size_t               block_size_;


  void* do_allocate( std::size_t, std::size_t );
  void  do_deallocate( void*, std::size_t, std::size_t );
  bool  do_is_equal( const UpstreamResource& other ) const noexcept {
    return (void*)this == (void*)&other;
  }


  void add_block( std::list< memory_block >::iterator );

public:

  segregated_memory_resource() = delete;
  segregated_memory_resource( std::size_t, std::size_t,
    UpstreamResource* );
  segregated_memory_resource( std::size_t, std::size_t );

  ~segregated_memory_resource() noexcept;

  segregated_memory_resource( const segregated_memory_resource& ) = delete;
  segregated_memory_resource( segregated_memory_resource&& ) noexcept;


  auto        mem_blocks() const noexcept { return mem_blocks_; }
  auto        free_list()  const noexcept { return free_list_;  }
  std::size_t block_size() const noexcept { return block_size_; }

};

} // namespace dvcxx
