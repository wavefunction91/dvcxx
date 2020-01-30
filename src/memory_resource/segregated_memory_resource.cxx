#include <dvcxx/memory/memory_resource/typedefs.hpp>
#include <algorithm>
#include <cassert>

namespace dvcxx {

namespace detail {

constexpr inline int64_t bytes_distance(
  void* first, void* last ) {
  return (char*)last - (char*)first;
}

inline size_t div_ceil( size_t n, size_t m ) {
  auto div = std::div( (long long)n, (long long)m );
  return div.quot + !!div.rem;
}

}

template <class UpstreamResource>
segregated_memory_resource<UpstreamResource>::segregated_memory_resource(
  std::size_t      nalloc,
  std::size_t      block_size,
  UpstreamResource* upstream
) : block_size_( block_size ), upstream_( upstream ) {

  std::size_t nblocks = nalloc / block_size_;
  std::size_t nbytes  = nblocks * block_size_;

  void* ptr = upstream_->allocate( nalloc, 8 );
  mem_blocks_.push_back(
    { ptr, nbytes, block_size_, nblocks }
  );

  add_block( mem_blocks_.begin() );

}

template <class UpstreamResource>
segregated_memory_resource<UpstreamResource>::~segregated_memory_resource()
  noexcept {

  for( auto& block : mem_blocks_ )
    upstream_->deallocate( block.top, block.nbytes, 8 );

}

template <class UpstreamResource>
segregated_memory_resource<UpstreamResource>::segregated_memory_resource(
  segregated_memory_resource&& other
) noexcept = default;


template <class UpstreamResource>
void segregated_memory_resource<UpstreamResource>::add_block(
  std::list< memory_block >::iterator block_it
) {

  // Add new blocks to the free list
  free_list_.push_back( { block_it, block_it->top } );
  for( std::size_t i = 1; i < block_it->nblocks; ++i ) {
    void* next_ptr =
      (unsigned char*)free_list_.back().ptr + block_size_;
    free_list_.push_back( { block_it, next_ptr } );
  }

  // Make sure that the free list is sorted
  free_list_.sort([](const auto& a, const auto& b) {
    return (unsigned char*)a.ptr < (unsigned char*)b.ptr;
  });

}


template <class UpstreamResource>
void* segregated_memory_resource<UpstreamResource>::do_allocate(
  std::size_t nbytes,
  std::size_t alignment
) {

  size_t nblocks = detail::div_ceil( nbytes, block_size_ );

  // Search for free memory block consisting of nblocks
  // contiguous memory chunks

  auto   ptr_it        = free_list_.begin();
  size_t ncontig_found = 1;

  using detail::bytes_distance;
  auto cur_it = ptr_it;
  while( ncontig_found != nblocks and
         cur_it != free_list_.end() ) {

    auto next_it = std::next( cur_it );
    if( next_it != free_list_.end() ) {

      auto dist = bytes_distance( cur_it->ptr, next_it->ptr );
      if( dist == block_size_ ) ncontig_found++;
      else {
        ptr_it = next_it;
        ncontig_found = 1;
      }

    }
    cur_it++;

  }

  void* ptr_return = nullptr;
  if( ncontig_found == nblocks ) {
    // A proper block was found
    ptr_return = ptr_it->ptr;

    // Remove it from the free list
    auto end_ptr_it = std::next( ptr_it, nblocks );
    free_list_.erase( ptr_it, end_ptr_it );
  } else throw std::bad_alloc();

  return ptr_return;

}


template <class UpstreamResource>
void segregated_memory_resource<UpstreamResource>::do_deallocate(
  void*       ptr,
  std::size_t nbytes,
  std::size_t alignment
) {

  unsigned char* ptr_as_char = (unsigned char*)ptr;

  // Find ptr in block
  auto mem_block_it =
    std::find_if( mem_blocks_.begin(), mem_blocks_.end(),
      [ptr_as_char]( const auto& blk ){
        unsigned char* top_as_char = (unsigned char*)blk.top;
        return ptr_as_char >= top_as_char and
               ptr_as_char  < top_as_char + blk.nbytes;
      });

  if( mem_block_it == mem_blocks_.end() ) {
    throw std::runtime_error("POOL ATTEMPTING TO FREE A POINTER NOT WITHIN MEMORY BLOCK");
    //throw std::out_of_range("POOL ATTEMPTING TO FREE A POINTER NOT WITHIN MEMORY BLOCK");
    //throw std::bad_alloc();
  }


  size_t nblocks = detail::div_ceil( nbytes, block_size_ );
  std::list< memory_chunk > ptrs_to_add;
  for(int i = 0; i < nblocks; ++i)
    ptrs_to_add.push_back(
      {mem_block_it, (unsigned char*)ptr + i*block_size_}
    );

#ifndef NDEBUG
  assert(
    std::is_sorted( free_list_.begin(), free_list_.end(),
      []( const auto& a, const auto& b ) {
        return a.ptr < b.ptr;
      })
  );

  // CHECK if frees are the right size
  for( auto &x : ptrs_to_add )
    if( std::find_if( free_list_.begin(), free_list_.end(),
          [&x]( const auto& chunk ) { return x.ptr == chunk.ptr; }) !=
          free_list_.end() )
      throw
        std::runtime_error( "Attempting to free pointer already in the free list" );
#endif


  // Find first free pointer larger than inserted block
  auto next_ptr =
    std::find_if( free_list_.begin(), free_list_.end(),
      [ptr]( const auto& chunk ){
        return chunk.ptr > ptr;
      });

  // Insert into the free list
  free_list_.splice( next_ptr, ptrs_to_add );

}

template class segregated_memory_resource< pmr::device_memory_resource >;
template class segregated_memory_resource< pmr::host_memory_resource >;

} // namespace dvcxx
