#pragma once
#include <vector>
#include <dvcxx/memory/allocator/sfinae.hpp>

namespace dvcxx {

template <typename T, typename Alloc = std::allocator<T>,
  typename = detail::enable_if_host_allocator_t<Alloc>
>
using host_vector = std::vector< T, Alloc >;

}
