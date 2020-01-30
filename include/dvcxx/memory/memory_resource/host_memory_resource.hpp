#pragma once
#if __has_include(<memory_resource>)
  #include <memory_resource>
#else
  #include <experimental/memory_resource>
#endif

namespace dvcxx::pmr {

#if __has_include(<memory_resource>)
using host_memory_resource = std::pmr::memory_resource;
#else
using host_memory_resource = std::experimental::pmr::memory_resource;
#endif

} // namespace dvcxx::pmr
