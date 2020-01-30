#pragma once

#include "device_allocator.hpp"
#include "device_polymorphic_allocator.hpp"
#include "unified_allocator.hpp"
#include "pinned_allocator.hpp"

#include <dvcxx/util/sfinae.hpp>

namespace dvcxx  {
namespace detail {

template <typename Alloc, typename = std::void_t<>>
struct is_device_allocator : public std::false_type { };

template <typename T>
struct is_device_allocator< device_allocator<T> > :
  public std::true_type { };

template <typename T>
struct is_device_allocator< unified_allocator<T> > :
  public std::true_type { };

template <typename T>
struct is_device_allocator< 
  pmr::device_polymorphic_allocator<T> 
> : public std::true_type { };


template <typename Alloc>
inline constexpr bool is_device_allocator_v =
  is_device_allocator<Alloc>::value;

template <typename Alloc>
struct enable_if_device_allocator : 
  public std::enable_if< is_device_allocator_v<Alloc> > {

  using type = typename std::enable_if_t<
    is_device_allocator_v<Alloc>
  >;

};

template <typename Alloc>
using enable_if_device_allocator_t =
  typename enable_if_device_allocator< Alloc >::type;



template <typename Alloc, typename = std::void_t<>>
struct is_host_allocator : public std::false_type { };

template <typename Alloc>
struct is_host_allocator< Alloc, 
  std::enable_if_t<
    not is_device_allocator_v<Alloc> and
    not std::is_same_v< Alloc, unified_allocator< typename Alloc::value_type > >
  >
> : public std::true_type { };

template <typename T>
struct is_host_allocator< unified_allocator<T> > :
  public std::true_type { };

template <typename Alloc>
inline constexpr bool is_host_allocator_v =
  is_host_allocator<Alloc>::value;

template <typename Alloc>
struct enable_if_host_allocator : 
  public std::enable_if< is_host_allocator_v<Alloc> > {

  using type = typename std::enable_if_t<
    is_host_allocator_v<Alloc>
  >;

};

template <typename Alloc>
using enable_if_host_allocator_t =
  typename enable_if_host_allocator< Alloc >::type;

}


}
