#pragma once
#include <type_traits>

namespace dvcxx {
namespace detail {

template <typename T>
struct enable_if_trivially_copyable : public std::enable_if< std::is_trivially_copyable<T>::value > {
  using type = typename std::enable_if< std::is_trivially_copyable<T>::value >::type;
};

template <typename T>
using enable_if_trivially_copyable_t = typename enable_if_trivially_copyable<T>::type;





template <typename T>
struct enable_if_trivial : public std::enable_if< std::is_trivial<T>::value > {
  using type = typename std::enable_if< std::is_trivial<T>::value >::type;
};

template <typename T>
using enable_if_trivial_t = typename enable_if_trivial<T>::type;



}
}
