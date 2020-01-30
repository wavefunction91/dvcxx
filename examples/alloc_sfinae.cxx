#include <dvcxx/memory/allocator/sfinae.hpp>
#include <iostream>

int main() {

  std::cout << std::boolalpha;

  std::cout << "DEVICE ALLOCATOR TESTS" << std::endl;

  std::cout << "device_allocator:  ";
  std::cout << dvcxx::detail::is_device_allocator_v< dvcxx::device_allocator<int> > << std::endl;
  std::cout << "unified_allocator: ";
  std::cout << dvcxx::detail::is_device_allocator_v< dvcxx::unified_allocator<int> > << std::endl;
  std::cout << "pinned_allocator:  ";
  std::cout << dvcxx::detail::is_device_allocator_v< dvcxx::pinned_allocator<int> > << std::endl;
  std::cout << "std::allocator:    ";
  std::cout << dvcxx::detail::is_device_allocator_v< std::allocator<int> > << std::endl;



  std::cout << "HOST ALLOCATOR TESTS" << std::endl;

  std::cout << "device_allocator:  ";
  std::cout << dvcxx::detail::is_host_allocator_v< dvcxx::device_allocator<int> > << std::endl;
  std::cout << "unified_allocator: ";
  std::cout << dvcxx::detail::is_host_allocator_v< dvcxx::unified_allocator<int> > << std::endl;
  std::cout << "pinned_allocator:  ";
  std::cout << dvcxx::detail::is_host_allocator_v< dvcxx::pinned_allocator<int> > << std::endl;
  std::cout << "std::allocator:    ";
  std::cout << dvcxx::detail::is_host_allocator_v< std::allocator<int> > << std::endl;
}
