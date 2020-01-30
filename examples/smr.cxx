#include <dvcxx/memory/allocator/device_polymorphic_allocator.hpp>
#include <dvcxx/memory/memory_resource.hpp>
#include <iostream>

int main() {

  dvcxx::pmr::device_malloc_free_memory_resource r;
  dvcxx::pmr::device_segregated_memory_resource  sr( 2 * 1024, 128, &r );
  dvcxx::pmr::device_polymorphic_allocator<double> alloc( &sr );

  double* ptr1 = alloc.allocate( 33 );
  double* ptr2 = alloc.allocate( 10 );

  std::cout << ptr1 << std::endl << ptr2 << std::endl;
  std::cout << (std::distance( ptr1, ptr2 ) * 8)/128 << std::endl;

  alloc.deallocate( ptr2, 10 );
  alloc.deallocate( ptr1, 10 );

}
