#include <dvcxx/memory/allocator/device_polymorphic_allocator.hpp>
#include <dvcxx/memory/memory_resource.hpp>
#include <iostream>

int main() {

  auto r = dvcxx::pmr::device_malloc_free_resource();
  dvcxx::pmr::device_polymorphic_allocator<double> alloc( r );

  double* ptr = alloc.allocate( 10 );
  alloc.deallocate( ptr, 10 );

}
