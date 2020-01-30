#include <dvcxx/container/vector/device_vector.hpp>
#include <dvcxx/memory/memory_resource.hpp>
#include <dvcxx/backend/cuda/types.hpp>
#include <iostream>

int main() {


  //auto* r = dvcxx::pmr::device_malloc_free_resource();
  auto* r = dvcxx::pmr::unified_malloc_free_resource();
  dvcxx::pmr::device_segregated_memory_resource  sr( 2 * 1024, 128, r );

  //dvcxx::device_vector<double> v( 10 );
  //dvcxx::pmr::device_vector<double> v_pmr( 10, &r );

  dvcxx::pmr::device_vector<double> v1_smr( 10, &sr );
  dvcxx::pmr::device_vector<double> v2_smr( 10, &sr );
  dvcxx::pmr::device_vector<double> v3_smr( 10, &sr );

  dvcxx::cuda::pointer_attributes att( v1_smr.data() );
  std::cout << std::boolalpha << att.is_host() << std::endl;
  std::cout << std::boolalpha << att.is_device() << std::endl;
}
