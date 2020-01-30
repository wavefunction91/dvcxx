#include <dvcxx/container/vector/device_vector.hpp>
#include <dvcxx/memory/memory_resource.hpp>
#include <dvcxx/backend/cuda/types.hpp>
#include <iostream>

int main() {


  //auto* r = dvcxx::pmr::device_malloc_free_resource();
  //auto* r = dvcxx::pmr::unified_malloc_free_resource();
  dvcxx::pmr::device_segregated_memory_resource  sr( 2 * 1024, 128 );

  dvcxx::pmr::set_default_device_resource( &sr );

  //dvcxx::device_vector<double> v( 10 );
  //dvcxx::pmr::device_vector<double> v_pmr( 10, &r );

  dvcxx::pmr::device_vector<double> v1_smr( 10 );
  dvcxx::pmr::device_vector<double> v2_smr( 32 );
  dvcxx::pmr::device_vector<double> v3_smr( 10 );

  std::cout << v1_smr.data() << std::endl;
  std::cout << v2_smr.data() << ", " << (std::distance( v1_smr.data(), v2_smr.data() ) * sizeof(double)) / sr.block_size() << std::endl;
  std::cout << v3_smr.data() << ", " << (std::distance( v2_smr.data(), v3_smr.data() ) * sizeof(double)) / sr.block_size() << std::endl;

  dvcxx::cuda::pointer_attributes att( v1_smr.data() );
  std::cout << std::boolalpha << att.is_host() << std::endl;
  std::cout << std::boolalpha << att.is_device() << std::endl;
}
