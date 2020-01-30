#include <dvcxx/container/vector/device_vector.hpp>
#include <dvcxx/memory/memory_resource.hpp>

int main() {


  //dvcxx::pmr::device_malloc_free_memory_resource r;
  dvcxx::pmr::unified_malloc_free_memory_resource r;
  dvcxx::pmr::device_segregated_memory_resource  sr( 2 * 1024, 128, &r );

  //dvcxx::device_vector<double> v( 10 );
  //dvcxx::pmr::device_vector<double> v_pmr( 10, &r );

  dvcxx::pmr::device_vector<double> v1_smr( 10, &sr );
  dvcxx::pmr::device_vector<double> v2_smr( 10, &sr );
  dvcxx::pmr::device_vector<double> v3_smr( 10, &sr );
}
