#pragma once

#include "device_memory_resource.hpp"
#include "host_memory_resource.hpp"
#include "segregated_memory_resource.hpp"

namespace dvcxx {

extern template class segregated_memory_resource< pmr::device_memory_resource >;
extern template class segregated_memory_resource< pmr::host_memory_resource   >;

namespace pmr {

using device_segregated_memory_resource = 
  segregated_memory_resource< device_memory_resource >;

using host_segregated_memory_resource = 
  segregated_memory_resource< host_memory_resource >;


}

}
