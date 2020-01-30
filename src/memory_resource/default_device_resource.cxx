#include <dvcxx/memory/memory_resource/device_malloc_free_memory_resource.hpp>
//#include "unified_malloc_free_memory_resource.hpp"
//#include "pinned_malloc_free_memory_resource.hpp"

namespace dvcxx::pmr {

namespace detail {


class default_device_memory_resource {

  device_memory_resource* resource_ = nullptr;

  default_device_memory_resource() noexcept = default;

public:

  default_device_memory_resource( const default_device_memory_resource& )         = delete;
  default_device_memory_resource( default_device_memory_resource&&     ) noexcept = delete;

  void operator=( const default_device_memory_resource& )         = delete;
  void operator=( default_device_memory_resource&&     ) noexcept = delete;

  static default_device_memory_resource& get_instance() noexcept;
  
  void set_resource( device_memory_resource* r );
  device_memory_resource* get_resource() const noexcept;

};






using ddmr = default_device_memory_resource;

ddmr& ddmr::get_instance() noexcept {
  static ddmr dr;
  if( !dr.get_resource() )
    dr.set_resource( device_malloc_free_resource() );
  return dr;
}

void ddmr::set_resource( device_memory_resource* r ) {
  resource_ = r;
}

device_memory_resource* ddmr::get_resource() const noexcept {
  return resource_;
}





}


device_memory_resource* get_default_device_resource() noexcept {
  return detail::default_device_memory_resource::get_instance().get_resource();
}

void set_default_device_resource(device_memory_resource* r) {
  detail::default_device_memory_resource::get_instance().set_resource( r );
}

}
