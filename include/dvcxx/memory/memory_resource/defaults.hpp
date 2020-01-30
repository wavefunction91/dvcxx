#pragma once

namespace dvcxx::pmr {

device_memory_resource* get_default_device_resource() noexcept;
void set_default_device_resource(device_memory_resource* r);


} // namespace dvcxx::pmr
