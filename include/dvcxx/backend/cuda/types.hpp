
#pragma once
#include <memory>
#include "types_fwd.hpp"

namespace dvcxx {
namespace cuda  {

namespace detail {

  struct event_pimpl;
  struct stream_pimpl;
  struct pointer_attributes_pimpl;

} // namespace detail





class stream {

  friend event;
  using impl_ptr = std::shared_ptr< detail::stream_pimpl >;

  impl_ptr pimpl_;

  stream( const impl_ptr& );
  stream( impl_ptr&& ) noexcept;

public:

  stream();
  ~stream() noexcept;

  stream( const stream& );
  stream( stream&&      ) noexcept;

  void synchronize() const ;

};





class event {

  using impl_ptr = std::shared_ptr< detail::event_pimpl >;
  impl_ptr pimpl_;

  event( const impl_ptr& );
  event( impl_ptr&& ) noexcept;

public:

  event();
  ~event() noexcept;

  event( const event& );
  event( event&&      ) noexcept;

  void record( const stream& );
  void record( );

  void synchronize() const;

  static float elapsed_time( const event&, const event& );

};






struct pointer_attributes {

  using impl_ptr = std::shared_ptr< detail::pointer_attributes_pimpl >;
  impl_ptr pimpl_;

public:

  pointer_attributes() = delete;
  pointer_attributes( const void* ptr ); 
  ~pointer_attributes() noexcept;

  pointer_attributes( const pointer_attributes& );
  pointer_attributes( pointer_attributes&&      ) noexcept;

  bool is_device() const;
  bool is_host()   const;
  int device_id()  const;


};





} // namespace cuda
} // namespace dvcxx
