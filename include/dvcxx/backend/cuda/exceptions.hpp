#pragma once

#include <exception>
#include <cassert>
#include <string>

//#define CUDA_THROW_AS_ASSERT

#define CUDA_ASSERT(err)  { assert( err == cudaSuccess ); }


#ifdef CUDA_THROW_AS_ASSERT
  #define CUDA_THROW(err) CUDA_ASSERT(err);
#else
  #define CUDA_THROW(err)  { if(err != cudaSuccess) throw cuda::exception( std::string(cudaGetErrorString(err)) + std::string(" @ Line: " + std::string(__FILE__)) + std::to_string(__LINE__) );           }
#endif

namespace dvcxx {
namespace cuda {

/**
 *  \brief CUDA Exception
 *
 *  Wraps the cudaError_t error code and translates to std::exception
 */
class exception : public std::exception {

  std::string message;

  virtual const char* what() const throw() {
    return message.c_str();
  }

public:

  exception( const char* msg ) : std::exception(), message( msg ) { };
  exception( std::string msg ) : std::exception(), message( msg ) { };

};

} // namespace cuda
} // namespace dvcxx


