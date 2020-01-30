#pragma once
#include <cstddef>
#include <dvcxx/util/sfinae.hpp>

namespace dvcxx    {
namespace cuda     {

namespace wrappers {



/**
 *  \brief Copy data between host and device
 *
 *  Copy data between host and device through CUDA API
 *  (cudaMemcpy)
 *
 *  @param[in/out] dest Pointer to destination memory
 *  @param[in]     src  Pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
void memcpy( void* dest, const void* src, size_t len );





/**
 *  \brief Copy 2D data between host and device
 *
 *  Copy 2D data between host and device through CUDA API
 *  (cudaMemcpy2D)
 *
 *   | - - - - - - |       pitch  
 *   |- - - -|             width
 *  [ x x x x - - - ]              
 *  [ y y y y - - - ]
 *  ...
 *  [ z z z z - - - ]
 *
 *  @param[in/out] dest   Pointer to destination memory
 *  @param[in]     dpitch Pitch of destination memory (bytes)
 *  @param[in]     src    Pointer to source memory
 *  @param[in]     dpitch Pitch of source memory (bytes)
 *  @param[in]     width  Width (contiguous) of segement in 
 *                        bytes
 *  @param[in]     height Height (noncontiguous) of segment 
 */
void memcpy2d( void* dest, size_t dpitch, const void* src, 
  size_t spitch, size_t width, size_t height );





/**
 *  \brief Allocate memory on the device
 *
 *  Allocate a memory segment on the device through the CUDA
 *  API (cudaMalloc)
 *
 *  @param[in] len Length of memory segment in bytes
 *  @returns   pointer to allocated memory segment
 */
void* malloc( size_t len );





/**
 *  \brief Allocate unified memory accessable from both
 *  host and device.
 *
 *  Allocate a segment of unified memory accessable from
 *  both host and device through the CUDA API 
 *  (cudaMallocManaged)
 *
 *  @param[in] len Length of memory segment in bytes
 *  @returns   pointer to allocated memory segment
 */
void* malloc_unified( size_t len );





/**
 *  \brief Allocate page locked memory on the host.
 *
 *  Allocate a page locked memory segment on the host
 *  through the CUDA API (cudaMallocHost). 
 *
 *  Yields better data movement performance in select 
 *  situations.
 *
 *  @param[in] len Length of memory segment in bytes
 *  @returns   pointer to allocated memory segment
 */
void* malloc_pinned( size_t len );





/**
 *  \brief Free (deallocate) a device memory segment
 *
 *  Deallocate a memory segment previously allocated by
 *  either malloc or malloc_unified (cudaFree).
 *
 *  @param[in] ptr Device pointer to memory segment to be
 *                 deallocated.
 */
void  free( void* ptr ); 





/**
 *  \brief Free (deallocate) a page locked host memory
 *  segment
 *
 *  Deallocate a memory segment previously allocated by
 *  malloc_pinned (cudaFreeHost).
 *
 *  @param[in] ptr Host pointer to memory segment to be
 *                 deallocated.
 */
void  free_pinned( void* ptr );





/**
 *  \brief Synchronize device and host
 *
 *   Synchronize the host and device through the CUDA API
 *   (cudaDeviceSynchronize). Host will wait until device
 *   has finished all enqueued operations.
 */
void device_sync();





/**
 *  \brief Initialize a raw device memory segment to an 
 *  integral value.
 *
 *  Initialize a raw device memory segment to an 
 *  integral value through the CUDA API (cudaMemset).
 *
 *  @param[in/out] data Device pointer of memory to 
 *                      manipulate
 *  @param[in]     val  Value to initialize elements of data
 *  @param[in]     len  Length of memory segment in bytes
 */
void memset( void* data, int val, size_t len );

} // namespace wrappers

/**
 *  \brief Copy data between host and device
 *
 *  Copy data between host and device through CUDA API
 *  (cudaMemcpy). Templated wrapper around 
 *  wrappers::memcpy. Data type must satisfy
 *  is_trivially_copyable
 *
 *  @tparam        T    Type of data to transfer
 *  @param[in/out] dest Device pointer to destination memory
 *  @param[in]     src  Host pointer to source memory
 *  @param[in]     len  Length of memory segment in bytes
 */
template <typename T>
detail::enable_if_trivially_copyable_t< T >
memcpy( T* dest, const T* src, size_t len ) {
  wrappers::memcpy( dest, src, len * sizeof(T) );
}





/**
 *  \brief Copy 2D data between host and device
 *
 *  Copy 2D data between host and device through CUDA API
 *  (cudaMemcpy2D) Templated wrapper around 
 *  wrappers::memcpy2d. Data type must satisfy 
 *  is_trivially_copyable
 *  
 *
 *   | - - - - - - |       pitch  
 *   |- - - -|             width
 *  [ x x x x - - - ]              
 *  [ y y y y - - - ]
 *  ...
 *  [ z z z z - - - ]
 *
 *  @param[in/out] dest   Pointer to destination memory
 *  @param[in]     dpitch Pitch of destination memory (T)
 *  @param[in]     src    Pointer to source memory
 *  @param[in]     dpitch Pitch of source memory (T)
 *  @param[in]     width  Width (contiguous) of segement in 
 *                        terms of sizeof(T)
 *  @param[in]     height Height (noncontiguous) of segment 
 */
template <typename T>
detail::enable_if_trivially_copyable_t< T >
memcpy2d( T* dest, size_t dpitch, const T* src, 
  size_t spitch, size_t width, size_t height ) {

  wrappers::memcpy2d( dest, dpitch * sizeof(T), src, 
    spitch * sizeof(T), width * sizeof(T), height );

}





/**
 *  \brief Allocate memory on the device
 *
 *  Allocate a memory segment on the device through the CUDA
 *  API (cudaMalloc)
 *
 *  @param[in] len Length of memory segment in terms of T
 *  @returns       pointer to allocated memory segment
 */
template <typename T>
T* device_malloc( size_t len ) {
  return (T*)wrappers::malloc( len * sizeof(T) );
}





/**
 *  \brief Allocate unified memory accessable from both
 *  host and device.
 *
 *  Allocate a segment of unified memory accessable from
 *  both host and device through the CUDA API 
 *  (cudaMallocManaged)
 *
 *  @param[in] len Length of memory segment in terms of T
 *  @returns   pointer to allocated memory segment
 */
template <typename T>
T* unified_malloc( size_t len ) {
  return (T*)wrappers::malloc_unified( len * sizeof(T) );
}





/**
 *  \brief Allocate page locked memory on the host.
 *
 *  Allocate a page locked memory segment on the host
 *  through the CUDA API (cudaMallocHost). 
 *
 *  Yields better data movement performance in select 
 *  situations.
 *
 *  @param[in] len Length of memory segment in terms of T
 *  @returns   pointer to allocated memory segment
 */
template <typename T>
T* pinned_malloc( size_t len ) {
  return (T*)wrappers::malloc_pinned( len * sizeof(T) );
}




/**
 *  \brief Free (deallocate) a device memory segment
 *
 *  Deallocate a memory segment previously allocated by
 *  device_malloc.
 *
 *  @param[in] ptr Device pointer to memory segment to be
 *                 deallocated.
 */
template <typename T>
void device_free( T* ptr ) {
  wrappers::free( ptr );
}





/**
 *  \brief Free (deallocate) a unified memory segment
 *
 *  Deallocate a memory segment previously allocated by
 *  unified_malloc.
 *
 *  @param[in] ptr Device pointer to memory segment to be
 *                 deallocated.
 */
template <typename T>
void unified_free( T* ptr ) {
  wrappers::free( ptr );
}





/**
 *  \brief Free (deallocate) a page locked host memory
 *  segment
 *
 *  Deallocate a memory segment previously allocated by
 *  pinned_malloc (cudaFreeHost).
 *
 *  @param[in] ptr Host pointer to memory segment to be
 *                 deallocated.
 */
template <typename T>
void pinned_free( T* ptr ) {
  wrappers::free_pinned( ptr );
}





/**
 *  \brief Synchronize device and host
 *
 *   Synchronize the host and device through the CUDA API
 *   (cudaDeviceSynchronize). Host will wait until device
 *   has finished all enqueued operations.
 */
inline void device_sync() { 
  wrappers::device_sync(); 
}





} // namespace cuda
} // namespace dvcxx
