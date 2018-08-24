//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef AFQMC_CUDA_POINTERS_HPP 
#define AFQMC_CUDA_POINTERS_HPP

#include "Numerics/detail/raw_pointers.hpp"
#ifdef QMC_CUDA 
#include <cuda_runtime.h>
#include "Utilities/type_conversion.hpp"
#include "Numerics/detail/cuda_gpu_pointer.hpp"
#include "Numerics/detail/cuda_um_pointer.hpp"
#include "Numerics/detail/cuda_ooc_pointer.hpp"
#include <type_traits>

namespace cuda {


template<class ptrA, class Size, class ptrB, 
         typename = typename std::enable_if_t<not std::is_pointer<ptrA>::value>, 
         typename = typename std::enable_if_t<not std::is_pointer<ptrB>::value> 
        > 
ptrB copy_n(ptrA const A, Size n, ptrB B) {
  if(cudaSuccess != cudaMemcpy(to_address(B),to_address(A),n*sizeof(typename ptrA::value_type),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n; 
}  
template<class T, class Size, class ptrA,
         typename = typename std::enable_if_t<not std::is_pointer<ptrA>::value> 
>
T* copy_n(ptrA const A, Size n, T* B) {
  if(cudaSuccess != cudaMemcpy(B,to_address(A),n*sizeof(typename ptrA::value_type),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n;
}
template<class T, class Size, class ptrB,
         typename = typename std::enable_if_t<not std::is_pointer<ptrB>::value> 
>
ptrB copy_n(T const* A, Size n, ptrB B) {
  if(cudaSuccess != cudaMemcpy(to_address(B),A,n*sizeof(T),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n;
}
template<class T, class Size>
T* copy_n(T const* A, Size n, T* B) {
  return std::copy_n(A,n,B);    
//  if(cudaSuccess != cudaMemcpy(B,A,n*sizeof(T),cudaMemcpyDefault))
//   throw std::runtime_error("Error: cudaMemcpy returned error code.");
//  return B+n;
}

/*
template<class T, class Size> 
cuda_gpu_ptr<T> copy_n(cuda_gpu_ptr<T> const A, Size n, cuda_gpu_ptr<T> B) {
  if(cudaSuccess != cudaMemcpy(to_address(B),to_address(A),n*sizeof(T),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n; 
}  
template<class T, class Size>
T* copy_n(cuda_gpu_ptr<T> const A, Size n, T* B) {
  if(cudaSuccess != cudaMemcpy(B,to_address(A),n*sizeof(T),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n;
}
template<class T, class Size>
cuda_gpu_ptr<T> copy_n(T const* A, Size n, cuda_gpu_ptr<T> B) {
  if(cudaSuccess != cudaMemcpy(to_address(B),A,n*sizeof(T),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n;
}
*/

}

#endif

#endif
