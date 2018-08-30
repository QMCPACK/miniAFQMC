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

#include "Kernels/fill_n.cuh"
#include "Kernels/print.cuh"

namespace cuda {

/*
 * NEED TO FIX THIS, RIGHT NOW WILL TAKE ALL ITERATORS WHICH IS WRONG!!!!!
 * NEED A is_cuda_pointer predicate!!!!
 */
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
}




/* fill_n */
template<class ptrA, class T, 
         typename = typename std::enable_if_t<(ptrA::memory_type!=CPU_OUTOFCARS_POINTER_TYPE)> 
        >
void fill_n(ptrA A, int n, int stride, const T& value) {
  kernels::fill_n(to_address(A),n,stride,value);
}

template<class ptrA, class T,
         typename = typename std::enable_if_t<(ptrA::memory_type==CPU_OUTOFCARS_POINTER_TYPE)>,
         typename = void 
        >
void fill_n(ptrA A, int n, int stride, const T& value) {
  typename ptrA::value_type* p = to_address(A);
  for(int i=0; i<n; i++, p+=stride)
    *p = static_cast<typename ptrA::value_type>(value);
}

template<class T>
void fill_n(T* p, int n, int stride, const T& value) {
  for(int i=0; i<n; i++, p+=stride)
    *p = value; 
}


// print
template<class ptr, 
         typename = typename std::enable_if_t<(ptr::memory_type!=CPU_OUTOFCARS_POINTER_TYPE)>
        >
void print(std::string str, ptr p, int n) {
  kernels::print(str,to_address(p),n);
}

template<class ptr, 
         typename = typename std::enable_if_t<(ptr::memory_type==CPU_OUTOFCARS_POINTER_TYPE)>,
         typename = void
        >
void print(std::string str, ptr p, int n) {
  std::cout<<str <<" ";
  for(int i=0; i<n; i++)
    std::cout<<*(to_address(p)+i) <<" "; 
  std::cout<<std::endl;
}

template<typename T>
void print(std::string str, T const* p, int n) {
  std::cout<<str <<" ";
  for(int i=0; i<n; i++)
    std::cout<<*(p+i) <<" ";
  std::cout<<std::endl;
}


}

#endif

#endif
