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
#include <algorithm>
#include <cuda_runtime.h>
#include "Utilities/type_conversion.hpp"
#include "Numerics/detail/cuda_gpu_pointer.hpp"
#include "Numerics/detail/cuda_um_pointer.hpp"
#include "Numerics/detail/cuda_ooc_pointer.hpp"
#include <type_traits>

#include "Kernels/fill_n.cuh"
#include "Kernels/print.cuh"
#include "Kernels/copy_n_cast.cuh"

namespace cuda {

/*
 * NEED TO FIX THIS, RIGHT NOW WILL TAKE ALL ITERATORS WHICH IS WRONG!!!!!
 * NEED A is_cuda_pointer predicate!!!!
 */
template<class ptrA, class Size, class ptrB, 
         typename = typename std::enable_if_t<(not std::is_pointer<ptrA>::value) and
                                  (not std::is_pointer<ptrB>::value)> > 
ptrB copy_n(ptrA const A, Size n, ptrB B) {
  if(cudaSuccess != cudaMemcpy(to_address(B),to_address(A),n*sizeof(typename ptrA::value_type),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n; 
}  
template<class T, class Size, class ptrA,
         typename = typename std::enable_if_t<(not std::is_pointer<ptrA>::value)> > 
T* copy_n(ptrA const A, Size n, T* B) {
  if(cudaSuccess != cudaMemcpy(B,to_address(A),n*sizeof(typename ptrA::value_type),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n;
}
template<class T, class Size, class ptrB,
         typename = typename std::enable_if_t<(not std::is_pointer<ptrB>::value)> >
ptrB copy_n(T const* A, Size n, ptrB B) {
  if(cudaSuccess != cudaMemcpy(to_address(B),A,n*sizeof(T),cudaMemcpyDefault))
   throw std::runtime_error("Error: cudaMemcpy returned error code.");
  return B+n;
}
template<class T, class Size>
T* copy_n(T const* A, Size n, T* B) {
  return std::copy_n(A,n,B);    
}

// array casting ( e.g. change precision) 
template<class ptrA, class ptrB, class Size,
     typename = typename std::enable_if_t< (ptrA::memory_type != GPU_MEMORY_POINTER_TYPE) and
                                           (ptrB::memory_type != GPU_MEMORY_POINTER_TYPE) > >   
ptrB copy_n_cast(ptrA const A, Size n, ptrB B) {
  using T = typename ptrA::value_type;
  using Q = typename ptrB::value_type;
  for(Size i=0; i<n; i++, ++A, ++B)
    *B = static_cast<Q>(*A);
  return B;
}
template<class ptrA, class ptrB, class Size,
     typename = typename std::enable_if_t< (ptrA::memory_type == GPU_MEMORY_POINTER_TYPE) and
                                           (ptrB::memory_type == GPU_MEMORY_POINTER_TYPE) >,
     typename = void    >  
ptrB copy_n_cast(ptrA const A, Size n, ptrB B) {
  kernels::copy_n_cast(to_address(A),n,to_address(B));
  return B;
}

template<class Q, class ptrA, class Size,
     typename = typename std::enable_if_t< (ptrA::memory_type != GPU_MEMORY_POINTER_TYPE)> > 
Q* copy_n_cast(ptrA const A, Size n, Q * B) {
  for(Size i=0; i<n; i++, ++A, ++B)
    *B = static_cast<Q>(*A);
  return B;
}
template<class Q, class ptrA, class Size,
     typename = typename std::enable_if_t< (ptrA::memory_type == GPU_MEMORY_POINTER_TYPE)>,
     typename = void    >
Q* copy_n_cast(ptrA const A, Size n, Q * B) {
  throw std::runtime_error(" Error: copy_n_cast(gpu_ptr,n,T*) is disabled."); 
  return B;
}

template<class T, class ptrB, class Size,
     typename = typename std::enable_if_t< (ptrB::memory_type != GPU_MEMORY_POINTER_TYPE)> >
ptrB copy_n_cast(T * A, Size n, ptrB B) {
  using Q = typename ptrB::value_type;
  for(Size i=0; i<n; i++, ++A, ++B)
    *B = static_cast<Q>(*A);
  return B;
}
template<class T, class ptrB, class Size,
     typename = typename std::enable_if_t< (ptrB::memory_type == GPU_MEMORY_POINTER_TYPE)>, 
     typename = void    >  
ptrB copy_n_cast(T * A, Size n, ptrB B) {
  throw std::runtime_error(" Error: copy_n_cast(gpu_ptr,n,T*) is disabled.");  
  return B;
}

template<class T, class Q, class Size>
Q* copy_n_cast(T const* A, Size n, Q* B) {
  for(Size i=0; i<n; i++, A++, B++)
    *B = static_cast<Q>(*A);
  return B;
}


/* fill_n */
/*
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
*/

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
