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

#ifndef AFQMC_NUMERICS_HELPERS_HPP
#define AFQMC_NUMERICS_HELPERS_HPP

#include<cassert>
#include "Numerics/detail/cuda_utilities.hpp"
#include "Numerics/detail/cuda_pointers.hpp"
#include "Kernels/determinant.cuh"

namespace ma 
{
  template<class T>
  inline static T determinant_from_getrf(int n, T* M, int lda, int* pivot)
  {
    T res(1.0);
    for(int i=0, ip=1; i != n; i++, ip++){
      if(pivot[i]==ip){
        res *= +static_cast<T>(M[i*lda+i]);
      }else{
        res *= -static_cast<T>(M[i*lda+i]);
      }
    }
    return res;
  } 

  template<class T,
           class ptr,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE)  and 
                                                 (ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE) > 
          >
  inline static T determinant_from_getrf(int n, ptr A, int lda, ptrI piv) 
  {
    T res(1.0); 
    auto pivot = to_address(piv);
    auto M = to_address(A);
    for(int i=0, ip=1; i != n; i++, ip++){
      if(pivot[i]==ip){
        res *= +static_cast<T>(M[i*lda+i]);
      }else{
        res *= -static_cast<T>(M[i*lda+i]);
      }
    }
    return res;
  }

  // using thrust for now to avoid kernels!!!
  template<class T,
           class ptr,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type != CPU_OUTOFCARS_POINTER_TYPE)  and 
                                                 (ptr::memory_type != CPU_OUTOFCARS_POINTER_TYPE) >,
          typename = void 
          >
  inline static T determinant_from_getrf(int n, ptr A, int lda, ptrI piv)
  {
    return kernels::determinant_from_getrf_gpu(n,to_address(A),lda,to_address(piv));
  }

  template<class T>
  inline static void determinant_from_getrf(int n, T* M, int lda, int* pivot, T* res)
  {
    *res = T(1.0);
    for(int i=0, ip=1; i != n; i++, ip++){
      if(pivot[i]==ip){
        *res *= +static_cast<T>(M[i*lda+i]);
      }else{
        *res *= -static_cast<T>(M[i*lda+i]);
      }
    }
  }

  template<class T,
           class ptr,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE)  and
                                                 (ptr::memory_type == CPU_OUTOFCARS_POINTER_TYPE) >
          >
  inline static void determinant_from_getrf(int n, ptr A, int lda, ptrI piv, T* res)
  {
    *res = T(1.0);
    auto pivot = to_address(piv);
    auto M = to_address(A);
    for(int i=0, ip=1; i != n; i++, ip++){
      if(pivot[i]==ip){
        *res *= +static_cast<T>(M[i*lda+i]);
      }else{
        *res *= -static_cast<T>(M[i*lda+i]);
      }
    }
  }

  // using thrust for now to avoid kernels!!!
  template<class T,
           class ptr,
           class ptrI,
           typename = typename std::enable_if_t< (ptr::memory_type != CPU_OUTOFCARS_POINTER_TYPE)  and
                                                 (ptr::memory_type != CPU_OUTOFCARS_POINTER_TYPE) >,
          typename = void
          >
  inline static void determinant_from_getrf(int n, ptr A, int lda, ptrI piv, T*res)
  {
    kernels::determinant_from_getrf_gpu(n,to_address(A),lda,to_address(piv),res);
  }

}

#endif
