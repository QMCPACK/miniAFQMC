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

#ifndef AFQMC_RAW_POINTERS_DETAIL_HPP 
#define AFQMC_RAW_POINTERS_DETAIL_HPP

#include <type_traits>
#include <complex>
#include "Utilities/type_conversion.hpp"

namespace detail {
/*
  inline static int* to_address(int* ptr) { return ptr; }
  inline static unsigned int* to_address(unsigned int* ptr) { return ptr; }
  inline static long* to_address(long* ptr) { return ptr; }
  inline static unsigned long* to_address(unsigned long* ptr) { return ptr; }
  inline static float* to_address(float* ptr) { return ptr; }
  inline static double* to_address(double* ptr) { return ptr; }
  inline static std::complex<float>* to_address(std::complex<float>* ptr) { return ptr; }
  inline static std::complex<double>* to_address(std::complex<double>* ptr) { return ptr; }
*/
  template<class T,
           typename = typename std::enable_if_t<std::is_fundamental<T>::value>>
  inline static T* to_address(T* p) { return p; }

  template<class T>
  inline static std::complex<T>* to_address(std::complex<T>* p) { return p; }

  template<class Q, class T,
           typename = typename std::enable_if_t<std::is_fundamental<Q>::value>,
           typename = typename std::enable_if_t<std::is_fundamental<T>::value>>
  inline static Q* pointer_cast(T* p) { return reinterpret_cast<Q*>(p); }

  template<class Q, class T,
           typename = typename std::enable_if_t<std::is_fundamental<Q>::value>>
  inline static Q* pointer_cast(std::complex<T>* p) { return reinterpret_cast<Q*>(p); }


  // checks if a type is a pointer to a fundamental type (including std::complex<T>*)
  template< class ptr >
  struct is_cuda_pointer
    : std::integral_constant<
        bool,
        not std::is_pointer<ptr>::value
  > {};


}

#endif
