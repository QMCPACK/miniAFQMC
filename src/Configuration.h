////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_TRAITS_H
#define QMCPLUSPLUS_TRAITS_H

#include <iostream>
#include <config.h>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <Utilities/OhmmsInfo.h>
#include <Message/Communicate.h>

#define byRows   999
#define byCols   111

// careful here that RealType is consistent with this!!!
#define MKL_INT         int
#define MKL_Complex8    std::complex<float> 
#define MKL_Complex16   std::complex<double> 

// define empty DEBUG_MEMORY
#define DEBUG_MEMORY(msg)
// uncomment this out to trace the call tree of destructors
//#define DEBUG_MEMORY(msg) std::cerr << "<<<< " << msg << std::endl;

#if defined(DEBUG_PSIBUFFER_ON)
#define DEBUG_PSIBUFFER(who, msg)                              \
  std::cerr << "PSIBUFFER " << who << " " << msg << std::endl; \
  std::cerr.flush();
#else
#define DEBUG_PSIBUFFER(who, msg)
#endif

#include<multi/array.hpp>

namespace qmcplusplus
{

  //using boost::multi_array_types::index_gen;

  typedef OHMMS_INDEXTYPE                 IndexType;
  typedef OHMMS_INDEXTYPE                 OrbitalType;
  typedef OHMMS_PRECISION_FULL            RealType;
  typedef OHMMS_PRECISION                 SPRealType;

#if defined(QMC_COMPLEX)
  typedef std::complex<RealType>  ValueType;
  typedef std::complex<SPRealType>       SPValueType;
#else
  typedef RealType                       ValueType;
  typedef SPRealType                     SPValueType;
#endif
  typedef std::complex<RealType>         ComplexType;
  typedef std::complex<SPRealType>       SPComplexType;

  enum WALKER_TYPES {UNDEFINED_WALKER_TYPE, CLOSED, COLLINEAR, NONCOLLINEAR};

  // [nwalk][2][NMO][NAEA]
  template< class Alloc = std::allocator<ComplexType> >
  using WalkerContainer =  boost::multi::array<ComplexType,4,Alloc>;

  template< class Alloc = std::allocator<int> >
  using IntegerVector =  boost::multi::array<int,1,Alloc>;
  template< class Alloc = std::allocator<ComplexType> >
  using ComplexVector =  boost::multi::array<ComplexType,1,Alloc>;
  template< class Alloc = std::allocator<SPComplexType> >
  using SPComplexVector =  boost::multi::array<SPComplexType,1,Alloc>;
  template< class Ptr = ComplexType* >
  using ComplexVector_ref =  boost::multi::array_ref<ComplexType,1,Ptr>;
  template< class Ptr = SPComplexType* >
  using SPComplexVector_ref =  boost::multi::array_ref<SPComplexType,1,Ptr>;

  template< class Alloc = std::allocator<int> >
  using IntegerMatrix =  boost::multi::array<int,2,Alloc>;
  template< class Alloc = std::allocator<ComplexType> >
  using ComplexMatrix =  boost::multi::array<ComplexType,2,Alloc>;
  template< class Alloc = std::allocator<SPComplexType> >
  using SPComplexMatrix =  boost::multi::array<SPComplexType,2,Alloc>;
  template< class Ptr = ComplexType* >
  using ComplexMatrix_ref =  boost::multi::array_ref<ComplexType,2,Ptr>;
  template< class Ptr = SPComplexType* >
  using SPComplexMatrix_ref =  boost::multi::array_ref<SPComplexType,2,Ptr>;

  template< class Alloc = std::allocator<ComplexType> >
  using Complex3Tensor =  boost::multi::array<ComplexType,3,Alloc>;
  template< class Alloc = std::allocator<SPComplexType> >
  using SPComplex3Tensor =  boost::multi::array<SPComplexType,3,Alloc>;
  template< class Ptr = ComplexType* >
  using Complex3Tensor_ref =  boost::multi::array_ref<ComplexType,3,Ptr>;
  template< class Ptr = SPComplexType* >
  using SPComplex3Tensor_ref =  boost::multi::array_ref<SPComplexType,3,Ptr>;

  template<std::ptrdiff_t D, class Alloc = std::allocator<ComplexType> >
  using ComplexArray =  boost::multi::array<ComplexType,D,Alloc>;
  template<std::ptrdiff_t D, class Alloc = std::allocator<SPComplexType> >
  using SPComplexArray =  boost::multi::array<SPComplexType,D,Alloc>;
  template<std::ptrdiff_t D, class Ptr = ComplexType* >
  using ComplexArray_ref =  boost::multi::array_ref<ComplexType,D,Ptr>;
  template<std::ptrdiff_t D, class Ptr = SPComplexType* > 
  using SPComplexArray_ref =  boost::multi::array_ref<SPComplexType,D,Ptr>;

/*
namespace detail {
  inline static int* get(int* ptr) { return ptr; }
  inline static unsigned int* get(unsigned int* ptr) { return ptr; }
  inline static long* get(long* ptr) { return ptr; }
  inline static unsigned long* get(unsigned long* ptr) { return ptr; }
  inline static float* get(float* ptr) { return ptr; }
  inline static double* get(double* ptr) { return ptr; }
  inline static std::complex<float>* get(std::complex<float>* ptr) { return ptr; }
  inline static std::complex<double>* get(std::complex<double>* ptr) { return ptr; }
}
*/
inline std::ostream &app_log() { return OhmmsInfo::Log->getStream(); } 
//inline std::ostream &app_log() { return std::cout; }

inline std::ostream &app_error()
{
  return std::cerr; 
}

inline std::ostream &app_warning()
{
  return std::cout; 
}

inline std::ostream &app_debug() { return std::cout; } 
}

namespace std{

inline  void swap(std::tuple<int &, int &, qmcplusplus::RealType &> const& a, std::tuple<int &, int &, qmcplusplus::RealType &> const& b) {
    using std::swap;
    swap(std::get<0>(a), std::get<0>(b));
    swap(std::get<1>(a), std::get<1>(b));
    swap(std::get<2>(a), std::get<2>(b));
  }

inline  void swap(std::tuple<int &, int &, std::complex<qmcplusplus::RealType> &> const & a, std::tuple<int &, int &, std::complex<qmcplusplus::RealType> &> const& b) {
    using std::swap;
    swap(std::get<0>(a), std::get<0>(b));
    swap(std::get<1>(a), std::get<1>(b));
    swap(std::get<2>(a), std::get<2>(b));
  }

}

#endif
