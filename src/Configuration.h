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

#include <config.h>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <Utilities/OhmmsInfo.h>
#include <Message/Communicate.h>

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


#include <OhmmsPETE/OhmmsVector.h>
#include <OhmmsPETE/OhmmsMatrix.h>
#include "Matrix/SparseMatrix.hpp"
//#include "Matrix/SMSparseMatrix.hpp"
//#include "Matrix/SMDenseVector.hpp"

namespace qmcplusplus
{

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


  typedef Vector<IndexType>     IndexVector;
  typedef Vector<RealType>      RealVector;
  typedef Vector<ValueType>     ValueVector;
  typedef Vector<SPValueType>   SPValueVector;
  typedef Vector<ComplexType>   ComplexVector;
  typedef Vector<SPComplexType>   SPComplexVector;
/*
  typedef SMDenseVector<IndexType>     IndexSMVector;
  typedef SMDenseVector<RealType>      RealSMVector;
  typedef SMDenseVector<ValueType>     ValueSMVector;
  typedef SMDenseVector<SPValueType>   SPValueSMVector;
  typedef SMDenseVector<ComplexType>   ComplexSMVector;
  typedef SMDenseVector<SPComplexType>   SPComplexSMVector;
*/
  typedef Matrix<IndexType>     IndexMatrix;
  typedef Matrix<RealType>      RealMatrix;
  typedef Matrix<ValueType>     ValueMatrix;
  typedef Matrix<SPValueType>     SPValueMatrix;
  typedef Matrix<ComplexType>   ComplexMatrix;
  typedef Matrix<SPComplexType>   SPComplexMatrix;

  typedef SparseMatrix<IndexType>     IndexSpMat;
  typedef SparseMatrix<RealType>      RealSpMat;
  typedef SparseMatrix<ValueType>     ValueSpMat;
  typedef SparseMatrix<SPValueType>   SPValueSpMat;
  typedef SparseMatrix<ComplexType>   ComplexSpMat;
/*
  typedef SMSparseMatrix<IndexType>     IndexSMSpMat;
  typedef SMSparseMatrix<RealType>      RealSMSpMat;
  typedef SMSparseMatrix<ValueType>     ValueSMSpMat;
  typedef SMSparseMatrix<SPValueType>   SPValueSMSpMat;
  typedef SMSparseMatrix<ComplexType>   ComplexSMSpMat;
  typedef SMSparseMatrix<SPComplexType>   SPComplexSMSpMat;
*/

inline std::ostream &app_log() { return OhmmsInfo::Log->getStream(); }

inline std::ostream &app_error()
{
  OhmmsInfo::Log->getStream() << "ERROR ";
  return OhmmsInfo::Error->getStream();
}

inline std::ostream &app_warning()
{
  OhmmsInfo::Log->getStream() << "WARNING ";
  return OhmmsInfo::Warn->getStream();
}

inline std::ostream &app_debug() { return OhmmsInfo::Debug->getStream(); }
}

#endif
