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

#ifndef AFQMC_SETIDENTITY_KERNELS_HPP
#define AFQMC_SETIDENTITY_KERNELS_HPP

#include<cassert>
#include<cuda.h>
#include<cuda_runtime.h>
#include <complex>

namespace kernels
{

void setIdentity(int n, double * A, int lda);
void setIdentity(int n, float * A, int lda);
void setIdentity(int n, std::complex<double> * A, int lda);
void setIdentity(int n, std::complex<float> * A, int lda);

}

#endif
