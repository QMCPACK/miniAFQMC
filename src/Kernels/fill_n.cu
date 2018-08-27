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

#include<cassert>
#include <complex>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace kernels 
{

void fill_n(int * first, int N, int const value)
{ thrust::fill_n(thrust::device_ptr<int>(first),N,value); }

void fill_n(float * first, int N, float const value)
{ thrust::fill_n(thrust::device_ptr<float>(first),N,value); }

void fill_n(double * first, int N, double const value)
{ thrust::fill_n(thrust::device_ptr<double>(first),N,value); }

void fill_n(std::complex<float> * first, int N, std::complex<float> const value)
{ thrust::fill_n(thrust::device_ptr<thrust::complex<float> >(
                    reinterpret_cast<thrust::complex<float> *>(first)),N,
                    static_cast<thrust::complex<float> const >(value)); }

void fill_n(std::complex<double> * first, int N, std::complex<double> const value)
{ thrust::fill_n(thrust::device_ptr<thrust::complex<double> >(
                    reinterpret_cast<thrust::complex<double> *>(first)),N,
                    static_cast<thrust::complex<double> const >(value)); }


}

