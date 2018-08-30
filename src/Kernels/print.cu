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

template<typename T>
__global__ void kernel_print( thrust::complex<T> const* p, int n) 
{
  for(int i=0; i<n; i++)
    printf("(%g, %g) ",(p+i)->real(),(p+i)->imag());
}

void print(std::string str, std::complex<double> const* p, int n)
{
  std::cout<<str <<" "; std::cout.flush();
  kernel_print<<<1,1>>>( reinterpret_cast<thrust::complex<double> const*>(p), n);
  cudaDeviceSynchronize ();
  std::cout.flush();
  std::cout<<std::endl;
  std::cout.flush();
}

}
