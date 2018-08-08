//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef AFQMC_WAVEFUNCTION_WAVEFUNCTIONFACTORY_HPP
#define AFQMC_WAVEFUNCTION_WAVEFUNCTIONFACTORY_HPP

#include "io/hdf_archive.h"
#include "Wavefunctions/Wavefunction.hpp"

namespace qmcplusplus
{

namespace afqmc
{

Wavefunction getWavefunction(hdf_archive& dump, TaskGroup_& TGprop, TaskGroup_& TGwfn, WALKER_TYPES& walker_type, 
                             int& NMO, int& NAEA, int targetNW, RealType cutvn, RealType cutv2);

}

}

#endif
