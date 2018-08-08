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

#ifndef QMCPLUSPLUS_AFQMC_PROPAGATORFACTORY_H
#define QMCPLUSPLUS_AFQMC_PROPAGATORFACTORY_H

#include<iostream>
#include<vector>
#include<map>
#include<fstream>
#include "Utilities/RandomGenerator.h"

#include "af_config.h"
#include "Utilities/taskgroup.h"
#include "Wavefunctions/Wavefunction.hpp"
#include "Propagators/Propagator.hpp"

namespace qmcplusplus
{

namespace afqmc
{

Propagator getPropagator(TaskGroup_& TG, int, int, Wavefunction& wfn, RandomGenerator_t* rng); 

}

}

#endif
