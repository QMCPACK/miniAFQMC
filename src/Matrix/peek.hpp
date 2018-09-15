//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
// Alfredo Correa, correaa@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_PEEK_HPP
#define QMCPLUSPLUS_AFQMC_PEEK_HPP

#include<string>
#include<vector>

#include "Configuration.h"
#include "io/hdf_archive.h"

namespace qmcplusplus
{

namespace afqmc
{

inline std::tuple<int,int,int,WALKER_TYPES> peek(hdf_archive& dump)
{
  int NMO, NAEA, NAEB;

  std::vector<int> dims(10);

  // read from HDF
  if(!dump.push("Wavefunction",false)) {
    app_error()<<" Error in initialize: Group Wavefunction not found. \n";
    APP_ABORT("");
  }
  if(!dump.push("NOMSD",false)) {
    app_error()<<" Error in initialize: Group NOMSD not found. \n";
    APP_ABORT("");
  }
  if(!dump.push("HamiltonianOperations",false)) {
    app_error()<<" Error in initialize: Group HamiltonianOperations not found. \n";
    APP_ABORT("");
  }
  if(!dump.push("THCOps",false)) {
    app_error()<<" Error in initialize: Group THCOps not found. \n";
    APP_ABORT("");
  }
  if(!dump.read(dims,"dims")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  assert(dims.size()==7);
  NMO = dims[0];
  NAEA = dims[1];
  NAEB = dims[2];
  WALKER_TYPES wtype = WALKER_TYPES(dims[4]);
  if(NAEA!=NAEB) {
    app_error()<<" Error in initialize: NAEA != NAEB. \n"; 
    APP_ABORT("");
  }
  dump.pop();
  dump.pop();
  dump.pop();
  dump.pop();

  return std::tuple<int,int,int,WALKER_TYPES>{NMO,NAEA,NAEB,wtype};
  
}

}

}

#endif
