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

#include "io/hdf_archive.h"
#include "Matrix/csr_hdf5_readers.hpp"
#include "Wavefunctions/WavefunctionFactory.h"
#include "Wavefunctions/Wavefunction.hpp"
#include "Wavefunctions/NOMSD.hpp"
#include "Wavefunctions/rotateHamiltonian.hpp"
#include "SlaterDeterminantOperations/rotate.hpp"
#include "HamiltonianOperations/HamOpsIO.hpp"

namespace qmcplusplus
{

namespace afqmc
{

Wavefunction getWavefunction(hdf_archive& dump, TaskGroup_& TGprop, TaskGroup_& TGwfn, WALKER_TYPES& walker_type, 
                             int& NMO, int& NAEA, int targetNW, RealType cutvn, RealType cutv2)
{
  int nci;
  ValueType NCE;
  int ndets_to_read(-1);

  std::vector<ComplexType> ci;
  std::vector<PsiT_Matrix> PsiT;
  std::vector<int> excitations;

  using Alloc = boost::mpi3::intranode::allocator<ComplexType>;
//  if(type=="msd") {

    // HOps, ci, PsiT, NCE
    if(!dump.push("Wavefunction",false)) {
      app_error()<<" Error in WavefunctionFactory: Group Wavefunction not found. \n";
      APP_ABORT("");
    }
    if(!dump.push("NOMSD",false)) {
      app_error()<<" Error in WavefunctionFactory: Group NOMSD not found. \n";
      APP_ABORT("");
    }

    // check for consistency in parameters
    std::vector<int> dims(5); //{NMO,NAEA,NAEB,walker_type,ndets_to_read};
    if(TGwfn.Global().root()) {
      if(!dump.read(dims,"dims")) {
        app_error()<<" Error in WavefunctionFactory::fromHDF5(): Problems reading dims. \n";
        APP_ABORT("");
      }
      if(!dump.read(ci,"CICOEFFICIENTS")) {
        app_error()<<" Error in WavefunctionFactory::fromHDF5(): Problems reading CICOEFFICIENTS. \n";
        APP_ABORT("");
      }
      ci.resize(dims[4]);
      std::vector<ValueType> dum;
      if(!dump.read(dum,"NCE")) {
        app_error()<<" Error in WavefunctionFactory::fromHDF5(): Problems reading NCE. \n";
        APP_ABORT("");
      }
      NCE = dum[0];
    }
    TGwfn.Global().broadcast_n(dims.data(),dims.size());
    NMO = dims[0];
    NAEA = dims[1];    
    walker_type = WALKER_TYPES(dims[3]);
    ndets_to_read=dims[4];
    ci.resize(ndets_to_read);
    TGwfn.Global().broadcast_n(ci.data(),ci.size());
    TGwfn.Global().broadcast_value(NCE);

    int nd = (walker_type==COLLINEAR?2*ndets_to_read:ndets_to_read);
    PsiT.reserve(nd);
    using Alloc = boost::mpi3::intranode::allocator<ComplexType>;
    for(int i=0; i<nd; ++i) {
      if(!dump.push(std::string("PsiT_")+std::to_string(i),false)) {
        app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
        APP_ABORT("");
      }
      PsiT.emplace_back(csr_hdf5::HDF2CSR<PsiT_Matrix,Alloc>(dump,TGwfn.Node())); //,Alloc(TGwfn.Node())));
      dump.pop();
    }

    HamiltonianOperations HOps(loadHamOps(dump,walker_type,NMO,NAEA,NAEA,PsiT,TGprop,TGwfn,cutvn,cutv2));

    AFQMCInfo AFinfo("AFinfo",NMO,NAEA,NAEA);

    return Wavefunction(NOMSD(AFinfo,TGwfn,std::move(HOps),std::move(ci),std::move(PsiT),
                        walker_type,NCE,targetNW));
//  }

}

}

}
