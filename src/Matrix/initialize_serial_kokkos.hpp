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

#ifndef QMCPLUSPLUS_AFQMC_INITIALIZE_HPP
#define QMCPLUSPLUS_AFQMC_INITIALIZE_HPP

#include<string>
#include<vector>

#include "Configuration.h"
#include "io/hdf_archive.h"

#include "AFQMC/afqmc_sys.hpp"
#include "AFQMC/THCOps.hpp"
#include "Matrix/hdfcsr2ma.hpp"

namespace qmcplusplus
{

namespace afqmc
{

template<class Mat>
inline THCOps Initialize(hdf_archive& dump, const double dt, base::afqmc_sys& sys, Mat& Propg1) 
{
  int NMO, NAEA, NAEB;

  std::cout<<"  Serial hdf5 read. \n";

  ComplexMatrix rotMuv("rotMuv", 1, 1);   // Muv in half-rotated basis   
  ComplexMatrix rotPiu("rotPiu", 1, 1);   // Interpolating orbitals in half-rotated basis 
  ComplexMatrix rotcPua("rotcPua", 1, 1);  // (half-rotated) Interpolating orbitals in half-rotated basis   
  ComplexMatrix Luv("Luv", 1, 1);      // Cholesky decomposition of Muv 
  ComplexMatrix Piu("Piu", 1, 1);      // Interpolating orbitals 
  ComplexMatrix cPua("cPua", 1, 1);     // (half-rotated) Interpolating orbitals 
  ComplexMatrix haj("haj", 1, 1);      // rotated 1-Body Hamiltonian Matrix

  // fix later for multidet case
  std::vector<int> dims(10);
  ValueType E0;
  int global_ncvecs=0;
  std::size_t nmu,rotnmu;

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
  if(NAEA!=NAEB) {
    app_error()<<" Error in initialize: NAEA != NAEB. \n"; 
    APP_ABORT("");
  }
  
  if(dims[3] != 1) {
    app_error()<<" Error in initialize: Inconsistent data in file: ndet. \n";
    APP_ABORT("");
  }
  int walker_type = dims[4];
  std::vector<ValueType> et;
  if(!dump.read(et,"E0")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  E0=et[0];
  nmu = size_t(dims[5]);
  rotnmu = size_t(dims[6]);

  Kokkos::resize(rotMuv, rotnmu, rotnmu);
  Kokkos::resize(rotPiu, NMO, rotnmu);
  Kokkos::resize(rotcPua, rotnmu, NAEA+NAEB);
  Kokkos::resize(Luv, nmu, nmu);
  Kokkos::resize(Piu, NMO, nmu);
  Kokkos::resize(cPua, nmu, NAEA+NAEB);

  // read Half transformed first
  /***************************************/
  if(!dump.read(rotPiu,"HalfTransformedFullOrbitals")) {
    app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
               <<" Problems reading HalfTransformedFullOrbitals. \n";
    APP_ABORT("");
  }
  /***************************************/
  if(!dump.read(rotMuv,"HalfTransformedMuv")) {
    app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
              <<" Problems reading HalfTransformedMuv. \n";
    APP_ABORT("");
  }
  /***************************************/
  if(!dump.read(Piu,"Orbitals")) {
    app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
               <<" Problems reading Orbitals. \n";
    APP_ABORT("");
  }
  /***************************************/
  if(!dump.read(Luv,"Luv")) {
    app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
               <<" Problems reading Luv. \n";
    APP_ABORT("");
  }
  /***************************************/

  dump.pop();
  dump.pop();

  // set values in sys
  sys.setup(NMO,NAEA);
  {
    Kokkos::resize(sys.trialwfn_alpha, NMO, NAEA);
    Kokkos::resize(sys.trialwfn_beta, NMO, NAEA);
    if(!dump.push(std::string("PsiT_0"),false)) {
      app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
      APP_ABORT("");
    }
    sys.trialwfn_alpha = hdfcsr2ma<ComplexMatrix>(dump,NMO,NAEA);
    dump.pop();
    if(walker_type == 2) {
      if(!dump.push(std::string("PsiT_1"),false)) {
        app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
        APP_ABORT("");
      }
      sys.trialwfn_beta = hdfcsr2ma<ComplexMatrix>(dump,NMO,NAEA);
      dump.pop();
    } else 
      sys.trialwfn_beta = sys.trialwfn_alpha;
  }

  if(!dump.push("HamiltonianOperations",false)) {
    app_error()<<" Error in initialize: Group HamiltonianOperations not found. \n";
    APP_ABORT("");
  }
  if(!dump.push("THCOps",false)) {
    app_error()<<" Error in initialize: Group THCOps not found. \n";
    APP_ABORT("");
  }

  ComplexMatrix& PsiT_Alpha = sys.trialwfn_alpha;
  ComplexMatrix& PsiT_Beta = sys.trialwfn_beta;
  // half-rotated Pia
  // simple
  using ma::H;
  using ma::T;
  // cPua = H(Piu) * conj(A)
  ma::product(H(Piu),PsiT_Alpha,cPua[indices[range_t()][range_t(0,NAEA)]]);
  ma::product(H(rotPiu),PsiT_Alpha,rotcPua[indices[range_t()][range_t(0,NAEA)]]);
  ma::product(H(Piu),PsiT_Beta,cPua[indices[range_t()][range_t(NAEA,NAEA+NAEB)]]);
  ma::product(H(rotPiu),PsiT_Beta,rotcPua[indices[range_t()][range_t(NAEA,NAEA+NAEB)]]);

  // build the propagator
  Kokkos::resize(Propg1, NMO, NMO);
  ComplexMatrix H1("H1", NMO, NMO);
  if(!dump.read(H1,"H1")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  // rotated 1 body hamiltonians
  Kokkos::resize(haj, NAEA+NAEB, NMO);
  ma::product(T(PsiT_Alpha),H1,haj[indices[range_t(0,NAEA)][range_t()]]);
  ma::product(T(PsiT_Beta),H1,haj[indices[range_t(NAEA,NAEA+NAEB)][range_t()]]);

  // read v0 in Propg1 temporarily
  if(!dump.read(Propg1,"v0")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  for(int i=0; i<NMO; i++) {
    H1(i, i) = -0.5*dt*(H1(i, i) + Propg1(i, i));
   for(int j=i+1; j<NMO; j++) {
     using std::conj;
     H1(i, j) += Propg1(i, j);
     H1(j, i) += Propg1(j, i);
     H1(i, j) = -0.5*dt*(0.5*(H1(i, j)+conj(H1(j, i))));
     H1(j, i) = conj(H1(i, j));
   }
  }
  Propg1 = ma::exp(H1);

  return THCOps(NMO,NAEA,NAEA,COLLINEAR,std::move(haj),std::move(rotMuv),
                std::move(rotPiu),std::move(rotcPua),std::move(Luv),std::move(Piu),
                std::move(cPua),E0);
} 

}  // afqmc


} // qmcplusplus


#endif
