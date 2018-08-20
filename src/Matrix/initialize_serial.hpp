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
#include "Matrix/hdfcsr2ma.hpp"
#include "Numerics/ma_blas.hpp"

namespace qmcplusplus
{

namespace afqmc
{

template<class THCOps,
         class af_sys = base::afqmc_sys<std::allocator<ComplexType>>
            >
inline THCOps Initialize(hdf_archive& dump, const double dt, af_sys& sys, ComplexMatrix<typename af_sys::Alloc>& Propg1)
{
  // the allocator of af_sys must be consistent with Alloc, otherwise LA operations will not work 
  using Alloc = typename af_sys::Alloc;
  using SpCAlloc = typename Alloc::template rebind<SPComplexType>::other;
  Alloc& alloc(sys.allocator_); 
  SpCAlloc spc_alloc(alloc); 

  int NMO, NAEA, NAEB;

  std::cout<<"  Serial hdf5 read. \n";

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

  SPComplexMatrix<SpCAlloc> rotMuv( {rotnmu,rotnmu},spc_alloc );   // Muv in half-rotated basis   
  SPComplexMatrix<SpCAlloc> rotPiu( {NMO,rotnmu},spc_alloc );   // Interpolating orbitals in half-rotated basis 
  SPComplexMatrix<SpCAlloc> rotcPua( {rotnmu,NAEA+NAEB},spc_alloc ); // (half-rotated) Interpolating orbitals in half-rotated basis   
  SPComplexMatrix<SpCAlloc> Luv( {nmu,nmu},spc_alloc );      // Cholesky decomposition of Muv 
  SPComplexMatrix<SpCAlloc> Piu( {NMO,nmu},spc_alloc );      // Interpolating orbitals 
  SPComplexMatrix<SpCAlloc> cPua( {nmu,NAEA+NAEB},spc_alloc );     // (half-rotated) Interpolating orbitals 

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
  {
    if(!dump.push(std::string("PsiT_0"),false)) {
      app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
      APP_ABORT("");
    }
// FIX FIX FIX problems with GPU memory
    sys.trialwfn_alpha = hdfcsr2ma<ComplexMatrix<Alloc>,Alloc>(dump,NMO,NAEA,alloc);
    dump.pop();
    if(walker_type == 2) {
      if(!dump.push(std::string("PsiT_1"),false)) {
        app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
        APP_ABORT("");
      }
// FIX FIX FIX problems with GPU memory
      sys.trialwfn_beta = hdfcsr2ma<ComplexMatrix<Alloc>,Alloc>(dump,NMO,NAEA,alloc);
      dump.pop();
    } else 
      sys.trialwfn_beta = sys.trialwfn_alpha;
  }

  /***************************************/
  if(!dump.push("HamiltonianOperations",false)) {
    app_error()<<" Error in initialize: Group HamiltonianOperations not found. \n";
    APP_ABORT("");
  }
  /***************************************/
  if(!dump.push("THCOps",false)) {
    app_error()<<" Error in initialize: Group THCOps not found. \n";
    APP_ABORT("");
  }
  /***************************************/

  // half-rotated Pia
  // simple
  using ma::H;
  using ma::T;
  // cPua = H(Piu) * conj(A)
  ma::product(H(Piu),sys.trialwfn_alpha,cPua( cPua.extension(0), {0,NAEA} ));
  ma::product(H(rotPiu),sys.trialwfn_alpha,rotcPua( rotcPua.extension(0), {0,NAEA} )); 
  ma::product(H(Piu),sys.trialwfn_beta,cPua( cPua.extension(0), {NAEA,NAEA+NAEB} ));
  ma::product(H(rotPiu),sys.trialwfn_beta,rotcPua( rotcPua.extension(0), {NAEA,NAEA+NAEB} ));

  // build the propagator (on the CPU)
  ComplexMatrix<std::allocator<ComplexType>> H1( {NMO,NMO} );
  ComplexMatrix<std::allocator<ComplexType>> v0( {NMO,NMO} );
  /***************************************/
  if(!dump.read(H1,"H1")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  /***************************************/
  // rotated 1 body hamiltonians
  ComplexMatrix<Alloc> haj( {NAEA+NAEB,NMO}, alloc );      // rotated 1-Body Hamiltonian Matrix
  ma::product(T(sys.trialwfn_alpha),H1,haj( {0, NAEA}, haj.extension(1)));
  ma::product(T(sys.trialwfn_beta),H1,haj( {NAEA, NAEA+NAEB}, haj.extension(1))); 

  // read v0 in Propg1 temporarily
  /***************************************/
  if(!dump.read(v0,"v0")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  /***************************************/

  // This step must be done on the CPU right now
  // this will not work with GPU memory!!! Need to call a routine!
  for(int i=0; i<NMO; i++) {
   H1[i][i] = -0.5*dt*(H1[i][i] + v0[i][i]);
   for(int j=i+1; j<NMO; j++) {
     using std::conj;
     H1[i][j] += v0[i][j];
     H1[j][i] += v0[j][i];
     H1[i][j] = -0.5*dt*(0.5*(H1[i][j]+conj(H1[j][i])));
     H1[j][i] = conj(H1[i][j]);
   }
  }
  v0 = ma::exp(H1);
  Propg1.reextent( {NMO,NMO} );
  // need specialized copy routine
  Propg1 = v0;

  return THCOps(NMO,NAEA,NAEA,COLLINEAR,std::move(haj),std::move(rotMuv),
                std::move(rotPiu),std::move(rotcPua),std::move(Luv),std::move(Piu),
                std::move(cPua),E0);
} 

}  // afqmc


} // qmcplusplus


#endif
