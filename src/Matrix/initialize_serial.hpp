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

#include "Numerics/detail/cuda_pointers.hpp"

namespace qmcplusplus
{

namespace afqmc
{

template<class THCOps,
         class af_sys
            >
inline THCOps Initialize(hdf_archive& dump, const double dt, af_sys& sys, ComplexMatrix<typename af_sys::Alloc>& Propg1)
{
  // the allocator of af_sys must be consistent with Alloc, otherwise LA operations will not work 
  using Alloc = typename af_sys::Alloc;
  using Alloc_ooc = typename af_sys::Alloc_ooc;
  using SpAlloc = typename Alloc::template rebind<SPComplexType>::other;
  using SpAlloc_ooc = typename Alloc_ooc::template rebind<SPComplexType>::other;
  using stdAlloc = std::allocator<ComplexType>; 
  using stdSpAlloc = std::allocator<SPComplexType>; 
  Alloc& alloc(sys.allocator_); 
  SpAlloc spc_alloc(alloc); 
  Alloc_ooc& alloc_ooc(sys.allocator_ooc_); 
  SpAlloc_ooc spc_alloc_ooc(alloc_ooc); 
  

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
    app_error()<<" Error in initialize: Inconsistent data in file: ndet != 1. \n";
    APP_ABORT("");
  }
  std::vector<ValueType> et;
  if(!dump.read(et,"E0")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  E0=et[0];
  nmu = size_t(dims[5]);
  rotnmu = size_t(dims[6]);

  int NEL = (sys.walker_type == COLLINEAR)?2*NAEA:NAEA;

  SPComplexMatrix<SpAlloc> rotMuv( {rotnmu,rotnmu}, spc_alloc );   // Muv in half-rotated basis 
  SPComplexMatrix<SpAlloc> rotPiu( {NMO,rotnmu},spc_alloc );   // Interpolating orbitals in half-rotated basis 
  SPComplexMatrix<SpAlloc> rotcPua( {rotnmu,NEL},spc_alloc ); // (half-rotated) Interpolating orbitals in half-rotated basis
  SPComplexMatrix<SpAlloc> Piu( {NMO,nmu},spc_alloc );      // Interpolating orbitals 
  SPComplexMatrix<SpAlloc> cPua( {nmu,NEL},spc_alloc );     // (half-rotated) Interpolating orbitals 

  // (Possibly) Out-of-core data
  SPComplexMatrix<SpAlloc_ooc> Luv( {nmu,nmu},spc_alloc_ooc );      // Cholesky decomposition of Muv

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
    {
      ComplexMatrix<stdAlloc> buff(hdfcsr2ma<ComplexMatrix<stdAlloc>,stdAlloc>(dump,NMO,NAEA,stdAlloc{}));
      // can't figure out why std::copy_n is used here, forcing cuda::copy_n for now
      cuda::copy_n(buff.origin(),buff.num_elements(),sys.trialwfn_alpha.origin());
      dump.pop();
    }
    if(sys.walker_type == COLLINEAR) {
      if(!dump.push(std::string("PsiT_1"),false)) {
        app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
        APP_ABORT("");
      }
      ComplexMatrix<stdAlloc> buff(hdfcsr2ma<ComplexMatrix<stdAlloc>,stdAlloc>(dump,NMO,NAEA,stdAlloc{}));
      cuda::copy_n(buff.origin(),buff.num_elements(),sys.trialwfn_beta.origin());
      dump.pop();
    } 
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
#ifdef MIXED_PRECISION
  SPComplexMatrix<SpAlloc> psiA({NMO,NAEA},spc_alloc );
  cuda::copy_n_cast(sys.trialwfn_alpha.origin(),sys.trialwfn_alpha.num_elements(),psiA.origin());
  ma::product(H(Piu),psiA,cPua( cPua.extension(0), {0,NAEA} ));
  ma::product(H(rotPiu),psiA,rotcPua( rotcPua.extension(0), {0,NAEA} )); 
  if(sys.walker_type == COLLINEAR) {
    SPComplexMatrix<SpAlloc> psiB({NMO,NAEB},spc_alloc );
    cuda::copy_n_cast(sys.trialwfn_beta.origin(),sys.trialwfn_beta.num_elements(),psiB.origin());
    ma::product(H(Piu),psiB,cPua( cPua.extension(0), {NAEA,NAEA+NAEB} ));
    ma::product(H(rotPiu),psiB,rotcPua( rotcPua.extension(0), {NAEA,NAEA+NAEB} ));
  }
#else 
  ma::product(H(Piu),sys.trialwfn_alpha,cPua( cPua.extension(0), {0,NAEA} ));
  ma::product(H(rotPiu),sys.trialwfn_alpha,rotcPua( rotcPua.extension(0), {0,NAEA} )); 
  if(sys.walker_type == COLLINEAR) {
    ma::product(H(Piu),sys.trialwfn_beta,cPua( cPua.extension(0), {NAEA,NAEA+NAEB} ));
    ma::product(H(rotPiu),sys.trialwfn_beta,rotcPua( rotcPua.extension(0), {NAEA,NAEA+NAEB} ));
  }
#endif

  // build the propagator (on the CPU)
  ComplexMatrix<Alloc> H1( {NMO,NMO}, alloc );
  /***************************************/
  if(!dump.read(H1,"H1")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  /***************************************/
  // rotated 1 body hamiltonians
  ComplexMatrix<Alloc> haj( {NEL,NMO}, alloc );      // rotated 1-Body Hamiltonian Matrix
  ComplexType a = (sys.walker_type==CLOSED)?ComplexType(2.0):ComplexType(1.0);
  ma::product(a, T(sys.trialwfn_alpha),H1,ComplexType(0.0),haj( {0, NAEA}, haj.extension(1)));
  if(sys.walker_type == COLLINEAR) 
    ma::product(T(sys.trialwfn_beta),H1,haj( {NAEA, NAEA+NAEB}, haj.extension(1))); 

  // Hack for gpu
  // This step must be done on the CPU right now
  // this will not work with GPU memory!!! Need to call a routine!
// implement ma::conjugate_transpose(A) using geam on the GPU  
  ComplexMatrix<stdAlloc> buff( {NMO,NMO}, stdAlloc{});
  cuda::copy_n(H1.origin(),H1.num_elements(),buff.origin());
  ComplexMatrix<stdAlloc> v0( {NMO,NMO}, stdAlloc{} );
  /***************************************/
  if(!dump.read(v0,"v0")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  /***************************************/
  for(int i=0; i<NMO; i++) {
   buff[i][i] = -0.5*dt*(buff[i][i] + v0[i][i]);
   for(int j=i+1; j<NMO; j++) {
     using std::conj;
     buff[i][j] += v0[i][j];
     buff[j][i] += v0[j][i];
     buff[i][j] = -0.5*dt*(0.5*(buff[i][j]+conj(buff[j][i])));
     buff[j][i] = conj(buff[i][j]);
   }
  }
  v0 = ma::exp(buff);
  //Propg1.reextent( {NMO,NMO} );
  // need specialized copy routine
  using std::copy_n;
  cuda::copy_n(v0.origin(),v0.num_elements(),Propg1.origin());

  return THCOps(NMO,NAEA,NAEA,sys.walker_type,std::move(haj),std::move(rotMuv),
                std::move(rotPiu),std::move(rotcPua),std::move(Luv),std::move(Piu),
                std::move(cPua),E0,alloc);
} 

}  // afqmc


} // qmcplusplus


#endif
