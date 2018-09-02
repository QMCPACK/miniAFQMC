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
#include "Utilities/UtilityFunctions.h"

#include "AFQMC/afqmc_sys.hpp"
#include "Matrix/hdfcsr2ma.hpp"
#include "Numerics/ma_blas.hpp"

#include "Numerics/detail/cuda_pointers.hpp"

#include "mpi.h"

namespace qmcplusplus
{

namespace afqmc
{

template<class THCOps,
         class af_sys 
            >
inline THCOps Initialize(hdf_archive& dump, const double dt, af_sys& sys, ComplexMatrix<typename af_sys::Alloc>& Propg1, MPI_Comm comm)
{
  // the allocator of af_sys must be consistent with Alloc, otherwise LA operations will not work 
  using Alloc = typename af_sys::Alloc;
  using SpAlloc = typename Alloc::template rebind<SPComplexType>::other;
  using stdAlloc = std::allocator<ComplexType>; 
  using stdSpAlloc = std::allocator<SPComplexType>; 
  Alloc& alloc(sys.allocator_); 
  SpAlloc spc_alloc(alloc); 
  
  int rank, nproc;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&nproc);

  int NMO, NAEA, NAEB;

  app_log()<<"  Concurrent hdf5 read. \n";

  // fix later for multidet case
  std::vector<int> dims(10);
  ValueType E0;
  int global_ncvecs=0;
  std::size_t gnmu,grotnmu;
  std::size_t nmu,rotnmu;
  std::size_t nmu0,rotnmu0;
  std::size_t nmuN,rotnmuN;

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
  gnmu = size_t(dims[5]);
  grotnmu = size_t(dims[6]);

  // setup partition, in general matrices are partitioned along 'u' 
  {
    std::tie(rotnmu0,rotnmuN) = FairDivideBoundary(size_t(rank),grotnmu,size_t(nproc));
    rotnmu = rotnmuN-rotnmu0;

    std::tie(nmu0,nmuN) = FairDivideBoundary(size_t(rank),gnmu,size_t(nproc));
    nmu = nmuN-nmu0;
  }

  using Matrix = SPComplexMatrix<SpAlloc>;
  SPComplexMatrix<SpAlloc> rotMuv( {rotnmu,grotnmu},spc_alloc );   // Muv in half-rotated basis   
  SPComplexMatrix<SpAlloc> rotPiu( {NMO,grotnmu},spc_alloc );   // Interpolating orbitals in half-rotated basis 
  SPComplexMatrix<SpAlloc> rotcPua( {grotnmu,NAEA+NAEB},spc_alloc ); // (half-rotated) Interpolating orbitals in half-rotated basis   
  SPComplexMatrix<SpAlloc> Luv( {nmu,gnmu},spc_alloc );      // Cholesky decomposition of Muv 
  SPComplexMatrix<SpAlloc> Piu( {NMO,nmu},spc_alloc );      // Interpolating orbitals 
  SPComplexMatrix<SpAlloc> cPua( {nmu,NAEA+NAEB},spc_alloc );     // (half-rotated) Interpolating orbitals 

  // read Half transformed first
  /***************************************/
  if(!dump.read(rotPiu,"HalfTransformedFullOrbitals")) {
    app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
               <<" Problems reading HalfTransformedFullOrbitals. \n";
    APP_ABORT("");
  }
  /***************************************/
  hyperslab_proxy<Matrix,2> hslab_rotMuv(rotMuv,
                                         std::array<size_t,2>{grotnmu,grotnmu}, 
                                         std::array<size_t,2>{rotnmu,grotnmu}, 
                                         std::array<size_t,2>{rotnmu0,0}); 
  if(!dump.read(hslab_rotMuv,"HalfTransformedMuv")) {
  //if(!dump.read(rotMuv,"HalfTransformedMuv")) {
    app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
            <<" Problems reading HalfTransformedMuv. \n";
    APP_ABORT("");
  }
  hyperslab_proxy<Matrix,2> hslab_Piu(Piu,
                                      std::array<size_t,2>{size_t(NMO),gnmu},
                                      std::array<size_t,2>{size_t(NMO),nmu}, 
                                      std::array<size_t,2>{0,nmu0});  
  /***************************************/
  if(!dump.read(hslab_Piu,"Orbitals")) {
    app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
               <<" Problems reading Orbitals. \n";
    APP_ABORT("");
  }
  hyperslab_proxy<Matrix,2> hslab_Luv(Luv,
                                      std::array<size_t,2>{gnmu,gnmu},
                                      std::array<size_t,2>{nmu,gnmu}, 
                                      std::array<size_t,2>{nmu0,0});  
  /***************************************/
  if(!dump.read(hslab_Luv,"Luv")) {
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
    if(walker_type == 2) {
      if(!dump.push(std::string("PsiT_1"),false)) {
        app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
        APP_ABORT("");
      }
      ComplexMatrix<stdAlloc> buff(hdfcsr2ma<ComplexMatrix<stdAlloc>,stdAlloc>(dump,NMO,NAEA,stdAlloc{}));
      cuda::copy_n(buff.origin(),buff.num_elements(),sys.trialwfn_beta.origin());
      dump.pop();
    } else 
      cuda::copy_n(sys.trialwfn_alpha.origin(),sys.trialwfn_alpha.num_elements(),sys.trialwfn_beta.origin());
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
  ComplexMatrix<Alloc> H1( {NMO,NMO}, alloc );
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

  dump.pop();
  dump.pop();
  dump.pop();
  dump.pop();

  return THCOps(NMO,NAEA,NAEA,COLLINEAR,std::move(haj),std::move(rotMuv),
                std::move(rotPiu),std::move(rotcPua),std::move(Luv),std::move(Piu),
                std::move(cPua),E0,alloc,nmu0,rotnmu0);
} 

}  // afqmc


} // qmcplusplus


#endif
