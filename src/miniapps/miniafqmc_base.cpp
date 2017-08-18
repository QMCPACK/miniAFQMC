////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
// clang-format off
/** @file miniafqmc.cpp
    @brief Miniapp 
 
 @mainpage MiniAFQMC: miniapp for QMCPACK AFQMC kernels

 */

 /*!
 \page 
 */
// clang-format on

#include <Configuration.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <getopt.h>
#include "io/hdf_archive.h"

#include "AFQMC/AFQMCInfo.hpp"
#include "Matrix/initialize_serial.hpp"
#include "AFQMC/mixed_density_matrix.hpp"

// temporary
#include "Numerics/SparseMatrixOperations.hpp"

using namespace std;
using namespace qmcplusplus;

enum MiniQMCTimers
{
  Timer_Total,
};

TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames = {
    {Timer_Total, "Total"},
};

void print_help()
{
  printf("miniafqmc - QMCPACK AFQMC miniapp\n");
  printf("\n");
  printf("Options:\n");
  printf("-i                Number of MC steps (default: 100)\n");
  printf("-s                Number of substeps (default: 1)\n");
  printf("-w                Number of walkers (default: 16)\n");
  printf("-v                Verbose output\n");
}

int main(int argc, char **argv)
{

  int nsteps=100;
  int nsubsteps=1; 
  int nwalk=16;
  const double dt = 0.01;  // 1-body propagators are assumed to be generated with a timestep = 0.01

  bool verbose = false;
  int iseed   = 11;
  std::string init_file = "afqmc.h5";

  bool transposed_Spvn = true;

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "hdvs:g:i:b:c:a:r:")) != -1)
  {
    switch (opt)
    {
    case 'h': print_help(); return 1;
    case 'i': // number of MC steps
      nsteps = atoi(optarg);
      break;
    case 's': // the number of sub steps for drift/diffusion
      nsubsteps = atoi(optarg);
      break;
    case 'v': verbose  = true; break;
    }
  }

  Random.init(0, 1, iseed);
  int ip = 0;
  PrimeNumberSet<uint32_t> myPrimes;
  // create generator within the thread
  RandomGenerator<RealType> random_th(myPrimes[ip]);

  TimerManager.set_timer_threshold(timer_level_coarse);
  TimerList_t Timers;
  setup_timers(Timers, MiniQMCTimerNames, timer_level_coarse);

  // Important Data Structures
  AFQMCInfo SysInfo; // Basic information about the simulation, e.g. #orbitals, #electrons, etc.
  ValueSpMat Spvn; // (Symmetric) Factorized Hamiltonian, e.g. <ij|kl> = sum_n Spvn(ik,n) * Spvn(jl,n)
  ValueMatrix haj;    // 1-Body Hamiltonian Matrix
  ValueSpMat Vakbl;   // 2-Body Hamiltonian Matrix: (Half-Rotated) 2-electron integrals 
  ValueMatrix trialwfn; // Slater Matrix representing the trial wave-function 
  ValueMatrix Propg1; // propagator for 1-body hamiltonian 

  index_gen indices;

  hdf_archive dump;
  if(!dump.open(init_file,H5F_ACC_RDONLY)) 
    APP_ABORT("Error: problems opening hdf5 file. \n");

  cout<<" Hello World. \n";

  if(!afqmc::Initialize(dump,dt,SysInfo,Propg1,Spvn,haj,Vakbl,trialwfn)) {
    app_error()<<" Error initalizing data structures from hdf5 file: " <<init_file <<std::endl;
    APP_ABORT(" Abort. \n\n\n");
  }
   
  SysInfo.print(cout);

  RealType Eshift = 0;
  int NMO = SysInfo.NMO;              // number of molecular orbitals
  int NAEA = SysInfo.NAEA;            // number of up electrons
  int nchol = Spvn.cols();            // number of cholesky vectors  
  int NIK = 2*NMO*NMO;                // dimensions of linearized green function
  int NAK = 2*NAEA*NMO;               // dimensions of linearized "compacted" green function

  // check assumptions 
  // nup==ndown
  assert(SysInfo.NAEA == SysInfo.NAEB);
  // UHF calculation 
  assert(trialwfn.shape()[0]==2*NMO);           // enforce UHF 
  assert(trialwfn.shape()[1]==NAEA);   

  ValueMatrix vbias(extents[nchol][nwalk]);     // bias potential
  ValueMatrix vHS(extents[NIK][nwalk]);        // Hubbard-Stratonovich potential
  ValueMatrix G(extents[NIK][nwalk]);           // density matrix
  ValueMatrix Gc(extents[NAK][nwalk]);           // compact density matrix for energy evaluation
  ValueMatrix Gcloc(extents[NAK][nwalk]);           // density matrix
  ValueMatrix X(extents[nchol][nwalk]);         // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 

  ValueMatrix DM(extents[NMO][NMO]);            // density matrix for a single walker
  ValueMatrix DMc(extents[NAEA][NMO]);          // compact density matrix for a single walker
  ValueVector hybridW(extents[nwalk]);         // stores weight factors
  ValueVector eloc(extents[nwalk]);         // stores local energies

  // work arrays 
  ValueMatrix TWORK1(extents[NAEA][NAEA]);
  ValueMatrix TWORK2(extents[NAEA][NMO]);
  ValueMatrix TWORK3(extents[NAEA][NMO]);
  IndexVector IWORK1(extents[2*NMO]); 
  ValueMatrix S0(extents[NAEA][NMO]);

  boost::multi_array_ref<ValueType,1> haj_ref(haj.data(), extents[haj.num_elements()]);

  boost::multi_array_ref<ValueType,3> G_3D(G.data(), extents[2*NMO][NMO][nwalk]);
  boost::multi_array_ref<ValueType,3> Gc_3D(Gc.data(), extents[2*NAEA][NMO][nwalk]);
  boost::multi_array_ref<ValueType,3> vHS_3D(vHS.data(), extents[NMO][NMO][nwalk]);

  boost::multi_array_ref<ValueType,2> trialwfn_alpha(trialwfn.data(), extents[NMO][NAEA]);
  boost::multi_array_ref<ValueType,2> trialwfn_beta(trialwfn.data()+NAEA*NMO, extents[NMO][NAEA]);

  assert(haj_ref.shape()[0] == Gc.shape()[0]);


  {
    ValueMatrix SM(extents[NMO][NAEA]);
    for(int i=0; i<NMO; i++)
     for(int j=0; j<NAEA; j++)
      SM[i][j] = std::conj(trialwfn_alpha[i][j]);
    ValueType ova = base::MixedDensityMatrix<ValueType>(trialwfn_alpha,SM,DMc,IWORK1,TWORK1,TWORK2,true);
    Gc_3D[ indices[range_t(0,NAEA)][range_t(0,NMO)][0] ] = DMc;
    for(int i=0; i<NMO; i++)
     for(int j=0; j<NAEA; j++)
      SM[i][j] = std::conj(trialwfn_beta[i][j]);
    ValueType ovb = base::MixedDensityMatrix<ValueType>(trialwfn_beta,SM,DMc,IWORK1,TWORK1,TWORK2,true);
    Gc_3D[ indices[range_t(NAEA,2*NAEA)][range_t(0,NMO)][0] ] = DMc;

    ma::product(ma::T(Gc),haj_ref,eloc); 

    ValueType el2=0.;
    // Vakbl * Gc(bl,nw) = T1(ak,nw)
    SparseMatrixOperators::product_SpMatM( NAK, nwalk, NAK, ValueType(1.), Vakbl.values(), Vakbl.column_data(), Vakbl.row_index(), Gc.data(), Gc.strides()[0], ValueType(0.), Gcloc.data(), Gcloc.strides()[0] );
    for(int i=0; i<NAK; i++) el2 += Gc[i][0]*Gcloc[i][0]; 
    el2*=0.5;

    std::cout<<"eloc: " <<eloc[0] <<" " <<el2 <<std::endl;
  }

/*
  for(int step = 0; step < nsteps; step++) {
  
    for(int substep = 0; substep < nsubsteps; substep++) {

      // propagate walker forward 
      
      // 1. calculate density matrix 
      if(transposed_Spvn) {

        for(int nw=0; nw<nwalk; nw++) {

          // alpha:  SM(nw, k) : k=0:alpha/alpha block, 1:beta/beta block, 2:full matrix (2N,M)
          ValueType ova = MixedDensityMatrix<ValueType>(trialwfn,wset.SM(nw,0),DMc,IWORK1,TWORK1,TWORK2,true);
          Gc_3D[ indices[range_t(0,NAEA)][range_t(0,NMO)][nw] ] = DMc;
          ValueType ovb = MixedDensityMatrix<ValueType>(trialwfn,wset.SM(nw,1),DMc,IWORK1,TWORK1,TWORK2,true);
          Gc_3D[ indices[range_t(NAEA,2*NAEA)][range_t(0,NMO)][nw] ] = DMc;

        }

        // 2. calculate bias potential
        get_vbias<true>(1.,SpvnT,Gc[ indices[range_t(0,NAEA*NMO)][range_t(0,nwalk)] ]    ,0.,vbias);      
        get_vbias<true>(1.,SpvnT,Gc[ indices[range_t(NAEA*NMO,2*NAEA*NMO)][range_t(0,nwalk)] ],1.,vbias);      
  
      } else {

        for(int nw=0; nw<nwalk; nw++) {

          // alpha:  SM(nw, k) : k=0:alpha/alpha block, 1:beta/beta block, 2:full matrix (2N,M)
          ValueType ova = MixedDensityMatrix<ValueType>(trialwfn,wset.SM(nw,0),DM,IWORK1,TWORK1,TWORK2,false);
          G_3D[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ] = DM;
          ValueType ovb = MixedDensityMatrix<ValueType>(trialwfn,wset.SM(nw,1),DM,IWORK1,TWORK1,TWORK2,false);
          G_3D[ indices[range_t(NMO,2*NMO)][range_t(0,NMO)][nw] ] = DM;

        }

        // 2. calculate bias potential
        get_vbias<false>(1.,Spvn,G[ indices[range_t(0,NMO*NMO)][range_t(0,nwalk)] ]    ,0.,vbias);      
        get_vbias<false>(1.,Spvn,G[ indices[range_t(NMO*NMO,2*NMO*NMO)][range_t(0,nwalk)] ],1.,vbias);      

      } 

      // 3. generate random numbers       
      get_random(X);        

      // 4. calculate X and weight
      for(int n=0; n<nchol; n++)
       for(int nw=0; nw<nwalk; nw++) 
         hybridW[nw] -= vbias[n][nw]*(X[n][nw]+0.5*vbias[n][nw]);
      ma::axpy(1.,vbias,X);     

      // 5. calculate vHS
      get_vHS(Spvn,X,vHS);      

      // 6. propagate walker
      for(int nw=0; nw<nwalk; nw++) { 

        // this will make a deep copy correct?
        //DM = vHS_3D[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ];

        // alpha
        ma::product(Propg1,wset.SM(nw,0),S0);                   
        apply_vHS( vHS_3D[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ] ,S0,TWORK1,TWORK3,6);       
        ma::product(Propg1,S0,TWORK1);                  
        wset.setSM(TWORK1,0);

        ma::product(Propg1,wset.SM(nw,1),S0);                   
        apply_vHS( vHS_3D[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ] ,S0,TWORK1,TWORK3,6);       
        ma::product(Propg1,S0,TWORK1);                  
        wset.setSM(TWORK1,1);
      }

      // 7. calculate new overlaps
      get_overlaps();

      // 8. adjust weights and walker data      
    }

    orthogonalize(wset);

    RealType eloc_ave = get_energy(wset), e_=eloc_ave;

    // Branching in real code would happen here!!!
  
  }    

*/

  return 0;
}
