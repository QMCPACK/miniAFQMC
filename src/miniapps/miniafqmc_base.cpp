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
#include <random>

#include <Configuration.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <getopt.h>
#include "io/hdf_archive.h"

#include "AFQMC/afqmc_sys.hpp"
#include "Matrix/initialize_serial.hpp"
#include "AFQMC/mixed_density_matrix.hpp"
#include "AFQMC/energy.hpp"
#include "AFQMC/vHS.hpp"
#include "AFQMC/vbias.hpp"

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

  // if half-tranformed transposed Spvn is on file  
  bool transposed_Spvn = false;

  ValueType one(1.),zero(0.),half(0.5);
  ComplexType cone(1.),czero(0.);
  ComplexType im(0.0,1.);
  ComplexType halfim(0.0,0.5);

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "hdvs:g:i:b:c:a:r:w:")) != -1)
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
    case 'w': // the number of sub steps for drift/diffusion
      nwalk = atoi(optarg);
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
  afqmc_sys AFQMCSys; // Basic information about the simulation, e.g. #orbitals, #electrons, etc.
  ValueSpMat Spvn; // (Symmetric) Factorized Hamiltonian, e.g. <ij|kl> = sum_n Spvn(ik,n) * Spvn(jl,n)
  ValueSpMat SpvnT; // (Symmetric) Factorized Hamiltonian, e.g. <ij|kl> = sum_n Spvn(ik,n) * Spvn(jl,n)
  ValueMatrix haj;    // 1-Body Hamiltonian Matrix
  ValueSpMat Vakbl;   // 2-Body Hamiltonian Matrix: (Half-Rotated) 2-electron integrals 
  ValueMatrix Propg1; // propagator for 1-body hamiltonian 

  index_gen indices;

  hdf_archive dump;
  if(!dump.open(init_file,H5F_ACC_RDONLY)) 
    APP_ABORT("Error: problems opening hdf5 file. \n");

  cout<<" Hello World. \n";

  if(!afqmc::Initialize(dump,dt,AFQMCSys,Propg1,Spvn,haj,Vakbl)) {
    app_error()<<" Error initalizing data structures from hdf5 file: " <<init_file <<std::endl;
    APP_ABORT(" Abort. \n\n\n");
  }
   
  AFQMCSys.print(cout);

  RealType Eshift = 0;
  int NMO = AFQMCSys.NMO;              // number of molecular orbitals
  int NAEA = AFQMCSys.NAEA;            // number of up electrons
  int nchol = Spvn.cols();            // number of cholesky vectors  
  int NIK = 2*NMO*NMO;                // dimensions of linearized green function
  int NAK = 2*NAEA*NMO;               // dimensions of linearized "compacted" green function

  ValueMatrix vbias(extents[nchol][nwalk]);     // bias potential
  ValueMatrix vHS(extents[NMO*NMO][nwalk]);        // Hubbard-Stratonovich potential
  ValueMatrix G(extents[NIK][nwalk]);           // density matrix
  ValueMatrix Gc(extents[NAK][nwalk]);           // compact density matrix for energy evaluation
  ValueMatrix X(extents[nchol][nwalk]);         // X(n,nw) = rand(n,nw) ( + vbias(n,nw)) 

  ValueVector hybridW(extents[nwalk]);         // stores weight factors
  ValueVector eloc(extents[nwalk]);         // stores local energies

  WalkerContainer W(extents[nwalk][2][NMO][NAEA]);
  // 0: eloc, 1: weight, 2: ovlp_up, 3: ovlp_down, 4: old_eloc, 5: old_ovlp_alpha, 6: old_ovlp_beta
  ValueMatrix W_data(extents[nwalk][7]);  
  // initialize walkers to trial wave function
  for(int n=0; n<nwalk; n++) 
    for(int nm=0; nm<NMO; nm++) 
      for(int na=0; na<NAEA; na++) {
        using std::conj;
        W[n][0][nm][na] = conj(AFQMCSys.trialwfn_alpha[nm][na]);
        W[n][1][nm][na] = conj(AFQMCSys.trialwfn_beta[nm][na]);
      }

  // set weights to 1
  for(int n=0; n<nwalk; n++) 
    W_data[n][1] = ValueType(1.);

  // initialize overlaps and energy
  AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc,true);
  RealType Eav = AFQMCSys.calculate_energy(W_data,Gc,haj,Vakbl);
  
  std::cout<<" Starting steps. \n";

//for(int i=0; i<NMO; i++)
//for(int j=0; j<NMO; j++)
//std::cout<<i <<" " <<j <<" " <<Propg1[i][j] <<std::endl;

  for(int step = 0; step < nsteps; step++) {
  
    for(int substep = 0; substep < nsubsteps; substep++) {

      // propagate walker forward 
      
      // 1. calculate density matrix 
      
      if(transposed_Spvn) {

        APP_ABORT(" transposed Spvn not implemented: \n\n\n");

        AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc,true);

        // 2. calculate bias potential
        base::get_vbias(SpvnT,Gc,vbias,true);  
  
      } else {

        AFQMCSys.calculate_mixed_density_matrix(W,W_data,G,false); 

        // 2. calculate bias potential
        base::get_vbias(Spvn,G,vbias,false);

      } 

      // 3. calculate X and weight
      //  X(chol,nw) = rand + i*vbias(chol,nw)
      random_th.generate_normal(X.data(),X.num_elements()); 

      for(int n=0; n<nchol; n++)
        for(int nw=0; nw<nwalk; nw++) { 
          hybridW[nw] -= im*vbias[n][nw]*(X[n][nw]+halfim*vbias[n][nw]);
          X[n][nw] += im*vbias[n][nw];
        }

      // 4. calculate vHS
      // vHS(i,k,nw) = sum_n Spvn(i,k,n) * X(n,nw) 
      base::get_vHS(Spvn,X,vHS);      

      // 5. propagate walker
      // W(new) = Propg1 * exp(vHS) * Propg1 * W(old)
      AFQMCSys.propagate(W,Propg1,vHS);

      // 6. update overlaps
      AFQMCSys.calculate_overlaps(W,W_data);

      // 7. adjust weights and walker data      
    }

//    orthogonalize(wset);

    AFQMCSys.calculate_mixed_density_matrix(W,W_data,Gc,true);
    Eav = AFQMCSys.calculate_energy(W_data,Gc,haj,Vakbl);
    std::cout<<" step: " <<step <<" " <<Eav <<"\n";

    // Branching in real code would happen here!!!
  
  }    

  return 0;
}
