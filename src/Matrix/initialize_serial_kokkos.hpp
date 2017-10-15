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

#include <Kokkos_Core.hpp>

#include "Configuration.h"
#include "io/hdf_archive.h"

#include "AFQMC/afqmc_sys.hpp"

namespace qmcplusplus
{

namespace afqmc
{

template< class SpMat,
          class Mat>
inline bool Initialize(hdf_archive& dump, const double dt, base::afqmc_sys& sys, Mat& Propg1, SpMat& Spvn, Mat& haj, SpMat& Vakbl)
{
  int NMO, NAEA;

  std::cout<<"  Serial hdf5 read. \n";

  if(!dump.is_group( std::string("/Wavefunctions/PureSingleDeterminant") )) {
    app_error()<<" ERROR: H5Group /Wavefunctions/PureSingleDeterminant does not exist. \n";
    return false;
  }
  if(!dump.is_group( std::string("/Propagators/phaseless_ImpSamp_ForceBias") )) {
    app_error()<<" ERROR: H5Group /Wavefunctions/PureSingleDeterminant does not exist. \n";
    return false;
  }
  if(!dump.push("Wavefunctions")) return false;
  if(!dump.push("PureSingleDeterminant")) return false;

  /*
   * 0: ignore
   * 1: global # terms in Vakbl 
   * 2: Vakbl #rows
   * 3: Vakbl #cols
   * 4: NMO
   * 5: NAEA
   * 6: NAEB
   * 7: should be 0
   */    
  std::vector<int> Idata(8);
  if(!dump.read(Idata,"dims")) return false;

  if(Idata[5] != Idata[6] ) {
    std::cerr<<" Error: Expecting NAEA==NAEB. \n"; 
;
    return false;
  } 
  sys.setup(Idata[4],Idata[5]); 
  NMO = Idata[4];
  NAEA = Idata[5];
  if(Idata[7] != 0) {
    std::cerr<<" Error: Found spin unrestricted integrals." <<std::endl;  
    return false;
  }
  if(Idata[2] != 2*NMO*NMO || Idata[3] != 2*NMO*NMO) {
    std::cerr<<" Incorrect dimensions on 2-body hamiltonian: " 
             <<Idata[2] <<" " <<Idata[3] <<std::endl; 
    return false;
  }


  // read 1-body hamiltonian.
  //  - in sparse format in the hdf5
  //  - will be assumed in compact notation in the miniapp, needs to be "compacted"
  std::vector<ValueType> vvec;
  std::vector<IndexType> ivec;
  if(!dump.read(ivec,"hij_indx")) return false;
  if(!dump.read(vvec,"hij")) return false;
  // resize haj 
  // haj.resize(extents[2*NAEA][NMO]);
  Kokkos::resize(haj, 2*NAEA, NMO);
  for(int n=0; n<ivec.size(); n++)
  {
    // ivec[n] = i*NMO+j, with i in [0:2*NMO), j in [0:NMO)
    // vvec[n] = h1(i,j)
    // haj[a][j] = h1(i,j)
    // a = i for i < NMO (which must be in [0:NAEA)) 
    //   = (i - NMO)+NAEA for i >= NMO (which must be in [NMO:NMO+NAEB))
    int i = ivec[n]/NMO;
    int j = ivec[n]%NMO;
    int a = (i<NMO)?i:(i-NMO+NAEA);
    if( i < NMO ) assert(i < NAEA);
    else assert(i-NMO < NAEA);
    // haj[a][j] = vvec[n];
    haj(a, j) = vvec[n];
  } 

  // read trial wave function.
  // trial[i][j] written as a continuous array in C-format
  // i:[0,2*NMO) , j:[0,NAEA)
  if(!dump.read(vvec,"Wavefun")) return false;
  if(vvec.size() != 2*NMO*NAEA) {
    std::cerr<<" Inconsistent dimensions in Wavefun. " <<std::endl;
    return false;
  }
  // sys.trialwfn_alpha.resize(extents[NMO][NAEA]);
  // Kokkos::resize(sys.trialwfn_alpha, NMO, NAEA);
  sys.trialwfn_alpha = ComplexMatrixKokkos("trialwfn_alpha", NMO, NAEA);
  // sys.trialwfn_beta.resize(extents[NMO][NAEA]);
  // Kokkos::resize(sys.trialwfn_beta, NMO, NAEA);
  sys.trialwfn_beta = ComplexMatrixKokkos("trialwfn_beta", NMO, NAEA);
  int ij=0;
  for(int i=0; i<NMO; i++)
   for(int j=0; j<NAEA; j++, ij++) {
    using std::conj;
    // sys.trialwfn_alpha[i][j] = conj(vvec[ij]);
    sys.trialwfn_alpha(i, j) = conj(vvec[ij]);
   }
  for(int i=0; i<NMO; i++)
   for(int j=0; j<NAEA; j++, ij++) {
    using std::conj;
    // sys.trialwfn_beta[i][j] = conj(vvec[ij]);
    sys.trialwfn_beta(i, j) = conj(vvec[ij]);
   }

  // read half-rotated hamiltonian
  // careful here!!!
  Vakbl.setDims(Idata[2],Idata[3]);
  Vakbl.resize(Idata[1]);
  if(!dump.read(*(Vakbl.getVals()),"SpHijkl_vals")) return false;
  if(!dump.read(*(Vakbl.getCols()),"SpHijkl_cols")) return false;
  if(!dump.read(*(Vakbl.getRowIndex()),"SpHijkl_rowIndex")) return false;
  Vakbl.setRowsFromRowIndex();
  // morph to "compacted" notation for miniapp
  { 
    typename SpMat::int_iterator it = Vakbl.cols_begin();
    typename SpMat::int_iterator itend = Vakbl.cols_end();
    for(; it!=itend; ++it) {
      int i = (*it)/NMO;
      int j = (*it)%NMO;
      int a = (i<NMO)?i:(i-NMO+NAEA);
      if( i < NMO ) assert(i < NAEA);
      else assert(i-NMO < NAEA);
      *it = a*NMO+j;
    }
    it = Vakbl.rows_begin();    
    itend = Vakbl.rows_end();
    for(; it!=itend; ++it) {
      int i = (*it)/NMO;
      int j = (*it)%NMO;
      int a = (i<NMO)?i:(i-NMO+NAEA);
      if( i < NMO ) assert(i < NAEA);
      else assert(i-NMO < NAEA);
      *it = a*NMO+j;
    }    
  }
  Vakbl.setDims(2*NMO*NAEA,2*NMO*NAEA);
  Vakbl.compress();  // Should already be compressed, but just in case

  dump.pop();
  dump.pop();
  // done reading wavefunction 

  // read propagator 
  if(!dump.push("Propagators",false)) return false;
  if(!dump.push("phaseless_ImpSamp_ForceBias",false)) return false;

  std::vector<long> Ldims(5);
  if(!dump.read(Ldims,"Spvn_dims")) return false;
    
  long ntot = Ldims[0];       // total number of terms
  int nrows = int(Ldims[1]);  // number of rows 
  int nvecs = int(Ldims[2]);  // number of cholesky vectors
  int nblk = int(Ldims[4]);   // number of blocks

  assert(nrows == NMO*NMO);
  assert(static_cast<int>(Ldims[3]) == NMO);

  // read 1-body propagator
  if(!dump.read(vvec,"Spvn_propg1")) return false;
  if(vvec.size() != NMO*NMO) {
    std::cerr<<" Incorrect dimensions on 1-body propagator: " <<vvec.size() <<std::endl;
    return false;
  }
  // Propg1.resize(extents[NMO][NMO]);
  Kokkos::resize(Propg1, NMO, NMO);
  for(int i=0,ij=0; i<NMO; i++)       
    for(int j=0; j<NMO; j++, ij++)
      // Propg1[i][j] = vvec[ij];
      Propg1(i, j) = vvec[ij];

  // reserve space
  Spvn.setDims(nrows,nvecs);
  Spvn.reserve(ntot);

  std::vector<int> counts(nblk);
  if(!dump.read(counts,"Spvn_block_sizes")) return false;  

  int maxsize = *std::max_element(counts.begin(), counts.end());
  vvec.reserve(maxsize);
  ivec.reserve(2*maxsize);

  // read blocks
  for(int i=0; i<nblk; i++) {
   
    // read index and data
    vvec.resize(counts[i]);
    ivec.resize(2*counts[i]);
    if(!dump.read(ivec,std::string("Spvn_index_")+std::to_string(i))) return false;
    if(!dump.read(vvec,std::string("Spvn_vals_")+std::to_string(i))) return false;

    for(int n=0; n<counts[i]; n++)
      Spvn.add(ivec[2*n],ivec[2*n+1],vvec[n]);
    
  }
  Spvn.compress();
  Spvn *= std::sqrt(dt);

  dump.pop();
  dump.pop();
  // done reading propagator

  return true;
} 

}  // afqmc


} // qmcplusplus


#endif
