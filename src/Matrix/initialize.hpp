//////////////////////////////////////////////////////////////////////////////////////
////// This file is distributed under the University of Illinois/NCSA Open Source License.
////// See LICENSE file in top directory for details.
//////
////// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//////
////// File developed by: 
//////
////// File created by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory 
//////////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_INITIALIZE_HPP
#define QMCPLUSPLUS_AFQMC_INITIALIZE_HPP

#include<string>
#include<vector>

#include "Configuration.h"
#include "io/hdf_archive.h"

#include "AFQMC/AFQMCInfo.hpp"
#include "Utilities/taskgroup.hpp"
#include "array_partition.hpp"
#include "hdf5_readers.hpp"

namespace qmcplusplus
{

namespace afqmc
{

template< class SpMat,
          class Mat,
          class task_group  >
inline bool Initialize(hdf_archive& dump, AFQMCInfo& sys, task_group& TG, Mat& Propg1, SpMat& Spvn, Mat& hij, SpMat& Vakbl, Mat& trialwfn, int nread=0)
{
  int nnodes = TG.getTotalNodes();
  int nodeid = TG.getNodeID();
  int ncores = TG.getTotalCores();
  int coreid = TG.getCoreID();
  if(nread<1) nread = ncores; // how many cores in a node read??? 
  int NMO, NAEA, NAEB;
  int head = (nodeid==0)&&(coreid==0);
  bool reader = (coreid<nread);
  int scl = sizeof(ValueType)/sizeof(RealType);

  if(head) {

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
    MPI_Bcast(Idata.data(),Idata.size(),MPI_INT,0,MPI_COMM_WORLD);

    if(Idata[2] != 2*NMO*NMO || Idata[3] != 2*NMO*NMO) {
      std::cerr<<" Incorrect dimensions on 2-body hamiltonian: " <<Idata[2] <<" " <<Idata[3] <<std::endl; 
      return false;
    }
    NMO = Idata[4];
    NAEA = Idata[5];
    NAEB = Idata[6];
    if(Idata[7] != 0) {
      std::cerr<<" Error: Found spin unrestricted integrals." <<std::endl;  
      return false;
    }

    Vakbl.setDims(Idata[2],Idata[3]);

    // read 1-body hamiltonian 
    std::vector<ValueType> vec;
    if(!dump.read(vec,"Wavefun_hij")) return false;
    if(vec.size() != NMO*NMO) {
      std::cerr<<" Incorrect dimensions on 1-body hamiltonian: " <<vec.size() <<std::endl; 
      return false;
    }
    MPI_Bcast(vec.data(),vec.size()*scl,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for(int i=0,ij=0; i<NMO; i++) 
      for(int j=0; j<NMO; j++, ij++)
        hij(i,j) = vec[ij];


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

  

    // read 1-body propagator
    if(!dump.read(vec,"Spvn_propg1")) return false;
    if(vec.size() != NMO*NMO) {
      std::cerr<<" Incorrect dimensions on 1-body propagator: " <<vec.size() <<std::endl;
      return false;
    }
    MPI_Bcast(vec.data(),vec.size()*scl,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for(int i=0,ij=0; i<NMO; i++)       
      for(int j=0; j<NMO; j++, ij++)
        Propg1(i,j) = vec[ij];

    simple_matrix_partition<afqmc::TaskGroup,IndexType,RealType> split(nrows,nvecs,1e-6);
    std::vector<IndexType> counts;
    // count dimensions of sparse matrix
    afqmc::count_entries_hdf5_SpMat(dump,split,std::string("Spvn"),nblk,false,counts,TG,true,nread);
 

    dump.pop();
    dump.pop();
    // done reading propagator

  } else {


  } 

  return true;
} 

}  // afqmc


} // qmcplusplus


#endif
