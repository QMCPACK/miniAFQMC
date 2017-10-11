/////////////////////////////////////////////////////////////////////
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
#include <Utilities/NewTimer.h>

#include "AFQMC/afqmc_sys_shm.hpp"
#include "array_partition.hpp"
#include "hdf5_readers.hpp"

namespace qmcplusplus
{

template< class SpMat,
          class Mat,
          class task_group
        >
inline bool Initialize(hdf_archive& dump, const double dt, task_group& TG, shm::afqmc_sys& sys, Mat& Propg1, SpMat& Spvn, Mat& haj, SpMat& Vakbl, std::vector<int>& sets, int nread)
{

  enum InitTimers
  {
    read_Spvn,
    sort_Spvn,
    read_Vakbl,
    sort_Vakbl,
  };
  TimerNameList_t<InitTimers> TimerNames = {
    {read_Spvn,"read_Spvn"},
    {sort_Spvn,"sort_Spvn"},
    {read_Vakbl,"read_Vakbl"},
    {sort_Vakbl,"sort_Vakbl"},
  };  
  TimerList_t Timers;

  int NMO, NAEA;
  int nnodes = TG.getTotalNodes();
  int nodeid = TG.getNodeID();
  int ncores = TG.getTotalCores();
  int coreid = TG.getCoreID();
  int ncores_per_TG = TG.getNCoresPerTG();
  int nnodes_per_TG = TG.getNNodesPerTG();
  if(nread<1) nread = ncores; // how many cores in a node read??? 
  nread = std::min(nread,ncores); 
  int head = (nodeid==0)&&(coreid==0);
  bool reader = (coreid<nread);
  int scl = sizeof(ValueType)/sizeof(RealType); // no templated mpi wrapper, so do it by hand for now
  sets.resize(TG.getNNodesPerTG()+1);

  // heads reads and bcasts everything except Vakbl and Spvn.
  // coreid<nread read simultaneously Vakbl and Spvn 
  
  app_log()<<"  Distributed read of hdf5 using all nodes and " <<nread <<" cores per node. \n";

  if(head) {

    setup_timers(Timers, TimerNames, timer_level_coarse); 

    if(!dump.is_group( std::string("/Wavefunctions/PureSingleDeterminant") )) {
      app_error()<<" ERROR: H5Group /Wavefunctions/PureSingleDeterminant does not exist."<<std::endl;
      return false;
    }
    if(!dump.is_group( std::string("/Propagators/phaseless_ImpSamp_ForceBias") )) {
      app_error()<<" ERROR: H5Group /Wavefunctions/PureSingleDeterminant does not exist."<<std::endl;
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
    NMO = Idata[4];
    NAEA = Idata[5];
    if(Idata[7] != 0) {
      std::cerr<<" Error: Found spin unrestricted integrals." <<std::endl;  
      MPI_Abort(MPI_COMM_WORLD,2);
    }
    if(Idata[2] != 2*NMO*NMO || Idata[3] != 2*NMO*NMO) {
      std::cerr<<" Incorrect dimensions on 2-body hamiltonian: " 
             <<Idata[2] <<" " <<Idata[3] <<std::endl; 
      MPI_Abort(MPI_COMM_WORLD,2);
    }
    MPI_Bcast(Idata.data(),Idata.size(),MPI_INT,0,MPI_COMM_WORLD);

    // setup AFQMCSys
    sys.setup(Idata[4],Idata[5]); 


    // read 1-body hamiltonian.
    //  - in sparse format in the hdf5
    //  - will be assumed in compact notation in the miniapp, needs to be "compacted"
    std::vector<ValueType> vvec;
    std::vector<IndexType> ivec;
    if(!dump.read(ivec,"hij_indx")) return false;
    if(!dump.read(vvec,"hij")) return false;
    // resize haj 
    haj.resize(extents[2*NAEA][NMO]);
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
      haj[a][j] = vvec[n];
    } 
    // I REALLY NEED AN MPI CLASS!!!
    typedef typename std::decay<Mat>::type::element MatType; 
    MPI_Bcast(reinterpret_cast<char*>(haj.data()),sizeof(MatType)*haj.num_elements(),
        MPI_CHAR,0,MPI_COMM_WORLD);

    // read trial wave function.
    // trial[i][j] written as a continuous array in C-format
    // i:[0,2*NMO) , j:[0,NAEA)
    if(!dump.read(vvec,"Wavefun")) return false;
    if(vvec.size() != 2*NMO*NAEA) {
      std::cerr<<" Inconsistent dimensions in Wavefun. " <<std::endl;
      return false;
    }
    sys.trialwfn_alpha.resize(extents[NMO][NAEA]);
    sys.trialwfn_beta.resize(extents[NMO][NAEA]);
    int ij=0;
    for(int i=0; i<NMO; i++)
      for(int j=0; j<NAEA; j++, ij++) {
        using std::conj;
        sys.trialwfn_alpha[i][j] = conj(vvec[ij]);
      } 
    for(int i=0; i<NMO; i++)
      for(int j=0; j<NAEA; j++, ij++) {
        using std::conj;
        sys.trialwfn_beta[i][j] = conj(vvec[ij]);
      }
    MPI_Bcast(reinterpret_cast<char*>(sys.trialwfn_alpha.data()),
        sys.trialwfn_alpha.num_elements()*sizeof(ComplexType),
        MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(reinterpret_cast<char*>(sys.trialwfn_beta.data()),
        sys.trialwfn_beta.num_elements()*sizeof(ComplexType),
        MPI_CHAR,0,MPI_COMM_WORLD);

    // read half-rotated hamiltonian
    // careful here!!!
    Timers[read_Vakbl]->start();
    {
      simple_matrix_partition<task_group,IndexType,RealType> split(Idata[2],Idata[3],1e-8);
      std::vector<IndexType> counts;
      // count dimensions of sparse matrix
      if(!count_entries_hdf5_SpMat(dump,split,std::string("SpHijkl"),true,counts,TG,true,nread))
        return false;

      split.partition(TG,true,counts,sets);

      app_log()<<"  Partitioning of Hamiltonian Matrix (rows): ";
      for(int i=0; i<=nnodes_per_TG; i++)
        app_log()<<sets[i] <<" ";
      app_log()<<std::endl;
      app_log()<<"  Number of terms in each partitioning: ";
      std::size_t mmin = std::accumulate(counts.begin()+sets[0],counts.begin()+sets[1],std::size_t(0));
      std::size_t mmax = mmin; 
      for(int i=0; i<nnodes_per_TG; i++) {
        std::size_t n = std::accumulate(counts.begin()+sets[i],counts.begin()+sets[i+1],std::size_t(0));
        mmin = std::min(n,mmin);
        mmax = std::max(n,mmax);
        app_log()<<n <<" ";
      }
      app_log()<<"\n  Largest work load variation (%): " <<double(mmax-mmin)*100.0/double(std::accumulate(counts.begin(),counts.end(),std::size_t(0))) <<std::endl;

      MPI_Bcast(sets.data(), sets.size(), MPI_INT, 0, MPI_COMM_WORLD);

      // resize Vakbl 
      int r0 = sets[TG.getLocalNodeNumber()];
      int rN = sets[TG.getLocalNodeNumber()+1];
      int sz = std::accumulate(counts.begin()+r0,counts.begin()+rN,0);
      MPI_Bcast(&sz, 1, MPI_INT, 0, TG.getNodeCommLocal());

      Vakbl.setup("miniAFQMC_Vakbl",TG.getNodeCommLocal());
      Vakbl.setDims(Idata[2],Idata[3]);
      Vakbl.reserve(sz);

      // read Spvn
      if(!read_hdf5_SpMat(Vakbl,split,dump,std::string("SpHijkl"),TG,true,nread))
        return false;
      
      // morph to "compacted" notation for miniapp
      // reset r0/rN  
      r0 = std::numeric_limits<int>::max();
      rN = 0; 
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
        if(a*NMO+j < r0) r0=a*NMO+j;
        if(a*NMO+j+1 > rN) rN=a*NMO+j+1;
      }    
      // shift rows to range [0,rN-r0)  
      for(it = Vakbl.rows_begin(),itend = Vakbl.rows_end(); it!=itend; ++it) *it -= r0; 
      MPI_Bcast(&r0, 1, MPI_INT, 0, TG.getNodeCommLocal());  
      MPI_Bcast(&rN, 1, MPI_INT, 0, TG.getNodeCommLocal());  
      Vakbl.setOffset(r0,0);  
      Vakbl.setDims(rN-r0,2*NMO*NAEA);

    }
    MPI_Barrier(MPI_COMM_WORLD);
    Timers[read_Vakbl]->stop();
    Timers[sort_Vakbl]->start();
    Vakbl.compress(TG.getNodeCommLocal());
    Timers[sort_Vakbl]->stop();

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

    if(nrows != NMO*NMO) return false;
    if(static_cast<int>(Ldims[3]) != NMO) return false;

    MPI_Bcast(Ldims.data(),Ldims.size(),MPI_LONG,0,MPI_COMM_WORLD);
    

    // read 1-body propagator
    Timers[read_Spvn]->start();
    if(!dump.read(vvec,"Spvn_propg1")) return false;
    if(vvec.size() != NMO*NMO) {
      std::cerr<<" Incorrect dimensions on 1-body propagator: " <<vvec.size() <<std::endl;
      return false;
    }
    Propg1.resize(extents[NMO][NMO]);
    for(int i=0,ij=0; i<NMO; i++)       
      for(int j=0; j<NMO; j++, ij++)
        Propg1[i][j] = vvec[ij];
    MPI_Bcast(reinterpret_cast<char*>(Propg1.data()),sizeof(MatType)*Propg1.num_elements(),
        MPI_CHAR,0,MPI_COMM_WORLD);

    simple_matrix_partition<task_group,IndexType,RealType> split(nrows,nvecs,1e-8);
    std::vector<IndexType> counts;
    // count dimensions of sparse matrix
    if(!count_entries_hdf5_SpMat(dump,split,std::string("Spvn"),false,counts,TG,true,nread))    
      return false;

    split.partition(TG,false,counts,sets);

    app_log()<<"  Partitioning of Cholesky Vectors: ";
    for(int i=0; i<=nnodes_per_TG; i++)
      app_log()<<sets[i] <<" ";
    app_log()<<std::endl;
    app_log()<<"  Number of terms in each partitioning: ";
    std::size_t mmin = std::accumulate(counts.begin()+sets[0],counts.begin()+sets[1],std::size_t(0));
    std::size_t mmax = mmin;
    for(int i=0; i<nnodes_per_TG; i++) {
      std::size_t n = std::accumulate(counts.begin()+sets[i],counts.begin()+sets[i+1],std::size_t(0));
      mmin = std::min(n,mmin);
      mmax = std::max(n,mmax);
      app_log()<<n <<" ";
    }
    app_log()<<"\n  Largest work load variation (%): " <<double(mmax-mmin)*100.0/double(std::accumulate(counts.begin(),counts.end(),std::size_t(0))) <<std::endl;

    MPI_Bcast(sets.data(), sets.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // resize Spvn
    int cvec0 = sets[TG.getLocalNodeNumber()];
    int cvecN = sets[TG.getLocalNodeNumber()+1];
    int sz = std::accumulate(counts.begin()+cvec0,counts.begin()+cvecN,0);
    MPI_Bcast(&sz, 1, MPI_INT, 0, TG.getNodeCommLocal());

    Spvn.setup("miniAFQMC_Spvn",TG.getNodeCommLocal());
    Spvn.setDims(nrows,cvecN-cvec0);
    Spvn.setOffset(0,cvec0);
    Spvn.reserve(sz);

    // read Spvn
    if(!read_hdf5_SpMat(Spvn,split,dump,std::string("Spvn"),TG,true,nread))
      return false;
    Timers[read_Spvn]->stop();

    Timers[sort_Spvn]->start();
    Spvn.compress(TG.getNodeCommLocal());
    Timers[sort_Spvn]->stop();

    // scale by sqrt(dt)
    Spvn *= std::sqrt(dt);

    dump.pop();
    dump.pop();
    // done reading propagator

  } else {

    if(reader) {
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
    }

    std::vector<int> Idata(8);
    MPI_Bcast(Idata.data(),Idata.size(),MPI_INT,0,MPI_COMM_WORLD);

    NMO = Idata[4];
    NAEA = Idata[5];
    sys.setup(Idata[4],Idata[5]);

    // resize haj 
    haj.resize(extents[2*NAEA][NMO]);
    typedef typename std::decay<Mat>::type::element MatType;
    MPI_Bcast(reinterpret_cast<char*>(haj.data()),sizeof(MatType)*haj.num_elements(),
        MPI_CHAR,0,MPI_COMM_WORLD);

    // trial wave function
    sys.trialwfn_alpha.resize(extents[NMO][NAEA]);
    sys.trialwfn_beta.resize(extents[NMO][NAEA]);
    MPI_Bcast(reinterpret_cast<char*>(sys.trialwfn_alpha.data()),
        sys.trialwfn_alpha.num_elements()*sizeof(ComplexType),
        MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(reinterpret_cast<char*>(sys.trialwfn_beta.data()),
        sys.trialwfn_beta.num_elements()*sizeof(ComplexType),
        MPI_CHAR,0,MPI_COMM_WORLD);

    // read Vakbl
    std::vector<IndexType> counts;
    {
      simple_matrix_partition<task_group,IndexType,RealType> split(Idata[2],Idata[3],1e-8);
      // count dimensions of sparse matrix
      if(!count_entries_hdf5_SpMat(dump,split,std::string("SpHijkl"),true,counts,TG,true,nread))
        return false;

      MPI_Bcast(sets.data(), sets.size(), MPI_INT, 0, MPI_COMM_WORLD);

      // resize Vakbl 
      int sz;
      int r0 = sets[TG.getLocalNodeNumber()];
      int rN = sets[TG.getLocalNodeNumber()+1];
      if( coreid==0 )
        sz = std::accumulate(counts.begin()+r0,counts.begin()+rN,0);
      MPI_Bcast(&sz, 1, MPI_INT, 0, TG.getNodeCommLocal());

      Vakbl.setup("miniAFQMC_Vakbl",TG.getNodeCommLocal());
      Vakbl.setDims(Idata[2],Idata[3]);
      Vakbl.reserve(sz);

      // Read Vakbl 
      if(reader) {
        split.partition(TG,true,counts,sets);

        // read Vakbl 
        if(!read_hdf5_SpMat(Vakbl,split,dump,std::string("SpHijkl"),TG,true,nread))
          return false;
      } else {
        // read Vakbl
        if(!read_hdf5_SpMat(Vakbl,split,dump,std::string("SpHijkl"),TG,true,nread))
          return false;
      }

      if(coreid==0) {
        // morph to "compacted" notation for miniapp
        r0 = std::numeric_limits<int>::max();
        rN = 0; 
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
          if(a*NMO+j < r0) r0=a*NMO+j;
          if(a*NMO+j+1 > rN) rN=a*NMO+j+1;
        }
        // shift rows to range [0,rN-r0)  
        for(it = Vakbl.rows_begin(),itend = Vakbl.rows_end(); it!=itend; ++it) *it -= r0; 
      }
      MPI_Bcast(&r0, 1, MPI_INT, 0, TG.getNodeCommLocal());  
      MPI_Bcast(&rN, 1, MPI_INT, 0, TG.getNodeCommLocal());  
      Vakbl.setOffset(r0,0);
      Vakbl.setDims(rN-r0,2*NMO*NAEA);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    Vakbl.compress(TG.getNodeCommLocal());

    if(reader) {
      dump.pop();
      dump.pop();
    }

    if(reader) {
      if(!dump.push("Propagators",false)) return false;
      if(!dump.push("phaseless_ImpSamp_ForceBias",false)) return false;
    }

    std::vector<long> Ldims(5);
    MPI_Bcast(Ldims.data(),Ldims.size(),MPI_LONG,0,MPI_COMM_WORLD);

    long ntot = Ldims[0];       // total number of terms
    int nrows = int(Ldims[1]);  // number of rows 
    int nvecs = int(Ldims[2]);  // number of cholesky vectors

    // propagator
    Propg1.resize(extents[NMO][NMO]);
    MPI_Bcast(reinterpret_cast<char*>(Propg1.data()),sizeof(MatType)*Propg1.num_elements(),
        MPI_CHAR,0,MPI_COMM_WORLD);

    simple_matrix_partition<task_group,IndexType,RealType> split(nrows,nvecs,1e-8);
    // count dimensions of sparse matrix
    if(!count_entries_hdf5_SpMat(dump,split,std::string("Spvn"),false,counts,TG,true,nread))
      return false;

    MPI_Bcast(sets.data(), sets.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // resize Spvn
    int sz;
    int cvec0 = sets[TG.getLocalNodeNumber()];
    int cvecN = sets[TG.getLocalNodeNumber()+1];
    if( coreid==0 )
      sz = std::accumulate(counts.begin()+cvec0,counts.begin()+cvecN,0);
    MPI_Bcast(&sz, 1, MPI_INT, 0, TG.getNodeCommLocal());

    Spvn.setup("miniAFQMC_Spvn",TG.getNodeCommLocal());
    Spvn.setDims(nrows,cvecN-cvec0);
    Spvn.setOffset(0,cvec0);
    Spvn.reserve(sz);

    // Read Spvn
    if(reader) {
      split.partition(TG,false,counts,sets);

      // read Spvn
      if(!read_hdf5_SpMat(Spvn,split,dump,std::string("Spvn"),TG,true,nread))
        return false;
    } else {
      // read Spvn
      if(!read_hdf5_SpMat(Spvn,split,dump,std::string("Spvn"),TG,true,nread))
        return false;
    }

    Spvn.compress(TG.getNodeCommLocal());

    // scale by sqrt(dt)
    if(coreid==0)
      Spvn *= std::sqrt(dt);

    if(reader) {
      dump.pop();
      dump.pop();
    }

  } 

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
} 


} // qmcplusplus


#endif
