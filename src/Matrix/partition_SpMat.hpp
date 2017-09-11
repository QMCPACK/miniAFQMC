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

#ifndef QMCPLUSPLUS_AFQMC_PARTITION_SPMAT_HPP
#define QMCPLUSPLUS_AFQMC_PARTITION_SPMAT_HPP

#include<string>
#include<vector>

#include "Configuration.h"
#include "Utilities/UtilityFunctions.h"
#include "Utilities/balanced_partition.hpp"

namespace qmcplusplus
{

namespace shm
{

template< class SpMat,
          class SpMat_ref,
          class task_group
        >
inline bool balance_partition_SpMat(task_group& TG, int type, SpMat& M, SpMat_ref& M_ref)  
{

  typedef typename SpMat::intType intType;
  typedef typename SpMat::indxType indxType;
  typedef typename SpMat::indxPtr indxPtr;
  typedef typename SpMat::intPtr intPtr;
  typedef typename SpMat::const_int_iterator int_iterator;

  // no partitioning
  if(TG.getNCoresPerTG() == 1) {
    int nr = M.rows();
    std::vector<intType> indx_b(nr);
    std::vector<intType> indx_e(nr);
    std::copy(M.index_begin(),M.index_begin()+nr,indx_b.data());
    std::copy(M.index_end(),M.index_end()+nr,indx_e.data());
    M_ref.setup(M.rows(),M.cols(),M.rows(),M.cols(),0,0,M.values(),M.column_data(),indx_b,indx_e);
    return true;
  }

  if(type == byRows) {

    std::vector<indxType> subsets(TG.getNCoresPerTG()+1);
    if( TG.getCoreID() == 0 ) {
      int nr = M.rows();
      indxPtr indx = M.row_index();
      balance_partition_ordered_set(nr,indx,subsets);

/*
      if(TG.getTGNumber()==0) {
        app_log()<<"   SpMat split over cores in TG: \n";
        app_log()<<"     Index: ";
        for(int i=0, iend=TG.getNCoresPerTG(); i<iend+1; i++)
          app_log()<<subsets[i] <<" ";
        app_log()<<std::endl;
        app_log()<<"     Number of terms per core: ";
        for(int i=0, iend=TG.getNCoresPerTG(); i<iend; i++)
          app_log()<<*(indx+subsets[i+1]) - *(indx+subsets[i]) <<" ";
        app_log()<<std::endl;
      }
*/
    }
    // MPI WRAPPER!!! 
    MPI_Bcast(reinterpret_cast<char*>(subsets.data()),sizeof(indxType)*subsets.size(),
      MPI_CHAR,0,TG.getNodeCommLocal());

    int rank=TG.getCoreRank();
    int r0 = subsets[rank];
    int nr = subsets[rank+1]-subsets[rank];
    indxType n0 = *M.row_index(r0); 
    std::vector<intType> indx_b(nr);
    std::vector<intType> indx_e(nr);
    for(int i=0; i<nr; i++) {
      indxType b = *M.row_index(r0+i);
      indxType e = *M.row_index(r0+i+1);
      // don't allow truncation due to range of intType
      if( (b-n0) > static_cast<indxType>(std::numeric_limits<intType>::max()) ) {
        std::cerr<<" Error partitioning matrix: Index > maximum range. " <<std::endl;
        return false;
      }
      if( (e-n0) > static_cast<indxType>(std::numeric_limits<intType>::max()) ) {
        std::cerr<<" Error partitioning matrix: Index > maximum range. " <<std::endl;
        return false;
      }
      indx_b[i] = static_cast<intType>(b-n0);
      indx_e[i] = static_cast<intType>(e-n0);
    }
    M_ref.setup(nr, M.cols(), M.rows(),M.cols(),r0,0,M.values()+n0,M.column_data()+n0,indx_b,indx_e);

  } else if(type == byCols) {

    // slightly more complicated. need to build counts by hand
    std::vector<indxType> subsets(TG.getNCoresPerTG()+1);
    std::vector<indxType> counts(M.cols()+1); 
    if( TG.getCoreID() == 0 ) {
      int nc = M.cols();
      for(int_iterator it=M.cols_begin(), itend = M.cols_end(); it!=itend; ++it)
        counts[ (*it)+1 ]++; 
      indxType cnt=0;
      for(int i=1; i<=nc; i++) {
        cnt+=counts[i];
        counts[i]=cnt;
      }       
      assert(cnt==M.size());
      balance_partition_ordered_set(nc,counts.data(),subsets);
/*
      if(TG.getTGNumber()==0) {
        app_log()<<"   SpMat split over cores in TG: \n";
        app_log()<<"     Index: ";
        for(int i=0, iend=TG.getNCoresPerTG(); i<iend+1; i++)
          app_log()<<subsets[i] <<" ";
        app_log()<<std::endl;
        app_log()<<"     Number of terms per core: ";
        for(int i=0, iend=TG.getNCoresPerTG(); i<iend; i++)
          app_log()<<counts[subsets[i+1]] - counts[subsets[i]] <<" ";
        app_log()<<std::endl;
      }
*/
    }
    // MPI WRAPPER PLEASE!!!
    MPI_Bcast(reinterpret_cast<char*>(subsets.data()),sizeof(indxType)*subsets.size(),
      MPI_CHAR,0,TG.getNodeCommLocal());
    MPI_Bcast(reinterpret_cast<char*>(counts.data()),sizeof(indxType)*counts.size(),
      MPI_CHAR,0,TG.getNodeCommLocal());

    int nr = M.rows();
    int rank=TG.getCoreRank();
    int c0 = subsets[rank];
    int cN = subsets[rank+1];
    int nc = cN-c0; 
    intPtr clm = M.column_data(*M.row_index(0));
    intPtr n0 = std::lower_bound(clm+(*M.row_index(0)),clm+(*M.row_index(1)),c0); 
    indxType dn0 = std::distance(clm+(*M.row_index(0)),n0);
    std::vector<intType> indx_b(nr);
    std::vector<intType> indx_e(nr);
    indxType cnt=0;
    for(int i=0; i<nr; i++) {
      intPtr b = std::lower_bound(clm+(*M.row_index(i)),clm+(*M.row_index(i+1)),c0);
      intPtr e = std::lower_bound(clm+(*M.row_index(i)),clm+(*M.row_index(i+1)),cN);
      indxType db = std::distance(n0,b); 
      indxType de = std::distance(n0,e); 
      cnt+= std::distance(b,e); 
      if( db > static_cast<indxType>(std::numeric_limits<intType>::max()) ) {
        std::cerr<<" Error partitioning matrix: Index > maximum range. " <<std::endl;
        return false;
      }
      if( de > static_cast<indxType>(std::numeric_limits<intType>::max()) ) {
        std::cerr<<" Error partitioning matrix: Index > maximum range. " <<std::endl;
        return false;
      }
      indx_b[i] = static_cast<intType>(db);
      indx_e[i] = static_cast<intType>(de);
    }
    assert(cnt == (counts[cN]-counts[c0]));
    M_ref.setup(nr, nc, M.rows(),M.cols(),0,c0,M.values()+dn0,M.column_data()+dn0,indx_b,indx_e);

  } else 
    return false;

  return true;
};

}

}

#endif
