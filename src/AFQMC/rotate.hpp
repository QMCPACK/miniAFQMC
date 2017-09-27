////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////


#ifndef  AFQMC_ROTATE_HPP 
#define  AFQMC_ROTATE_HPP 

#include "Numerics/ma_operations.hpp"
#include "mpi.h"

namespace qmcplusplus
{

namespace shm 
{

/*
 *  Performs a (left) half rotation and a transposition of a Cholesky matrix. 
 *  The rotated matrix is stored in sparse format. The output matrix is resized to accomodate
 *  the necesary number of non-zero terms. 
 *  Input:
 *    -alpha: Rotation matrix for spin up.
 *    -beta: Rotation matrix for spin down.
 *    -A: Input sparse cholesky matrix.
 *    -cutoff: Value below which elements of rotated matrix are ignored.
 *  Output:
 *    -B: Rotated cholesky matrix. 
 *
 *  If transposed==true:
 *     B(n,ak) = sum_i^M alpha(i,a) * Spvn(ik,n) 
 *     B(n,ak+N*M) = sum_i^M beta(i,a) * Spvn(ik,n) 
 *  else:
 *     B(ak,n) = sum_i A(i,a) * Spvn(ik,n) 
 *     B(ak+N*M,n) = sum_i^M beta(i,a) * Spvn(ik,n), 
 *  where M/N is the number of rows/columns of alpha and beta.
 *  The number of rows of Spvn should be equal to M*M.
 */ 
// Serial Implementation
template< class task_group,
          class Mat,
          class SpMatA,  
	  class SpMatB	
        >
inline void halfrotate_cholesky(task_group& TG, const Mat& alpha, const Mat& beta, SpMatA& A, SpMatB& B, double cutoff=1e-6)
{
  assert(Mat::dimensionality == 2); 
  int M = alpha.shape()[0]; 
  int N = alpha.shape()[1];

  assert( M == beta.shape()[0]); 
  assert( N == beta.shape()[1]); 
  assert( A.rows() == M*M );
  
  auto zero = typename SpMatB::value_type(0);
  int nchol = A.cols();

  int ncores = TG.getTotalCores();
  int coreid = TG.getCoreID();

  B.setup("miniAFQMC_SpvnT",TG.getNodeCommLocal());
  B.setDims(A.cols(),2*N*M); 

  using Type = typename SpMatB::value_type;
  using IType = typename SpMatB::intType;

  boost::multi_array<Type,2> An(extents[M][M]);
  boost::multi_array<Type,2> C(extents[N][M]);
  std::size_t nz=0;
  // count number of non-zero terms

  // transpose A for easy access
  A.transpose(TG.getNodeCommLocal());

  auto col = A.indx();
  auto val = A.val();
  auto pb = A.pntrb();
  auto pe = A.pntre();
  for(int i=0, iend=nchol; i<iend; i++, pb++,pe++) {

    int nterms = *pe - *pb;
    if(nterms==0) continue;

    if(i%ncores != coreid) {
      col+=nterms;
      val+=nterms;
      continue;
    }

//    std::fill(An.begin(), An.end(), zero);
    for(int a=0; a<M; a++)
      for(int k=0; k<M; k++)
        An[a][k]=zero;

    // extract Cholesky vector i
    for(int nt=0; nt<nterms; nt++, col++, val++) 
      // *col == a*M+b
      An[(*col)/M][(*col)%M] = static_cast<Type>(*val);

    using ma::T;
 
    // rotate the matrix: C = alpha^H * An
    ma::product(T(alpha),An,C);
    for(int a=0; a<N; a++)
      for(int k=0; k<M; k++)
        if(std::abs(C[a][k]) > cutoff)  
          nz++;

    ma::product(T(beta),An,C);
    for(int a=0; a<N; a++)
      for(int k=0; k<M; k++)
        if(std::abs(C[a][k]) > cutoff)  
          nz++;
  }

  // C++ wrapper please!!!
  long sz1=static_cast<long>(nz);
  MPI_Allreduce(MPI_IN_PLACE,&sz1,1,MPI_LONG,MPI_SUM,TG.getNodeCommLocal());
  nz = static_cast<std::size_t>(sz1);
  B.reserve(nz);

  int maxn = 100000;
  std::vector<std::tuple<IType,IType,Type>> buff;
  buff.reserve(maxn);

  // generate rotated matrix
  col = A.indx();
  val = A.val();
  pb = A.pntrb();
  pe = A.pntre();
  for(int i=0, iend=nchol; i<iend; i++, pb++,pe++) {

    int nterms = *pe - *pb;
    if(nterms==0) continue;

    if(i%ncores != coreid) {
      col+=nterms;
      val+=nterms;
      continue;
    }

//    std::fill(An.begin(), An.end(), zero);
    for(int a=0; a<M; a++)
      for(int k=0; k<M; k++)
        An[a][k]=zero;

    // extract Cholesky vector i
    for(int nt=0; nt<nterms; nt++, col++, val++) 
      // *col == a*M+b
      An[(*col)/M][(*col)%M] = static_cast<Type>(*val);

    using ma::T;

    // rotate the matrix: C = alpha^H * An
    ma::product(T(alpha),An,C);
    for(int a=0; a<N; a++)
      for(int k=0; k<M; k++)
        if(std::abs(C[a][k]) > cutoff) { 
          //B.add(i,a*M+k,C[a][k]); 
          buff.push_back( std::make_tuple(i,a*M+k,C[a][k]) );
          if(buff.size() == maxn) {
            B.add(buff,ncores>1);
            buff.clear();
          }  
        } 

    ma::product(T(beta),An,C);
    for(int a=0; a<N; a++)
      for(int k=0; k<M; k++)
        if(std::abs(C[a][k]) > cutoff) {
          //B.add(i,N*M+a*M+k,C[a][k]); 
          buff.push_back( std::make_tuple(i,N*M+a*M+k,C[a][k]) );
          if(buff.size() == maxn) {
            B.add(buff,ncores>1);
            buff.clear();
          } 
        } 
  }
  if(buff.size() > 0) {
    B.add(buff,ncores>1);
    buff.clear();
  }

  // transpose A back
  A.transpose(TG.getNodeCommLocal());
  B.compress(TG.getNodeCommLocal());
}

}

}

#endif
