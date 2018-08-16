//////////////////////////////////////////////////////////////////////////////////////
//// This file is distributed under the University of Illinois/NCSA Open Source License.
//// See LICENSE file in top directory for details.
////
//// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
////
//// File developed by: 
////
//// File created by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_CSR_HDF5_READERS_HPP
#define QMCPLUSPLUS_AFQMC_CSR_HDF5_READERS_HPP


#include<cassert>
#include<complex>
#include<cstdlib>
#include<algorithm>
#include<utility>
#include<vector>
#include<numeric>

#include "Configuration.h"
#include "io/hdf_archive.h"

namespace qmcplusplus
{

namespace afqmc
{

template<class MultiArray2D, 
         typename = typename std::enable_if<std::decay<MultiArray2D>::type::dimensionality == 2>::type
        >
inline MultiArray2D hdfcsr2ma(hdf_archive& dump, int nr, int nc) 
{
  using value_type = std::complex<double>; 
  using index_type = int; 
  using int_type = int; 

  // Need to read:
  // - dims: nrow,ncols, nnz
  // - data_
  // - jdata_
  // - pointers_begin_
  // - pointers_end_

  using size_type = std::size_t; 

  size_type nrows, ncols, nnz; 
  std::vector<size_type> dims(3);
  if(!dump.read(dims,"dims")) 
    APP_ABORT("Problems reading dims in csr_from_hdf5. \n");
  assert(dims.size()==3);
  ncols=dims[1];
  nnz=dims[2];
  nrows=dims[0];
  // matrix is stored as conjugate transpose of PsiT, we want only conjugate
  if(nc != nrows || nr != ncols)
    APP_ABORT("Problems with matrix dimensions in hdfcsr2ma. \n"); 

  MultiArray2D A(extents[nr][nc]);
  std::fill_n(A.origin(),A.num_elements(),value_type(0));

  std::vector<int_type> nnz_per_row(nrows);
  std::vector<int_type> ptrb;
  std::vector<int_type> ptre;
  if(!dump.read(ptrb,"pointers_begin_")) 
    APP_ABORT("Problems reading pointers_begin_ in csr_from_hdf5. \n");
  if(!dump.read(ptre,"pointers_end_")) 
    APP_ABORT("Problems reading pointers_end_ in csr_from_hdf5. \n");
  assert(ptrb.size()==nrows);
  assert(ptre.size()==nrows);
  for(size_type i = 0; i<nrows; i++) 
    nnz_per_row[i] = ptre[i]-ptrb[i]; 

  std::vector<value_type> data;
  std::vector<index_type> jdata;
  if(!dump.read(data,"data_"))
    APP_ABORT("Problems reading data_ in csr_from_hdf5. \n");
  if(!dump.read(jdata,"jdata_"))
    APP_ABORT("Problems reading jdata_ in csr_from_hdf5. \n");
  if(data.size() != nnz)
    APP_ABORT("Problems with data_ array in csr_from_hdf5. \n");
  if(jdata.size() != nnz)
    APP_ABORT("Problems with jdata_ array in csr_from_hdf5. \n");
  size_type cnt=0;
  for(index_type i = 0; i<static_cast<index_type>(nrows); i++) {
      size_type nt = static_cast<size_type>(ptre[i]-ptrb[i]);
      for( size_type nn=0; nn<nt; ++nn, ++cnt)
          A[jdata[cnt]][i] = data[cnt];  
  }

  return A;
}

}

}

#endif
