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

#ifndef MA_COMMUNICATIONS_HPP
#define MA_COMMUNICATIONS_HPP

#include<type_traits> // enable_if
#include<mpi.h>

namespace ma{

template< class MultiArray2DA, 
          class MultiArray2DB, 
        typename = typename std::enable_if<
                MultiArray2DA::dimensionality == 2 and
                MultiArray2DB::dimensionality == 2> 
>
inline void allgather_matrix( MPI_Comm comm, const MultiArray2DA& source, MultiArray2DB&& dest, int type)
{
  typedef typename MultiArray2DA::element TypeA;
  typedef typename std::decay<MultiArray2DB>::type::element TypeB;
  assert(sizeof(TypeA) == sizeof(TypeB));
  assert(source.strides()[1]==1);  // not sure how to handle a different stride()[1] yet
  assert(dest.strides()[1]==1);  // not sure how to handle a different stride()[1] yet

  int size;
  MPI_Comm_size(comm,&size);

  if(type == byRows) {

    assert(source.shape()[0]*size == dest.shape()[0]);
    assert(source.shape()[1] == dest.shape()[1]);
    assert(source.strides()[0] == source.shape()[1]);
    assert(dest.strides()[0] == dest.shape()[1]);
    // MPI!!!
    MPI_Allgather(source.origin(),source.num_elements()*sizeof(TypeA), MPI_CHAR
        ,dest.origin(),source.num_elements()*sizeof(TypeA), MPI_CHAR, comm);

  } else if(type == byCols) {

    assert(source.shape()[0] == dest.shape()[0]);
    assert(source.shape()[1]*size == dest.shape()[1]);

#if 1 
    // using Allreduce

    assert(dest.strides()[0] == dest.shape()[1]);
    assert(source.strides()[0] == source.shape()[1]);

    int rank;
    MPI_Comm_rank(comm,&rank);

    using boost::indices;
    using range_t = boost::multi_array_types::index_range;

    std::fill_n(dest.origin(),dest.num_elements(),0);
    dest[indices[range_t()][range_t(rank*source.shape()[1],(rank+1)*source.shape()[1])]] = source;

    // hard-coded to std::complex<double>
    MPI_Allreduce(MPI_IN_PLACE,dest.origin(),2*dest.num_elements(),
                 MPI_DOUBLE,MPI_SUM,comm);

#else
    // using MPI_Datatype

    MPI_Datatype typeA, typeB_, typeB;
    MPI_Type_vector(source.shape()[0], source.shape()[1]*sizeof(TypeA), source.strides()[0]*sizeof(TypeA), MPI_CHAR, &typeA);
    MPI_Type_commit(&typeA);
    MPI_Type_vector(source.shape()[0], source.shape()[1]*sizeof(TypeA), dest.strides()[0]*sizeof(TypeA), MPI_CHAR, &typeB_);
    MPI_Type_commit(&typeB_);
    MPI_Type_create_resized(typeB_, 0, source.shape()[1]*sizeof(TypeA), &typeB);
    MPI_Type_commit(&typeB);
    MPI_Type_free(&typeB_);

    MPI_Allgather(source.origin(),1,typeA,
                  dest.origin(),1,typeB,
                  comm);

    MPI_Type_free(&typeA);
    MPI_Type_free(&typeB);

#endif
        

  } else {

    std::cerr<<" Error: unknown gather type in ma::gather_matrix(). " <<std::endl;
    APP_ABORT(" Error in ma::gather_matrix. \n");

  }

} 

}

#endif
