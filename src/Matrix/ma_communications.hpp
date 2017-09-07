#ifndef MA_COMMUNICATIONS_HPP
#define MA_COMMUNICATIONS_HPP

#include"Configuration.h"
#include<type_traits> // enable_if
#include<mpi.h>

namespace ma{

template< class MultiArray2DA, 
          class MultiArray2DB, 
        typename = typename std::enable_if<
                MultiArray2DA::dimensionality == 2 and
                MultiArray2DB::dimensionality == 2> 
>
inline void gather_matrix( MPI_Comm comm, MultiArray2DA& A, MultiArray2DB& B, int type)
{
  typedef typename std::decay<MultiArray2DA>::type::element TypeA;
  typedef typename std::decay<MultiArray2DB>::type::element TypeB;
  assert(sizeof(TypeA) == sizeof(TypeB));

  if(type == byRows) {

    int size;
    MPI_Comm_size(comm,&size);
    assert(A.shape()[0]*size == B.shape()[0]);
    assert(A.shape()[1] == B.shape()[1]);
    // MPI!!!
    MPI_Allgather(A.data(),A.shape()[0]*A.shape()[1]*sizeof(TypeA), MPI_CHAR
        ,B.data(),A.shape()[0]*A.shape()[1]*sizeof(TypeA), MPI_CHAR, comm);

  } else if(type == byCols) {

    int size, rank;
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);
    assert(A.shape()[0] == B.shape()[0]);
    assert(A.shape()[1]*size == B.shape()[1]);
    std::vector<TypeA> buff(A.shape()[0]*A.shape()[1]);

    // learn how to use MPI_TYPE_..._SUBARRAY
    for(int i=0, col0=0; i<size; i++, col0+=A.shape()[1]) {
      if( rank == i) std::copy(A.data(), A.data()+buff.size(), buff.data());  
      MPI_Bcast(buff.data(), buff.size()*sizeof(TypeA), MPI_CHAR, i, comm );
      for(int k=0, kj=0; k<A.shape()[0]; k++)
        for(int j=0; j<A.shape()[1]; j++, kj++)
          B[k][col0+j] = buff[kj];     
    }
        

  } else {

    std::cerr<<" Error: unknown gather type in ma::gather_matrix(). " <<std::endl;
    APP_ABORT(" Error in ma::gather_matrix. \n");

  }

} 

}

#endif
