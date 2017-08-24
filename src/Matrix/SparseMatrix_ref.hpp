
#ifndef QMCPLUSPLUS_AFQMC_SPARSEMATRIX_REF_HPP
#define QMCPLUSPLUS_AFQMC_SPARSEMATRIX_REF_HPP

#include <tuple>
#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <sys/time.h>
#include <ctime>

namespace qmcplusplus
{

/*
 * This class implements a lightweight view of a SparseMatrix. 
 * It does not manage memory. It contains a view of a submatrix of an already
 * allocated matrix managed by a separate object.
 * Used for explicit shared-memory-based linear algebra.  
 * Right now only allows access to a constant pointer.
 * No mutation of sub-matrix through the reference object is currently allowed.
 */  
template<class T>
class SparseMatrix_ref
{
  public:

  typedef T                Type_t;
  typedef T                value_type;
  typedef T*               pointer;
  typedef const T*         const_pointer;
  typedef const int*       const_intPtr;
  typedef int              intType;
  typedef int*             intPtr;
  typedef SparseMatrix_ref<T>  This_t;

  SparseMatrix_ref<T>():gnr(0),gnc(0),vals(NULL),colms(NULL)
  {
    shape_[0]=0;
    shape_[1]=0;
  }

  ~SparseMatrix_ref<T>()
  {
  }

  // disable copy constructor and operator=  
  SparseMatrix_ref<T>(const SparseMatrix_ref<T> &rhs) = delete;
  inline This_t& operator=(const SparseMatrix_ref<T> &rhs) = delete; 

  // for now single setup function
  inline void setup(intType nr_, intType nc_, intType gnr_, intType gnc_, intType r0_, intType c0_, 
        pointer v_, intPtr c_, std::vector<intType>& indx_b_, std::vector<intType>& indx_e_)
  {
    assert(gnr_ > 0 && gnc_ > 0);
    assert(nr_ > 0 && nr_ <= gnr_);
    assert(nc_ > 0 && nc_ <= gnc_);
    assert(v_ != NULL);
    assert(c_ != NULL);
    assert( indx_b_.size() == nr_ );
    assert( indx_e_.size() == nr_ );
    shape_[0]=nr_;
    nc0=nc_;
    shape_[1]=c0_+nc_;  // nc must be set to cN since column indices are global
    gnr=gnr_;
    gnc=gnc_;
    r0=r0_;
    c0=c0_;
    vals=v_;
    colms=c_;
    indx_b=indx_b_;
    indx_e=indx_e_;
  }

  inline int rows() const
  {
    return shape_[0];
  }
  // CAREFUL HERE!!! 
  inline int cols() const
  {
    return shape_[1];
  }

  intPtr shape() const
  {
    return shape_; 
  } 

  inline int global_row() const
  {
    return gnr;
  }
  inline int global_col() const
  {
    return gnc;
  }
  inline int global_r0() const
  {
    return r0;
  }
  inline int global_c0() const
  {
    return c0;
  }
  inline int global_rN() const
  {
    return r0+shape_[0];
  }
  inline int global_cN() const
  {
    return c0+nc0;
  }

  inline const_pointer values() const 
  {
    return vals;
  }

  inline const_intPtr column_data() const 
  {
    return colms; 
  }

  inline const_intPtr index_begin() const
  {
    return indx_b.data(); 
  }

  inline const_intPtr index_end() const
  {
    return indx_e.data(); 
  }


  // use binary search PLEASE!!! Not really used anyway
  inline intType find_element(int i, int j) const {
    for (intType k = indx_b[i]; k<indx_e[i]; k++) {
      if (colms[k] == j) return k;
    }
    return -1;
  }

  inline Type_t operator()( int i, int j) const
  {
    assert(i>=0 && i<shape_[0] && j>=0 && j<nc0); 
    intType idx = find_element(i,j+c0);
    if (idx == intType(-1)) return T(0);
    return vals[idx]; 
  }

  private:

  // dimensions of sub-matrix
  // shape_ = {nr, nc}
  intType shape_[2];

  // dimensions of global matrix 
  intType gnr, gnc;

  // coordinates of top-left corner of sub-matrix
  intType r0, c0, nc0;
 
  // pointer to data
  pointer vals;
  intPtr colms;
  std::vector<intType> indx_b;
  std::vector<intType> indx_e;
  
};


}

#endif
