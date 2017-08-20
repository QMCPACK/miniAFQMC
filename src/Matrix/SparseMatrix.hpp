
#ifndef QMCPLUSPLUS_AFQMC_SPARSEMATRIX_H
#define QMCPLUSPLUS_AFQMC_SPARSEMATRIX_H

#include<iostream>
#include<vector>
#include<tuple>
#include<assert.h>
#include<algorithm>
#include <mpi.h>

#include "Utilities/tuple_iterator.hpp"

#define ASSERT_SPARSEMATRIX 

namespace qmcplusplus
{

// class that implements a sparse matrix in CSR format
template<class T>
class SparseMatrix
{
  public:

  typedef T            Type_t;
  typedef T            value_type;
  typedef T*           pointer;
  typedef const T*     const_pointer;
  typedef const int    const_intType;
  typedef const int*   const_intPtr;
  typedef int           intType;
  typedef int*           intPtr;
  typedef typename std::vector<T>::iterator iterator;
  typedef typename std::vector<T>::const_iterator const_iterator;
  typedef typename std::vector<intType>::iterator int_iterator;
  typedef typename std::vector<intType>::const_iterator const_int_iterator;
  typedef SparseMatrix<T>  This_t;

  const static int dimensionality = 2;

  SparseMatrix<T>():vals(),colms(),myrows(),rowIndex(),nr(0),nc(0),compressed(false),zero_based(true)
  {
  }

  SparseMatrix<T>(int n):vals(),colms(),myrows(),rowIndex(),nr(n),nc(n),compressed(false),zero_based(true)
  {
  }

  SparseMatrix<T>(int n,int m):vals(),colms(),myrows(),rowIndex(),nr(n),nc(m),compressed(false),zero_based(true)
  {
  }

  ~SparseMatrix<T>()
  {
  }

  SparseMatrix<T>(const SparseMatrix<T> &rhs) = delete;

  inline void reserve(unsigned long n)
  {
    vals.reserve(n);
    myrows.reserve(n);
    colms.reserve(n); 
    rowIndex.resize(nr+1);
  }

  inline void resize(unsigned long nnz)
  {
    vals.resize(nnz);
    myrows.resize(nnz);
    colms.resize(nnz);
    rowIndex.resize(nr+1);
  }

  inline void clear() { 
    vals.clear();
    colms.clear();
    myrows.clear();
    rowIndex.clear();
    compressed=false;
    zero_based=true;
  }

  // does nothing, needed for compatibility with shared memory version
  inline void setup(bool hd=true, std::string ii=std::string(""), MPI_Comm comm_=MPI_COMM_SELF) {}

  inline void setDims(int n, int m)
  {
    nr=n;
    nc=m;
    compressed=false;
    zero_based=true;
  }

  inline void setCompressed() 
  {
    compressed=true;
  }

  inline bool isCompressed() const
  {
    return compressed;
  }
  inline unsigned long size() const
  {
    return vals.size();
  }
  inline int rows() const
  {
    return nr;
  }
  inline int cols() const
  {
    return nc;
  }

  inline const_pointer values(long n=0) const 
  {
    return vals.data()+n;
  }

  inline pointer values(long n=0) 
  {
    return vals.data()+n;
  }

  inline const_intPtr column_data(long n=0) const 
  {
    return colms.data()+n;
  }
  inline intPtr column_data(long n=0) 
  {
    return colms.data()+n;
  }

  inline const_intPtr row_data(long n=0) const 
  {
    return myrows.data()+n;
  }
  inline intPtr row_data(long n=0) 
  {
    return myrows.data()+n;
  }

  inline const_intPtr row_index(long n=0) const 
  {
    return rowIndex.data()+n;
  }
  inline intPtr row_index(long n=0) 
  {
    return rowIndex.data()+n;
  }

  inline This_t& operator=(const SparseMatrix<T> &rhs) = delete; 

  // should be using binary search, but this should not be used in performance critical 
  // areas in any case
  inline intType find_element(int i, int j) {
    for (int k = rowIndex[i]; k < rowIndex[i+1]; k++) {
      if (colms[k] == j) return k;
    }
    return -1;
  }

  // DANGER: This returns a reference, which could allow changes to the stored value.
  // If a zero element is changed, it will change zero everywhere in the matrix.
  // For now this method is only used for testing so it should not be a problem.
  inline Type_t& operator()(int i, int j)
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(i>=0 && i<nr && j>=0 && j<nc && compressed); 
#endif
    intType idx = find_element(i,j);
    if (idx == -1) return zero;
    return vals[idx];
  }

  inline Type_t operator()( int i, int j) const
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(i>=0 && i<nr && j>=0 && j<nc && compressed); 
#endif
    intType idx = find_element(i,j);
    if (idx == -1) return 0;
    return vals[idx];
  }

  inline void add(const int i, const int j, const T& v, bool dummy=false) 
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(i>=0 && i<nr && j>=0 && j<nc);
#endif
    compressed=false;
    myrows.push_back(i);
    colms.push_back(j);
    vals.push_back(v);
  }

  inline void add(const std::vector<std::tuple<intType,intType,T>>& v, bool dummy=false)
  {
    compressed=false;
    for(auto&& a: v) {
#ifdef ASSERT_SPARSEMATRIX
      assert(std::get<0>(a)>=0 && std::get<0>(a)<nr && std::get<1>(a)>=0 && std::get<1>(a)<nc);
#endif
      myrows.push_back(std::get<0>(a));
      colms.push_back(std::get<1>(a));
      vals.push_back(std::get<2>(a));
    }
    assert(vals->size()<static_cast<unsigned long>(INT_MAX)); // right now limited to INT_MAX due to indexing problem.
  }

  inline void compress()
  {
    // define comparison operator for tuple_iterator
    auto comp = [](std::tuple<intType, intType, value_type> const& a, std::tuple<intType, intType, value_type> const& b){return std::get<0>(a) < std::get<0>(b) || (!(std::get<0>(b) < std::get<0>(a)) && std::get<1>(a) < std::get<1>(b));};

    // use std::sort on tuple_iterator
    std::sort(make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows.begin(),colms.begin(),vals.begin()),
              make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows.end(),colms.end(),vals.end()),
              comp);

    // define rowIndex
    rowIndex.resize(nr+1);
    intType curr=-1;
    for(intType n=0; n<myrows.size(); n++) {
      if( myrows[n] != curr ) {
        intType old = curr;
        curr = myrows[n];
        for(int i=old+1; i<=curr; i++) rowIndex[i] = n;
      }
    }
    for(int i=myrows.back()+1; i<rowIndex.size(); i++)
      rowIndex[i] = static_cast<intType>(vals.size());
    compressed=true;

  }

  inline bool remove_repeated_and_compress()
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(myrows.size() == colms.size() && myrows.size() == vals.size());
#endif

    if(myrows.size() <= 1) return true;
    compress();

      int_iterator first_r=myrows.begin(), last_r=myrows.end();
      int_iterator first_c=colms.begin(), last_c=colms.end();
      iterator first_v=vals.begin(), last_v = vals.end();
      int_iterator result_r = first_r;
      int_iterator result_c = first_c;
      iterator result_v = first_v;

      while ( ( (first_r+1) != last_r)  )
      {
        ++first_r;
        ++first_c;
        ++first_v;
        if (!( (*result_r == *first_r) && (*result_c == *first_c) ) ) {
          *(++result_r)=*first_r;
          *(++result_c)=*first_c;
          *(++result_v)=*first_v;
        } else {
          if( std::abs(*result_v - *first_v) > 1e-8 ) { //*result_v != *first_v) {
            std::cerr<<" Error in call to SMSparseMatrix::remove_repeate_and_compressd. Same indexes with different values. \n";
            std::cerr<<"ri, ci, vi: "
                       <<*result_r <<" "
                       <<*result_c <<" "
                       <<*result_v <<" "
                       <<"rj, cj, vj: "
                       <<*first_r <<" "
                       <<*first_c <<" "
                       <<*first_v <<std::endl;
            std::cerr.flush();
            return false;
          }
        }
      }
      ++result_r;
      ++result_c;
      ++result_v;

      long sz1 = std::distance(myrows.begin(),result_r);
      long sz2 = std::distance(colms.begin(),result_c);
      long sz3 = std::distance(vals.begin(),result_v);
      if(sz1 != sz2 || sz1 != sz2) {
        std::cerr<<"Error: Different number of erased elements in SMSparseMatrix::remove_repeate_and_compressed. \n" <<std::endl;
        return false;
      }
      myrows.resize(sz1);
      colms.resize(sz1);
      vals.resize(sz1);

      // define rowIndex
      intType curr=-1;
      for(intType n=0; n<myrows.size(); n++) {
        if( myrows[n] != curr ) {
          intType old = curr;
          curr = myrows[n];
          for(int i=old+1; i<=curr; i++) rowIndex[i] = n;
        }
      }
      for(int i=myrows.back()+1; i<rowIndex.size(); i++)
        rowIndex[i] = static_cast<intType>(vals.size());

    return true;
  }

  inline void transpose() {
    assert(myrows.size() == colms.size() && myrows.size() == vals.size());
    for(std::vector<intType>::iterator itR=myrows.begin(),itC=colms.begin(); itR!=myrows.end(); ++itR,++itC)
      std::swap(*itR,*itC);
    std::swap(nr,nc);
    compress();
  }

  inline SparseMatrix<T>& operator*=(const double rhs ) 
  {
    for(iterator it=vals.begin(); it!=vals.end(); it++)
      (*it) *= rhs;
    return *this; 
  }

  inline SparseMatrix<T>& operator*=(const std::complex<double> rhs ) 
  {
    for(iterator it=vals.begin(); it!=vals.end(); it++)
      (*it) *= rhs;
    return *this; 
  }

  inline SparseMatrix<T>& operator*=(const float rhs )  
  {
    for(iterator it=vals.begin(); it!=vals.end(); it++)
      (*it) *= rhs;
    return *this;
  }

  inline SparseMatrix<T>& operator*=(const std::complex<float> rhs )  
  {
    for(iterator it=vals.begin(); it!=vals.end(); it++)
      (*it) *= rhs;
    return *this;
  }

  inline void toZeroBase() {
    if(zero_based) return;
    zero_based=true;
    for (intType& i : colms ) i--; 
    for (intType& i : myrows ) i--; 
    for (intType& i : rowIndex ) i--; 
  }

  inline void toOneBase() {
    if(!zero_based) return;
    zero_based=false;
    for (intType& i : colms ) i++; 
    for (intType& i : myrows ) i++; 
    for (intType& i : rowIndex ) i++; 
  }

  friend std::ostream& operator<<(std::ostream& out, const SparseMatrix<T>& rhs)
  {
    for(unsigned long i=0; i<rhs.vals.size(); i++)
      out<<"(" <<rhs.myrows[i] <<"," <<rhs.colms[i] <<":" <<rhs.vals[i] <<")\n"; 
    return out;
  }

  // this is ugly, but I need to code quickly 
  // so I'm doing this to avoid adding hdf5 support here 
  inline std::vector<T>* getVals() { return &vals; } 
  inline std::vector<intType>* getRows() { return &myrows; }
  inline std::vector<intType>* getCols() { return &colms; }
  inline std::vector<intType>* getRowIndex() { return &rowIndex; }

  inline iterator vals_begin() { return vals.begin(); }
  inline int_iterator rows_begin() { return myrows.begin(); }
  inline int_iterator cols_begin() { return colms.begin(); }
  inline int_iterator rowIndex_begin() { return rowIndex.begin(); }
  inline const_iterator vals_begin() const { return vals.begin(); }
  inline const_int_iterator cols_begin() const { return colms.begin(); }
  inline const_int_iterator rowIndex_begin() const { return rowIndex.begin(); }
  inline const_iterator vals_end() const { return vals.end(); }
  inline const_int_iterator rows_end() const { return myrows.end(); }
  inline const_int_iterator cols_end() const { return colms.end(); }
  inline const_int_iterator rowIndex_end() const { return rowIndex.end(); }
  inline iterator vals_end() { return vals.end(); }
  inline int_iterator rows_end() { return myrows.end(); }
  inline int_iterator cols_end() { return colms.end(); }
  inline int_iterator rowIndex_end() { return rowIndex.end(); }

  void setRowsFromRowIndex()
  {
    intType shift = zero_based?0:1;
    myrows.resize(vals.size());
    for(int i=0; i<nr; i++)
     for(intType j=rowIndex[i]; j<rowIndex[i+1]; j++)
      myrows[j]=i+shift;
  }
  bool zero_base() const { return zero_based; }

  private:

  bool compressed;
  int nr,nc;
  std::vector<T> vals;
  std::vector<intType> colms,myrows,rowIndex;
  bool zero_based;
  Type_t zero; // zero for return value

};


}

#endif
