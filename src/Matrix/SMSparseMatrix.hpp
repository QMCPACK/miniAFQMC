
#ifndef QMCPLUSPLUS_AFQMC_SMSPARSEMATRIX_H
#define QMCPLUSPLUS_AFQMC_SMSPARSEMATRIX_H

#include <tuple>
#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <mpi.h>
#include <sys/time.h>
#include <ctime>

#include "Utilities/tuple_iterator.hpp"
#include "Utilities/UtilityFunctions.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/exceptions.hpp>

#define ASSERT_SPARSEMATRIX 

namespace qmcplusplus
{

/*
 * This class implements a SHM sparse matrix in CSR format. 
 * Its main function should be to generate objects of type SMSparseMatrix_ref
 * which contain sparse sub-matrices to be used in sparse blas operations. 
 */  
template<class T>
class SMSparseMatrix
{
  public:

  template<typename spT> using ShmemAllocator = boost::interprocess::allocator<spT, boost::interprocess::managed_shared_memory::segment_manager>;
  template<typename spT> using SMVector = boost::interprocess::vector<spT, ShmemAllocator<spT>>;

  typedef T                Type_t;
  typedef T                value_type;
  typedef T*               pointer;
  typedef const T*         const_pointer;
  typedef const int*       const_intPtr;
  typedef const long*      const_indxPtr;
  typedef int              intType;
  typedef int*             intPtr;
  typedef long             indxType;
  typedef long*            indxPtr;
  typedef typename SMVector<T>::iterator iterator;
  typedef typename SMVector<T>::const_iterator const_iterator;
  typedef typename SMVector<intType>::iterator int_iterator;
  typedef typename SMVector<intType>::const_iterator const_int_iterator;
  typedef typename SMVector<indxType>::iterator indx_iterator;
  typedef typename SMVector<indxType>::const_iterator const_indx_iterator;
  typedef SMSparseMatrix<T>  This_t;

  SMSparseMatrix<T>():nr(0),nc(0),compressed(false),zero_based(true),head(false),ID(""),SMallocated(false),vals(NULL),rowIndexFull(NULL),rowIndex(NULL),myrows(NULL),colms(NULL),mutex(NULL),segment(NULL) 
  {
    remover.ID="NULL";
    remover.head=false;
  }

  SMSparseMatrix<T>(int n, int m):nr(n),nc(m),compressed(false),zero_based(true),head(false),ID(""),SMallocated(false),vals(NULL),rowIndexFull(NULL),rowIndex(NULL),myrows(NULL),colms(NULL),mutex(NULL),segment(NULL) 
  {
    remover.ID="NULL";
    remover.head=false;
  }

  ~SMSparseMatrix<T>()
  {
    if(segment != NULL) {
     delete segment;
     boost::interprocess::shared_memory_object::remove(ID.c_str());
    }
  }

  // disable copy constructor and operator=  
  SMSparseMatrix<T>(const SMSparseMatrix<T> &rhs) = delete;
  inline This_t& operator=(const SMSparseMatrix<T> &rhs) = delete; 

  inline void setup(std::string ii, MPI_Comm comm_=MPI_COMM_SELF) {
    int rk;
    MPI_Comm_rank(comm_,&rk);
    head=(rk==0);
    ID=ii;
    remover.ID=ii;
    remover.head=head;
    comm=comm_;
  }

  inline void barrier() {
    MPI_Barrier(comm);
  }

  inline void reserve(unsigned long n)
  {
    if(vals==NULL) { 
      allocate(n);
    } else if(vals->capacity() < n) {
      std::cerr<<" Error: SMSparseMatrix already allocated without enough capacity. \n";
      std::cerr<<" ID: " <<ID <<std::endl;
      MPI_Abort(MPI_COMM_WORLD,1);  
    }
    if(head) {
      assert(n < vals->max_size());
      assert(n < colms->max_size());
      vals->reserve(n);
      myrows->reserve(n);
      colms->reserve(n); 
      rowIndex->resize(std::max(nr,nc)+1);
      rowIndexFull->resize(std::max(nr,nc)+1);
    }
    barrier();
  }

  inline void resize(unsigned long n)
  {
    reserve(n);
    if(head) {
      assert(n < vals->max_size());
      assert(n < colms->max_size());
      vals->resize(n);
      myrows->resize(n);
      colms->resize(n);
      rowIndex->resize(nr+1);
      rowIndexFull->resize(nr+1);
    }
    barrier();
  }

  // all processes must call this routine
  inline bool allocate(unsigned long n)
  {
    if(SMallocated || segment!=NULL) {
      std::cerr<<" Error: Reallocation of SMSparseMatrix is not allowed. \n"; 
      std::cerr<<" ID: " <<ID <<std::endl;
      MPI_Abort(MPI_COMM_WORLD,1);  
    }
    barrier();    
    if(head) { 
      memory = sizeof(boost::interprocess::interprocess_mutex)+n*sizeof(T)+(2*n+std::max(nr,nc)+1)*sizeof(intType)+(std::max(nr,nc)+1)*sizeof(indxType)+100000; // some empty space just in case

      try {
        segment = new boost::interprocess::managed_shared_memory(boost::interprocess::create_only, ID.c_str(), memory);
      } catch(boost::interprocess::interprocess_exception &ex) {
        std::cout<<"\n Found managed_shared_memory segment, removing. CAREFUL WITH PERSISTENT SHM MEMORY. !!! \n" <<std::endl;
        boost::interprocess::shared_memory_object::remove(ID.c_str());
        segment=NULL;
      }
    
      if(segment==NULL) {
        try {
          segment = new boost::interprocess::managed_shared_memory(boost::interprocess::create_only, ID.c_str(), memory);
        } catch(boost::interprocess::interprocess_exception &ex) {
          std::cerr<<"Problems setting up managed_shared_memory in SMSparseMatrix." <<std::endl;
          std::cerr<<"ID: " <<ID <<std::endl;
          MPI_Abort(MPI_COMM_WORLD,1);
          return false;
        }
      }

      try {

        alloc_indx = new ShmemAllocator<indxType>(segment->get_segment_manager());
        alloc_int = new ShmemAllocator<int>(segment->get_segment_manager());
        alloc_T = new ShmemAllocator<T>(segment->get_segment_manager());

        mutex = segment->construct<boost::interprocess::interprocess_mutex>("mutex")();

        rowIndexFull = segment->construct<SMVector<indxType>>("rowIndexFull")(*alloc_indx);
        assert((std::max(nr,nc)+1) < rowIndexFull->max_size());
        if( !((std::max(nr,nc)+1) < rowIndexFull->max_size()) ) {
          std::cerr<<" Error: Problems with container construction in SMSparseMatrix. \n";
          std::cerr<<" ID: " <<ID <<std::endl;
          MPI_Abort(MPI_COMM_WORLD,1);
        }
        rowIndexFull->resize(std::max(nr,nc)+1);

        rowIndex = segment->construct<SMVector<intType>>("rowIndex")(*alloc_int);
        assert((std::max(nr,nc)+1) < rowIndex->max_size());
        if( !((std::max(nr,nc)+1) < rowIndex->max_size()) ) {
          std::cerr<<" Error: Problems with container construction in SMSparseMatrix. \n"; 
          std::cerr<<" ID: " <<ID <<std::endl;
          MPI_Abort(MPI_COMM_WORLD,1);
        }
        rowIndex->resize(std::max(nr,nc)+1);

        myrows = segment->construct<SMVector<int>>("myrows")(*alloc_int);
        assert(n < myrows->max_size());
        if( !(n < myrows->max_size()) ) {
          std::cerr<<" Error: Problems with container construction in SMSparseMatrix. \n"; 
          std::cerr<<" ID: " <<ID <<std::endl;
          MPI_Abort(MPI_COMM_WORLD,1);
        }
        myrows->reserve(n);

        colms = segment->construct<SMVector<int>>("colms")(*alloc_int);
        assert(n < colms->max_size());
        if( !(n < colms->max_size()) ) {
          std::cerr<<" Error: Problems with container construction in SMSparseMatrix. \n"; 
          std::cerr<<" ID: " <<ID <<std::endl;
          MPI_Abort(MPI_COMM_WORLD,1);
        }
        colms->reserve(n);

        vals = segment->construct<SMVector<T>>("vals")(*alloc_T);
        assert(n < vals->max_size());
        if( !(n < vals->max_size()) ) {
          std::cerr<<" Error: Problems with container construction in SMSparseMatrix. \n"; 
          std::cerr<<" ID: " <<ID <<std::endl;
          MPI_Abort(MPI_COMM_WORLD,1);
        }
        vals->reserve(n);

        // resizing to n to make sure enough space is actually allocated.
        // sometimes I don't get errors with segment space until resize is actually called 
        
        myrows->resize(n);
        colms->resize(n);
        vals->resize(n);
        colms->resize(0);
        myrows->resize(0);
        vals->resize(0);

      } catch(std::bad_alloc& ex) {
        std::cerr<<"Problems allocating shared memory in SMSparseMatrix." <<std::endl;
        std::cerr<<" ID: " <<ID <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
        return false;
      }
      barrier();
    } else {

      barrier();
      try {
        segment = new boost::interprocess::managed_shared_memory(boost::interprocess::open_only, ID.c_str());
        vals = segment->find<SMVector<T>>("vals").first;
        colms = segment->find<SMVector<int>>("colms").first;
        myrows = segment->find<SMVector<int>>("myrows").first;
        rowIndex = segment->find<SMVector<intType>>("rowIndex").first;
        rowIndexFull = segment->find<SMVector<indxType>>("rowIndexFull").first;
        mutex = segment->find<boost::interprocess::interprocess_mutex>("mutex").first;
      } catch(std::bad_alloc& ex) {
        std::cerr<<"Problems allocating shared memory in SMSparseMatrix at children." <<std::endl;
        std::cerr<<" ID: " <<ID <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
        return false;
      }
      assert(mutex != 0);
      assert(vals != 0);
      assert(myrows != 0);
      assert(colms != 0);
      assert(rowIndex != 0);
      assert(rowIndexFull != 0);
      if( mutex==0 || vals==0 || myrows==0 || colms==0 || rowIndex==0 || rowIndexFull==0 ) {
        std::cerr<<"Problems allocating shared memory in SMSparseMatrix at children." <<std::endl;
        std::cerr<<" ID: " <<ID <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }
    
    }
    barrier();
    SMallocated=true;
    return true;
  }

  inline void clear() { 
    compressed=false;
    zero_based=true;
    if(!SMallocated) return; 
    if(!head) return;
    vals->clear();
    colms->clear();
    myrows->clear();
    rowIndex->clear();
    rowIndexFull->clear();
  }

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

  inline unsigned long memoryUsage() { return memory; }

  inline unsigned long capacity() const
  {
    return (vals!=NULL)?(vals->capacity()):0;
  }
  inline unsigned long size() const
  {
    return (vals!=NULL)?(vals->size()):0;
  }
  inline int rows() const
  {
    return nr;
  }
  inline int cols() const
  {
    return nc;
  }

  inline const_pointer values(indxType n=indxType(0)) const 
  {
    return (vals!=NULL)?(&((*vals)[n])):NULL;
  }

  inline pointer values(indxType n=indxType(0)) 
  {
    return (vals!=NULL)?(&((*vals)[n])):NULL;
  }

  inline const_intPtr column_data(indxType n=indxType(0)) const 
  {
    return (colms!=NULL)?(&((*colms)[n])):NULL;
  }
  inline intPtr column_data(indxType n=indxType(0)) 
  {
    return (colms!=NULL)?(&((*colms)[n])):NULL;
  }

  inline const_intPtr row_data(indxType n=indxType(0)) const 
  {
    return (myrows!=NULL)?(&((*myrows)[n])):NULL;
  }
  inline intPtr row_data(indxType n=indxType(0)) 
  {
    return (myrows!=NULL)?(&((*myrows)[n])):NULL;
  }

  inline const_intPtr index_begin(indxType n=indxType(0)) const
  {
    return (rowIndex!=NULL)?(&((*rowIndex)[n])):NULL;
  }
  inline intPtr index_begin(indxType n=indxType(0))
  {
    return  (rowIndex!=NULL)?(&((*rowIndex)[n])):NULL;
  }

  inline const_intPtr index_end(indxType n=indxType(0)) const
  {
    return (rowIndex!=NULL)?(&((*rowIndex)[n+1])):NULL;
  }
  inline intPtr index_end(indxType n=indxType(0))
  {
    return  (rowIndex!=NULL)?(&((*rowIndex)[n+1])):NULL;
  }

  inline const_indxPtr row_index(indxType n=indxType(0)) const 
  {
    return (rowIndexFull!=NULL)?(&((*rowIndexFull)[n])):NULL;
  }
  inline indxPtr row_index(indxType n=indxType(0)) 
  {
    return  (rowIndexFull!=NULL)?(&((*rowIndexFull)[n])):NULL;
  }

  // use binary search PLEASE!!! Not really used anyway
  inline indxType find_element(int i, int j) const {
    for (indxType k = (*rowIndexFull)[i]; k<(*rowIndexFull)[i+1]; k++) {
      if ((*colms)[k] == j) return k;
    }
    return -1;
  }

  inline Type_t operator()( int i, int j) const
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(i>=0 && i<nr && j>=0 && j<nc && compressed); 
#endif
    indxType idx = find_element(i,j);
    if (idx == indxType(-1)) return T(0);
    return (*vals)[idx]; 
  }

  inline void add(const int i, const int j, const T& v, bool needs_locks=false) 
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(i>=0 && i<nr && j>=0 && j<nc);
#endif
    compressed=false;
    if(needs_locks) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex);
      myrows->push_back(i);
      colms->push_back(j);
      vals->push_back(v);
    } else {
      if(!head) return;
      myrows->push_back(i);
      colms->push_back(j);
      vals->push_back(v);
    }
  }

  inline void add(const std::vector<std::tuple<int,int,T>>& v, bool needs_locks=false)
  {
    compressed=false;
    if(needs_locks) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex);
      for(auto&& a: v) {
#ifdef ASSERT_SPARSEMATRIX
        assert(std::get<0>(a)>=0 && std::get<0>(a)<nr && std::get<1>(a)>=0 && std::get<1>(a)<nc);
#endif
        myrows->push_back(std::get<0>(a));
        colms->push_back(std::get<1>(a));
        vals->push_back(std::get<2>(a));
      }
    } else {
      if(!head) return;
      for(auto&& a: v) {
#ifdef ASSERT_SPARSEMATRIX
        assert(std::get<0>(a)>=0 && std::get<0>(a)<nr && std::get<1>(a)>=0 && std::get<1>(a)<nc);
#endif
        myrows->push_back(std::get<0>(a));
        colms->push_back(std::get<1>(a));
        vals->push_back(std::get<2>(a));
      }
    }
  }

  inline void setup_index()
  {
    if(head) {
      rowIndex->resize(nr+1);
      rowIndexFull->resize(nr+1);
      intType curr=-1;
      for(indxType n=0; n<myrows->size(); n++) {
        if( (*myrows)[n] != curr ) {
          intType old = curr;
          curr = (*myrows)[n];
          for(intType i=old+1; i<=curr; i++) (*rowIndexFull)[i] = n;
        }
      }
      for(intType i=myrows->back()+1; i<rowIndexFull->size(); i++)
        (*rowIndexFull)[i] = static_cast<intType>(vals->size());

      // now setup rowIndex until INT_MAX, additional elements are truncated

      for(int i=0, iend=rowIndexFull->size(); i<iend; i++) {
        indxType k = (*rowIndexFull)[i];  
        if(k >= static_cast<indxType>(std::numeric_limits<intType>::max())) {
          std::cout<<" CAREFULL: SMSparseMatrix " <<ID <<" is truncated if used directly. \n"
                   <<" Too many elements. Use SMSparseMatrix_ref for sub_matrix operations." 
                   <<std::endl;
          indxType klast = (*rowIndexFull)[i-1];  
          for( int j=i; j<iend; j++)
            (*rowIndex)[j] = static_cast<intType>(klast);
          return;  
        }
        (*rowIndex)[i] = static_cast<intType>(k);
      }
    }
  }

  inline void compress(MPI_Comm local_comm=MPI_COMM_SELF)
  {

    auto comp = [](std::tuple<intType, intType, value_type> const& a, std::tuple<intType, intType, value_type> const& b){return std::get<0>(a) < std::get<0>(b) || (!(std::get<0>(b) < std::get<0>(a)) && std::get<1>(a) < std::get<1>(b));};

    if(local_comm==MPI_COMM_SELF) {
      if(!head) {
        std::cerr<<" Error in SMSparseMatrix::compress: Calling with MPI_COMM_SELF from node that is not head of node." <<std::endl;
        std::cerr<<" ID: " <<ID <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      // sort
      std::sort(make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows->begin(),colms->begin(),vals->begin()),
                make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows->end(),colms->end(),vals->end()),
                comp);

      setup_index();
      compressed=true;   
      return;
    }

    int npr,rank;
    MPI_Comm_rank(local_comm,&rank); 
    MPI_Comm_size(local_comm,&npr); 

    MPI_Barrier(local_comm);

    assert(myrows->size() == colms->size() && myrows->size() == vals->size());
    if(vals->size() == 0) return;

    int nlvl = static_cast<int>(  std::floor(std::log2(npr*1.0)) );
    int nblk = pow(2,nlvl);

    // sort equal segments in parallel
    std::vector<int> pos(nblk+1);     
    // a core processes elements from pos[rank]-pos[rank+1]
    FairDivide(vals->size(),nblk,pos); 
    
    // sort local segment
    if(rank < nblk)   
      std::sort(make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows->begin()+pos[rank],colms->begin()+pos[rank],vals->begin()+pos[rank]),
          make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows->begin()+pos[rank+1],colms->begin()+pos[rank+1],vals->begin()+pos[rank+1]),
          comp);
    MPI_Barrier(local_comm);

    for(int i=0, k=rank, sz=1; i<nlvl; i++, sz*=2 ) {
      if(k%2==0 && rank<nblk) {
        std::inplace_merge( 
           make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows->begin()+pos[rank],colms->begin()+pos[rank],vals->begin()+pos[rank]),
           make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows->begin()+pos[rank+sz],colms->begin()+pos[rank+sz],vals->begin()+pos[rank+sz]),
           make_tuple_iterator<int_iterator,int_iterator,iterator>(myrows->begin()+pos[rank+sz*2],colms->begin()+pos[rank+sz*2],vals->begin()+pos[rank+sz*2]),
           comp);
        k/=2;
      } else
        k=1;
      MPI_Barrier(local_comm);
    }

    if(head) setup_index(); 
    MPI_Barrier(local_comm);
    compressed=true;   
  }

  inline void transpose(MPI_Comm local_comm=MPI_COMM_SELF)   
  {
    if(!SMallocated) return;
    assert(myrows->size() == colms->size() && myrows->size() == vals->size());
    if(head) {
      // can parallelize this if you want
      for(int_iterator itR=myrows->begin(),itC=colms->begin(); itR!=myrows->end(); ++itR,++itC)
        std::swap(*itR,*itC);
    } else {
      if(local_comm == MPI_COMM_SELF) {
        std::cerr<<" Error in SMSparseMatrix::compress: Calling with MPI_COMM_SELF from node that is not head of node." <<std::endl;
        std::cerr<<" ID: " <<ID <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }
    }
    std::swap(nr,nc);
    compress(local_comm);
  }

  inline void check()
  {
    if(!head) return; 
    for(int i=0; i<rowIndexFull->size()-1; i++)
    {
      if((*rowIndexFull)[i+1] < (*rowIndexFull)[i]) std::cout<<"Error: SMSparseMatrix::check(): rowIndex-> \n" <<std::endl; 
  
    }
  }

  template<typename Tp>
  inline SMSparseMatrix<T>& operator*=(const Tp rhs ) 
  {
    if(!head) return *this; 
    for(iterator it=vals->begin(); it!=vals->end(); it++)
      (*it) *= rhs;
    return *this; 
  }

  template<typename Tp>
  inline SMSparseMatrix<T>& operator*=(const std::complex<Tp> rhs ) 
  {
    if(!head) return *this; 
    for(iterator it=vals->begin(); it!=vals->end(); it++)
      (*it) *= rhs;
    return *this; 
  }

  inline void toZeroBase() {
    if(!head) return; 
    if(zero_based) return;
    zero_based=true;
    for (intType& i : *colms ) i--; 
    for (intType& i : *myrows ) i--; 
    for (intType& i : *rowIndex ) i--; 
    for (indxType& i : *rowIndexFull ) i--; 
  }

  inline void toOneBase() {
    if(!head) return; 
    if(!zero_based) return;
    zero_based=false;
    for (intType& i : *colms ) i++; 
    for (intType& i : *myrows ) i++; 
    for (intType& i : *rowIndex ) i++; 
    for (indxType& i : *rowIndexFull ) i++; 
  }

  // this is ugly, but I need to code quickly 
  // so I'm doing this to avoid adding hdf5 support here 
  inline SMVector<T>* getVals() const { return vals; } 
  inline SMVector<intType>* getRows() const { return myrows; }
  inline SMVector<intType>* getCols() const { return colms; }
  inline SMVector<indxType>* getRowIndex() const { return rowIndexFull; }

  inline iterator vals_begin() { assert(vals!=NULL); return vals->begin(); } 
  inline int_iterator rows_begin() { assert(myrows!=NULL); return myrows->begin(); }
  inline int_iterator cols_begin() { assert(colms!=NULL); return colms->begin(); }
  inline indx_iterator rowIndex_begin() { assert(rowIndexFull!=NULL); return rowIndexFull->begin(); }
  inline const_iterator vals_begin() const { return vals->begin(); } 
  inline const_int_iterator cols_begin() const { assert(colms!=NULL); return colms->begin(); }
  inline const_indx_iterator rowIndex_begin() const { assert(rowIndexFull!=NULL); return rowIndexFull->begin(); }
  inline const_iterator vals_end() const { assert(vals!=NULL); return vals->end(); } 
  inline const_int_iterator rows_end() const { assert(myrows!=NULL); return myrows->end(); }
  inline const_int_iterator cols_end() const { assert(colms!=NULL); return colms->end(); }
  inline const_indx_iterator rowIndex_end() const { assert(rowIndexFull!=NULL); return rowIndexFull->end(); }
  inline iterator vals_end() { assert(vals!=NULL); return vals->end(); } 
  inline int_iterator rows_end() { assert(myrows!=NULL); return myrows->end(); }
  inline int_iterator cols_end() { assert(colms!=NULL); return colms->end(); }
  inline indx_iterator rowIndex_end() { assert(rowIndexFull!=NULL); return rowIndexFull->end(); }

  inline bool isAllocated() {
    return (SMallocated)&&(vals!=NULL)&&(segment!=NULL);
  }

  void setRowsFromRowIndex()
  {
    if(!head) return;
    intType shift = zero_based?0:1;
    myrows->resize(vals->size());
    for(int i=0; i<nr; i++)
     for(indxType j=(*rowIndexFull)[i]; j<(*rowIndexFull)[i+1]; j++)
      (*myrows)[j]=i+shift;
  }

  bool zero_base() const { return zero_based; }

  private:

  boost::interprocess::interprocess_mutex *mutex;
  bool compressed;
  int nr,nc;
  SMVector<T> *vals;
  SMVector<intType> *colms,*myrows;
  SMVector<indxType> *rowIndexFull;   // row index table for full matrix
  SMVector<intType> *rowIndex;        // row index table in intType. Limit to INT_MAX elements
  bool head;
  std::string ID; 
  bool SMallocated;
  bool zero_based;
  unsigned long memory=0;

  //_mySort_snD_ my_sort;

  boost::interprocess::managed_shared_memory *segment;
  ShmemAllocator<T> *alloc_T;
  ShmemAllocator<boost::interprocess::interprocess_mutex> *alloc_mutex;
  ShmemAllocator<intType> *alloc_int;
  ShmemAllocator<indxType> *alloc_indx;

  // using MPI for barrier calls until I find solution
  MPI_Comm comm;
 
  struct shm_remove
  {
    bool head;
    std::string ID; 
    shm_remove() {
      if(head) boost::interprocess::shared_memory_object::remove(ID.c_str());
    }
    ~shm_remove(){
      if(head) boost::interprocess::shared_memory_object::remove(ID.c_str());
    }
  } remover;

};


}

#endif
