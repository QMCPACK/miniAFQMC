
#ifndef QMCPLUSPLUS_AFQMC_SMDENSEVECTOR_H
#define QMCPLUSPLUS_AFQMC_SMDENSEVECTOR_H

#include<iostream>
#include<vector>
#include<tuple>
#include <cassert>
#include<algorithm>
#include<complex>
#include<cmath>
#include<string>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>

#define ASSERT_VECTOR

namespace qmcplusplus
{

// wrapper for boost::interprocess::vector 
// only allows allocation once
template<class T>
class SMDenseVector
{
  public:

  template<typename spT> using ShmemAllocator = boost::interprocess::allocator<spT, boost::interprocess::managed_shared_memory::segment_manager>;
  template<typename spT> using boost_SMVector = boost::interprocess::vector<spT, ShmemAllocator<spT>>;


  typedef T              Type_t;
  typedef T              value_type;
  typedef T*             pointer;
  typedef const T*       const_pointer;
  typedef unsigned long  indxType;
  typedef typename boost_SMVector<T>::iterator iterator;
  typedef typename boost_SMVector<T>::const_iterator const_iterator;
  typedef boost_SMVector<T>  This_t;

  SMDenseVector<T>():head(false),ID(""),SMallocated(false),vals(NULL),mutex(NULL),
                      segment(NULL),alloc_T(NULL),alloc_mutex(NULL)
  {
    remover.ID="NULL";
    remover.head=false;
  }

  SMDenseVector<T>(std::string ii, MPI_Comm comm_, int n):head(false),ID(""),
                      SMallocated(false),vals(NULL),mutex(NULL),
                      segment(NULL),alloc_T(NULL),alloc_mutex(NULL)
  {
    setup(ii,comm_);
    resize(n);
  }

  ~SMDenseVector<T>()
  {
    if(segment!=NULL) {
     delete segment;
     boost::interprocess::shared_memory_object::remove(ID.c_str());
    }
  }

  SMDenseVector<T>(const SMDenseVector<T> &rhs) = delete;

  inline void setup(std::string ii, MPI_Comm comm_) {
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

  inline bool deallocate()
  {
    SMallocated = false;
    barrier();
    if(!head) {
      try{
        delete segment;
        segment=NULL;
      } catch(std::bad_alloc&) {
        std::cerr<<"Problems deleting segment in SMDenseVector::deallocate()." <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,20);
      }
    }
    barrier();
    if(head) {
      try{
        delete segment;
        segment=NULL;
        boost::interprocess::shared_memory_object::remove(ID.c_str());
      } catch(std::bad_alloc&) {
        std::cerr<<"Problems de-allocating shared memory in SMDenseVector." <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,20);
      }
    }
    barrier();
    return true;
  } 

  inline bool reserve(unsigned long n)
  {
    assert(ID != std::string(""));
    assert(!SMallocated);
    assert(segment==NULL);
    if(segment!=NULL || SMallocated) {
      std::cerr<<" Error: SMDenseVector: " <<ID <<" already allocated. " <<std::endl;
      MPI_Abort(MPI_COMM_WORLD,20);
    }
    barrier();
    if(head) {
      // some extra space just in case
      memory = sizeof(boost::interprocess::interprocess_mutex)+n*sizeof(T)+8000; 

      try {
        segment = new boost::interprocess::managed_shared_memory(boost::interprocess::create_only, ID.c_str(), memory);
      } catch(boost::interprocess::interprocess_exception &ex) {
        std::cout<<" Found managed_shared_memory segment, removing. Careful with persistent SHM segment. \n";
        boost::interprocess::shared_memory_object::remove(ID.c_str());
        segment=NULL;
      }

      if(segment==NULL) {
        try {
          segment = new boost::interprocess::managed_shared_memory(boost::interprocess::create_only, ID.c_str(), memory);
        } catch(boost::interprocess::interprocess_exception &ex) {
          std::cerr<<"Problems setting up managed_shared_memory in SMDenseVector." <<std::endl;
          MPI_Abort(MPI_COMM_WORLD,20);
          return false;
        }
      }

      try {
        alloc_T = new ShmemAllocator<T>(segment->get_segment_manager());

        mutex = segment->construct<boost::interprocess::interprocess_mutex>("mutex")();

        vals = segment->construct<boost_SMVector<T>>("vals")(*alloc_T);
        assert(n < vals->max_size());
        vals->reserve(n);
      } catch(std::bad_alloc&) {
        std::cerr<<"Problems allocating shared memory in SMDenseVector." <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,20);
        return false;
      }
    }
    barrier();
    if(!head) {
      try {
        segment = new boost::interprocess::managed_shared_memory(boost::interprocess::open_only, ID.c_str());
        vals = segment->find<boost_SMVector<T>>("vals").first;
        mutex = segment->find<boost::interprocess::interprocess_mutex>("mutex").first;
        assert(vals != 0);
        assert(mutex != 0);
      } catch(std::bad_alloc&) {
        std::cerr<<"Problems allocating shared memory in SMDenseVector: initializeChildren() ." <<std::endl;
        MPI_Abort(MPI_COMM_WORLD,20);
        return false;
      }
    }
    barrier();
    SMallocated=true;
    return true;
  }

  // resize is probably the best way to setup the vector 
  inline void resize(unsigned long nnz) 
  {
    if(!SMallocated || vals==NULL) { 
      if(!reserve(nnz)) {
        std::cerr<<" Error with reserve in SMDenseVector: " <<ID <<std::endl; 
        MPI_Abort(MPI_COMM_WORLD,20);
      }
    }
    if(vals->capacity() < nnz) {  
      std::cerr<<" Error in SMDenseVector: " <<ID <<" Resizing beyond capacity." <<std::endl; 
      MPI_Abort(MPI_COMM_WORLD,20);
    }
    if(head) {
      assert(SMallocated);
      assert(vals->capacity() >= nnz); 
      vals->resize(nnz);
    }
    barrier();
  }

  inline void clear() { 
    assert(SMallocated);
    if(!head) return;
    vals->clear();
  }

  inline unsigned long size() const
  {
    assert(SMallocated && vals!=NULL);
    return vals->size();
  }

  inline const_pointer values() const 
  {
    assert(SMallocated && vals!=NULL);
    return &((*vals)[0]);
  }

  inline pointer values() 
  {
    assert(SMallocated && vals!=NULL);
    return &((*vals)[0]);
  }

  inline pointer data()
  {
    assert(SMallocated && vals!=NULL);
    return &((*vals)[0]);
  }

  inline bool isAllocated() {
    return (SMallocated)&&(vals!=NULL); 
  }

  inline This_t& operator=(const SMDenseVector<T> &rhs) = delete; 

  inline Type_t& operator()(unsigned long i)
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(SMallocated && vals!=NULL);
    assert(i>=0 && i<vals->size());
#endif
    return (*vals)[i]; 
  }

  inline Type_t& operator[](unsigned long i)
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(SMallocated && vals!=NULL);
    assert(i>=0 && i<vals->size());
#endif
    return (*vals)[i]; 
  }

  template<typename IType>
  inline void add(const IType i, const T& v, bool needs_locks=false) 
  {
#ifdef ASSERT_SPARSEMATRIX
    assert(SMallocated && vals!=NULL);
    assert(i>=0 && i<vals->size());
#endif
    if(needs_locks) {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex);
        (*vals)[i]=v;
    } else {
      if(!head) return;
      (*vals)[i]=v;
    }
  }

  inline unsigned long memoryUsage() { return memory; }

  inline unsigned long capacity() { return (vals==NULL)?0:vals->capacity(); }

  inline void push_back(const T& v, bool needs_locks=false)             
  {
    assert(SMallocated && vals!=NULL);
    if(needs_locks) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex);
      assert(vals->capacity() >= vals->size()+1 );
      vals->push_back(v);
    } else {
      assert(vals->capacity() >= vals->size()+1 );
      vals->push_back(v);
    }
  }

  inline void push_back(const std::vector<T>& v, bool needs_locks=false)
  {
    assert(SMallocated && vals!=NULL);
    if(needs_locks) {
      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex);
      assert(vals->capacity() >= vals->size()+v.size() );
      for(auto&& i: v) 
        vals->push_back(i);
    } else {
      assert(vals->capacity() >= vals->size()+v.size() );
      for(auto&& i: v) 
        vals->push_back(i);
    }
  }

  inline void sort() {
    assert(SMallocated && vals!=NULL);
    if(!head) return;
    std::sort(vals->begin(),vals->end());
  }

  template<class Compare>
  inline void sort(Compare comp, MPI_Comm local_comm=MPI_COMM_SELF, bool inplace=true) {

    assert(SMallocated && vals!=NULL);
    if(vals->size() == 0) return;
    if(local_comm==MPI_COMM_SELF) {
      std::sort(vals->begin(),vals->end(),comp);
    }

    int npr,rank;
    MPI_Comm_rank(local_comm,&rank);
    MPI_Comm_size(local_comm,&npr);

    MPI_Barrier(local_comm);

    int nlvl = static_cast<int>(  std::floor(std::log2(npr*1.0)) );
    int nblk = pow(2,nlvl);

    // sort equal segments in parallel
    std::vector<int> pos(nblk+1);
    // a core processes elements from pos[rank]-pos[rank+1]
    FairDivide(vals->size(),nblk,pos);

    // sort local segment
    if(rank < nblk)
      std::sort(vals->begin()+pos[rank], vals->begin()+pos[rank+1], comp);

    MPI_Barrier(local_comm);

    std::vector<T> temp;
    if (!inplace && rank<nblk) {
      int nt=0;
      for(int i=0, k=rank, sz=1; i<nlvl; i++, sz*=2 ) {
        if(k%2==0 && rank<nblk) {
          nt = std::max(nt,pos[rank+sz*2] - pos[rank]);
          k/=2;
        } else
          k=1;
      }
      temp.resize(nt);
    }
    
    for(int i=0, k=rank, sz=1; i<nlvl; i++, sz*=2 ) {
      if(k%2==0 && rank<nblk) {
        if(inplace) {
          std::inplace_merge( vals->begin()+pos[rank], vals->begin()+pos[rank+sz], vals->begin()+pos[rank+sz*2], comp);  
        } else { 
          unsigned long nt = pos[rank+sz*2] - pos[rank];
           assert( temp.size() >= nt );  
          std::merge( vals->begin()+pos[rank], vals->begin()+pos[rank+sz], vals->begin()+pos[rank+sz], vals->begin()+pos[rank+sz*2], temp.begin(), comp);  
          std::copy(temp.begin(),temp.begin()+nt,vals->begin()+pos[rank]);
        }
        k/=2;
      } else
        k=1;  
      MPI_Barrier(local_comm);
    }

  }

  template<typename Tp>
  inline SMDenseVector<T>& operator*=(const Tp rhs ) 
  {
    assert(SMallocated && vals!=NULL);
    if(!head) return *this; 
    for(iterator it=vals->begin(); it!=vals->end(); it++)
      (*it) *= rhs;
    return *this; 
  }

  template<typename Tp>
  inline SMDenseVector<T>& operator*=(const std::complex<Tp> rhs ) 
  {
    assert(SMallocated && vals!=NULL);
    if(!head) return *this; 
    for(iterator it=vals->begin(); it!=vals->end(); it++)
      (*it) *= rhs;
    return *this; 
  }

  // this is ugly, but I need to code quickly 
  // so I'm doing this to avoid adding hdf5 support here 
  inline boost_SMVector<T>* getVector() const { return vals; } 

  inline iterator begin() { assert(SMallocated && vals!=NULL); return vals->begin(); } 
  inline const_iterator begin() const { assert(SMallocated && vals!=NULL); return vals->begin(); } 
  inline const_iterator end() const { assert(SMallocated && vals!=NULL); return vals->end(); } 
  inline iterator end() { assert(SMallocated && vals!=NULL); return vals->end(); } 
  inline T& back() { assert(SMallocated && vals!=NULL); return vals->back(); } 

  boost::interprocess::interprocess_mutex* getMutex()
  {
    return mutex;
  } 

  private:

  boost::interprocess::interprocess_mutex *mutex;
  boost_SMVector<T> *vals;
  bool head;
  std::string ID; 
  bool SMallocated;
  uint64_t memory=0;
  std::pair<int, int> dims;   // kind of cheating, but allows me to use as a matrix when needed 

  boost::interprocess::managed_shared_memory *segment;
  ShmemAllocator<T> *alloc_T;
  ShmemAllocator<boost::interprocess::interprocess_mutex> *alloc_mutex;

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
