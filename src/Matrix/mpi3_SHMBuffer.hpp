////////////////////////////////////////////////////////////////////////
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


#ifndef  AFQMC_MPI3_SHMBUFFER_HPP 
#define  AFQMC_MPI3_SHMBUFFER_HPP 

#include "mpi.h"
#include "alf/boost/mpi3/shared_window.hpp"
#include "alf/boost/mpi3/shared_communicator.hpp"
#include <memory>

#ifdef __bgq__
#include "Utilities/simple_stack_allocator.hpp"
extern std::unique_ptr<simple_stack_allocator> bgq_dummy_allocator;
extern std::size_t BG_SHM_ALLOCATOR;
#endif

namespace qmcplusplus
{

namespace afqmc
{

#ifdef __bgq__
// hack for BGQ
// right now it leaks, fix later!!!!
template<typename T>
class mpi3_SHMBuffer
{ 
  using communicator = boost::mpi3::shared_communicator;
  using shared_window = boost::mpi3::shared_window<T>;
  
  public:
    
    mpi3_SHMBuffer(communicator& comm_, size_t n=0):
        base(nullptr),comm(std::addressof(comm_)),capacity(n)
    {
      if(bgq_dummy_allocator==nullptr) {
        // this leaks memory right now!!! needs fixing!!!
        MPI_Win impl_;
        void* base_ptr = nullptr;
        boost::mpi3::size_t size_ = (comm_.root()?BG_SHM_ALLOCATOR*1024*1024:0);
        int dunit;
        int s = MPI_Win_allocate_shared(size_, 1, MPI_INFO_NULL, comm_.impl_, &base_ptr, &impl_);
        if(s != MPI_SUCCESS) throw std::runtime_error("cannot create shared window");
        MPI_Win_shared_query(impl_, 0, &size_, &dunit, &base_ptr);
        bgq_dummy_allocator = std::move(std::make_unique<simple_stack_allocator>(base_ptr,size_));
      }
      if(n>0)
        base = bgq_dummy_allocator->allocate<T>(n);
    }
    
    mpi3_SHMBuffer<T>(const mpi3_SHMBuffer<T>& other) = delete;
    mpi3_SHMBuffer<T>& operator=(const mpi3_SHMBuffer<T>& other) = delete;
    
    mpi3_SHMBuffer<T>(mpi3_SHMBuffer<T>&& other) {
      *this = std::move(other);
    }
    
    mpi3_SHMBuffer<T>& operator=(mpi3_SHMBuffer<T>&& other) {
      if(this != &other) {
        comm = std::exchange(other.comm,nullptr);
        base = std::exchange(other.base,nullptr);
        capacity = std::exchange(other.capacity,0);
        comm->barrier();
      }
      return *this;
    }
    
    void resize(size_t n) {
      if(size() == n) return;
      T* base_ = bgq_dummy_allocator->allocate<T>(n);  
      comm->barrier();
      if(comm->rank()==0 && base != nullptr && capacity > 0) {
        if(size() < n) 
          std::copy(this->data(),this->data()+size(),base_);
        else
          std::copy(this->data(),this->data()+n,base_);
      }
      comm->barrier();
      base = base_;  
      capacity = n;
      comm->barrier();
    }

    T* data() {return base;}
    T const* data() const{return base;}
    size_t size() const{return static_cast<size_t>(capacity);}

    communicator& getCommunicator() const{return *comm;}

  private:

    communicator* comm;
    size_t capacity;
    T* base;

};
#else
template<typename T>
class mpi3_SHMBuffer
{
  using communicator = boost::mpi3::shared_communicator;
  using shared_window = boost::mpi3::shared_window<T>;  

  public:

    mpi3_SHMBuffer(communicator& comm_, size_t n=0):
        comm(std::addressof(comm_)),win(comm_,(comm_.root()?n:0),sizeof(T)) 
    {}

    mpi3_SHMBuffer<T>(const mpi3_SHMBuffer<T>& other) = delete;
    mpi3_SHMBuffer<T>& operator=(const mpi3_SHMBuffer<T>& other) = delete;

    mpi3_SHMBuffer<T>(mpi3_SHMBuffer<T>&& other) {
      *this = std::move(other);
    }

    mpi3_SHMBuffer<T>& operator=(mpi3_SHMBuffer<T>&& other)
    {
      assert(comm==other.comm);
      if(this != &other) {
        // this should be in mpi3 namespace
        auto tmp = win.impl_;
        win.impl_ = other.impl_;
        other.impl_ = tmp;
        //swap(win.impl_,other.impl_);
        comm->barrier();
      }
      return *this;
    }

    void resize(size_t n) {
      if(size() == n) return;
      shared_window w0(*comm,(comm->root()?n:0),sizeof(T)); 
      comm->barrier();
      if(comm->rank()==0) {
        if(size() < n) 
          std::copy(this->data(),this->data()+size(),w0.base(0));
        else
          std::copy(this->data(),this->data()+static_cast<size_t>(w0.size(0)),w0.base(0));
      }   
      comm->barrier();
      auto tmp = win.impl_;
      win.impl_ = w0.impl_;
      w0.impl_ = tmp;
      comm->barrier();
    }

    T* data() {return win.base(0);} 
    T const* data() const{return win.base(0);} 
    size_t size() const{return static_cast<size_t>(win.size(0));} 

    communicator& getCommunicator() const{return *comm;}

  private:

    // mpi3_SHMBuffer does not own the communicator comm points to.
    // This class assumes that objects of this class live in the same scope
    // as the communicator comm points to and  that it will remain valid and equivalent. 
    communicator* comm;
    
    shared_window win;

};
#endif

}

}

#endif
