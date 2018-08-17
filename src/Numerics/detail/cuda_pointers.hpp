//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef AFQMC_CUBLAS_POINTERS_HPP 
#define AFQMC_CUBLAS_POINTERS_HPP

#define CUBLAS_BLAS_TYPE    100
#define CUBLASXT_BLAS_TYPE  200
#define CUSTOM1_BLAS_TYPE   300
#define CUSTOM2_BLAS_TYPE   400
#define CUSTOM3_BLAS_TYPE   500

#define CPU_MEMORY_POINTER_TYPE      1001
#define GPU_MEMORY_POINTER_TYPE      2001
#define MANAGED_MEMORY_POINTER_TYPE  3001
#define CPU_OUTOFCARS_POINTER_TYPE   4001

#include<cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"

template<class T>
struct cuda_um_ptr{
  using value_type = T;
  using pointer = T*;
  static const int blas_type = CUBLAS_BLAS_TYPE;
  T* impl_;
  cublasHandle_t* cublas_handle;
  cuda_um_ptr() = default;
  cuda_um_ptr(T* impl__, cublasHandle_t* cublas_handle_=nullptr):impl_(impl__),cublas_handle(cublas_handle_) {}
// eventually check if memory types and blas types are convertible, e.g. CPU_MEMORY to CPU_OUTOFCARD
  template<typename Q,
           typename = typename std::enable_if_t<cuda_um_ptr<Q>::blas_type == blas_type>
              >
  cuda_um_ptr(cuda_um_ptr<Q> const& ptr):impl_(ptr.impl_),cublas_handle(ptr.cublas_handle) {} 
  T& operator*() const{return *impl_;}
  T& operator[](std::ptrdiff_t n) const{return impl_[n];}
  T* operator->() const{return impl_;}
  explicit operator bool() const{return (impl_!=nullptr);}
  auto operator+(std::ptrdiff_t n) const{return cuda_um_ptr{impl_ + n,cublas_handle};}
  std::ptrdiff_t operator-(cuda_um_ptr other) const{return std::ptrdiff_t(impl_-other.impl_);}
  cuda_um_ptr& operator++() {++impl_; return *this;} 
  cuda_um_ptr& operator--() {--impl_; return *this;} 
  cuda_um_ptr& operator+=(std::ptrdiff_t d){impl_ += d; return *this;}
  cuda_um_ptr& operator-=(std::ptrdiff_t d){impl_ -= d; return *this;}
  bool operator==(cuda_um_ptr const& other) const{ return impl_==other.impl_; }
  bool operator!=(cuda_um_ptr const& other) const{ return not (*this == other); } 
  bool operator<=(cuda_um_ptr<T> const& other) const{
    return impl_ <= other.impl_;
  }
  T* get() const {return impl_;}
  friend decltype(auto) get(cuda_um_ptr const& self){return self.get();}
};

  
template<class T> struct cuda_um_allocator{
  template<class U> struct rebind{typedef cuda_um_allocator<U> other;};
  using value_type = T;
  using pointer = cuda_um_ptr<T>;
  using const_pointer = cuda_um_ptr<T const>;
  using reference = T&;
  using const_reference = T const&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  cublasHandle_t* cublas_handle_;
  cuda_um_allocator(cublasHandle_t* cublas_handle__) : cublas_handle_(cublas_handle__){}
  cuda_um_allocator() = delete;
  ~cuda_um_allocator() = default;
  cuda_um_allocator(cuda_um_allocator const& other) : cublas_handle_(other.cublas_handle_){}
  template<class U>
  cuda_um_allocator(cuda_um_allocator<U> const& other) : cublas_handle_(other.cublas_handle_) {} 

  cuda_um_ptr<T> allocate(size_type n, const void* hint = 0){
    if(n == 0) return cuda_um_ptr<T>{};
    T* p;
    cudaMallocManaged ((void**)&p,n*sizeof(T),cudaMemAttachGlobal);
    return cuda_um_ptr<T>{p,cublas_handle_};
  }
  void deallocate(cuda_um_ptr<T> ptr, size_type){
    cudaFree(ptr.impl_); 
  }
  bool operator==(cuda_um_allocator const& other) const{
    cublas_handle_ == other.cublas_handle_;   
  }
  bool operator!=(cuda_um_allocator const& other) const{
    return not (other == *this);
  }
  template<class U, class... Args>
  void construct(U* p, Args&&... args){
    ::new((void*)p) U(std::forward<Args>(args)...);
  }
  template< class U >
  void destroy(U* p){
    p->~U();
  }
};
  
#endif
