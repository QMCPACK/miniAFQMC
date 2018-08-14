#ifndef AFQMC_SIMPLE_STACK_ALLOCATOR
#define AFQMC_SIMPLE_STACK_ALLOCATOR

#include <memory>

//#define __TESTING__
#ifdef __TESTING__
// just a hack for bgq right now, do something proper later
class simple_stack_allocator
{
  double *ptr;
  std::size_t left;

  public:

  simple_stack_allocator(void* ptr_, std::size_t cap_): ptr(reinterpret_cast<double*>(ptr_)), left(cap_/sizeof(double)) {} 

  ~simple_stack_allocator() = default;

  simple_stack_allocator(simple_stack_allocator const& other) = delete;  
  simple_stack_allocator(simple_stack_allocator && other) = delete;  
  simple_stack_allocator& operator=(simple_stack_allocator const& other) = delete;
  simple_stack_allocator& operator=(simple_stack_allocator && other) = delete;

  template<class T>
  T* allocate(std::size_t n) {
    if(n==0) return nullptr;
    int m = (int)std::ceil(sizeof(T)*double(n)/sizeof(double));
    if(m > left) {
      throw std::out_of_range("simple_stack_allocator::allocate(n) exceeded the maximum capacity");  
    }
    T* result = reinterpret_cast<T*>(ptr);
    ptr += m;
    left -= m;
    return result;
  }

};
#else
#include "Configuration.h"
class simple_stack_allocator
{
  void *ptr;
  std::size_t left;

  public:

  simple_stack_allocator(void* ptr_, std::size_t cap_): ptr(ptr_), left(cap_) {}

  ~simple_stack_allocator() = default;

  simple_stack_allocator(simple_stack_allocator const& other) = delete;
  simple_stack_allocator(simple_stack_allocator && other) = delete;
  simple_stack_allocator& operator=(simple_stack_allocator const& other) = delete;
  simple_stack_allocator& operator=(simple_stack_allocator && other) = delete;

  template<class T>
  T* allocate(std::size_t n) {
    if(n==0) return nullptr;
    if (std::align(alignof(T), sizeof(T), ptr, left))
    {
        if(not ptr)
          throw std::out_of_range("simple_stack_allocator::problems in simple_stack_allocator:;allocate");
        if(n*sizeof(T) > left)
          throw std::out_of_range("simple_stack_allocator::allocate(n) exceeded the maximum capacity");
        T* result = reinterpret_cast<T*>(ptr);
        ptr = (char*)ptr + n*sizeof(T);
        left -= n*sizeof(T);
        qmcplusplus::app_log()<<" Remaining shared memory in simple_stack_allocator: " <<left/1024.0/1024.0 <<" MB. " <<"\n";
        return result;
    } else
      throw std::out_of_range("simple_stack_allocator::problems in simple_stack_allocator:;allocate");
    return nullptr;
  }

};
#endif

#endif

