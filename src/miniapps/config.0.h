#ifndef AFQMC_CONFIG_0_H 
#define AFQMC_CONFIG_0_H 

#include <string>
#include <algorithm>
#include<cstdlib>
#include<ctype.h>
#include <vector>
#include <map>
#include <complex>
#include <tuple>
#include <fstream>

/*
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
*/

namespace qmcplusplus
{

template<typename T>
inline bool isComplex(const T& a) 
{
  return std::is_same<T,std::complex<RealType>>::value; 
}

template<typename T>
inline ComplexType toComplex(const T& a);

template<>
inline ComplexType toComplex(const RealType& a) 
{
  return ComplexType(a,0.0); 
}

template<>
inline ComplexType toComplex(const std::complex<RealType>& a) 
{
  return a; 
}

template<typename T>
inline void setImag(T& a, RealType b);

template<>
inline void setImag(RealType& a, RealType b)
{
}

template<>
inline void setImag(std::complex<RealType>& a, RealType b)
{
  a.imag(b);
}

template<typename T>
inline T myconj(const T& a) 
{
  return a;
}

template<typename T>
inline std::complex<T> myconj(const std::complex<T>& a) 
{
  return std::conj(a);
}

template<typename T>
inline RealType mynorm(const T& a)
{
  return a*a;
}

template<typename T>
inline RealType mynorm(const std::complex<T> &a)
{
  return std::norm(a);
}

template<typename T>
inline std::complex<T> operator*(const int &lhs, const std::complex<T> &rhs)
{
  return T(lhs) * rhs;
}

template<typename T>
inline std::complex<T> operator*(const std::complex<T> &lhs, const int &rhs)
{
  return lhs * T(rhs);
}

inline bool sortDecreasing (int i,int j) { return (i>j); }

}


namespace std {
template<typename T>
inline bool operator<(const std::complex<T> &lhs, const std::complex<T> &rhs)
{
/*
  if (lhs.real() != rhs.real())
  {
    return lhs.real() < rhs.real();
  }
  return lhs.imag() < rhs.imag();
*/
  return std::abs(lhs) < std::abs(rhs);
}
}

#endif
