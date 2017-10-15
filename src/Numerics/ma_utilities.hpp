#ifndef MA_UTILITIES_HPP
#define MA_UTILITIES_HPP

#include <Kokkos_Core.hpp>



// Translate a 2d boost::multi_array to a Kokkos::View
template<typename T>
void boostMultiArrayToKokkosView(boost::multi_array<T, 2> const& a, Kokkos::View<T**> &b)
{
  // get dimension of the multi_array and resize the View
  size_t dim0 = a.shape()[0];
  size_t dim1 = a.shape()[1];
  if(dim0 != b.dimension(0) && dim1 != b.dimension(1))
    Kokkos::resize(b, dim0, dim1);

  // Copy element wise to the View (inefficient, but ultimately we want to get rid of the multi_arrays anyway)
  for(size_t i = 0; i < dim0; ++i)
    for(size_t j = 0; j < dim1; ++j)
      b(i,j) = a[i][j];
}

templae<typename T>
void kokkosViewToBoostMultiArray(Kokkos::View<T**> a, boost::multi_array<T, 2> &b)
{

  // get dimension of the View and resize the multi_array
  size_t dim0 = a.dimension(0);
  size_t dim1 = a.dimension(1);
  if(dim0 != b.shape()[0] && dim1 != b.shape()[1])
    b.resize(extents[dim0][dim1]);

  // Copy element wise to the multi_array (inefficient, but ultimately we want to get rid of the multi_arrays anyway)
  for(size_t i = 0; i < dim0; ++i)
    for(size_t j = 0; j < dim1; ++j)
      b[i][j] = a(i,j);
}

// Translate a 4d boost::multi_array to a Kokkos::View
template<typename T>
void boostMultiArrayToKokkosView(boost::multi_array<T, 4> const& a, Kokkos::View<T****> &b)
{
  // get dimension of the multi_array and resize the View
  size_t dim0 = a.shape()[0];
  size_t dim1 = a.shape()[1];
  size_t dim2 = a.shape()[2];
  size_t dim3 = a.shape()[3];
  if(dim0 != b.dimension(0) && dim1 != b.dimension(1) &&
     dim2 != b.dimension(2) && dim3 != b.dimension(3))
    Kokkos::resize(b, dim0, dim1, dim2, dim3);

  // Copy element wise to the View (inefficient, but ultimately we want to get rid of the multi_arrays anyway)
  for(size_t i = 0; i < dim0; ++i)
    for(size_t j = 0; j < dim1; ++j)
      for(size_t k = 0; k < dim2; ++k)
        for(size_t l = 0; l < dim3; ++l)
          b(i,j,k,l) = a[i][j][k][l];
}

templae<typename T>
void kokkosViewToBoostMultiArray(Kokkos::View<T**> a, boost::multi_array<T, 2> &b)
{

  // get dimension of the View and resize the multi_array
  size_t dim0 = a.dimension(0);
  size_t dim1 = a.dimension(1);
  size_t dim2 = a.dimension(2);
  size_t dim3 = a.dimension(3);
  if(dim0 != b.shape()[0] && dim1 != b.shape()[1] &&
     dim2 != b.shape()[2] && dim3 != b.shape()[3])
    b.resize(extents[dim0][dim1][dim2][dim3]);

  // Copy element wise to the multi_array (inefficient, but ultimately we want to get rid of the multi_arrays anyway)
  for(size_t i = 0; i < dim0; ++i)
    for(size_t j = 0; j < dim1; ++j)
      for(size_t k = 0; k < dim2; ++k)
        for(size_t l = 0; l < dim3; ++l)
          b[i][j][k][l] = a(i,j,k,l);
}

#endif
