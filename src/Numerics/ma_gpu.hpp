#ifndef MA_GPU
#define MA_GPU

#include <vector>
#include <array>
#include <boost/multi_array.hpp>
#include <cuda_runtime.h>
#include "Matrix/SparseMatrix.hpp"

template <typename ArrayType>
class constGPUArray {

public:

  typedef typename ArrayType::index index;
  typedef typename ArrayType::size_type size_type;
  typedef typename ArrayType::element element;
  
  const static std::size_t dimensionality = ArrayType::dimensionality;
  
  constGPUArray(const ArrayType & array):
    const_base(array),
    data_(array)
  {
    cudaMalloc(&gpu_data_, sizeof(element)*data_.num_elements());
    cudaMemcpy(gpu_data_, data_.data(), sizeof(element)*data_.num_elements(), cudaMemcpyHostToDevice);
  }

  ~constGPUArray(){
    cudaFree(gpu_data_);
  }

  const index * strides() const {
    return data_.strides();
  }

  const size_type * shape() const {
    return data_.shape();
  }

  const element * origin() const {
    return gpu_data_;
  }

protected:
  const ArrayType & const_base;
  boost::multi_array<element, dimensionality> data_;
  element * gpu_data_;
  
};


template <typename ArrayType>
class GPUArray: public constGPUArray<ArrayType> {

public:

  typedef typename ArrayType::index index;
  typedef typename ArrayType::size_type size_type;
  typedef typename ArrayType::element element;
  
  const static std::size_t dimensionality = ArrayType::dimensionality;
  
  GPUArray(ArrayType & array):
    constGPUArray<ArrayType>(array),
    base(array)
  {
  }

  ~GPUArray(){
    cudaMemcpy(data_.data(), gpu_data_, sizeof(element)*data_.num_elements(), cudaMemcpyDeviceToHost);
    base = data_;
  }

  using constGPUArray<ArrayType>::origin;
  
  element * origin() {
    return gpu_data_;
  }

private:
  using constGPUArray<ArrayType>::data_;
  using constGPUArray<ArrayType>::gpu_data_;
  ArrayType & base;
  
};

using qmcplusplus::SparseMatrix;
  
template <typename Type>
class constGPUSparseMatrix {
  
public:
 
  typedef typename SparseMatrix<Type>::intType intType;
  typedef typename SparseMatrix<Type>::const_pointer const_pointer;
  typedef typename SparseMatrix<Type>::const_intPtr const_intPtr;
  
  const static int dimensionality = SparseMatrix<Type>::dimensionality;
  
  constGPUSparseMatrix(const SparseMatrix<Type> & matrix):
    const_base_(matrix)
  {
    cudaMalloc(&vals_, const_base_.size()*sizeof(Type));
    cudaMemcpy(vals_, const_base_.values(), const_base_.size()*sizeof(Type), cudaMemcpyHostToDevice);

    cudaMalloc(&cols_, const_base_.size()*sizeof(typename SparseMatrix<Type>::intType));
    cudaMemcpy(cols_, const_base_.column_data(), const_base_.size()*sizeof(typename SparseMatrix<Type>::intType), cudaMemcpyHostToDevice);

    cudaMalloc(&rows_, (const_base_.rows() + 1)*sizeof(typename SparseMatrix<Type>::intType));
    cudaMemcpy(rows_, const_base_.row_index(), (const_base_.rows() + 1)*sizeof(typename SparseMatrix<Type>::intType), cudaMemcpyHostToDevice);
    
  }

  ~constGPUSparseMatrix(){

    cudaFree(vals_);
    cudaFree(cols_);
    cudaFree(rows_);

  }

  int rows() const {
    return const_base_.rows();
  }
  
  int cols() const {
    return const_base_.cols();
  }

  int nnz() const {
    return const_base_.nnz();
  }
  
  const_pointer val() const {
    return vals_;
  }
  
  const_intPtr indx() const {
    return cols_;
  }

  const_intPtr pntrb(long n=0) const {
    return rows_;
  }

  const_intPtr pntre(long n=0) const {
    return rows_ + 1;
  }
  
protected:
  const SparseMatrix<Type> & const_base_;
  Type * vals_;
  intType * cols_;
  intType * rows_;
  
};


template <typename Type>
constGPUSparseMatrix<Type> gpu(const SparseMatrix<Type> & matrix){
  return constGPUSparseMatrix<Type>(matrix);
}

template <typename ArrayType>
constGPUArray<ArrayType> gpu(const ArrayType & array){
  return constGPUArray<ArrayType>(array);
}


template <typename ArrayType>
GPUArray<ArrayType> gpu(ArrayType & array){
  return GPUArray<ArrayType>(array);
}

template <typename ArrayType>
bool in_gpu(const constGPUArray<ArrayType> & array){
  return true;
}

template <typename ArrayType>
bool in_gpu(const GPUArray<ArrayType> & array){
  return true;
}

template <typename Type>
bool in_gpu(const constGPUSparseMatrix<Type> &){
  return true;
}
template <typename ArrayType>
bool in_gpu(const ArrayType & array){
  return false;
}

#endif

