#include <vector>
#include <array>
#include<boost/multi_array.hpp>

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
  }

  ~constGPUArray(){
  }

  const index * strides() const {
    data_.strides();
  }

  const size_type * shape() const {
    data_.shape();
  }

  const element * origin() const {
    return data_.origin();
  }

protected:
  const ArrayType & const_base;
  boost::multi_array<element, dimensionality> data_;
  
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
    base = constGPUArray<ArrayType>::data_;
  }

  using constGPUArray<ArrayType>::origin;
  
  element * origin() {
    return constGPUArray<ArrayType>::data_.origin();
  }

private:
  ArrayType & base;
  
};


template <typename ArrayType>
constGPUArray<ArrayType> gpu(const ArrayType & array){
  return constGPUArray<ArrayType>(array);
}


template <typename ArrayType>
GPUArray<ArrayType> gpu(ArrayType & array){
  return GPUArray<ArrayType>(array);
}
  
  
