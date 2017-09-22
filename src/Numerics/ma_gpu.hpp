template <typename ArrayType>
class GPUArray {

public:

  typedef typename ArrayType::index index;
  typedef typename ArrayType::size_type size_type;
  typedef typename ArrayType::element element;
  
  const static std::size_t dimensionality = ArrayType::dimensionality;
  
  GPUArray(ArrayType & array):
    base(array)
  {
  }

  const index * strides() const {
    return base.strides();
  }

  const size_type * shape() const {
    return base.shape();
  }

  const element * origin() const {
    return base.origin();
  }

  element * origin() {
    return base.origin();
  }
  
private:
  ArrayType & base;
};

template <typename ArrayType>
GPUArray<ArrayType> gpu(ArrayType array){
  return GPUArray<ArrayType>(array);
}


  
  
