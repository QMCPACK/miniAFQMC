#include <vector>
#include <array>

template <typename ArrayType>
class GPUArray {

public:

  typedef typename ArrayType::index index;
  typedef typename ArrayType::size_type size_type;
  typedef typename ArrayType::element element;
  
  const static std::size_t dimensionality = ArrayType::dimensionality;
  
  GPUArray(ArrayType & array):
    base(array),
    dirty(false)
  {

    size_type total_size = 1;
    
    for(size_type idim = dimensionality - 1; idim-- > 0;){
      shape_[idim] = base.shape()[idim];
      total_size *= shape_[idim];

      if(idim == dimensionality - 1){
	strides_[idim] = 1;
      } else {
	strides_[idim] = strides_[idim + 1]*shape_[idim + 1];
      }
      
    }

    data_.resize(total_size);

    std::array<size_type, dimensionality> coords;
    for(size_type idim = 0; idim < dimensionality; idim++) coords[idim] = 0;

    for(size_type ii = 0; ii < total_size; ii++){

      size_type i1 = 0;
      size_type i2 = 0;

      for(size_type idim = 0; idim < dimensionality; idim++){
	i1 += strides_[idim]*coords[idim];
	i2 += base.strides()[idim]*coords[idim];
      }
      
      data_[i1] = base.origin()[i2];

      coords[dimensionality - 1]++;
      for(size_type idim = dimensionality - 1; idim-- > 0;){
	if(coords[idim] == shape_[idim]){
	  coords[idim] = 0;
	  if(idim > 0) coords[idim - 1]++;
	}
      }

    }
      
  }

  ~GPUArray(){

    if(dirty){
      
      auto origin = const_cast<element *>(base.origin());
      
      std::array<size_type, dimensionality> coords;
      for(size_type idim = 0; idim < dimensionality; idim++) coords[idim] = 0;
      
      for(size_type ii = 0; ii < data_.size(); ii++){
	
	size_type i1 = 0;
	size_type i2 = 0;
	
	for(size_type idim = 0; idim < dimensionality; idim++){
	  i1 += strides_[idim]*coords[idim];
	  i2 += base.strides()[idim]*coords[idim];
	}
      
	origin[i2] = data_[i1];
	
	coords[dimensionality - 1]++;
	for(size_type idim = dimensionality - 1; idim-- > 0;){
	  if(coords[idim] == shape_[idim]){
	  coords[idim] = 0;
	  if(idim > 0) coords[idim - 1]++;
	  }
	}
	
      }
     
    }
    
  }
  
  const index * strides() const {
    return strides_.data();
  }

  const size_type * shape() const {
    return shape_.data();
  }

  const element * origin() const {
    return data_.data();
    //return base.origin();
  }

  element * origin() {
    dirty = true;
    return data_.data();
    //return base.origin();
  }
  
private:
  std::array<index, dimensionality> strides_;
  std::array<size_type, dimensionality> shape_;
  ArrayType & base;
  std::vector<element> data_;
  bool dirty;
  
};

template <typename ArrayType>
GPUArray<ArrayType> gpu(ArrayType array){
  return GPUArray<ArrayType>(array);
}


  
  
