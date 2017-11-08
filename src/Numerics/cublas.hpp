#ifndef CUBLAS_HPP
#define CUBLAS_HPP

#include <cublas_v2.h>

namespace cublas {

  cublasHandle_t handle = NULL;

  void check_status(cublasStatus_t status){
    switch(status) {
    case CUBLAS_STATUS_SUCCESS :
      return;
    case CUBLAS_STATUS_ALLOC_FAILED	:
      std::cout << "cublas: the resources could not be allocated."  << std::endl;
      break;
    case CUBLAS_STATUS_NOT_INITIALIZED :
      std::cout << "cublas: the library was not initialized."  << std::endl;
      break;
    case CUBLAS_STATUS_INVALID_VALUE :
      std::cout << "cublas: invalid parameters were passed."  << std::endl;
      break;
    case CUBLAS_STATUS_ARCH_MISMATCH :
      std::cout << "cublas: the device does not support double precision."  << std::endl;
      break;
    case CUBLAS_STATUS_EXECUTION_FAILED :
      std::cout << "cublas: the function failed to launch on the GPU."  << std::endl;
      break;
    case CUBLAS_STATUS_INTERNAL_ERROR :
      std::cout << "cublas: an internal operation failed."  << std::endl;
      break;
    default :
      std::cout << "cublas: Unknown error."  << std::endl;
      break;
    }
    exit(1);
  }

  void init(){
    check_status(cublasCreate(&handle));
  }
  
  void end(){
    check_status(cublasDestroy(handle));
  }

  template<typename op_tag_type>
  cublasOperation_t op_tag(){
    if(op_tag_type::value == 'N') return CUBLAS_OP_N;
    if(op_tag_type::value == 'T') return CUBLAS_OP_T;
    if(op_tag_type::value == 'C') return CUBLAS_OP_C;
    assert(false);
  }

  void transpose(const int m, const int n, double *A, const int lda, double *B, const int ldb){
    const double one = 1.0;
    const double zero = 0.0;

    check_status(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			     m, n, &one, A, lda, &zero, NULL, 0, A, lda));
    
  }
  
  void transpose(const int m, const int n, const std::complex<double> *A, const int lda,
		 const std::complex<double> *B, const int ldb){
    
    const std::complex<double> one = 1.0;
    const std::complex<double> zero = 0.0;

    std::complex<double> * C;

    std::cout << "m = " << m << " n = " << n << " lda = " << lda << " ldb = " << ldb << std::endl;
    std::cout << A << '\t' << B << std::endl;
    
    check_status(cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
			     m, n, (cuDoubleComplex *) &one, (cuDoubleComplex *) A, lda,
			     (cuDoubleComplex *) &zero, (cuDoubleComplex *) A, m, (cuDoubleComplex *) B, ldb));

  }
}
#endif
