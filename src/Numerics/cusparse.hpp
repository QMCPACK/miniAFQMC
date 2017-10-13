#ifndef CUSPARSE_HPP
#define CUSPARSE_HPP

#include <cusparse.h>

namespace cusparse {

  cusparseHandle_t handle = NULL;

  void check_status(cusparseStatus_t status){
    switch(status) {
    case CUSPARSE_STATUS_SUCCESS :
      return;
    case CUSPARSE_STATUS_ALLOC_FAILED	:
      std::cout << "cusparse: the resources could not be allocated."  << std::endl;
      break;
    case CUSPARSE_STATUS_NOT_INITIALIZED :
      std::cout << "cusparse: the library was not initialized."  << std::endl;
      break;
    case CUSPARSE_STATUS_INVALID_VALUE :
      std::cout << "cusparse: invalid parameters were passed (m,n <0)."  << std::endl;
      break;
    case CUSPARSE_STATUS_ARCH_MISMATCH :
      std::cout << "cusparse: the device does not support double precision."  << std::endl;
      break;
    case CUSPARSE_STATUS_EXECUTION_FAILED :
      std::cout << "cusparse: the function failed to launch on the GPU."  << std::endl;
      break;
    case CUSPARSE_STATUS_INTERNAL_ERROR :
      std::cout << "cusparse: an internal operation failed."  << std::endl;
      break;
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED :
      std::cout << "cusparse: the matrix type is not supported."  << std::endl;
      break;
    default :
      std::cout << "cusparse: Unknown error."  << std::endl;
      break;
    }
    exit(1);
  }

  void init(){
    check_status(cusparseCreate(&handle));
  }
  
  void end(){
    check_status(cusparseDestroy(handle));
  }
  
  void csrmm2(cusparseOperation_t      transA,    
	      cusparseOperation_t      transB,    
	      int                      m,         
	      int                      n,         
	      int                      k,         
	      int                      nnz,       
	      const float              *alpha,    
	      const cusparseMatDescr_t descrA, 
	      const float              *csrValA, 
	      const int                *csrRowPtrA, 
	      const int                *csrColIndA,
	      const float              *B,
	      int                      ldb,
	      const float              *beta,
	      float                    *C,
	      int                      ldc){
    
    check_status(cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
  }
  
  
  
  void csrmm2(cusparseOperation_t      transA, 
	      cusparseOperation_t      transB,
	      int                      m,
	      int                      n,
	      int                      k,
	      int                      nnz, 
	      const double             *alpha, 
	      const cusparseMatDescr_t descrA, 
	      const double             *csrValA, 
	      const int                *csrRowPtrA,
	      const int                *csrColIndA,
	      const double             *B,
	      int                      ldb,
	      const double             *beta,
	      double                   *C,
	      int                      ldc){
    
    check_status(cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));
  }

  void csrmm2(cusparseOperation_t       transA, 
	      cusparseOperation_t       transB,
	      int                       m,
	      int                       n,
	      int                       k,
	      int                       nnz, 
	      const std::complex<float> *alpha, 
	      const cusparseMatDescr_t  descrA, 
	      const std::complex<float> *csrValA, 
	      const int                 *csrRowPtrA,
	      const int                 *csrColIndA,
	      const std::complex<float> *B,
	      int                       ldb,
	      const std::complex<float> *beta,
	      std::complex<float>       *C,
	      int                       ldc){
   
    check_status(cusparseCcsrmm2(handle, transA, transB, m, n, k, nnz, (const cuComplex*) alpha, descrA, (const cuComplex*) csrValA, csrRowPtrA, csrColIndA, (const cuComplex*) B, ldb, (const cuComplex*) beta, (cuComplex*) C, ldc));
  }
  
  void csrmm2(cusparseOperation_t        transA, 
	      cusparseOperation_t        transB,
	      int                        m,
	      int                        n,
	      int                        k,
	      int                        nnz, 
	      const std::complex<double> *alpha, 
	      const cusparseMatDescr_t   descrA, 
	      const std::complex<double> *csrValA, 
	      const int                  *csrRowPtrA,
	      const int                  *csrColIndA,
	      const std::complex<double> *B,
	      int                        ldb,
	      const std::complex<double> *beta,
	      std::complex<double>       *C,
	      int                        ldc){

    check_status(cusparseZcsrmm2(handle, transA, transB, m, n, k, nnz, (const cuDoubleComplex*) alpha, descrA, (const cuDoubleComplex*) csrValA, csrRowPtrA, csrColIndA, (const cuDoubleComplex*) B, ldb, (const cuDoubleComplex*) beta, (cuDoubleComplex*) C, ldc));
  }

  template<typename op_tag_type>
  cusparseOperation_t op_tag(){
    if(op_tag_type::value == 'N') return CUSPARSE_OPERATION_NON_TRANSPOSE;
    if(op_tag_type::value == 'T') return CUSPARSE_OPERATION_TRANSPOSE;
    if(op_tag_type::value == 'C') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    assert(false);
  }

  
}
#endif
