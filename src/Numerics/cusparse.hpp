#ifndef CUSPARSE_HPP
#define CUSPARSE_HPP

#include <cusparse.h>

namespace cusparse {

  cusparseHandle_t handle = NULL;

  void init(){
    cusparseStatus_t cusparseStat = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
  }

  void end(){
    cusparseDestroy(handle);
  }
  
  cusparseStatus_t csrmm2(cusparseOperation_t      transA,    
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
    
    return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
  }
  
  
  
  cusparseStatus_t csrmm2(cusparseOperation_t      transA, 
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
  
    return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
  }

  cusparseStatus_t csrmm2(cusparseOperation_t       transA, 
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

    return cusparseCcsrmm2(handle, transA, transB, m, n, k, nnz, (const cuComplex*) alpha, descrA, (const cuComplex*) csrValA, csrRowPtrA, csrColIndA, (const cuComplex*) B, ldb, (const cuComplex*) beta, (cuComplex*) C, ldc);
  }
  
  cusparseStatus_t csrmm2(cusparseOperation_t        transA, 
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
    
    return cusparseZcsrmm2(handle, transA, transB, m, n, k, nnz, (const cuDoubleComplex*) alpha, descrA, (const cuDoubleComplex*) csrValA, csrRowPtrA, csrColIndA, (const cuDoubleComplex*) B, ldb, (const cuDoubleComplex*) beta, (cuDoubleComplex*) C, ldc);
  }

}
#endif
