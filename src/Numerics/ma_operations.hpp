#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && clang++ -O3 -std=c++11 -Wfatal-errors -I.. -D_TEST_MA_OPERATIONS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif
#ifndef MA_OPERATIONS_HPP
#define MA_OPERATIONS_HPP

// File created and developed by:
// Alfredo Correa, correaa@llnl.gov
//    Lawrence Livermore National Laboratory

#include "ma_blas.hpp"

#include<type_traits> // enable_if
#include<vector>

namespace ma{

double const& conj(double const& d){return d;}
float const& conj(float const& f){return f;}

template<class MultiArray2D, typename = typename std::enable_if<(MultiArray2D::dimensionality > 1)>::type>
bool is_hermitian(MultiArray2D const& A){
	using std::conj;
	using ma::conj;
	if(A.shape()[0] != A.shape()[1]) return false;
	for(int i = 0; i != A.shape()[0]; ++i)
		for(int j = i + 1; j != A.shape()[1]; ++j)
			if( A[i][j] != conj(A[j][i]) ) return false;
	return true;
}

template<class MultiArray2D, typename = typename std::enable_if<(MultiArray2D::dimensionality > 1)>::type>
bool is_symmetric(MultiArray2D const& A){
	if(A.shape()[0] != A.shape()[1]) return false;
	for(int i = 0; i != A.shape()[0]; ++i)
		for(int j = i + 1; j != A.shape()[1]; ++j)
			if( A[i][j] != A[j][i] ) return false;
	return true;
}

template<class MultiArray2D, typename = typename std::enable_if<(std::decay<MultiArray2D>::type::dimensionality > 1)>::type>
MultiArray2D transpose(MultiArray2D&& A){
	assert(A.shape()[0] == A.shape()[1]);
	using std::swap;
	for(int i = 0; i != A.shape()[0]; ++i)
		for(int j = 0; j != i; ++j)
			swap(A[i][j], A[j][i]);
	return std::forward<MultiArray2D>(A);
}

template<class MA> struct op_tag : std::integral_constant<char, 'N'>{}; // see specializations
template<class MA> MA arg(MA&& ma){return std::forward<MA>(ma);} // see specializations below

template<class T, class MultiArray2DA, class MultiArray1DB, class MultiArray1DC,
	typename = typename std::enable_if<
		MultiArray2DA::dimensionality == 2 and 
		MultiArray1DB::dimensionality == 1 and
		std::decay<MultiArray1DC>::type::dimensionality == 1
	>::type, typename = void // TODO change to use dispatch
>
MultiArray1DC product(T alpha, MultiArray2DA const& A, MultiArray1DB const& B, T beta, MultiArray1DC&& C){
	return ma::gemv<
		(op_tag<MultiArray2DA>::value=='N')?'T':'N'
	>
	(alpha, arg(A), B, beta, std::forward<MultiArray1DC>(C));
}

template<class MultiArray2DA, class MultiArray1DB, class MultiArray1DC,
        typename = typename std::enable_if<
                MultiArray2DA::dimensionality == 2 and
                MultiArray1DB::dimensionality == 1 and
                std::decay<MultiArray1DC>::type::dimensionality == 1
        >::type, typename = void // TODO change to use dispatch 
>
MultiArray1DC product(MultiArray2DA const& A, MultiArray1DB const& B, MultiArray1DC&& C){
        return product(1., A, B, 0., std::forward<MultiArray1DC>(C));
}

template<class T, class MultiArray2DA, class MultiArray2DB, class MultiArray2DC,
	typename = typename std::enable_if<
		MultiArray2DA::dimensionality == 2 and 
		MultiArray2DB::dimensionality == 2 and
		std::decay<MultiArray2DC>::type::dimensionality == 2
	>::type
>
MultiArray2DC product(T alpha, MultiArray2DA const& A, MultiArray2DB const& B, T beta, MultiArray2DC&& C){
	return ma::gemm<
		op_tag<MultiArray2DB>::value,
		op_tag<MultiArray2DA>::value
	>
	(alpha, arg(B), arg(A), beta, std::forward<MultiArray2DC>(C));
}

template<class MultiArray2DA, class MultiArray2DB, class MultiArray2DC,
        typename = typename std::enable_if<
                MultiArray2DA::dimensionality == 2 and
                MultiArray2DB::dimensionality == 2 and
                std::decay<MultiArray2DC>::type::dimensionality == 2
        >::type
>
MultiArray2DC product(MultiArray2DA const& A, MultiArray2DB const& B, MultiArray2DC&& C){
	return product(1., A, B, 0., std::forward<MultiArray2DC>(C));
}

template<class MultiArray2D> struct normal_tag{
	MultiArray2D arg1;
	static auto const dimensionality = std::decay<MultiArray2D>::type::dimensionality;
	normal_tag(normal_tag const&) = delete;
	normal_tag(normal_tag&&) = default;
	static const char tag = 'N';
};

template<class MultiArray2D> struct op_tag<normal_tag<MultiArray2D>> : std::integral_constant<char, 'N'>{};

template<class MultiArray2D> normal_tag<MultiArray2D> normal(MultiArray2D&& arg){
	return {std::forward<MultiArray2D>(arg)};
}

template<class MultiArray2D>
MultiArray2D arg(normal_tag<MultiArray2D> const& nt){return nt.arg1;}

template<class MultiArray2D> struct transpose_tag{
	MultiArray2D arg1; 
	static auto const dimensionality = std::decay<MultiArray2D>::type::dimensionality;
	transpose_tag(transpose_tag const&) = delete;
	transpose_tag(transpose_tag&&) = default;
	static const char tag = 'T';
};

template<class MultiArray2D> struct op_tag<transpose_tag<MultiArray2D>> : std::integral_constant<char, 'T'>{};

template<class MultiArray2D> transpose_tag<MultiArray2D> transposed(MultiArray2D&& arg){
	return {std::forward<MultiArray2D>(arg)};
}

template<class MultiArray2D>
MultiArray2D arg(transpose_tag<MultiArray2D> const& tt){return tt.arg1;}

template<class MultiArray2D> struct hermitian_tag{
	MultiArray2D arg1; 
	static auto const dimensionality = std::decay<MultiArray2D>::type::dimensionality;
	hermitian_tag(hermitian_tag const&) = delete;
	hermitian_tag(hermitian_tag&&) = default;
	static const char tag = 'H';
};

template<class MultiArray2D> hermitian_tag<MultiArray2D> hermitian(MultiArray2D&& arg){
	return {std::forward<MultiArray2D>(arg)};
}

template<class MultiArray2D>
MultiArray2D arg(hermitian_tag<MultiArray2D> const& nt){return nt.arg1;}

template<class MultiArray2D> struct op_tag<hermitian_tag<MultiArray2D>> : std::integral_constant<char, 'H'>{};

template<class MultiArray2D>
MultiArray2D arg(hermitian_tag<MultiArray2D>& ht){return ht.arg1;}


template<class MA2D> auto T(MA2D&& arg)
->decltype(transposed(std::forward<MA2D>(arg))){
	return transposed(std::forward<MA2D>(arg));
}
template<class MA2D> auto H(MA2D&& arg)
->decltype(hermitian(std::forward<MA2D>(arg))){
	return hermitian(std::forward<MA2D>(arg));
}
template<class MA2D> auto N(MA2D&& arg)
->decltype(normal(std::forward<MA2D>(arg))){
	return normal(std::forward<MA2D>(arg));
}


#if 0
// * MAM: I need to somehow overload resize(n) for multi_array<T,1>
template<class MultiArray2D>
MultiArray2D invert_lu(MultiArray2D&& m){
	assert(m.shape()[0] == m.shape()[1]);
	boost::multi_array<typename std::decay<MultiArray2D>::type::element, 1> pivot(boost::extents[m.shape()[0]]);
//	std::vector<int> pivot(m.shape()[0]);
	getrf(std::forward<MultiArray2D>(m), pivot);
	std::vector<typename std::decay<MultiArray2D>::type::element> work(m.shape()[0]);
	getri(std::forward<MultiArray2D>(m), pivot, work);
	return std::forward<MultiArray2D>(m);
}
#endif

template<class MultiArray2D, class MultiArray1D, class Vector, class T = typename std::decay<MultiArray2D>::type::element>
T invert(MultiArray2D&& m, MultiArray1D&& pivot, Vector&& WORK){
	assert(m.shape()[0] == m.shape()[1]);
	assert(pivot.size() >= m.shape()[0]);
	assert(WORK.size() >= m.shape()[0]);
	
	getrf(std::forward<MultiArray2D>(m), pivot);
	T detvalue(1.0);
	for(int i=0, ip=1, m_ = m.shape()[0]; i != m_; i++, ip++){
		if(pivot[i]==ip){
			detvalue *= +static_cast<T>(m[i][i]);
		}else{
			detvalue *= -static_cast<T>(m[i][i]);
		}
	}
	getri(std::forward<MultiArray2D>(m), pivot, WORK);
	return detvalue;
}

/*
 * MAM: not sure how I want to handle this
template<class MultiArray2D, class MultiArray1D, class MultiArray1DW>
void qr(MultiArray2D& A, MultiArray1D& tau, MultiArray1DW& work){
        geqrf(std::forward<MultiArray2D>(A), tau, work);
        gqr(std::forward<MultiArray2D>(A), tau, work);
}

template<class MultiArray2D, class MultiArray1D, class MultiArray1DW>
void lq(MultiArray2D& A, MultiArray1D& tau, MultiArray1DW& work){
        gelqf(std::forward<MultiArray2D>(A), tau, work);
        glq(std::forward<MultiArray2D>(A), tau, work);
}
*/

template<class MultiArray2D, class MultiArray1D, class T = typename std::decay<MultiArray2D>::type::element>
T determinant(MultiArray2D&& m, MultiArray1D&& pivot){
	assert(m.shape()[0] == m.shape()[1]);
	assert(pivot.size() >= m.shape()[0]);
        
	getrf(std::forward<MultiArray2D>(m), std::forward<MultiArray1D>(pivot));
	T detvalue(1.0);
	for(int i=0,ip=1,m_=m.shape()[0]; i<m_; i++, ip++){
		if(pivot[i]==ip){
			detvalue *= static_cast<T>(m[i][i]);
		}else{
			detvalue *= -static_cast<T>(m[i][i]);
		}
	}
	return detvalue;
}

template<class MultiArray2D>
MultiArray2D set_identity(MultiArray2D&& m){
	assert(m.shape()[0] == m.shape()[1]);
	for(int i = 0; i != m.shape()[0]; ++i)
		for(int j = 0; j != m.shape()[1]; ++j)
			m[i][j] = ((i==j)?1:0);
	return std::forward<MultiArray2D>(m);
}

template<class MultiArray2DA, class MultiArray2DB, class T>
bool equal(MultiArray2DB const& a, MultiArray2DA const& b, T tol = 0){
	if(a.shape()[0] != b.shape()[0] or a.shape()[0] != b.shape()[0]) return false; 
	using std::abs;
	for(int i = 0; i != a.shape()[0]; ++i)
		for(int j = 0; j != a.shape()[1]; ++j)
			if(abs(a[i][j] - b[i][j]) > tol) return false;
	return true;
}

} 

#if 0
namespace DenseMatrixOperators
{

inline bool isHermitian(int N, std::complex<double>* A, int LDA)
{
  for(int i=0; i<N; i++) 
   for(int j=i+1; j<N; j++) 
    if( A[i*LDA+j] != myconj(A[j*LDA+i]) )
     return false; 
  return true;
}

inline bool isHermitian(int N, double* A, int LDA)
{
  for(int i=0; i<N; i++)
   for(int j=i+1; j<N; j++)
    if( A[i*LDA+j] != A[j*LDA+i] )
     return false;
  return true;
}

inline bool isHermitian(Matrix<std::complex<double> >& A)
{
  if(A.rows() != A.cols()) return false;
  for(int i=0; i<A.rows(); i++)
   for(int j=i+1; j<A.cols(); j++)
    if( A(i,j) != myconj(A(j,i)) )
     return false;
  return true;
}

inline bool isSymmetric(int N, std::complex<double>* A, int LDA)
{
  for(int i=0; i<N; i++) 
   for(int j=i+1; j<N; j++) 
    if( A[i*LDA+j] != A[j*LDA+i] )
     return false; 
  return true;
}

inline bool isSymmetric(Matrix<std::complex<double> >& A)
{
  if(A.rows() != A.cols()) return false;
  for(int i=0; i<A.rows(); i++)
   for(int j=i+1; j<A.cols(); j++)
    if( A(i,j) != A(j,i) )
     return false;
  return true;
}

template<typename T>
inline void transpose(int N, T* A, int LDA ) {
  for (int i=0; i<N; i++)
    for (int j=0; j<i; j++)
      std::swap(A[i*LDA+j],A[j*LDA+i]);
}

bool exponentiateHermitianMatrix(int N, std::complex<double>* A, int LDA, std::complex<double>* expA, int LDEXPA); 

bool symEigenSysAll(int N, std::complex<double>* A, int LDA, double* eigVal, std::complex<double>* eigVec, int LDV); 
bool symEigenSysAll(int N, double* A, int LDA, double* eigVal, double* eigVec, int LDV); 

bool symEigenSysSelect(int N, double* A, int LDA, int neig, double* eigVal, bool getEigV, double* eigVec, int LDV); 
bool symEigenSysSelect(int N, std::complex<double>* A, int LDA, int neig, double* eigVal, bool getEigV, std::complex<double>* eigVec, int LDV); 

bool genHermitianEigenSysSelect(int N, std::complex<double>* A, int LDA, std::complex<double>* B, int LDB, int neig, double* eigVal, bool getEigV, std::complex<double>* eigVec, int LDV, int* ifail);

template<typename T>
inline void product(const int M, const int N, const int K, const T one, const T* A, const int LDA, const T* B, const int LDB, const T zero, T* C, const int LDC )
{
  const char transa = 'N';
  const char transb = 'N';

  // C = A*B -> fortran -> C' = B'*A', 
  BLAS::gemm(transa,transb, N, M, K,
          one, B, LDB, A, LDA,
          zero, C, LDC);  
} 

template<typename T>
inline void product(const int M, const int N, const int K, const T one, const T* A, const int LDA, const std::complex<T>* B, const int LDB, const T zero, std::complex<T>* C, const int LDC )
{
  const char transa = 'N';
  const char transb = 'N';
  const T* B_ = reinterpret_cast<T*>(const_cast<std::complex<T>*>(B));
  T* C_ = reinterpret_cast<T*>(C);
  // C = A*B -> fortran -> C' = B'*A', 
  BLAS::gemm(transa,transb, 2*N, M, K,
            one, B_, 2*LDB, A, LDA,
            zero, C_, 2*LDC);
}

template<typename T>
inline void product_AtB(const int M, const int N, const int K, const T one, const T* A, const int LDA, const std::complex<T>* B, const int LDB, const T zero, std::complex<T>* C, const int LDC )
{
  const char transa = 'N';
  const char transb = 'T';
  const T* B_ = reinterpret_cast<T*>(const_cast<std::complex<T>*>(B));
  T* C_ = reinterpret_cast<T*>(C);
  // C = A'*B -> fortran -> C' = B'*A, 
  BLAS::gemm(transa,transb, 2*N, M, K,
            one, B_, 2*LDB, A, LDA,
            zero, C_, 2*LDC);
} 

template<typename T>
inline void product_AtB(const int M, const int N, const int K, const T one, const T* A, const int LDA, const T* B, const int LDB, const T zero, T* C, const int LDC )
{
  const char transa = 'N';
  const char transb = 'T';

  // C = A'*B -> fortran -> C' = B'*A, 
  BLAS::gemm(transa,transb, N, M, K,
            one, B, LDB, A, LDA,
            zero, C, LDC);
}

template<typename T>
inline void product_AtBt(const int M, const int N, const int K, const T one, const T* A, const int LDA, const T* B, const int LDB, const T zero, T* C, const int LDC )
{
  const char transa = 'T';
  const char transb = 'T';

  // C = A'*B' -> fortran -> C' = B*A, 
  BLAS::gemm(transa,transb, N, M, K,
            one, B, LDB, A, LDA,
            zero, C, LDC);
}

template<typename T>
inline void product_ABt(const int M, const int N, const int K, const T one, const T* A, const int LDA, const T* B, const int LDB, const T zero, T* C, const int LDC )
{
  const char transa = 'T';
  const char transb = 'N';

  // C = A*B' -> fortran -> C' = B*A', 
     BLAS::gemm(transa,transb, N, M, K,
            one, B, LDB, A, LDA,
            zero, C, LDC);
  
}

template<typename T>
inline void product_AhB(const int M, const int N, const int K, const T one, const T* A, const int LDA, const T* B, const int LDB, const T zero, T* C, const int LDC )
{
  const char transa = 'N';
  const char transb = 'C';

  // C = A'*B -> fortran -> C' = B'*A, 
     BLAS::gemm(transa,transb, N, M, K,
            one, B, LDB, A, LDA,
            zero, C, LDC);
  
}

template<typename T>
inline void product_ABh(const int M, const int N, const int K, const T one, const T* A, const int LDA, const T* B, const int LDB, const T zero, T* C, const int LDC )
{
  const char transa = 'C';
  const char transb = 'N';

  // C = A*B^H -> fortran -> C' = conjg(B)*A', 
  BLAS::gemm(transa,transb, N, M, K,
          one, B, LDB, A, LDA,
          zero, C, LDC);

}

template<typename T>
inline void product_Ax(const int N, const int K, const T one, const T* A, const int lda, const T* x, const T zero, T* restrict yptr)
{
  const char transa = 'T';
  BLAS::gemv(transa, K, N, one, A, lda, x, 1, zero, yptr, 1);
}

template<typename T>
inline void product_Ax(const int M, const int K, const T one, const T* A, const int lda, const std::complex<T>* x, const T zero, std::complex<T>* restrict yptr)
{
// add specialized routine if it exists 
#if defined(HAVE_MKL)
  const std::complex<T> one_ = one;
  const std::complex<T> zero_ = zero;
  const char transa = 'T';

  BLAS::gemv(transa, K, M, one_, A, lda, x, 1, zero_, yptr, 1);

#else 
  const T* x_ = reinterpret_cast<T*>(const_cast<std::complex<T>*>(x));
  T* yptr_ = reinterpret_cast<T*>(yptr);
  product(M,2,K,one,A,lda,x_,2,zero,yptr_,2);
#endif
}

template<typename T>
inline void product_Atx(const int N, const int K, const T one, const T* A, const int lda, const T* x, const T zero, T* restrict yptr)
{
  const char transa = 'N';
  BLAS::gemv(transa, K, N, one, A, lda, x, 1, zero, yptr, 1);
}

template<typename T>
inline void product_Atx(const int M, const int K, const T one, const T* A, const int lda, const std::complex<T>* x, const T zero, std::complex<T>* restrict yptr)
{
// add specialized routine if it exists 
#if defined(HAVE_MKL)
  const std::complex<T> one_ = one;
  const std::complex<T> zero_ = zero;
  const char transa = 'N';

  BLAS::gemv(transa, K, M, one_, A, lda, x, 1, zero_, yptr, 1);

#else
  const T* x_ = reinterpret_cast<T*>(const_cast<std::complex<T>*>(x));
  T* yptr_ = reinterpret_cast<T*>(yptr);
  product_AtB(K,2,M,one,A,lda,x_,2,zero,yptr_,2);
#endif
}
/*
inline std::complex<double> 
Determinant(std::complex<double>* restrict x, int n, int* restrict pivot)
{
  std::complex<double> detvalue(1.0);
  int status;
  zgetrf(n,n,x,n,pivot,status);
  for(int i=0,ip=1; i<n; i++, ip++)
  {
    if(pivot[i]==ip)
      detvalue *= x[i*n+i];
    else
      detvalue *= -x[i*n+i];
  }
  return detvalue;
}
*/


struct SmallDet
{

  template<typename T>
  inline static
  T D2x2(T a11, T a12,
               T a21, T a22)
  {
    return a11*a22-a21*a12;
  }

  template<typename T>
  inline static
  T D3x3(T a11, T a12, T a13,
        T a21, T a22, T a23,
        T a31, T a32, T a33)
  {
    return (a11*(a22*a33-a32*a23)-a21*(a12*a33-a32*a13)+a31*(a12*a23-a22*a13));
  }

  template<typename T>
  inline static
  T I3x3(T a11, T a12, T a13,
        T a21, T a22, T a23,
        T a31, T a32, T a33, T* M)
  {
    T det = (a11*(a22*a33-a32*a23)-a21*(a12*a33-a32*a13)+a31*(a12*a23-a22*a13));
    M[0] = (a22*a33-a32*a23)/det; 
    M[1] = (a13*a32-a12*a33)/det;
    M[2] = (a12*a23-a13*a22)/det;
    M[3] = (a23*a31-a21*a33)/det;
    M[4] = (a11*a33-a13*a31)/det;
    M[5] = (a13*a21-a11*a23)/det;
    M[6] = (a21*a32-a22*a31)/det;
    M[7] = (a12*a31-a11*a32)/det;
    M[8] = (a11*a22-a12*a21)/det;
    return det; 
  }

  template<typename T>
  inline static
  T D4x4(T a11, T a12, T a13, T a14,
        T a21, T a22, T a23, T a24,
        T a31, T a32, T a33, T a34,
        T a41, T a42, T a43, T a44)
  {
    return (a11*(a22*(a33*a44-a43*a34)-a32*(a23*a44-a43*a24)+a42*(a23*a34-a33*a24))-a21*(a12*(a33*a44-a43*a34)-a32*(a13*a44-a43*a14)+a42*(a13*a34-a33*a14))+a31*(a12*(a23*a44-a43*a24)-a22*(a13*a44-a43*a14)+a42*(a13*a24-a23*a14))-a41*(a12*(a23*a34-a33*a24)-a22*(a13*a34-a33*a14)+a32*(a13*a24-a23*a14)));
  }

  template<typename T>
  inline static
  T D5x5(T a11, T a12, T a13, T a14, T a15,
        T a21, T a22, T a23, T a24, T a25,
        T a31, T a32, T a33, T a34, T a35,
        T a41, T a42, T a43, T a44, T a45,
        T a51, T a52, T a53, T a54, T a55)
  {
    return (a11*(a22*(a33*(a44*a55-a54*a45)-a43*(a34*a55-a54*a35)+a53*(a34*a45-a44*a35))-a32*(a23*(a44*a55-a54*a45)-a43*(a24*a55-a54*a25)+a53*(a24*a45-a44*a25))+a42*(a23*(a34*a55-a54*a35)-a33*(a24*a55-a54*a25)+a53*(a24*a35-a34*a25))-a52*(a23*(a34*a45-a44*a35)-a33*(a24*a45-a44*a25)+a43*(a24*a35-a34*a25)))-a21*(a12*(a33*(a44*a55-a54*a45)-a43*(a34*a55-a54*a35)+a53*(a34*a45-a44*a35))-a32*(a13*(a44*a55-a54*a45)-a43*(a14*a55-a54*a15)+a53*(a14*a45-a44*a15))+a42*(a13*(a34*a55-a54*a35)-a33*(a14*a55-a54*a15)+a53*(a14*a35-a34*a15))-a52*(a13*(a34*a45-a44*a35)-a33*(a14*a45-a44*a15)+a43*(a14*a35-a34*a15)))+a31*(a12*(a23*(a44*a55-a54*a45)-a43*(a24*a55-a54*a25)+a53*(a24*a45-a44*a25))-a22*(a13*(a44*a55-a54*a45)-a43*(a14*a55-a54*a15)+a53*(a14*a45-a44*a15))+a42*(a13*(a24*a55-a54*a25)-a23*(a14*a55-a54*a15)+a53*(a14*a25-a24*a15))-a52*(a13*(a24*a45-a44*a25)-a23*(a14*a45-a44*a15)+a43*(a14*a25-a24*a15)))-a41*(a12*(a23*(a34*a55-a54*a35)-a33*(a24*a55-a54*a25)+a53*(a24*a35-a34*a25))-a22*(a13*(a34*a55-a54*a35)-a33*(a14*a55-a54*a15)+a53*(a14*a35-a34*a15))+a32*(a13*(a24*a55-a54*a25)-a23*(a14*a55-a54*a15)+a53*(a14*a25-a24*a15))-a52*(a13*(a24*a35-a34*a25)-a23*(a14*a35-a34*a15)+a33*(a14*a25-a24*a15)))+a51*(a12*(a23*(a34*a45-a44*a35)-a33*(a24*a45-a44*a25)+a43*(a24*a35-a34*a25))-a22*(a13*(a34*a45-a44*a35)-a33*(a14*a45-a44*a15)+a43*(a14*a35-a34*a15))+a32*(a13*(a24*a45-a44*a25)-a23*(a14*a45-a44*a15)+a43*(a14*a25-a24*a15))-a42*(a13*(a24*a35-a34*a25)-a23*(a14*a35-a34*a15)+a33*(a14*a25-a24*a15))));
  }
};

template<typename T>
inline T DeterminantSmall(T* M, int n, int* restrict pivot)
{
  switch(n)
  {
  case 0:
    return 1.0;
  case 1:
    return M[0]; 
    break;
  case 2:
  {
    return M[0]*M[3] - M[1]*M[2]; 
    break;
  }
  case 3:
  {
    return SmallDet::D3x3(M[0], M[1], M[2],
                        M[3], M[4], M[5],  
                        M[6], M[7], M[8]); 
    break;
  }
  case 4:
  {
    return SmallDet::D4x4(M[0], M[1], M[2], M[3],
                        M[4], M[5], M[6], M[7],
                        M[8], M[9], M[10], M[11], 
                        M[12], M[13], M[14], M[15]); 
    break;
    break;
  }
  case 5:
  {
    return SmallDet::D5x5(M[0], M[1], M[2], M[3], M[4],
                        M[5], M[6], M[7], M[8], M[9], 
                        M[10], M[11], M[12], M[13], M[14], 
                        M[15], M[16], M[17], M[18], M[19], 
                        M[20], M[21], M[22], M[23], M[24]); 
    break;
    break;
  }
  default:
  {
    return Determinant(M,n,n,pivot); 
  }
  }
  return 0.0;
}

template<typename T>
inline T InvertSmall(T* M, int n, T* restrict work, int* restrict pivot)
{
  switch(n)
  {
  case 0:
  {
    return T(1);
    break; 
  }
  case 1:
  {
    T det = M[0];
    M[0] = T(1)/det;
    return det;
    break;
  }
  case 2:
  {
    T one = T(1);
    T det = M[0]*M[3] - M[1]*M[2]; 
    std::swap(M[0],M[3]);  
    M[0]*=one/det;  
    M[3]*=one/det;  
    M[1]*=-one/det;  
    M[2]*=-one/det;  
    return det;
    break;
  }
  case 3:
  {
    return SmallDet::I3x3(M[0], M[1], M[2],
                        M[3], M[4], M[5],
                        M[6], M[7], M[8],M);
    break;
  }
  }
  return Invert(M,n,n,work,pivot);
}

void GeneralizedGramSchmidt(std::complex<double>* A, int LDA, int nR, int nC);

} // namespace DenseMatrixOperators

} // namespace qmcplusplus

}
#endif

#ifdef _TEST_MA_OPERATIONS

#include<boost/multi_array.hpp>

#include<vector>
#include<iostream>

using std::cout;

int main(){

	{
		std::vector<double> m = {
			9.,24.,30.,
			4.,10.,12.,
			14.,16.,36.//,
		//	9., 6., 1.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][3]);
		assert(M.num_elements() == m.size());
		std::vector<double> x = {1.,2.,3.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y(3);
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);

		using ma::T;
		ma::product(M, X, Y); // Y := M X
		
		std::vector<double> mx = {147., 60.,154.};
		boost::multi_array_ref<double, 1> MX(mx.data(), boost::extents[mx.size()]);
		assert( MX == Y );
	}
	{
		std::vector<double> m = {
			9.,24.,30., 2.,
			4.,10.,12., 1.,
			14.,16.,36., 20.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][4]);
		assert(M.num_elements() == m.size());
		std::vector<double> x = {1.,2.,3., 4.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y(3);
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);

		using ma::T;
		ma::product(M, X, Y); // Y := M X

		std::vector<double> mx = {155., 64.,234.};
		boost::multi_array_ref<double, 1> MX(mx.data(), boost::extents[mx.size()]);
		assert( MX == Y );
	}
	{
		std::vector<double> m = {
			9.,24.,30., 2.,
			4.,10.,12., 1.,
			14.,16.,36., 20.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][4]);
		assert(M.num_elements() == m.size());
		std::vector<double> x = {1.,2.,3.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y(4);
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);

		using ma::T;
		ma::product(T(M), X, Y); // Y := T(M) X
		
		std::vector<double> mx = {59., 92., 162., 64.};
		boost::multi_array_ref<double, 1> MX(mx.data(), boost::extents[mx.size()]);
		assert( MX == Y );
	}
	{
		std::vector<double> m = {
			9.,24.,30., 9.,
			4.,10.,12., 7.,
			14.,16.,36., 1.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][4]);
		std::vector<double> x = {1.,2.,3., 4.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y = {4.,5.,6.};
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);
		ma::product(M, X, Y); // y := M x
		
		std::vector<double> y2 = {183., 88.,158.};
		boost::multi_array_ref<double, 1> Y2(y2.data(), boost::extents[y2.size()]);
		assert( Y == Y2 );
	}

	{
	std::vector<double> m = {
		1.,2.,1.,
		2.,5.,8.,
		1.,8.,9.
	};
	boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][3]);
	assert( ma::is_hermitian(M) );
	}{
	std::vector<double> m = {
		1.,0.  , 2.,0. ,  1.,0.,
		2.,0.  , 5.,0. ,  8.,-1.,
		1.,0.  , 8.,1. ,  9.,0.,
	};
	boost::multi_array_ref<std::complex<double>, 2> M(reinterpret_cast<std::complex<double>*>(m.data()), boost::extents[3][3]);
	assert( ma::is_hermitian(M) );
	}{
	std::vector<double> m = {
		1.,2.,1.,
		2.,5.,8.,
		1.,8.,9.
	};
	boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][3]);
	assert( ma::is_hermitian(M) );
	}
	{
	std::vector<double> a = {
		1.,0.,1.,
		3.,5.,8., 
		4.,8.,9.
	};
	boost::multi_array_ref<double, 2> A(a.data(), boost::extents[3][3]);
	assert( A.num_elements() == a.size() );
	std::vector<double> b = {
		6.,2.,8.,
		9.,5.,5.,
		1.,7.,9.
	};
	boost::multi_array_ref<double, 2> B(b.data(), boost::extents[3][3]);
	assert( B.num_elements() == b.size() );

	std::vector<double> c(9);
	boost::multi_array_ref<double, 2> C(c.data(), boost::extents[3][3]);
	assert( C.num_elements() == c.size() );
	
	ma::product(A, B, C);

	std::vector<double> ab = {
		7., 9., 17.,
		71., 87., 121.,
		105., 111., 153.
	};
	boost::multi_array_ref<double, 2> AB(ab.data(), boost::extents[3][3]);
	assert( AB.num_elements() == ab.size() );

	for(int i = 0; i != C.shape()[0]; ++i, cout << '\n')
		for(int j = 0; j != C.shape()[1]; ++j)
			cout << C[i][j] << ' ';
	cout << '\n';

	assert(C == AB);


	using ma::N;
	ma::product(N(A), N(B), C); // same as ma::product(A, B, C);
	assert(C == AB);

	using ma::T;
	
	ma::product(T(A), B, C);
	std::vector<double> atb = {37., 45., 59., 53., 81., 97., 87., 105., 129.};
	boost::multi_array_ref<double, 2> AtB(atb.data(), boost::extents[3][3]);
	assert(C == AtB);
	
	ma::product(A, T(B), C);
	std::vector<double> abt = {14., 14., 10., 92., 92., 110., 112., 121., 141.};
	boost::multi_array_ref<double, 2> ABt(abt.data(), boost::extents[3][3]);
	assert(C == ABt);

	ma::product(T(A), T(B), C);
	std::vector<double> atbt = {44., 44., 58., 74., 65., 107., 94., 94., 138.};
	boost::multi_array_ref<double, 2> AtBt(atbt.data(), boost::extents[3][3]);
	assert(C == AtBt);

	
	}
	#if 0
	{
		std::vector<double> a = {37., 45., 59., 53., 81., 97., 87., 105., 129.};
		boost::multi_array_ref<double, 2> A(a.data(), boost::extents[3][3]);
		assert(A.num_elements() == a.size());
		boost::multi_array<double, 2> B = A;
		ma::invert_lu(A);

		boost::multi_array<double, 2> Id(boost::extents[3][3]);
		ma::set_identity(Id);

		boost::multi_array<double, 2> Id2(boost::extents[3][3]);
		ma::product(A, B, Id2);
						
		assert( ma::equal(Id, Id2, 1e-14) );
	}
	#endif
	cout << "test ended" << std::endl;
}
#endif

#endif


