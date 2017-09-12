#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && clang++ -O3 -std=c++14 -Wfatal-errors -I.. -D_TEST_SP_BLAS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -Wl,-rpath=/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -lmkl_core -lmkl_intel_ilp64 -lmkl_sequential -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif

// File created and developed by:
// Alfredo Correa, correaa@llnl.gov
//    Lawrence Livermore National Laboratory

#ifndef MA_BLAS_HPP
#define MA_BLAS_HPP

#define HAVE_MKL
#define MKL_INT long int

#include "Numerics/spblas.hpp"
#include<utility> //std::enable_if
#include<cassert>
#include<iostream>
#include<array>

namespace sp{

void csrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const float &alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT &ldb, const float &beta, float *c, const MKL_INT &ldc){
	mkl_scsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc);
}
void csrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const std::complex<float> &alpha, const char *matdescra, const std::complex<float> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const std::complex<float> *b, const MKL_INT &ldb, const std::complex<float> &beta, std::complex<float> *c, const MKL_INT &ldc){
	mkl_ccsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc);
}
void csrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const double &alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT &ldb, const double &beta, double *c, const MKL_INT &ldc){
	mkl_dcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc);
}
void csrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const std::complex<double> &alpha, const char *matdescra, const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const std::complex<double> *b, const MKL_INT &ldb, const std::complex<double> &beta, std::complex<double> *c, const MKL_INT &ldc){
	mkl_zcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc);
}

template<class UBlas>
auto num_non_zero_elements(UBlas&& m)
->decltype(m.nnz()){
	return m.nnz();
}

template<class UBlas>
auto shape(UBlas&& m)
->decltype(std::array<int, 2>{int(m.size1()), int(m.size2())} ){
	return std::array<int, 2>{int(m.size1()), int(m.size2())} ;
}

template<class UBlas>
auto non_zero_values(UBlas&& m)
->decltype(m.value_data().begin()){
	return m.value_data().begin();
}
template<class UBlas>
auto non_zero_indices2(UBlas&& m)
->decltype(reinterpret_cast<const long*>(m.index2_data().begin())){
	return reinterpret_cast<const long*>(m.index2_data().begin());
}
template<class UBlas>
auto pointers_begin(UBlas&& m)
->decltype(reinterpret_cast<const long*>(m.index1_data().begin())){
	return reinterpret_cast<const long*>(m.index1_data().begin());
}
template<class UBlas>
auto pointers_end(UBlas&& m)
->decltype(reinterpret_cast<const long*>(m.index1_data().begin() + 1)){
	return reinterpret_cast<const long*>(m.index1_data().begin() + 1);
}

template<class UBlas>
auto index_bases(UBlas&& m)
->decltype(std::array<int, 2>{int(m.index_base()), int(m.index_base())} ){
	return std::array<int, 2>{int(m.index_base()), int(m.index_base())} ;
}

template<class T>
auto num_non_zero_elements(T&& m)
->decltype(m.num_non_zero_elements()){
	return m.num_non_zero_elements();
}
template<class T>
auto shape(T&& m)
->decltype(m.shape()){
	return m.shape();
}
template<class T>
auto num_non_zero_values(T&& m)
->decltype(m.num_non_zero_values()){
	return m.num_non_zero_values();
}

template<char TA, class T, class CSRArray, class MultiArray2DB, class MultiArray2DC>
MultiArray2DC csrmm(
	T alpha, CSRArray const& A, MultiArray2DB const& B, T beta, MultiArray2DC&& C
){
	assert(B.strides()[1]==1);
	assert(C.strides()[1]==1);
	std::string code = "Gxx?";
	if(index_bases(A)[0] == 0 and index_bases(A)[1] == 0) code[3] = 'C';
	else if(index_bases(A)[0] == 1 and index_bases(A)[1] == 1) code[3] = 'F';
	else assert(0); // base not supported

	csrmm(
		TA, shape(A)[0], shape(C)[1], shape(A)[1], 
		alpha, code.c_str(), non_zero_values(A), non_zero_indices2(A), pointers_begin(A), pointers_end(A),
		B.origin(), B.strides()[0],
		beta,  
		C.origin(), C.strides()[0]
	);
	return std::forward<MultiArray2DC>(C);
}

template<char TA, class CSRArray, class MultiArray2DB, class MultiArray2DC>
MultiArray2DC csrmm(CSRArray const& A, MultiArray2DB const& B, MultiArray2DC&& C){
	return csrmm<TA>(1.0, A, B, 0., std::forward<MultiArray2DC>(C));
}

}

#ifdef _TEST_SP_BLAS

#include<boost/multi_array.hpp>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "../Matrix/sparse_array.hpp"
#include<iostream>

using std::cout;

int main(){
//	ma::csr_matrix<double> A(boost::extents[4][4]);
//	A[3][3] = 1.;
//	A[2][1] = 3.;
//	A[0][1] = 9.;

	boost::numeric::ublas::compressed_matrix<double> A(4, 4);
	A(3,3) = 1.;
	A(2,1) = 3.;
	A(0,1) = 9.;

//	cout << "Aval = ";	
//	for(int i = 0; i != A.num_non_zero_elements(); ++i) cout << A.non_zero_values()[i] << " ";
	
	std::vector<double> b = {
		1.,2.,1., 5.,
		2.,5.,8., 7.,
		1.,8.,9., 9.,
		4.,1.,2., 3.
	};
	boost::multi_array_ref<double, 2> B(b.data(), boost::extents[4][4]);
	assert(B.num_elements() == b.size());

	std::vector<double> c(16);
	boost::multi_array_ref<double, 2> C(c.data(), boost::extents[4][4]);
	assert(C.num_elements() == c.size());

	sp::csrmm<'N'>(A, B, C); // C = A*B

	std::vector<double> c2 = {
		18., 45., 72., 63.,
		0., 0., 0., 0., 
		6., 15., 24., 21., 
		4., 1., 2., 3.
	};
	boost::multi_array_ref<double, 2> C2(c2.data(), boost::extents[4][4]);
	assert(C2.num_elements() == c2.size());
	assert(C == C2);

	std::vector<double> d(16);
	boost::multi_array_ref<double, 2> D(d.data(), boost::extents[4][4]);
	assert(D.num_elements() == d.size());

	sp::csrmm<'T'>(A, B, D); // D = T(A)*B
	std::vector<double> d2 = {
		0, 0, 0, 0, 
		12, 42, 36, 72, 
		0, 0, 0, 0, 
		4, 1, 2, 3 
	};
	boost::multi_array_ref<double, 2> D2(d2.data(), boost::extents[4][4]);
	assert(D2.num_elements() == d2.size());

	assert(D2 == D);

}

#endif
#endif

