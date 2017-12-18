#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && clang++ -O3 -std=c++14 -Wfatal-errors -I.. -D_TEST_SPARSE_BLAS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -Wl,-rpath=/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -lmkl_core -lmkl_intel_ilp64 -lmkl_sequential -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif

#ifndef SPARSE_BLAS_HPP

#include "mkl.hpp"
#include<cassert>

namespace sblas{

template<class M> auto shape(M&& m) -> decltype(m.shape()){return m.shape();}

template<
	char TA, class T, class CSRArray, class MultiArray2DB, class MultiArray2DC
>
MultiArray2DC csrmm(
	T const& alpha, 
	CSRArray const& A, 
	MultiArray2DB const& B, 
	T const& beta, 
	MultiArray2DC&& C
){
	assert(B.strides()[1]==1);
	assert(C.strides()[1]==1);
	std::string code = "Gxx?";
	if(index_bases(A)[0] == 0 and index_bases(A)[1] == 0) code[3] = 'C';
	else if(index_bases(A)[0] == 1 and index_bases(A)[1] == 1) code[3] = 'F';
	else assert(0); // base not supported

	mkl::csrmm(
		TA, shape(A)[0], shape(C)[1], shape(A)[1], 
		alpha, code.c_str(), 
		non_zero_values_data(A), non_zero_indices2_data(A), 
		pointers_begin(A), pointers_end(A),
		B.origin(), B.strides()[0],
		beta,  
		C.origin(), C.strides()[0]
	);
	return std::forward<MultiArray2DC>(C);
}

template<char TA, class CSRArray, class MultiArray2DB, class MultiArray2DC>
MultiArray2DC csrmm(CSRArray const& A, MultiArray2DB const& B, MultiArray2DC&& C){
	return csrmm<TA>(1.0, A, B, 0.0, std::forward<MultiArray2DC>(C));
}

}

#ifdef _TEST_SPARSE_BLAS

#include "../Matrix/sparse/csr_matrix.hpp"
#include<boost/multi_array.hpp>

#include<iostream>

using std::cout;

int main(){

	using ma::sparse::csr_matrix;
	using ma::sparse::coo_matrix;

	csr_matrix<double const> A = coo_matrix<double>(
		{4,4}, 
		{
			{{3,3}, 1}, 
			{{2,1}, 3}, 
			{{0,1}, 9}
		}
	);
	
	std::vector<double> b = {
		1, 2, 1, 5,
		2, 5, 8, 7,
		1, 8, 9, 9,
		4, 1, 2, 3
	};
	boost::multi_array_ref<double, 2> B(b.data(), boost::extents[4][4]);
	assert(B.num_elements() == b.size());

	std::vector<double> c(16);
	boost::multi_array_ref<double, 2> C(c.data(), boost::extents[4][4]);
	assert(C.num_elements() == c.size());

	sblas::csrmm<'N'>(A, B, C); // C = A*B

	std::vector<double> c2 = {
		18, 45, 72, 63,
		 0,  0,  0,  0, 
		 6, 15, 24, 21, 
		 4,  1,  2,  3
	};
	boost::multi_array_ref<double, 2> C2(c2.data(), boost::extents[4][4]);
	assert(C2.num_elements() == c2.size());
	assert(C == C2);

	std::vector<double> d(16);
	boost::multi_array_ref<double, 2> D(d.data(), boost::extents[4][4]);
	assert(D.num_elements() == d.size());

	sblas::csrmm<'T'>(A, B, D); // D = T(A)*B
	std::vector<double> d2 = {
		 0,  0,  0,  0, 
		12, 42, 36, 72, 
		 0,  0,  0,  0, 
		 4,  1,  2,  3 
	};
	boost::multi_array_ref<double, 2> D2(d2.data(), boost::extents[4][4]);
	assert(D2.num_elements() == d2.size());

	assert(D2 == D);

}

#endif
#endif



