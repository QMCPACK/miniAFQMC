#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && clang++ -O3 -std=c++14 -Wfatal-errors -I.. -D_TEST_UBLAS_ADAPTORS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -Wl,-rpath=/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -lmkl_core -lmkl_intel_ilp64 -lmkl_sequential -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif

//////////////////////////////////////////////////////////////////////
// File developed by:
// Alfredo Correa, correaa@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Alfredo Correa, correaa@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////


#ifndef UBLAS_ADAPTORS_HPP
#define UBLAS_ADAPTORS_HPP

#include<array>

namespace sblas{

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
auto non_zero_values_data(UBlas&& m)
->decltype(m.value_data().begin()){
	return m.value_data().begin();
}
template<class UBlas>
auto non_zero_indices2_data(UBlas&& m)
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

}

#ifdef _TEST_UBLAS_ADAPTORS

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

#include<boost/multi_array.hpp>

#include "sparse_blas.hpp"

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

