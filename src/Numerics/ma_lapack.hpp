#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && c++ -O3 -std=c++11 -Wfatal-errors -I.. -D_TEST_MA_LAPACK -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif

// File created and developed by:
// Alfredo Correa, correaa@llnl.gov
//    Lawrence Livermore National Laboratory

#ifndef MA_LAPACK_HPP
#define MA_LAPACK_HPP

#include "Numerics/OhmmsBlas.h"
#include<cassert>

namespace ma{

template<class MultiArray2D, class Array1D>
MultiArray2D getrf(MultiArray2D&& m, Array1D& pivot){
	assert(m.strides()[0] >= std::max(std::size_t(1), m.shape()[1]));
	assert(m.strides()[1] == 1);
	assert(pivot.size() >= std::min(m.shape()[1], m.shape()[0]));
	
	int status = -1;
	BLAS::getrf(
		m.shape()[1], m.shape()[0], m.origin(), m.strides()[0], 
		pivot.data(), 
		status
	);
	assert(status==0);
	return std::forward<MultiArray2D>(m);
}

template<class MultiArray2D>
int getri_optimal_workspace_size(MultiArray2D const& A){
	typename MultiArray2D::element WORK;
	int status = -1;
	BLAS::getri(
		A.shape()[0], nullptr, A.strides()[0], 
		nullptr, 
		&WORK, /*lwork*/ -1, 
		status
	);
	assert(status == 0);
	return WORK;
}

template<class MultiArray2D, class MultiArray1D, class Buffer>
MultiArray2D getri(MultiArray2D&& A, MultiArray1D const& IPIV, Buffer&& WORK){
//	assert(A.strides()[0] > std::max(std::size_t(1), A.shape()[1]));
	assert(A.strides()[1] == 1);
	assert(IPIV.size() == A.shape()[0]);
	assert(WORK.capacity() >= std::max(std::size_t(1), A.shape()[0]));
	
//	WORK.reserve(getri_optimal_workspace_size(A));
	int status = -1;
	BLAS::getri(
		A.shape()[0], A.origin(), A.strides()[0], 
		IPIV.data(), 
		WORK.data(), WORK.capacity(), 
		status
	);
	assert(status == 0);
	return std::forward<MultiArray2D>(A);
}

template<class MultiArray2D, class Array1D>
int geqrf_optimal_workspace_size(MultiArray2D const& A){
	assert(A.strides()[0] > 0);
	assert(A.strides()[1] == 1);

	typename MultiArray2D::element WORK;
	int status = -1;
	BLAS::geqrf(
		A.shape()[1], A.shape()[0], nullptr, A.strides()[0], 
		nullptr, 
		&WORK, /*lwork*/ -1, 
		status
	);
	assert(status==0);
	return real(WORK);
}

template<class MultiArray2D, class Array1D, class Buffer>
MultiArray2D geqrf(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK){
	assert(A.strides()[0] > std::max(std::size_t(1), A.shape()[0]));
	assert(A.strides()[1] == 1);
	assert(TAU.strides()[0] == 1);
	assert(TAU.size() >= std::max(1, std::min(A.shape()[0], A.shape()[1])));
	assert(WORK.capacity() >= std::max(1, A.shape()[1]));
	
//	WORK.reserve(gelqf_optimal_workspace_size(A));
	int status = -1;
	BLAS::geqrf(
		A.shape()[1], A.shape()[0], A.origin(), A.strides()[0], 
		TAU.data(), 
		WORK.data(), WORK.capacity(),
		status
	);
	assert(status==0);
	return std::forward<MultiArray2D>(A);
}

template<class MultiArray2D>
int gelqf_optimal_workspace_size(MultiArray2D const& A){
	assert(A.strides()[0] > 0);
	assert(A.strides()[1] == 1);

	typename MultiArray2D::element WORK;
	int status = -1;
	BLAS::gelqf(
		A.shape()[1], A.shape()[0], nullptr, A.strides()[0], 
		nullptr, 
		&WORK, /*lwork*/ -1, 
		status
	);
	assert(status==0);
	return real(WORK);
}

template<class MultiArray2D, class Array1D, class Buffer>
MultiArray2D gelqf(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK){
	assert(A.strides()[1] > 0);
	assert(A.strides()[1] == 1);
	assert(TAU.strides()[0] == 1);
	assert(TAU.size() >= std::max(1, std::min(A.shape()[0], A.shape()[1])));
	assert(WORK.capacity() > 0);

	int status = -1;
//	WORK.reserve(gelqf_optimal_workspace_size(A));
	BLAS::gelqf(
		A.shape()[1], A.shape()[0], A.origin(), A.strides()[0], TAU.data(),
		WORK.data(), WORK.capacity(), 
		status
	);
	assert(status==0);
	return std::forward<MultiArray2D>(A);
}


template<class MultiArray2D, class Array1D>
int gqr_optimal_workspace_size(MultiArray2D const& A){
	assert(A.strides()[0] > 0);
	assert(A.strides()[1] == 1);

	typename MultiArray2D::element WORK;
	int status = -1;
	BLAS::gqr(
		A.shape()[1], A.shape()[0], nullptr, A.strides()[0], 
		nullptr, 
		&WORK, /*lwork*/ -1, 
		status
	);
	assert(status==0);
	return real(WORK);
}

template<class MultiArray2D, class Array1D, class Buffer>
MultiArray2D gqr(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK){
	assert(A.strides()[1] == 1);
	assert(TAU.strides()[0] == 1);
	assert(TAU.size() >= std::max(1, std::min(A.shape()[0], A.shape()[1])));
	assert(WORK.capacity() > 0);

	WORK.reserve(gqr_optimal_workspace_size(A));
	int status = -1;
	BLAS::gqr(
		A.shape()[1], A.shape()[0], TAU.size(), 
		A.origin(), A.strides()[0], TAU.data(), 
		WORK.data(), WORK.capacity(), 
		status
	);
	assert(status==0);
	return std::forward<MultiArray2D>(A);
}

template<class MultiArray2D, class Array1D, class Buffer>
MultiArray2D glq_optimal_workspace_size(MultiArray2D&& A){
	assert(A.strides()[0] > 0);
	assert(A.strides()[1] == 1);

	typename MultiArray2D::element WORK;
	int status = -1;
	BLAS::gqr(
		A.shape()[1], A.shape()[0], nullptr, A.strides()[0], 
		nullptr, 
		&WORK, /*lwork*/ -1, 
		status
	);
	assert(status==0);
	return real(WORK);
}

template<class MultiArray2D, class Array1D, class Buffer>
MultiArray2D glq(MultiArray2D&& A, Array1D&& TAU, Buffer&& WORK){
	assert(A.strides()[1] == 1);
	assert(TAU.strides()[0] == 1);
	assert(TAU.size() >= std::max(1, std::min(A.shape()[0], A.shape()[1])));
	assert(WORK.capacity() > 0);

	int status = -1, lwork = -1;
//	WORK.reserve(glq_optimal_workspace_size(A));
	BLAS::glq(
		A.shape()[1], A.shape()[0], TAU.size(), 
		A.origin(), A.strides()[0], TAU.data(), 
		WORK.data(), WORK.capacity(), 
		status
	);
	assert(status==0);
	return std::forward<MultiArray2D>(A);
}

}

#ifdef _TEST_MA_LAPACK

#include<boost/multi_array.hpp>
#include<iostream>

using std::cout;

int main(){
	{
		std::vector<double> a = {
			1.,2.,
			3.,4.,
		};
		boost::multi_array_ref<double, 2> A(a.data(), boost::extents[2][2]);
		std::vector<int> p(std::min(A.shape()[0], A.shape()[1]));
		ma::getrf(A, p);
		for(int i = 0; i != A.shape()[0]; ++i, std::cout << '\n')
			for(int j = 0; j != A.shape()[1]; ++j)
				std::cout << A[i][j] << ' ';
	}
	cout << "end test" << std::endl;
}

#endif
#endif

