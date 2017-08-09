#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && c++ -O3 -std=c++11 `#-Wfatal-errors` -I.. -D_TEST_MA_BLAS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif
#ifndef MA_BLAS_HPP
#define MA_BLAS_HPP

#include "Numerics/OhmmsBlas.h"
#include<utility> //std::enable_if
#include<cassert>
#include<iostream>

template<class T, class MultiArray1D, typename = typename std::enable_if<std::decay<MultiArray1D>::type::dimensionality == 1>::type >
MultiArray1D scal(T a, MultiArray1D&& x){
	BLAS::scal(x.size(), a, x.origin(), x.strides()[0]);
	return std::forward<MultiArray1D>(x);
}

template<class T, class MultiArray1D>
auto operator*=(MultiArray1D&& x, T a) -> decltype(scal(a, std::forward<MultiArray1D>(x))){
	return scal(a, std::forward<MultiArray1D>(x));
}

template<class T, class MultiArray1DA, class MultiArray1DB, 
	typename = typename std::enable_if<std::decay<MultiArray1DA>::type::dimensionality == 1 and std::decay<MultiArray1DB>::type::dimensionality == 1>::type
>
MultiArray1DB axpy(T x, MultiArray1DA const& a, MultiArray1DB&& b){
	assert( a.shape()[0] == b.shape()[0] );
	BLAS::axpy(a.shape()[0], x, a.origin(), a.strides()[0], b.origin(), b.strides()[0]);
	return std::forward<MultiArray1DB>(b);
}

template<char IN, class T, class MultiArray2DA, class MultiArray1DX, class MultiArray1DY,
	typename = typename std::enable_if< MultiArray2DA::dimensionality == 2 and MultiArray1DX::dimensionality == 1 and std::decay<MultiArray1DY>::type::dimensionality == 1>::type
>
MultiArray1DY gemv(T alpha, MultiArray2DA const& A, MultiArray1DX const& x, T beta, MultiArray1DY&& y){
	if(IN == 'T') assert( x.shape()[0] == A.shape()[1] and y.shape()[0] == A.shape()[0]);
	else if(IN == 'N') assert( x.shape()[0] == A.shape()[0] and y.shape()[0] == A.shape()[1]);
	assert( A.strides()[1] == 1 ); // gemv is not implemented for arrays with non-leading stride != 1
	BLAS::gemv(IN, A.shape()[1], A.shape()[0], alpha, A.origin(), A.strides()[0], x.origin(), x.strides()[0], beta, y.origin(), y.strides()[0]);
	return std::forward<MultiArray1DY>(y);
} //y := alpha*A*x + beta*y,

template<char IN, class MultiArray2DA, class MultiArray1DX, class MultiArray1DY>
MultiArray1DY gemv(MultiArray2DA const& A, MultiArray1DX const& x, MultiArray1DY&& y){
	return gemv<IN>(1., A, x, 0., y);
} //y := alpha*A*x

#ifdef _TEST_MA_BLAS

#include<boost/multi_array.hpp>
#include<iostream>

using std::cout;

int main(){

	std::vector<double> v = {1.,2.,3.};
	{
		boost::multi_array_ref<double, 1> V(v.data(), boost::extents[v.size()]);
		scal(2., V);
		{
			std::vector<double> v2 = {2.,4.,6.};
			boost::multi_array_ref<double, 1> V2(v2.data(), boost::extents[v2.size()]);
			assert( V == V2 );
		}
	}
	
	std::vector<double> m = {
		1.,2.,3.,
		4.,5.,6.,
		7.,8.,9.
	};
	boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][3]);
	assert( M.num_elements() == m.size());
	scal(2., M[2]);
	{
		std::vector<double> m2 = {
			1.,2.,3.,
			4.,5.,6.,
			14.,16.,18.
		};
		boost::multi_array_ref<double, 2> M2(m2.data(), boost::extents[3][3]);
		assert( M == M2 );
	}
	typedef boost::multi_array_types::index_range range_t;
	boost::multi_array_types::index_gen indices;

	scal(2., M[ indices[range_t(0,3)][2] ]);
	{
		std::vector<double> m2 = {
			1.,2.,6.,
			4.,5.,12.,
			14.,16.,36.
		};
		boost::multi_array_ref<double, 2> M2(m2.data(), boost::extents[3][3]);
		assert( M == M2 );
	}
	scal(2., M[ indices[range_t(0,2)][1] ]);
	{
		std::vector<double> m2 = {
			1.,4.,6.,
			4.,10.,12.,
			14.,16.,36.
		};
		boost::multi_array_ref<double, 2> M2(m2.data(), boost::extents[3][3]);
		assert( M == M2 );
	}
	axpy(2., M[1], M[0]); // M[0] += a*M[1]
	{
		std::vector<double> m2 = {
			9.,24.,30.,
			4.,10.,12.,
			14.,16.,36.
		};
		boost::multi_array_ref<double, 2> M2(m2.data(), boost::extents[3][3]);
		assert( M == M2 );
	}
	{
		std::vector<double> m = {
			9.,24.,30., 9.,
			4.,10.,12., 7.,
			14.,16.,36., 1.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][4]);
		assert( M[2][0] == 14. );
		std::vector<double> x = {1.,2.,3., 4.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y = {4.,5.,6.};
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);
		gemv<'T'>(1., M, X, 0., Y); // y := M x

		std::vector<double> y2 = {183., 88.,158.};
		boost::multi_array_ref<double, 1> Y2(y2.data(), boost::extents[y2.size()]);
		assert( Y == Y2 );
	}
	{
		std::vector<double> m = {
			9.,24.,30., 9.,
			4.,10.,12., 7.,
			14.,16.,36., 1.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][4]);
		typedef boost::multi_array_types::index_range range_t;
		boost::multi_array_types::index_gen indices;

		std::vector<double> x = {1.,2.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y = {4.,5.};
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);
		
		auto const& mm = M[ indices[range_t(0,2,1)][range_t(0,2,1)] ];//, X, 0., Y); // y := M x
		gemv<'T'>(1., M[ indices[range_t(0,2,1)][range_t(0,2,1)] ], X, 0., Y); // y := M x

		std::vector<double> y2 = {57., 24.};
		boost::multi_array_ref<double, 1> Y2(y2.data(), boost::extents[y2.size()]);
		assert( Y == Y2 );
	}
	{
		std::vector<double> m = {
			9.,24.,30.,
			4.,10.,12.,
			14.,16.,36.,
			4.,9.,1.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[4][3]);
		assert( M[2][0] == 14. );
		std::vector<double> x = {1.,2.,3.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y = {4.,5.,6., 7.};
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);
		gemv<'T'>(1., M, X, 0., Y); // y := M x
		std::vector<double> y2 = {147., 60.,154.,25.};
		boost::multi_array_ref<double, 1> Y2(y2.data(), boost::extents[y2.size()]);
		assert( Y == Y2 );
	}
	{
		std::vector<double> m = {
			9.,24.,30., 9.,
			4.,10.,12., 7.,
			14.,16.,36., 1.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][4]);
		assert( M[2][0] == 14. );
		std::vector<double> x = {1.,2.,3.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y = {4.,5.,6.,7.};
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);
		gemv<'N'>(1., M, X, 0., Y); // y := M^T x
		std::vector<double> y2 = {59., 92., 162., 26.};
		boost::multi_array_ref<double, 1> Y2(y2.data(), boost::extents[y2.size()]);
		assert( Y == Y2 );
	}
	{
		std::vector<double> m = {
			9.,24.,30., 3.,
			4.,10.,12., 1.,
			14.,16.,36., 5.
		};
		boost::multi_array_ref<double, 2> M(m.data(), boost::extents[3][4]);
		std::vector<double> x = {1.,2.,3., 4.};
		boost::multi_array_ref<double, 1> X(x.data(), boost::extents[x.size()]);
		std::vector<double> y = {4.,5.,6.};
		boost::multi_array_ref<double, 1> Y(y.data(), boost::extents[y.size()]);
		gemv<'T'>(1., M, X, 0., Y); // y := M x
		std::vector<double> y2 = {159.,64.,174.};
		boost::multi_array_ref<double, 1> Y2(y2.data(), boost::extents[y2.size()]);
		assert( Y == Y2 );
	}

}

#endif
#endif


