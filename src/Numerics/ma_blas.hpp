#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && c++ -O3 -std=c++11 `#-Wfatal-errors` -I.. -D_TEST_MA_BLAS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif
#ifndef MA_BLAS_HPP
#define MA_BLAS_HPP

#include "Numerics/OhmmsBlas.h"
#include<utility> //std::enable_if

template<class T, class MultiArray, typename = typename std::enable_if<std::decay<MultiArray>::type::dimensionality == 1>::type >
MultiArray& Scal(T a, MultiArray&& x){
	BLAS::scal(x.size(), a, x.origin(), x.strides()[0]);
	return x;
}

#ifdef _TEST_MA_BLAS

#include<boost/multi_array.hpp>
#include<iostream>

using std::cout;

int main(){

	std::vector<double> v = {1.,2.,3.};
	{
		boost::multi_array_ref<double, 1> V(v.data(), boost::extents[v.size()]);
		Scal(2., V);
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
	Scal(2., M[2]);
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

	Scal(2., M[ indices[range_t(0,3)][2] ]);
	{
		std::vector<double> m2 = {
			1.,2.,6.,
			4.,5.,12.,
			14.,16.,36.
		};
		boost::multi_array_ref<double, 2> M2(m2.data(), boost::extents[3][3]);
		assert( M == M2 );
	}
	Scal(2., M[ indices[range_t(0,2)][1] ]);
	{
		std::vector<double> m2 = {
			1.,4.,6.,
			4.,10.,12.,
			14.,16.,36.
		};
		boost::multi_array_ref<double, 2> M2(m2.data(), boost::extents[3][3]);
		assert( M == M2 );
	}

}

#endif
#endif


