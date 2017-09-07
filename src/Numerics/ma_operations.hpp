#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && clang++ -O3 -std=c++11 -Wfatal-errors -I.. -D_TEST_MA_OPERATIONS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif
#ifndef MA_OPERATIONS_HPP
#define MA_OPERATIONS_HPP

// File created and developed by:
// Alfredo Correa, correaa@llnl.gov
//    Lawrence Livermore National Laboratory

#include "ma_blas.hpp"
#include "ma_lapack.hpp"

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

template<class MA2D> auto trans(MA2D&& arg)
->decltype(transposed(std::forward<MA2D>(arg))){
	return transposed(std::forward<MA2D>(arg));
}
template<class MA2D> auto herm(MA2D&& arg)
->decltype(hermitian(std::forward<MA2D>(arg))){
	return hermitian(std::forward<MA2D>(arg));
}
//template<class MA2D> auto norm(MA2D&& arg)
//->decltype(normal(std::forward<MA2D>(arg))){
//	return normal(std::forward<MA2D>(arg));
//}


template<class MultiArray2D>
int invert_optimal_workspace_size(MultiArray2D const& m){
	return getri_optimal_workspace_size(m);
}

template<class MultiArray2D, class MultiArray1D, class Buffer, class T = typename std::decay<MultiArray2D>::type::element>
T invert(MultiArray2D&& m, MultiArray1D&& pivot, Buffer&& WORK){
	assert(m.shape()[0] == m.shape()[1]);
	assert(pivot.size() >= m.shape()[0]);

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

template<class MultiArray2D, class MultiArray1D, class T = typename std::decay<MultiArray2D>::type::element>
T invert(MultiArray2D&& m, MultiArray1D&& pivot){
	std::vector<typename std::decay<MultiArray2D>::type::element> WORK;
	WORK.reserve(invert_optimal_workspace_size(m));
	return invert(m, pivot, WORK);
}

template<class MultiArray2D, class T = typename std::decay<MultiArray2D>::type::element>
MultiArray2D invert(MultiArray2D&& m){
	std::vector<int> pivot(m.shape()[0]);
	auto det = invert(m, pivot);
	return std::forward<MultiArray2D>(m);
}

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
	{
		std::vector<double> a = {37., 45., 59., 53., 81., 97., 87., 105., 129.};
		boost::multi_array_ref<double, 2> A(a.data(), boost::extents[3][3]);
		assert(A.num_elements() == a.size());
		boost::multi_array<double, 2> B = A;
		ma::invert(A);

		boost::multi_array<double, 2> Id(boost::extents[3][3]);
		ma::set_identity(Id);

		boost::multi_array<double, 2> Id2(boost::extents[3][3]);
		ma::product(A, B, Id2);
						
		assert( ma::equal(Id, Id2, 1e-14) );
	}

	cout << "test ended" << std::endl;
}
#endif

#endif


