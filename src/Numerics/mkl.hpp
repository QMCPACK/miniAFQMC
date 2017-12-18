#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && clang++ -O3 -std=c++14 -Wfatal-errors -I.. -D_TEST_SP_BLAS -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -Wl,-rpath=/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/11.1/SystemFiles/Libraries/Linux-x86-64 -lmkl_core -lmkl_intel_ilp64 -lmkl_sequential -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif

////////////////////////////////////////////////////////////////////////////////
// File created and developed by:
// Alfredo Correa, correaa@llnl.gov
//    Lawrence Livermore National Laboratory
////////////////////////////////////////////////////////////////////////////////

#ifndef NUMERICS_MKL_HPP
#define NUMERICS_MKL_HPP

#include<complex>

#define MKL_INT long int

extern "C" {

void mkl_scsrmv(const char &transa, const MKL_INT &m, const MKL_INT &k, const float &alpha               , const char *matdescra, const float *val               , const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x               , const float &beta               , float *y               );
void mkl_ccsrmv(const char &transa, const MKL_INT &m, const MKL_INT &k, const std::complex<float> &alpha , const char *matdescra, const std::complex<float> *val , const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const std::complex<float> *x , const std::complex<float> &beta , std::complex<float> *y );
void mkl_dcsrmv(const char &transa, const MKL_INT &m, const MKL_INT &k, const double &alpha              , const char *matdescra, const double *val              , const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x              , const double &beta              , double *y              );
void mkl_zcsrmv(const char &transa, const MKL_INT &m, const MKL_INT &k, const std::complex<double> &alpha, const char *matdescra, const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const std::complex<double> *x, const std::complex<double> &beta, std::complex<double> *y);

void mkl_scsrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const float &alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT &ldb, const float &beta, float *c, const MKL_INT &ldc);
void mkl_ccsrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const std::complex<float> &alpha, const char *matdescra, const std::complex<float> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const std::complex<float> *b, const MKL_INT &ldb, const std::complex<float> &beta, std::complex<float> *c, const MKL_INT &ldc);
void mkl_dcsrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const double &alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT &ldb, const double &beta, double *c, const MKL_INT &ldc);
void mkl_zcsrmm(const char &transa, const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const std::complex<double> &alpha, const char *matdescra, const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const std::complex<double> *b, const MKL_INT &ldb, const std::complex<double> &beta, std::complex<double> *c, const MKL_INT &ldc);

}

namespace mkl{

template<class T> struct mkl_Tcsrmm;

template<> struct mkl_Tcsrmm<float>{
	template<class... As> static void run(As&&... as){mkl_scsrmm(as...);}
};
template<> struct mkl_Tcsrmm<std::complex<float>>{
	template<class... As> static void run(As&&... as){mkl_ccsrmm(as...);}
};
template<> struct mkl_Tcsrmm<double>{
	template<class... As> static void run(As&&... as){::mkl_dcsrmm(as...);}
};
template<> struct mkl_Tcsrmm<std::complex<double>>{
	template<class... As> static void run(As&&... as){mkl_zcsrmm(as...);}
};

template<class T>
void csrmm(
	char const &transa, 
	const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, 
	T const & alpha, 
	const char *matdescra, T const *val, const MKL_INT *indx, 
	MKL_INT const *pntrb, const MKL_INT *pntre, T const* b, 
	MKL_INT const &ldb, const T &beta, 
	T* c, MKL_INT const& ldc
){
	mkl_Tcsrmm<T>::run(
		transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, 
		b, ldb, beta, c, ldc
	);
}

}

#endif

