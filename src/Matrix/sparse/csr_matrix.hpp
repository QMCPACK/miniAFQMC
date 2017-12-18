#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && time c++ -O3 -std=c++1z -Wall `#-Wfatal-errors` -lboost_timer -D_TEST_SPARSE_CSR_MATRIX $0x.cpp -lstdc++fs -lboost_system -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif

#ifndef SPARSE_CSR_MATRIX_HPP
#define SPARSE_CSR_MATRIX_HPP

#include "compressed_vector.hpp"
#include "csr_matrix_const.hpp"

#include<array>
#include<cstddef>  // ptrdiff_t
#include<vector>

#include<iostream> // for debug

namespace ma{
namespace sparse{

using size_type           = std::ptrdiff_t; //std::size_t;
using difference_type     = std::ptrdiff_t;
using index               = std::ptrdiff_t;

template<class T, class Alloc>
class csr_matrix : public csr_matrix<T const, Alloc>{
	struct row_reference : csr_matrix<T const, Alloc>::template row_reference_base<csr_matrix>{
		using typename csr_matrix<T const, Alloc>::template row_reference_base<csr_matrix>::value_const_reference;
		struct value_reference : row_reference::template value_reference_base<row_reference>{
			using row_reference::template value_reference_base<row_reference>::self_;
			using row_reference::template value_reference_base<row_reference>::j_;
			using row_reference::template value_reference_base<row_reference>::find;
			template<class TT>
			value_reference&& operator=(TT&& tt)&&{
				auto it = find();
				if(it.first != self_.self_.cs_.begin() + self_.self_.pointers_[self_.i_+1] and *it.first == j_){
					*it.second = std::forward<TT>(tt);
				}else{
					self_.self_.vs_.emplace(it.second, std::forward<TT>(tt));
					self_.self_.cs_.emplace(it.first, j_);
					for(
						auto it = std::next(self_.self_.pointers_.begin() + self_.i_) ; 
						it!= self_.self_.pointers_.end(); 
						++it
					) ++*it;
				}
				return std::move(*this);
			}
		};
		using reference = value_reference;
		reference operator[](index j){return reference{{*this, j}};}
	};
	public:
	using typename csr_matrix<T const, Alloc>::const_reference;
	using reference = row_reference;
	using csr_matrix<T const, Alloc>::csr_matrix;
	using csr_matrix<T const, Alloc>::operator[];
	reference operator[](index i){return {{*this, i}};}
};

}
}

#ifdef _TEST_SPARSE_CSR_MATRIX

#include<random>

#include<boost/range/iterator_range_core.hpp>
#include<boost/timer/timer.hpp>

int main(){
	using std::cout;
	using ma::sparse::coo_matrix;
	using ma::sparse::csr_matrix;
	
	auto const M = 2000;
	auto const N = 2000;
	double const sparsity = 0.01;

	// generate tuples for reference
	auto const csource = [&](){
		std::vector<std::tuple<int, int, double>> source; 
		source.reserve(M*sparsity*N);
		std::default_random_engine gen;
		std::uniform_real_distribution<double> dist;
		std::uniform_real_distribution<double> dist2(0, 10);
		for(auto i = M; i != 0; --i)
			for(auto j = N; j != 0; --j)
				if(dist(gen) < sparsity) source.emplace_back(i - 1, j - 1, dist2(gen));
		return source;
	}();

	{
		boost::timer::auto_cpu_timer t("fill coo_matrix and move to csr %w seconds\n");
		coo_matrix<double> coom({M, N});
		using std::get;
		for(auto&& e : csource)
			coom[get<0>(e)][get<1>(e)] = get<2>(e);
		csr_matrix<double> csrm = std::move(coom);
	}
	cout << std::flush;
	{
		boost::timer::auto_cpu_timer t("fill csr matrix directly %w seconds\n");
		csr_matrix<double> csrm({M, N});
		using std::get;
		for(auto&& e : csource)
			csrm[get<0>(e)][get<1>(e)] = get<2>(e);
	}
#if 0
	{
		boost::timer::auto_cpu_timer t(std::cerr, "fill csr directly %w seconds\n");

		ma::sparse::csr_matrix<double> csrm({{M, N}});
		std::default_random_engine gen;
		std::uniform_real_distribution<double> dist;
		std::uniform_real_distribution<double> dist2(0, 10);
		for(ma::sparse::index i = 0; i != M; ++i)
			for(ma::sparse::index j = 0; j != N; ++j)
				if(dist(gen) < sparsity) csrm[i][j] = dist2(gen);
		assert(csrm == csrm_ref);
		for(int i = 0; i != M; ++i){
			for(int j = 0; j != N; ++j){
				std::cout << csrm[i][j] << ' ';
			}
		std::cout << '\n';
		}
	}
	return 0;
#if 0
	{
		std::cout << "natural syntax " << boost::time([&](){
			ma::sparse::csr_matrix<double> csrm2(std::array<ma::sparse::size_type, 2>{{M, N}});
			std::default_random_engine gen;
			std::uniform_real_distribution<double> dist;
			std::uniform_real_distribution<double> dist2(0, 10);
			for(ma::sparse::index i = 0; i != M; ++i)
				for(ma::sparse::index j = 0; j != N; ++j)
					if(dist(gen) > sparcity) csrm2[i][j] = dist2(gen);
			std::cout 
				<< "num elements " 
				<< csrm.num_non_zero_elements() << " vs " << csrm2.num_non_zero_elements()
				<< std::endl
			;
			assert(csrm.num_non_zero_elements() == csrm2.num_non_zero_elements());
//			assert(csrm == csrm2);

#if 0
			std::cout << "csrm = \n";
			for(int i = 0; i != M; ++i){
				for(int j = 0; j != N; ++j){
					std::cout << csrm[i][j] << ' ';
				}
				std::cout << '\n';
			}
			std::cout << "csrm2 = \n";
			for(int i = 0; i != M; ++i){
				for(int j = 0; j != N; ++j){
					std::cout << csrm2[i][j] << ' ';
				}
				std::cout << '\n';
			}
			std::cout << csrm2.num_non_zero_elements() << '\n';
#endif
			return 0;
		}, 1)().first.count()/1.e9 << " seconds\n";

	}
#endif
#endif
}

#endif
#endif

