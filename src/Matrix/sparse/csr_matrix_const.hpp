#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && time c++ -O3 -std=c++1z -Wall `#-Wfatal-errors` -lboost_timer -D_TEST_SPARSE_CSR_MATRIX_CONST $0x.cpp -lstdc++fs -lboost_system -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
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

#ifndef SPARSE_CSR_MATRIX_CONST_HPP
#define SPARSE_CSR_MATRIX_CONST_HPP

#include "compressed_vector.hpp"
#include "coo_matrix.hpp"

#include "detail/zipper.hpp"

#include<array>
#include<cstddef>  // ptrdiff_t
#include<vector>

#include<iostream> // for debug

namespace ma{
namespace sparse{

using size_type           = std::ptrdiff_t; //std::size_t;
using difference_type     = std::ptrdiff_t;
using index               = std::ptrdiff_t;

template<class T, class Alloc = std::allocator<std::remove_const_t<T>>> 
class csr_matrix;

struct nonmember{
	protected:
	template<class... T> static decltype(auto) num_non_zero_elements_(T&&... t){
		return num_non_zero_elements(std::forward<T>(t)...);
	}
	template<class T> static decltype(auto) non_zero_indices1_data_(T&& t){
		return non_zero_indices1_data(std::forward<T>(t));
	}
	template<class T> static decltype(auto) non_zero_indices2_data_(T&& t){
		return non_zero_indices2_data(std::forward<T>(t));
	}
	template<class T> static decltype(auto) non_zero_values_data_(T&& t){
		return non_zero_values_data(std::forward<T>(t));
	}
	template<class T> static decltype(auto) shape_(T&& t){
		return shape(std::forward<T>(t));
	}
	template<class T> static decltype(auto) clear_(T&& t){
		return shape(std::forward<T>(t));
	}
};

template<class T, class Alloc>
class csr_matrix<T const, Alloc> : nonmember{
	protected:
	using alloc_ts = std::allocator_traits<Alloc>; 
	using index_allocator = typename alloc_ts::template rebind_alloc<index>;
	int cols_;
	vector<index, index_allocator> pointers_;
	vector<index, index_allocator> cs_;
	vector<T, Alloc> vs_;
	public:
	using element =  T;
//	bool operator==(csr_matrix const& other) const = delete;
//	bool operator!=(csr_matrix const& other) const{return not (*this==other);}
	auto num_elements() const{return size()*cols_;}
	csr_matrix(std::array<index, 2> const& sizes = {{0,0}}) : 
		cols_(std::get<1>(sizes)),
		pointers_(std::get<0>(sizes) + 1)
	{}
	void reserve(size_type s){vs_.reserve(s); cs_.reserve(s);}
	operator coo_matrix<T, Alloc>()&&{
		coo_matrix<T, Alloc> ret(shape());
		ret.is_.resize(num_non_zero_elements());
		ret.js_ = std::move(cs_);
		ret.vs_ = std::move(vs_);
		for(std::size_t ip = 0; ip != pointers_.size() - 1; ++ip){
			for(auto i = pointers_[ip]; i != pointers_[ip+1]; ++i){
				ret.is_[i] = ip;
			}
			pointers_[ip] = 0;
		}
		pointers_.back() = 0;
		return ret;
	}
	operator coo_matrix<T, Alloc>() const&{return csr_matrix(*this);}
//	csr_matrix(coo_matrix<T> const& cm) : csr_matrix(coo_matrix<T>(cm)){}
	csr_matrix(coo_matrix<T, Alloc>&& cm) : csr_matrix(shape_(cm)){
		using boost::zip;
		auto begin = zip(
			non_zero_indices1_data(cm), 
			non_zero_indices2_data_(cm), 
			non_zero_values_data_(cm)
		);
		using std::sort;
		sort(
			begin, begin + num_non_zero_elements_(cm),
			[](auto&& a, auto&& b){
				using std::tie;
				using std::get;
				assert(tie(get<0>(a), get<1>(a)) != tie(get<0>(b), get<1>(b))); 
				return tie(get<0>(a), get<1>(a)) <  tie(get<0>(b), get<1>(b));
			}
		);
		index curr = -1;
		for(index n = 0; n != (index)num_non_zero_elements_(cm); ++n){
			if(non_zero_indices1_data(cm)[n] != curr){
				index old = curr;
				curr = non_zero_indices1_data(cm)[n];
				for(index i = old + 1; i <= curr; ++i) pointers_[i] = n;
			}
		}
		for(
			auto i = non_zero_indices1_data(cm)[num_non_zero_elements_(cm)-1]+1; 
			i < (index)pointers_.size(); ++i
		){
			pointers_[i] = num_non_zero_elements_(cm);
		}
		vs_ = std::move(cm).move_non_zero_values();
		cs_ = std::move(cm).move_non_zero_indices2();
		clear_(cm);
	}
	csr_matrix& operator=(coo_matrix<T, Alloc>&& cm){
		return operator=(csr_matrix(std::move(cm)));
	}
	auto num_non_zero_elements() const{return vs_.size();}
	auto non_zero_values_data() const{return vs_.data();}
	auto non_zero_indices2_data() const{return cs_.data();}
	auto pointers_begin() const{return pointers_.data();}
	auto pointers_end() const{return pointers_.data() + 1;}
	auto size() const{return size_type(pointers_.size() - 1);}
	std::array<size_type, 2> shape() const{return {{size(), cols_}};}
	protected:
	static T constexpr zero = 0;
	template<class Self> struct row_reference_base{
		Self& self_;
		index const i_;
		template<class SelfRow> struct value_reference_base{
			SelfRow& self_;
			index const j_;
			auto find() const{
				auto row_begin = self_.self_.cs_.begin() + self_.self_.pointers_[self_.i_  ];
				auto row_end   = self_.self_.cs_.begin() + self_.self_.pointers_[self_.i_+1];
				auto f = std::lower_bound(row_begin, row_end, j_);
				return std::make_pair(f, self_.self_.vs_.begin() + distance(self_.self_.cs_.begin(), f));
			}
			operator T const&() const{
				auto it = find();
				if(it.first != self_.self_.cs_.cbegin() + self_.self_.pointers_[self_.i_+1] and *it.first == j_)
					return *it.second;
				return zero; 
			}
			value_reference_base& operator=(value_reference_base const&) = delete;
			value_reference_base& operator=(value_reference_base&&) = delete;
		};
		using value_const_reference = value_reference_base<row_reference_base const>;
		using const_reference = value_const_reference;
		const_reference operator[](index j) const{return {*this, j};}
		using reference = const_reference;
	};
	using row_const_reference = row_reference_base<csr_matrix const>;
	public:
	using const_reference = row_const_reference;
	const_reference operator[](index i) const{return const_reference{*this, i};}
	friend std::array<index, 2> index_bases(csr_matrix const&){return {{0,0}};}
	friend auto non_zero_values_data(csr_matrix const& s){return s.non_zero_values_data();}
	friend auto non_zero_indices2_data(csr_matrix const& s){return s.non_zero_indices2_data();}
	friend auto pointers_begin(csr_matrix const& s){return s.pointers_begin();}
	friend auto pointers_end(csr_matrix const& s){return s.pointers_end();}
	friend auto shape(csr_matrix const& s){return s.shape();}
};


}
}

////////////////////////////////////////////////////////////////////////////////

#ifdef _TEST_SPARSE_CSR_MATRIX_CONST

#include<boost/range/iterator_range_core.hpp>
#include<boost/timer/timer.hpp>

#include<random>

int main(){

	using std::cout;
	using ma::sparse::coo_matrix;
	using ma::sparse::csr_matrix;
	
	auto const M = 10000;
	auto const N = 10000;
	double const sparsity = 0.08;

	{
		auto source_reverse = [&](){
			coo_matrix<double> source({M, N});
			source.reserve(num_non_zero_elements(source)*sparsity);
			std::default_random_engine gen;
			std::uniform_real_distribution<double> dist, dist2(0, 10);
			for(auto i = M; i != 0; --i)
				for(auto j = N; j != 0; --j)
					if(dist(gen) < sparsity) source[i - 1][j - 1] = dist2(gen);
			return source;}();
		{
			boost::timer::auto_cpu_timer t("coo (reverse) to csr: %t seconds\n");
			csr_matrix<double const> csrm = std::move(source_reverse);
		}
	}
	{
		auto source_sorted = [&](){
			coo_matrix<double> source({M, N});
			source.reserve(num_non_zero_elements(source)*sparsity);
			std::default_random_engine gen;
			std::uniform_real_distribution<double> dist;
			std::uniform_real_distribution<double> dist2(0, 10);
			for(auto i = 0; i != M; ++i)
				for(auto j = 0; j != N; ++j)
					if(dist(gen) < sparsity) source[i][j] = dist2(gen);
			return source;}();
		{
			boost::timer::auto_cpu_timer t("coo (sorted) to csr: %t seconds\n");
			csr_matrix<double const> csrm(std::move(source_sorted));
		}
	}
	
	csr_matrix<double const> small = coo_matrix<double>(
		{4,4}, 
		{
			{{3,3}, 1}, 
			{{2,1}, 3}, 
			{{0,1}, 9}
		}
	);
	
}

#endif
#endif

