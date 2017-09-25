#if COMPILATION_INSTRUCTIONS
(echo "#include<"$0">" > $0x.cpp) && clang++ -O3 -std=c++14 -Wfatal-errors -I.. -D_TEST_SPARSE_ARRAY -DADD_ -Drestrict=__restrict__ $0x.cpp -lblas -llapack -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif
#ifndef SPARSE_ARRAY_HPP
#define SPARSE_ARRAY_HPP

#include<boost/container/flat_set.hpp>
#include "../Utilities/tuple_iterator.hpp"
#include "alf/boost/iterator/zipper.hpp"
#include<algorithm>
#include<cstddef>  // ptrdiff_t
#include<vector>
#include<iosfwd>
#include<iostream>

namespace ma{

	using size_type           = std::ptrdiff_t; //std::size_t;
	using difference_type     = std::ptrdiff_t;
	using index               = std::ptrdiff_t;

	template<class T>
	class const_compressed_vector{
		protected:
		size_type size_;
		std::vector<T> vs_;
		boost::container::flat_set<index> js_;
		protected:
		const_compressed_vector(size_type s) : size_(s){}
		public:
		using element = T;
		template<class It>
		const_compressed_vector(size_type s, It first, It last) : size_(s){
			while(first != last){
				assert( std::get<0>(*first) < size_ );
				auto in = js_.insert(std::get<0>(*first));
				auto it = vs_.begin() + std::distance(js_.begin(), in.first);
				if(in.second)
					vs_.insert(it, std::get<1>(*first));
				else
					*it = std::get<1>(*first);
				++first;
			}
		}
		const_compressed_vector(size_type s, std::initializer_list<std::pair<index, T>> il)
		:	const_compressed_vector(s, il.begin(), il.end()){}
		
		T const* non_zero_values() const{return vs_.data();}
		index const* non_zero_indices() const{return &*js_.begin();}
		size_type num_non_zero_elements() const{
			assert(vs_.size() == js_.size());
			return vs_.size();
		}
		size_type const* shape() const{return &size_;}
		auto begin() const{return qmcplusplus::make_paired_iterator(js_.begin(), vs_.begin());}
		auto end() const{return qmcplusplus::make_paired_iterator(js_.end(), vs_.end());}
		bool operator==(const_compressed_vector const& other) const{
			return size_ == other.size_ and vs_ == other.vs_ and js_ == other.js_;
		}
		T const& operator[](index i) const{
			static T const zero(0);
			auto f = js_.find(i);
			if(f!= js_.end()) return vs_[std::distance(js_.begin(), f)];
			return zero;
		}
	};

	template<class T>
	class compressed_vector : public const_compressed_vector<T>{
		class reference{
			compressed_vector& sv_;
			const index j_;
			reference(compressed_vector& sv, index j) : sv_(sv), j_(j){}
			public:
			reference(reference const&) = delete;
			reference(reference&& other) : sv_(other.sv_), j_(other.j_){}
			operator T const&() const{
				static T const zero(0);
				auto f = sv_.js_.find(j_);
				if(f!= sv_.js_.end()) return sv_.vs_[std::distance(sv_.js_.begin(), f)];
				return zero;
			}
			reference operator=(void**** zero)&&{
				assert(zero == nullptr);
				auto f = sv_.js_.find(j_);
				if(f!= sv_.js_.end()){
					sv_.vs_.erase(sv_.vs_.begin() + std::distance(sv_.js_.begin(), f));
					sv_.js_.erase(f);
				}
				return std::move(*this);
			}
			operator T&() const{
				static T const zero(0);
				auto in = sv_.js_.insert(j_);
				if(in.second){
					sv_.vs_.insert(sv_.vs_.begin() + std::distance(sv_.js_.begin(), in.first), zero);
				}
				return *(sv_.vs_.begin() + std::distance(sv_.js_.begin(), in.first));
			}
			template<class O> auto operator==(O&& t) const{
				return operator T const&() == std::forward<O>(t);
			}
			template<class O> auto operator>(O&& t) const{
				return operator T const&() > std::forward<O>(t);
			} 
			reference const& operator=(T const& t) const{
				auto in = sv_.js_.insert(j_);
				if(in.second)
					sv_.vs_.insert(sv_.vs_.begin() + std::distance(sv_.js_.begin(), in.first), t);
				else
					*(sv_.vs_.begin() + std::distance(sv_.js_.begin(), in.first)) = t;
				return *this;
			}
			friend class compressed_vector;
		};
		public:
		using const_compressed_vector<T>::const_compressed_vector;
		using const_compressed_vector<T>::operator[];
		reference operator[](index i){return reference{*this, i};}
	};
	
	template<class CompressedVector>
	std::ostream& print(std::ostream& os, CompressedVector const& cv){
		std::for_each(
			boost::zip(cv.non_zero_indices(), cv.non_zero_values()), boost::zip(cv.non_zero_indices(), cv.non_zero_values()) + cv.num_non_zero_elements(), 
			[&os](auto&& p){os << std::get<0>(p) << "->" << std::get<1>(p) << " ";}
		);
		return os;
	}

	template<class T>
	class csr_matrix{
		std::vector<T> vs_;
		std::vector<index> cs_;
		int cols_;
		std::vector<index> pointers_;
		public:
		static std::size_t const dimensionality = 2;
		size_type num_non_zero_elements() const{
			assert(cs_.size() == vs_.size());
			return vs_.size();
		}
		T const* non_zero_values() const{return vs_.data();}
		index const* non_zero_indices2() const{return cs_.data();}
		index const* pointers_begin() const{return pointers_.data();}
		index const* pointers_end() const{return pointers_.data() + 1;}
		
		index size() const{return pointers_.size()-1;}
		std::array<index, 2> shape() const{return {size(), cols_};}
		template<class ExtentList>
		csr_matrix(ExtentList e) : pointers_(e.ranges_[0].finish() + 1), cols_(e.ranges_[1].finish()){
			assert(e.ranges_[0].start() == 0);
			assert(e.ranges_[1].start() == 0);
		}
		struct const_reference{
			std::vector<index>::const_iterator begin_;
			std::vector<index>::const_iterator end_;
			typename std::vector<T>::const_iterator vs_begin_;
			T const& operator[](index i){
				static T const zero(0);
				auto f = std::lower_bound(begin_, end_, i);
				if(f != end_ and i >= *begin_) return *(vs_begin_ + i);
				return zero;
			}
		};
		struct compressed_vector_ref{
			std::vector<T>& vs_;
			std::vector<index>& cs_;
			std::vector<index>& pointers_;
			std::vector<index>::const_iterator begin_;
			std::vector<index>::const_iterator end_;
			index i_;
			T const* non_zero_values() const{
				std::vector<index>::const_iterator bb = cs_.begin();
				return &*(vs_.begin() + std::distance(bb, begin_));
			}
			index const* non_zero_indices() const{return &*begin_;}
			int num_non_zero_elements() const{return std::distance(begin_, end_);}
			operator compressed_vector<T>() const{
				return compressed_vector<T>(
					boost::zip(non_zero_indices(), non_zero_values()), 
					boost::zip(non_zero_indices(), non_zero_values()) + num_non_zero_elements()
				);
			}
			struct reference{
				std::vector<T>& vs_;
				std::vector<index>& cs_;
				std::vector<index>& pointers_;
				std::vector<index>::const_iterator begin_;
				std::vector<index>::const_iterator end_;
				index i_;
				index j_;
				operator T const&() const{
					static T const zero(0);
					std::vector<index>::const_iterator f = std::lower_bound(begin_, end_, j_);
					std::vector<index>::const_iterator bb = cs_.begin();
					if(f != end_ and j_ >= *begin_) return *(vs_.begin() + std::distance(bb, f));
					return zero; 
				}
				template<class O>
				reference const& operator=(O&& t) const{
					auto f = std::lower_bound(begin_, end_, j_);
					if(f != end_ and j_ >= *begin_) *(vs_.begin() + std::distance(const_cast<long const*>(&*cs_.begin()), &*f)) = std::forward<O>(t);
					else{
						vs_.insert(vs_.begin() + std::distance(const_cast<long const*>(&*cs_.begin()), &*f), t);
						cs_.insert(f, j_);
						for(int i = i_; i != pointers_.size(); ++i) ++pointers_[i];
					}
					return *this;
				}
			};
			reference operator[](index j) const{
				return {vs_, cs_, pointers_, begin_, end_, i_, j};
			}
		};
		using reference = compressed_vector_ref;
		using value = compressed_vector<T>;
		compressed_vector_ref operator[](index i){
			return compressed_vector_ref{vs_, cs_, pointers_, cs_.begin() + pointers_[i], cs_.begin() + pointers_[i+1], i};
		}
		const_reference operator[](index i) const{
		//	assert(std::is_sorted(cs_.begin() + pB_, cs_.begin() + pE_));
			return {cs_.begin() + pointers_[i], cs_.begin() + pointers_[i+1], vs_.begin() + pointers_[i]};
		}
	};

}

#ifdef _TEST_SPARSE_ARRAY

#include "alf/boost/assert/simple_msg.hpp"

#include<iostream>
#include<sstream>
#include<numeric>

#include<boost/multi_array.hpp>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/assert.hpp>


int main(){

	{
		ma::compressed_vector<double> ccv(10, {{9, 5.}, {5, 5.}, {4, 2.}});
		assert(ccv.shape()[0] == 10);
		assert(ccv.num_non_zero_elements() == 3);
		assert(ccv[9] == 5.);
		assert(ccv[5] == 5.);
		assert(ccv[4] == 2.);
		
		ccv[4] = 0;
		BOOST_ASSERT(ccv[4] == 0.);
		BOOST_ASSERT(ccv.num_non_zero_elements() == 2);

		assert(ccv[3] == 0.);
		assert(ccv[6] == 0.);
		
		std::ostream_iterator<long> out_long (std::cout,", ");
		std::copy( ccv.non_zero_indices(), ccv.non_zero_indices() + ccv.num_non_zero_elements(), out_long);
		std::cout << '\n';
		std::ostream_iterator<double> out_double (std::cout,", ");
		std::copy( ccv.non_zero_values(), ccv.non_zero_values() + ccv.num_non_zero_elements(),  out_double);
		ma::compressed_vector<double> ccv2(10, {{9, 5.}, {5, 5.}});
		BOOST_ASSERT(ccv == ccv2);
	}
	
	std::cout << '\n';
	boost::numeric::ublas::compressed_matrix<double> m (3, 3, 3 * 3);
	for(unsigned i = 0; i < m.size1 (); ++i){
		for (unsigned j = 0; j < m.size2 (); ++j){
			m (i, j) = 3 * i + j;
		}
	}
	std::cout << m << std::endl;

	double* pp = m.value_data().begin();

	std::ostream_iterator<double> out_double (std::cout,", ");
	std::copy( m.value_data().begin(), m.value_data().begin() + m.nnz(),  out_double);
	std::cout << '\n';
	std::ostream_iterator<long> out_long (std::cout,", ");
	std::copy( m.index2_data().begin(), m.index2_data().begin() + m.nnz(),  out_long);
	std::cout << '\n';
	std::copy( m.index1_data().begin(), m.index1_data().begin() + m.size1(),  out_long);
	std::copy( m.index1_data().begin() + 1, m.index1_data().begin() + 1 + m.size1(),  out_long);
	std::cout << '\n';
	
	

#if 0
	// element write access
	v[9] = 5.;
	v[5] = 5.;
	v[4] = 2.;
	v[4] = 1.;

	// element read access
	assert(v[9] == 5.);
	assert(v[9] > 4.);
	assert(v[1] == 0.);
	assert(v[4] == 1.);

	// pointer access
	assert( v.num_non_zero_elements() == 3 );
	assert( v.non_zero_values()[0] == 1. );
	assert( v.non_zero_indices()[0] == 4 ); 

//	std::ostringstream oss; oss << v;
//	print(std::cout, v);
//	assert( oss.str() == "4->1 5->5 9->5 " );
	ma::csr_matrix<double> cm(boost::extents[10][20]);
#if 0
	cm[3][4] = 5.;
	cm[3][7] = 6.;
	assert( cm[3][4] == 5. );
	double cm34 = cm[3][4];
//	std::cout << "cm34 = " << cm34 << std::endl;
	std::cout << cm[3].size() << std::endl;
	print(std::cout, cm[3]);
	return 0;
	{
		boost::container::vector<double> v(10);
		std::iota(v.begin(), v.end(), 0);
		for(int i = 0; i != v.size(); ++i) std::cout << "v[" << i << "] = " << v[i] << std::endl;
		boost::container::flat_set<double>& fs = reinterpret_cast<boost::container::flat_set<double>&>(v);
		fs.insert(2.5);
		for(int i = 0; i != v.size(); ++i) std::cout << "v[" << i << "] = " << v[i] << std::endl;
	}
#endif
#endif
}

#endif
#endif

