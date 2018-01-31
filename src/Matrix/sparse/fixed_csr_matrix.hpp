#if COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && mpic++ -O3 -std=c++14 -Wall `#-Wfatal-errors` -D_TEST_SPARSE_FIXED_CSR_MATRIX $0x.cpp -o $0x.x && time mpirun -np 2 $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif

#if COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && clang++ -O3 -std=c++14 -Wall -Wfatal-errors -I.. -D_TEST_SPARSE_FIXED_CSR_MATRIX $0x.cpp -lstdc++fs -lboost_system -lboost_timer -o $0x.x && time $0x.x $@ && rm -f $0x.cpp; exit
#endif

////////////////////////////////////////////////////////////////////////////////
// File developed by:
// Alfredo Correa, correaa@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Alfredo Correa, correaa@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef SPARSE_CFO_MATRIX_HPP
#define SPARSE_CFO_MATRIX_HPP

#include<array>
#include<cassert>
#include<cstddef>  // ptrdiff_t
#include<vector>
#include<tuple>

#include<memory> 

namespace ma{
namespace sparse{

using size_type           = std::ptrdiff_t; //std::size_t;
using difference_type     = std::ptrdiff_t;
using index               = std::ptrdiff_t;

template<class T, class Alloc = std::allocator<T>>
class fixed_csr_matrix{
	using alloc_ts = std::allocator_traits<Alloc>; 
	using index_allocator = typename Alloc::template rebind<index>::other;//  alloc_ts::template rebind_alloc<index>;
	using ialloc_ts = std::allocator_traits<index_allocator>; 
	Alloc allocator_;
	index_allocator iallocator_;
	size_type size1_;
	size_type size2_;
	size_type max_num_non_zeros_per_row_;
	typename Alloc::pointer data_; 
	typename index_allocator::pointer jdata_;
	typename index_allocator::pointer pointers_begin_;
	typename index_allocator::pointer pointers_end_;
	public:
	using element = T;
	fixed_csr_matrix(
		std::tuple<size_type, size_type> const& arr = {0, 0}, 
		size_type max_num_non_zeros_per_row = 0,
		Alloc alloc = Alloc{}
	) : 
		allocator_(alloc), 
		iallocator_(alloc),
		size1_(std::get<0>(arr)), size2_(std::get<1>(arr)), 
		max_num_non_zeros_per_row_(max_num_non_zeros_per_row),
		data_(allocator_.allocate(size1_*max_num_non_zeros_per_row_)),
		jdata_(iallocator_.allocate(size1_*max_num_non_zeros_per_row_)),
		pointers_begin_(iallocator_.allocate(size1_)),
		pointers_end_(iallocator_.allocate(size1_))
	{
		for(index i = 0; i != size1_; ++i){
			ialloc_ts::construct(iallocator_, &*(pointers_begin_ + i), i*max_num_non_zeros_per_row_);
			ialloc_ts::construct(iallocator_, &*(pointers_end_   + i), i*max_num_non_zeros_per_row_);
		//	iallocator_.construct(pointers_begin_ + i, i*max_num_non_zeros_per_row_);
		//	iallocator_.construct(pointers_end_   + i, i*max_num_non_zeros_per_row_);
		}
	}
	~fixed_csr_matrix(){
		for(index i = 0; i != size1_; ++i){
		//	for(auto p = data_ + pointers_begin_[i]; p != data_ + pointers_end_[i]; ++p)
		//		allocator_.destroy(p);
		//	for(auto p = jdata_ + pointers_begin_[i]; p != jdata_ + pointers_end_[i]; ++p) iallocator_.destroy(p);
		//	iallocator_.destroy(pointers_begin_ + i);
		//	iallocator_.destroy(pointers_end_ + i);
		}
		allocator_.deallocate(data_, size1_*max_num_non_zeros_per_row_);
		iallocator_.deallocate(jdata_, size1_*max_num_non_zeros_per_row_);
		iallocator_.deallocate(pointers_begin_, size1_);
		iallocator_.deallocate(pointers_end_  , size1_);
	}
	auto pointers_begin() const{return pointers_begin_;}
	auto pointers_end() const{return pointers_end_;}
	auto size() const{return size1_;}
	auto num_elements() const{return size()*size2_;}
	auto num_non_zero_elements() const{
		size_type ret = 0;
		for(auto i = 0; i != size(); ++i) 
			ret += pointers_end_[i] - pointers_begin_[i];
		return ret;
	}
	template<class Pair = std::array<index, 2>, class... Args>
	void emplace(Pair&& indices, Args&&... args){
		using std::get;
		if(pointers_end_[get<0>(indices)] - pointers_begin_[get<0>(indices)] < max_num_non_zeros_per_row_){
			allocator_.construct(&*(data_ + pointers_end_[get<0>(indices)]), std::forward<Args>(args)...);
			ialloc_ts::construct(iallocator_, &*(jdata_ + pointers_end_[get<0>(indices)]), get<1>(indices));
			++pointers_end_[get<0>(indices)];
		} else throw std::out_of_range("row size exceeded the maximum");
	}
	auto shape() const{return std::array<size_type, 2>{{size(),size2_}};}
	auto non_zero_values_data() const{return data_;}
	auto non_zero_indices2_data() const{return jdata_;}
	protected:
	struct row_reference{
		fixed_csr_matrix& self_;
		index i_;
		struct element_reference{
			row_reference& self_;
			index j_;
			template<class TT>
			element_reference&&
			operator=(TT&& tt)&&{
				self_.self_.emplace({{self_.i_, j_}}, std::forward<TT>(tt));
				return std::move(*this);
			}
		};
		using reference = element_reference;
		reference operator[](index i)&&{return reference{*this, i};}
	};
	
	public:
	using reference = row_reference;
	reference operator[](index i){return reference{*this, i};}
	friend decltype(auto) size(fixed_csr_matrix const& s){return s.size();}
	friend decltype(auto) shape(fixed_csr_matrix const& s){return s.shape();}
//	friend decltype(auto) clear(fixed_csr_matrix& s){s.clear();}
	friend auto index_bases(fixed_csr_matrix const& s){return std::array<index, 2>{{0, 0}};}
	friend auto non_zero_values_data(fixed_csr_matrix const& s){return s.non_zero_values_data();}
	friend auto non_zero_indices2_data(fixed_csr_matrix const& s){return s.non_zero_indices2_data();}
	friend auto pointers_begin(fixed_csr_matrix const& s){return s.pointers_begin();}
	friend auto pointers_end(fixed_csr_matrix const& s){return s.pointers_end();}
};

}
}

#ifdef _TEST_SPARSE_FIXED_CSR_MATRIX

#include "alf/boost/iterator/zipper.hpp"
#include "alf/boost/timer/timed.hpp"

#include <boost/timer/timer.hpp>

#include<algorithm> // std::sort
#include<cassert>
#include<iostream>
#include<random>

#include "alf/boost/mpi3/main.hpp"
#include "alf/boost/mpi3/shared_window.hpp"
#include "alf/boost/mpi3/shared_communicator.hpp"

using std::cout;
using std::cerr;
using std::get;

namespace mpi3 = boost::mpi3; using std::cout;

//int main(){
int boost::mpi3::main(int, char*[], mpi3::communicator& world){

	mpi3::shared_communicator node = world.split_shared();

	using ma::sparse::fixed_csr_matrix;

	{
		using Alloc = boost::mpi3::intranode::allocator<double>;
		Alloc A(node);
	//	Alloc::pointer p = A.allocate(10);
		fixed_csr_matrix<double, Alloc> med({4,4}, 2, A);
		node.barrier();
		if(node.rank() == 0) med[3][3] = 1;
		if(node.rank() == 1) med[2][1] = 3;
		if(node.rank() == 2) med[0][1] = 9;
		node.barrier();
		if(node.rank() == 0){
			for(int i = 0; i != 8; ++i) cout << med.non_zero_values_data()[i] << ' ';
			cout << '\n';
			for(int i = 0; i != 8; ++i) cout << med.non_zero_indices2_data()[i] << ' ';
			cout << '\n';
			for(int i = 0; i != 4; ++i) cout << med.pointers_begin()[i] << ' ';
			cout << '\n';
			for(int i = 0; i != 4; ++i) cout << med.pointers_end()[i] << ' ';
			cout << '\n';
		}
	}
	return 0;

	fixed_csr_matrix<double> small({4,4}, 2);
	small[3][3] = 1;
	small[2][1] = 3;
	small[0][1] = 9;
	small[0][2] = 9;

//	try{
//		small[3][0] = 8;
//	}catch(std::out_of_range& e){
//		assert(e.what() == std::string("row size exceeded the maximum"));
//	}
//	small[2][1] = 3;
//	small[0][1] = 9;
	for(int i = 0; i != 8; ++i) cout << small.non_zero_values_data()[i] << ' ';
	cout << '\n';
	for(int i = 0; i != 8; ++i) cout << small.non_zero_indices2_data()[i] << ' ';
	cout << '\n';
	for(int i = 0; i != 4; ++i) cout << small.pointers_begin()[i] << ' ';
	cout << '\n';
	for(int i = 0; i != 4; ++i) cout << small.pointers_end()[i] << ' ';
	cout << '\n';
	return 0;
}

#endif
#endif

