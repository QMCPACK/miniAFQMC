#if COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && mpicxx -O3 -I./ -I./alf/boost/ -I./alf/boost/mpi3/ -std=c++14 -Wall -Wfatal-errors -D_TEST_SPARSE_FIXED_CSR_MATRIX $0x.cpp -o $0x.x && time mpirun -np 2 $0x.x $@ && rm -f $0x.x $0x.cpp; exit
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
#include<iostream>
//#include<cstddef>  // ptrdiff_t
#include<vector>
#include<tuple>
#include<numeric>
#include<memory> 
#include<type_traits> // enable_if
#include<algorithm>

#include "Configuration.h"
#include "AFQMC/Utilities/tuple_iterator.hpp"

#include "mpi.h"

namespace ma{
namespace sparse{

using size_type           = std::size_t;
//using difference_type     = std::ptrdiff_t;
//using index               = std::ptrdiff_t;

template<class Allocator>
struct null_is_root{
	null_is_root(Allocator){}
	bool root(){return true;}
	int size(){return 1;}
	int rank(){return 0;}
	void barrier() {};
};

template<
        class ValType,
        class IndxType = int,
        class IntType = size_type,
	class ValType_alloc = std::allocator<ValType>
>
class csr_matrix_ref {
	public:
	using value_type = ValType;
	using index_type = IndxType;
	using int_type = IntType;
        protected:
        using IndxType_alloc = typename ValType_alloc::template rebind<IndxType>::other;
        using IntType_alloc = typename ValType_alloc::template rebind<IntType>::other;
        using this_t = csr_matrix_ref<ValType,IndxType,IntType,ValType_alloc>;
	using ValTypePtr = typename ValType_alloc::pointer;
	using IndxTypePtr = typename IndxType_alloc::pointer;
	using IntTypePtr = typename IntType_alloc::pointer;
        size_type size1_;
        size_type size2_;
	size_type index_base1_;
	size_type index_base2_;
        ValTypePtr data_;
        IndxTypePtr jdata_;
        IntTypePtr pointers_begin_;
        IntTypePtr pointers_end_;
        IntTypePtr max_num_non_zeros_per_row_;
        public:
	static const bool sparse = true;
	static const int dimensionality = 2;
        csr_matrix_ref(
                std::tuple<size_type, size_type> const& arr,
                std::tuple<size_type, size_type> const& ind,
                ValTypePtr __data_,
                IndxTypePtr __jdata_,
                IntTypePtr __pointers_begin_,
                IntTypePtr __pointers_end_,
                IntTypePtr __max_num_non_zeros_per_row_
        ) :
                size1_(std::get<0>(arr)), size2_(std::get<1>(arr)),
                index_base1_(std::get<0>(ind)), index_base2_(std::get<1>(ind)),
                data_(__data_),
                jdata_(__jdata_),
                pointers_begin_(__pointers_begin_),
                pointers_end_(__pointers_end_),
                max_num_non_zeros_per_row_(__max_num_non_zeros_per_row_)
        {}
	csr_matrix_ref(this_t const& other) = delete;
	csr_matrix_ref& operator=(this_t const& other) = delete;
	// pointer movement is handled by derived classes
        csr_matrix_ref(this_t&& other) = default; //:csr_matrix_ref() { }
        csr_matrix_ref& operator=(this_t&& other) = default; 
        ~csr_matrix_ref() {}
        auto pointers_begin() const{return pointers_begin_;}
        auto pointers_end() const{return pointers_end_;}
        auto index_bases() const{return std::array<size_type, 2>{{index_base1_,index_base2_}};;}
        auto size() const{return size1_;}
	template<typename integer_type>
        auto num_elements(integer_type i) const{
		if(max_num_non_zeros_per_row_ == IntTypePtr(nullptr))  return size_type(0);
		return static_cast<size_type>(max_num_non_zeros_per_row_[i]);
	}
        auto num_elements() const{
		if(max_num_non_zeros_per_row_ == IntTypePtr(nullptr))  return size_type(0);
		return std::accumulate(max_num_non_zeros_per_row_,
                                                           max_num_non_zeros_per_row_ + size1_,
                                                           size_type(0));}
        auto num_non_zero_elements() const{
                size_type ret = 0;
                for(size_type i = 0; i != size(); ++i)
                        ret += pointers_end_[i] - pointers_begin_[i];
                return ret;
        }
        auto shape() const{return std::array<size_type, 2>{{size(),size2_}};}
        auto non_zero_values_data() const{return data_;}
        auto non_zero_indices2_data() const{return jdata_;}
	auto release_max_num_non_zeros_per_row() {
		auto t = max_num_non_zeros_per_row_;
		max_num_non_zeros_per_row_ = IntTypePtr(nullptr);
		return t;
	}
	auto release_non_zero_values_data() {
		auto t = data_;
		data_ = ValTypePtr(nullptr);
		return t;
	}
	auto release_non_zero_indices2_data() {
		auto t = jdata_; 
		jdata_ = IndxTypePtr(nullptr);
		return t;
	}
	auto release_pointer_begin() {
		auto t = pointers_begin_;
		pointers_begin_ = IntTypePtr(nullptr);
		return t;
	}
	auto release_pointer_end() {
		auto t = pointers_end_;
		pointers_end_ = IntTypePtr(nullptr);
		return t;
	}
        friend decltype(auto) size(this_t const& s){return s.size();}
        friend decltype(auto) shape(this_t const& s){return s.shape();}
        friend auto index_bases(this_t const& s){return s.index_bases();}
	friend auto num_elements(this_t const& s){return s.num_elements();}
	friend auto num_non_zero_elements(this_t const& s){return s.num_non_zero_elements();}
        friend auto non_zero_values_data(this_t const& s){return s.non_zero_values_data();}
        friend auto non_zero_indices2_data(this_t const& s){return s.non_zero_indices2_data();}
        friend auto pointers_begin(this_t const& s){return s.pointers_begin();}
        friend auto pointers_end(this_t const& s){return s.pointers_end();}
};


template<
	class ValType,
	class IndxType = int,
	class IntType = size_type,    
	class ValType_alloc = std::allocator<ValType>, 
	class IsRoot = null_is_root<ValType_alloc> 
>
class ucsr_matrix: public csr_matrix_ref<ValType,IndxType,IntType,ValType_alloc>{
	public:
	using value_type = ValType;
	using index_type = IndxType;
	using int_type = IntType;
	protected:
	using this_t = ucsr_matrix<ValType,IndxType,IntType,ValType_alloc,IsRoot>;
	using base = csr_matrix_ref<ValType,IndxType,IntType,ValType_alloc>;
	using IndxType_alloc = typename ValType_alloc::template rebind<IndxType>::other;
	using IntType_alloc = typename ValType_alloc::template rebind<IntType>::other;
	using ValTypePtr = typename ValType_alloc::pointer;
	using IndxTypePtr = typename IndxType_alloc::pointer;
	using IntTypePtr = typename IntType_alloc::pointer;
	using Valloc_ts = std::allocator_traits<ValType_alloc>; 
	using Ialloc_ts = std::allocator_traits<IndxType_alloc>; 
	using Palloc_ts = std::allocator_traits<IntType_alloc>; 
	ValType_alloc Valloc_;
	IndxType_alloc Ialloc_;
	IntType_alloc Palloc_;
	public:
        static const bool sparse = true;
        static const int dimensionality = 2;
	template<typename integer_type>
	ucsr_matrix(
		std::tuple<size_type, size_type> const& arr = {0, 0}, 
		integer_type nnzpr_unique = 0,
		ValType_alloc alloc = ValType_alloc{}
	) : 
		csr_matrix_ref<ValType,IndxType,IntType,ValType_alloc>(arr,
			{0,0},
			ValTypePtr(nullptr),
			IndxTypePtr(nullptr),
			IntTypePtr(nullptr),
			IntTypePtr(nullptr),
			IntTypePtr(nullptr)),
		Valloc_(alloc), 
		Ialloc_(alloc),
		Palloc_(alloc)
	{
		this->data_ = Valloc_.allocate(std::get<0>(arr)*nnzpr_unique);
		this->jdata_ = Ialloc_.allocate(std::get<0>(arr)*nnzpr_unique);
		this->pointers_begin_ = Palloc_.allocate(std::get<0>(arr));
		this->pointers_end_ = Palloc_.allocate(std::get<0>(arr));
		this->max_num_non_zeros_per_row_ = Palloc_.allocate(std::get<0>(arr));

		IsRoot r(Valloc_);
		if(r.root()){
			for(size_type i = 0; i != base::size1_; ++i){
				Palloc_ts::construct(Palloc_, std::addressof(base::max_num_non_zeros_per_row_[i]), nnzpr_unique); 
				Palloc_ts::construct(Palloc_, std::addressof(base::pointers_begin_[i]), i*nnzpr_unique);
				Palloc_ts::construct(Palloc_, std::addressof(base::pointers_end_[i]), i*nnzpr_unique);
			}
		}
		r.barrier();
	}
	template<typename integer_type>
        ucsr_matrix(
                std::tuple<size_type, size_type> const& arr = {0, 0},
                std::vector<integer_type> const& nnzpr = std::vector<integer_type>(0),
                ValType_alloc alloc = ValType_alloc{}
        ) :
                csr_matrix_ref<ValType,IndxType,IntType,ValType_alloc>(arr,
			{0,0},
			ValTypePtr(nullptr),
			IndxTypePtr(nullptr),
			IntTypePtr(nullptr),
			IntTypePtr(nullptr),
			IntTypePtr(nullptr)),
                Valloc_(alloc),
                Ialloc_(alloc),
                Palloc_(alloc)
        {
		size_type sz = std::accumulate(nnzpr.begin(),nnzpr.end(),integer_type(0));
		this->data_ = Valloc_.allocate(std::get<0>(arr)*sz);
		this->jdata_ = Ialloc_.allocate(std::get<0>(arr)*sz);
		this->pointers_begin_ = Palloc_.allocate(std::get<0>(arr));
		this->pointers_end_ = Palloc_.allocate(std::get<0>(arr));
		this->max_num_non_zeros_per_row_ = Palloc_.allocate(std::get<0>(arr));

		assert(nnzpr.size() >= base::size1_);
                IsRoot r(Valloc_);
                if(r.root()){
			IntType cnter(0);
                        for(size_type i = 0; i != base::size1_; ++i){
                                Palloc_ts::construct(Palloc_, std::addressof(base::max_num_non_zeros_per_row_[i]), static_cast<IntType>(nnzpr[i]));
                                Palloc_ts::construct(Palloc_, std::addressof(base::pointers_begin_[i]), cnter); 
                                Palloc_ts::construct(Palloc_, std::addressof(base::pointers_end_[i]), cnter);
				cnter += static_cast<IntType>(nnzpr[i]); 
                        }
                }
		r.barrier();
        }
	~ucsr_matrix(){
		IsRoot r(Valloc_);
		if(r.root()){
			if(base::data_ == ValTypePtr(nullptr) ||
                                        base::jdata_ == IndxTypePtr(nullptr) ||
                                        base::pointers_begin_ == IntTypePtr(nullptr) ||        
                                        base::pointers_end_ == IntTypePtr(nullptr) ||          
                                        base::max_num_non_zeros_per_row_ == IntTypePtr(nullptr) ) {
				if(base::data_ != ValTypePtr(nullptr) ||
					base::jdata_ != IndxTypePtr(nullptr) || 
					base::pointers_begin_ != IntTypePtr(nullptr) ||  
					base::pointers_end_ != IntTypePtr(nullptr) ||  
					base::max_num_non_zeros_per_row_ != IntTypePtr(nullptr) )  
					//throw std::runtime_error("bad matrix state");
                                        APP_ABORT("Problems: Inconsistent csr_matrix state in destructor. This isa bug!!!\n");
			} else {
				size_type tot_sz = base::num_elements(); 
				for(size_type i = 0; i != base::size1_; ++i){
					for(auto p = base::data_ + base::pointers_begin_[i]; p != base::data_ + base::pointers_end_[i]; ++p)
						Valloc_.destroy(std::addressof(*p));
					for(auto p = base::jdata_ + base::pointers_begin_[i]; p != base::jdata_ + base::pointers_end_[i]; ++p) 
						Ialloc_.destroy(std::addressof(*p));
					Palloc_.destroy(std::addressof(base::pointers_begin_[i]));
					Palloc_.destroy(std::addressof(base::pointers_end_[i]));
					Palloc_.destroy(std::addressof(base::max_num_non_zeros_per_row_[i]));
				}
				Valloc_.deallocate(base::data_, tot_sz); 
				Ialloc_.deallocate(base::jdata_, tot_sz); 
				Palloc_.deallocate(base::max_num_non_zeros_per_row_, base::size1_);
				Palloc_.deallocate(base::pointers_begin_, base::size1_);
				Palloc_.deallocate(base::pointers_end_  , base::size1_);
			}
		}
		r.barrier();
	}
	ucsr_matrix(const this_t& other) = delete;  
	ucsr_matrix& operator=(const this_t& other) = delete;  
	ucsr_matrix(this_t&& other):ucsr_matrix({0,0},0,other.Valloc_)
	{ *this = std::move(other); } 
        // Instead of moving allocators, require they are the same right now
	ucsr_matrix& operator=(this_t&& other) {
		if(this != std::addressof(other)) {
			base::size1_ = other.size1_;
			base::size2_ = other.size2_;
                        if(Valloc_ != other.Valloc_ ||
                           Ialloc_ != other.Ialloc_ ||
                           Palloc_ != other.Palloc_ )
                            APP_ABORT(" Error: Can only move assign between csr_matrices with equivalent allocators. \n");
//			Valloc_ = other.Valloc_;
//	                Ialloc_ = other.Ialloc_;
//        	        Palloc_ = other.Palloc_;
			IsRoot r(Valloc_);
	                if(r.root()){
				size_type tot_sz = 0;
				if(base::max_num_non_zeros_per_row_!=IntTypePtr(nullptr)) {
					tot_sz = base::num_elements();
					Palloc_.deallocate(base::max_num_non_zeros_per_row_, base::size1_);
				}
				if(base::data_!=ValTypePtr(nullptr))
                        		Valloc_.deallocate(base::data_, tot_sz);
				if(base::jdata_!=IndxTypePtr(nullptr))
		                        Ialloc_.deallocate(base::jdata_, tot_sz);
				if(base::pointers_begin_!=IntTypePtr(nullptr))
		                        Palloc_.deallocate(base::pointers_begin_, base::size1_);
				if(base::pointers_end_!=IntTypePtr(nullptr))
		                        Palloc_.deallocate(base::pointers_end_  , base::size1_);
			}
			r.barrier();
                	base::max_num_non_zeros_per_row_ = std::exchange(other.max_num_non_zeros_per_row_, IntTypePtr(nullptr));
                	base::data_ = std::exchange(other.data_,ValTypePtr(nullptr));
                	base::jdata_ = std::exchange(other.jdata_,IndxTypePtr(nullptr));
                	base::pointers_begin_ = std::exchange(other.pointers_begin_,IntTypePtr(nullptr));
                	base::pointers_end_ = std::exchange(other.pointers_end_,IntTypePtr(nullptr));
		}
		return *this;
	} 
	auto getAlloc() { return Valloc_; }
        template<typename integer_type>
        void reserve(integer_type nnzpr_unique){
                if( static_cast<IntType>(nnzpr_unique) <= *std::min_element(base::max_num_non_zeros_per_row_,
                                                      base::max_num_non_zeros_per_row_+base::size1_) )
                        return;
                this_t other({base::size1_,base::size2_},nnzpr_unique,Valloc_);
                IsRoot r(Valloc_);
                if(r.root()){
                        for(size_type i=0; i<base::size1_; i++)
			{
				size_type disp = static_cast<size_type>(base::pointers_end_[i]-
									base::pointers_begin_[i]); 
				std::copy_n(std::addressof(base::data_[base::pointers_begin_[i]]),
					    disp,
	       				    std::addressof(other.data_[other.pointers_begin_[i]]));
				std::copy_n(std::addressof(base::jdata_[base::pointers_begin_[i]]),
					    disp,
	       				    std::addressof(other.jdata_[other.pointers_begin_[i]]));
				other.pointers_end_[i] = other.pointers_begin_[i] + disp;
			}
                }
                r.barrier();
                *this = std::move(other);
        }
        template<typename integer_type>
        void reserve(std::vector<integer_type>& nnzpr){
                bool resz = false;
                assert(nnzpr.size() >= base::size1_);
                for(size_type i=0; i<base::size1_; i++)
                        if(static_cast<IntType>(nnzpr[i]) > base::max_num_non_zeros_per_row_[i]) {
                                resz=true;
                                break;
                        }
                if(not resz)
                        return;
                this_t other({base::size1_,base::size2_},nnzpr,Valloc_);
                IsRoot r(Valloc_);
                if(r.root()){
                        for(size_type i=0; i<base::size1_; i++)
                        {
                                size_type disp = static_cast<size_type>(base::pointers_end_[i]-
                                                                        base::pointers_begin_[i]);
                                std::copy_n(std::addressof(base::data_[base::pointers_begin_[i]]),
                                            disp,
                                            std::addressof(other.data_[other.pointers_begin_[i]]));
                                std::copy_n(std::addressof(base::jdata_[base::pointers_begin_[i]]),
                                            disp,
                                            std::addressof(other.jdata_[other.pointers_begin_[i]]));
                                other.pointers_end_[i] = other.pointers_begin_[i] + disp;
                        }
                }
                r.barrier();
                *this = std::move(other);
        }
	template<class Pair = std::array<IndxType, 2>, class... Args>
	void emplace(Pair&& indices, Args&&... args){
		using std::get;
		if(base::pointers_end_[get<0>(indices)] - base::pointers_begin_[get<0>(indices)] < base::max_num_non_zeros_per_row_[get<0>(indices)]){
			Valloc_ts::construct(Valloc_,std::addressof(base::data_[base::pointers_end_[get<0>(indices)]]), std::forward<Args>(args)...);
			Ialloc_ts::construct(Ialloc_, std::addressof(base::jdata_[base::pointers_end_[get<0>(indices)]]), get<1>(indices));
			++base::pointers_end_[get<0>(indices)];
		} else throw std::out_of_range("row size exceeded the maximum");
	}
	protected:
	struct row_reference{
		ucsr_matrix& self_;
		IndxType i_;
		struct element_reference{
			row_reference& self_;
			IndxType j_;
			template<class TT>
			element_reference&&
			operator=(TT&& tt)&&{
				self_.self_.emplace({{self_.i_, j_}}, std::forward<TT>(tt));
				return std::move(*this);
			}
		};
		using reference = element_reference;
		reference operator[](IndxType i)&&{return reference{*this, i};}
	};
	public:
	using reference = row_reference;
	template<typename integer_type>
	reference operator[](integer_type i){return reference{*this, static_cast<IndxType>(i)};}
	ValType_alloc& getValloc() {return Valloc_;}
	IndxType_alloc& getIalloc() {return Ialloc_;}
	IntType_alloc& getPalloc() {return Palloc_;}
};

template<
        class ValType,
        class IndxType = int,
        class IntType = size_type,   
        class ValType_alloc = std::allocator<ValType>,
        class IsRoot = null_is_root<ValType_alloc>
>
class csr_matrix: public ucsr_matrix<ValType,IndxType,IntType,ValType_alloc,IsRoot> 
{
	using base = ucsr_matrix<ValType,IndxType,IntType,ValType_alloc,IsRoot>;
	using this_t = csr_matrix<ValType,IndxType,IntType,ValType_alloc,IsRoot>;
	public:
	using value_type = ValType;
	using index_type = IndxType;
	using int_type = IntType;
	using ValTypePtr = typename ValType_alloc::pointer;
	using IndxTypePtr = typename base::IndxType_alloc::pointer;
	using IntTypePtr = typename base::IntType_alloc::pointer;
        static const bool sparse = true;
        static const int dimensionality = 2;
	template<typename integer_type>
        csr_matrix(
                std::tuple<size_type, size_type> const& arr = {0, 0},
                integer_type nnzpr_unique = 0,
                ValType_alloc alloc = ValType_alloc{}
        ):base(arr,nnzpr_unique,alloc)
	{}
        template<typename integer_type>
        csr_matrix(
                std::tuple<size_type, size_type> const& arr = {0, 0},
                std::vector<integer_type>& nnzpr = std::vector<integer_type>(0),
                ValType_alloc alloc = ValType_alloc{}
        ):base(arr,nnzpr,alloc)
        {}
	csr_matrix(this_t const& ucsr) = delete;
	csr_matrix& operator=(this_t const& ucsr) = delete;
	csr_matrix(this_t&& other):csr_matrix({0,0},0,other.Valloc_) { *this = std::move(other); }
	csr_matrix(ucsr_matrix<ValType,IndxType,IntType,ValType_alloc,IsRoot>&& ucsr):
		csr_matrix({0,0},0,ucsr.getAlloc()) {
		*this = std::move(ucsr);
	}
        csr_matrix& operator=(csr_matrix<ValType,IndxType,IntType,ValType_alloc,IsRoot>&& other) {
                base::size1_ = other.size1_;
                base::size2_ = other.size2_;
                if(base::Valloc_ != other.Valloc_ ||
                   base::Ialloc_ != other.Ialloc_ ||
                   base::Palloc_ != other.Palloc_ )
                        APP_ABORT(" Error: Can only move assign between csr_matrices with equivalent allocators. \n");
//                base::Valloc_ = std::move(other.Valloc_);
//                base::Ialloc_ = std::move(other.Ialloc_);
//                base::Palloc_ = std::move(other.Palloc_);
                IsRoot r(base::Valloc_);
                if(r.root()){
                        size_type tot_sz = 0; 
                        if(base::max_num_non_zeros_per_row_!=IntTypePtr(nullptr)) {
				tot_sz = base::num_elements();
                                base::Palloc_.deallocate(base::max_num_non_zeros_per_row_, base::size1_);
			}
                        if(base::data_!=ValTypePtr(nullptr))
                                base::Valloc_.deallocate(base::data_, tot_sz);
                        if(base::jdata_!=IndxTypePtr(nullptr))
                                base::Ialloc_.deallocate(base::jdata_, tot_sz);
                        if(base::pointers_begin_!=IntTypePtr(nullptr))
                                base::Palloc_.deallocate(base::pointers_begin_, base::size1_);
                        if(base::pointers_end_!=IntTypePtr(nullptr))
                                base::Palloc_.deallocate(base::pointers_end_  , base::size1_);
                }
                r.barrier();
                base::max_num_non_zeros_per_row_ = std::exchange(other.max_num_non_zeros_per_row_, IntTypePtr(nullptr));
                base::data_ = std::exchange(other.data_,ValTypePtr(nullptr));
                base::jdata_ = std::exchange(other.jdata_,IndxTypePtr(nullptr));
                base::pointers_begin_ = std::exchange(other.pointers_begin_,IntTypePtr(nullptr));
                base::pointers_end_ = std::exchange(other.pointers_end_,IntTypePtr(nullptr));
                r.barrier();
                return *this;
        }
	csr_matrix& operator=(ucsr_matrix<ValType,IndxType,IntType,ValType_alloc,IsRoot>&& other) {
		base::size1_ = other.shape()[0]; 
		base::size2_ = other.shape()[1]; 
                if(base::Valloc_ != other.getValloc() ||
                   base::Ialloc_ != other.getIalloc() ||
                   base::Palloc_ != other.getPalloc() )
                        APP_ABORT(" Error: Can only move assign between csr_matrices with equivalent allocators. \n");
//                base::Valloc_ = other.getValloc();
//                base::Ialloc_ = other.getIalloc(); 
//                base::Palloc_ = other.getPalloc(); 
                IsRoot r(base::Valloc_);
                if(r.root()){
                        size_type tot_sz = 0; 
                        if(base::max_num_non_zeros_per_row_!=IntTypePtr(nullptr)) {
                        	tot_sz = base::num_elements();
                                base::Palloc_.deallocate(base::max_num_non_zeros_per_row_, base::size1_);
			}
                        if(base::data_!=ValTypePtr(nullptr))
                                base::Valloc_.deallocate(base::data_, tot_sz);
                        if(base::jdata_!=IndxTypePtr(nullptr))
                                base::Ialloc_.deallocate(base::jdata_, tot_sz);
                        if(base::pointers_begin_!=IntTypePtr(nullptr))
                                base::Palloc_.deallocate(base::pointers_begin_, base::size1_);
                        if(base::pointers_end_!=IntTypePtr(nullptr))
                                base::Palloc_.deallocate(base::pointers_end_  , base::size1_);
                }
                r.barrier();
                base::max_num_non_zeros_per_row_ = other.release_max_num_non_zeros_per_row();
                base::data_ = other.release_non_zero_values_data();
                base::jdata_ = other.release_non_zero_indices2_data(); 
                base::pointers_begin_ = other.release_pointer_begin(); 
                base::pointers_end_ = other.release_pointer_end(); 
		using qmcplusplus::make_paired_iterator;
		for(size_type p=0; p<base::size1_; p++) {
			if(p%static_cast<size_type>(r.size()) == static_cast<size_type>(r.rank())) {
				auto i1 = base::pointers_begin_[p];
				auto i2 = base::pointers_end_[p];
				std::sort(make_paired_iterator(std::addressof(base::jdata_[i1]),std::addressof(base::data_[i1])),
					  make_paired_iterator(std::addressof(base::jdata_[i2]),std::addressof(base::data_[i2])),
					  [](auto const& a, auto const& b) {
						return std::get<0>(a)<std::get<0>(b);
					  });
			}	
		}	
		r.barrier();
		return *this;
	}
	template<class Pair = std::array<IndxType, 2>, class... Args>
        void emplace(Pair&& indices, Args&&... args){
                using std::get;
                if(base::pointers_end_[get<0>(indices)] - base::pointers_begin_[get<0>(indices)] < base::max_num_non_zeros_per_row_[get<0>(indices)]){
			auto loc = std::lower_bound(std::addressof(base::jdata_[base::pointers_begin_[get<0>(indices)]]),
						    std::addressof(base::jdata_[base::pointers_end_[get<0>(indices)]]),
						    get<1>(indices));
			size_type disp = std::distance(std::addressof(base::jdata_[base::pointers_begin_[get<0>(indices)]]),std::addressof(*loc));
			size_type disp_ = std::distance(std::addressof(*loc),std::addressof(base::jdata_[base::pointers_end_[get<0>(indices)]]));
			if( disp_ > 0 && *loc == get<1>(indices)) { 
				// value exists, construct in place 
                        	base::Valloc_ts::construct(base::Valloc_,std::addressof(base::data_[base::pointers_begin_[get<0>(indices)] + disp]), std::forward<Args>(args)...);
			} else {
				// new value, shift back and add in correct place
				if(disp_ > 0) {
					std::move_backward(std::addressof(base::data_[base::pointers_begin_[get<0>(indices)] + disp]), 
					   std::addressof(base::data_[base::pointers_end_[get<0>(indices)]]),
					   std::addressof(base::data_[base::pointers_end_[get<0>(indices)] + 1]));
					std::move_backward(std::addressof(base::jdata_[base::pointers_begin_[get<0>(indices)] + disp]), 
                                           std::addressof(base::jdata_[base::pointers_end_[get<0>(indices)]]),
                                           std::addressof(base::jdata_[base::pointers_end_[get<0>(indices)] + 1]));
				}
                        	++base::pointers_end_[get<0>(indices)];
                        	base::Valloc_ts::construct(base::Valloc_,std::addressof(base::data_[base::pointers_begin_[get<0>(indices)] + disp]), std::forward<Args>(args)...);
                        	base::Ialloc_ts::construct(base::Ialloc_, std::addressof(base::jdata_[base::pointers_begin_[get<0>(indices)] + disp]), get<1>(indices));
			}
                } else throw std::out_of_range("row size exceeded the maximum");
        }	
	void remove_empty_spaces() {
		IsRoot r(base::Valloc_);
		if(base::num_non_zero_elements() == base::num_elements()) {
                	r.barrier();
			return;
		}
                r.barrier();
                if(r.root()){
			for(size_type i=0; i<base::size1_-1; i++) {
				auto ni = static_cast<size_type>(base::pointers_end_[i]-base::pointers_begin_[i]);
				if(ni == base::max_num_non_zeros_per_row_[i]) continue;
				auto nip1 = static_cast<size_type>(base::pointers_end_[i+1]-base::pointers_begin_[i+1]);
				std::move(std::addressof(base::data_[base::pointers_begin_[i+1]]),
					  std::addressof(base::data_[base::pointers_end_[i+1]]),
					  std::addressof(base::data_[base::pointers_end_[i]]));
				std::move(std::addressof(base::jdata_[base::pointers_begin_[i+1]]),
					  std::addressof(base::jdata_[base::pointers_end_[i+1]]),
					  std::addressof(base::jdata_[base::pointers_end_[i]]));
				base::pointers_begin_[i+1] = base::pointers_end_[i];	
				base::pointers_end_[i+1] = base::pointers_begin_[i+1]+nip1;	
				base::max_num_non_zeros_per_row_[i+1] += base::max_num_non_zeros_per_row_[i]-IntType(ni);
				base::max_num_non_zeros_per_row_[i] = IntType(ni); 
			}
                }
                r.barrier();
	}
        protected:
        struct row_reference{
                this_t& self_;
                IndxType i_;
                struct element_reference{
                        row_reference& self_;
                        IndxType j_;
                        template<class TT>
                        element_reference&&
                        operator=(TT&& tt)&&{
                                self_.self_.emplace({{self_.i_, j_}}, std::forward<TT>(tt));
                                return std::move(*this);
                        }
                };
                using reference = element_reference;
		template<typename integer_type>
                reference operator[](integer_type i)&&{return reference{*this, static_cast<IndxType>(i)};}
        };

        public:
        using reference = row_reference;
	template<typename integer_type>
        reference operator[](integer_type i){return reference{*this, static_cast<IndxType>(i)};}
	using matrix_view = csr_matrix_ref<ValType,IndxType,IntType,ValType_alloc>;
	template<typename integer_type>
	matrix_view operator[](std::array<integer_type,4>& arr) {
		// limited right now
		assert(arr[0]>=0 && arr[1] <= integer_type(base::size1_));
		assert(arr[2]>=0 && arr[3] <= integer_type(base::size2_));
		assert(arr[0] < arr[1]);
		assert(arr[2] < arr[3]);
		// just row partitions right now
		assert(arr[2]==0 && arr[3] == integer_type(base::size2_));
		
	        size_type disp = static_cast<size_type>(base::pointers_begin_[arr[0]]-base::pointers_begin_[0]);
		// Note: This depends on how the view is used/interpreted.
		// e.g. MKL takes as the pointer values ptrb[i]-ptrb[0], so data/jdata
		//      has to be shifted by disp.
		//      If the shift in ptrb is not done, (e.g. some other library),
		//      then data/jdata should be the same (unshifted) value as *this.
		//      cuSparse uses a 3-index format with ptr shift, e.g. ptrb[i]-ptrb[0]
		return matrix_view({base::size1_-(arr[1]-arr[0]),base::size2_},
			{arr[0],arr[2]},
			base::data_ + disp,
			base::jdata_ + disp,
			base::pointers_begin_ + static_cast<size_type>(arr[0]),
			base::pointers_end_ + static_cast<size_type>(arr[0]),
			base::max_num_non_zeros_per_row_ + static_cast<size_type>(arr[0])
		);
	} 

};

}
}

#ifdef _TEST_SPARSE_FIXED_CSR_MATRIX

//#include "alf/boost/iterator/zipper.hpp"
//#include "alf/boost/timer/timed.hpp"

#include <boost/timer/timer.hpp>

#include<algorithm> // std::sort
#include<cassert>
#include<iostream>
#include<random>

//#include "alf/boost/mpi3/main.hpp"
//#include "alf/boost/mpi3/shared_window.hpp"
//#include "alf/boost/mpi3/shared_communicator.hpp"
#include "alf/boost/mpi3_new/main.hpp"
#include "alf/boost/mpi3_new/shared_window.hpp"
#include "alf/boost/mpi3_new/shared_communicator.hpp"

using std::endl;
using std::cout;
using std::cerr;
using std::get;

namespace mpi3 = boost::mpi3; using std::cout;

//int main(){
int boost::mpi3::main(int, char*[], mpi3::communicator& world){

	mpi3::shared_communicator node = world.split_shared();

	using Type = double;

/*
	using Alloc = std::allocator<Type>;
	using ucsr_matrix = ma::sparse::ucsr_matrix<Type,int,std::size_t,Alloc>;
	using csr_matrix = ma::sparse::csr_matrix<Type,int,std::size_t,Alloc>;
	bool serial = true;
	Alloc A();
*/
	using Alloc = boost::mpi3::intranode::allocator<Type>; 
	using is_root = boost::mpi3::intranode::is_root;
	using ucsr_matrix = ma::sparse::ucsr_matrix<Type,int,std::size_t,Alloc,is_root>;
	using csr_matrix = ma::sparse::csr_matrix<Type,int,std::size_t,Alloc,is_root>;
	bool serial = false;
	Alloc A(node);

  	std::vector<Type> v_ = {9,10,3,1};
	auto itv = v_.begin();
  	std::vector<int> c_ = {2,1,1,3};
	auto itc = c_.begin();
	std::vector<int> non_zero_per_row = {2,0,1,1};
	std::vector<int> max_non_zero_per_row = {5,3,4,2};
	
	{
		ucsr_matrix small({4,4}, 2, A);
		if(serial || node.rank()==0) small[3][3] = 1;
		if(serial || node.rank()==0) small[0][2] = 9;
		node.barrier();
		if(serial || node.rank()==node.size()-1) small[2][1] = 3;
		if(serial || node.rank()==node.size()-1) small[0][1] = 10;
		node.barrier();

		assert(small.num_non_zero_elements() == 4);
		auto val = small.non_zero_values_data();
		auto col = small.non_zero_indices2_data();
		for(std::size_t i=0; i<small.shape()[0]; i++) {
		  for(auto it=small.pointers_begin()[i]; it!=small.pointers_end()[i]; it++) {
		    assert(val[it] == *(itv++));	
		    assert(col[it] == *(itc++));	
		  }	
		}	

	        node.barrier();

	}

        {
                ucsr_matrix small({4,4}, non_zero_per_row, A);
                if(serial || node.rank()==0) small[3][3] = 1;
                if(serial || node.rank()==0) small[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small[0][1] = 10;
                node.barrier();

                assert(small.num_non_zero_elements() == 4);
                auto val = small.non_zero_values_data();
                auto col = small.non_zero_indices2_data();
		itv = v_.begin();
		itc = c_.begin();
                for(std::size_t i=0; i<small.shape()[0]; i++) {
                  for(auto it=small.pointers_begin()[i]; it!=small.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                node.barrier();

        }

        {
                ucsr_matrix small({4,4}, max_non_zero_per_row, A);
                if(serial || node.rank()==0) small[3][3] = 1;
                if(serial || node.rank()==0) small[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small[0][1] = 10;
                node.barrier();

                assert(small.num_non_zero_elements() == 4);
                auto val = small.non_zero_values_data();
                auto col = small.non_zero_indices2_data();
		itv = v_.begin();
		itc = c_.begin();
                for(std::size_t i=0; i<small.shape()[0]; i++) {
                  for(auto it=small.pointers_begin()[i]; it!=small.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                node.barrier();

        }

        {
                ucsr_matrix small({4,4}, 2, A);
                if(serial || node.rank()==0) small[3][3] = 1;
                if(serial || node.rank()==0) small[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small[0][1] = 10;
                node.barrier();

                assert(small.num_non_zero_elements() == 4);
                auto val = small.non_zero_values_data();
                auto col = small.non_zero_indices2_data();
		itv = v_.begin();
		itc = c_.begin();
                for(std::size_t i=0; i<small.shape()[0]; i++) {
                  for(auto it=small.pointers_begin()[i]; it!=small.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                node.barrier();

		ucsr_matrix small2(std::move(small));
                assert(small2.num_non_zero_elements() == 4);
                val = small2.non_zero_values_data();
                col = small2.non_zero_indices2_data();
		itv = v_.begin();
		itc = c_.begin();
                for(std::size_t i=0; i<small2.shape()[0]; i++) {
                  for(auto it=small2.pointers_begin()[i]; it!=small2.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                node.barrier();

		ucsr_matrix small3({0,0},0, A);
		small3 = std::move(small2);
                assert(small3.num_non_zero_elements() == 4);
                val = small3.non_zero_values_data();
                col = small3.non_zero_indices2_data();
		itv = v_.begin();
		itc = c_.begin();
                for(std::size_t i=0; i<small3.shape()[0]; i++) {
                  for(auto it=small3.pointers_begin()[i]; it!=small3.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }
                
                node.barrier();
        	// ordered
  		v_ = {10,9,3,1};
  		c_ = {1,2,1,3};
                csr_matrix small4(std::move(small3));
                assert(small4.num_non_zero_elements() == 4);
                val = small4.non_zero_values_data();
                col = small4.non_zero_indices2_data();
                itv = v_.begin();
                itc = c_.begin();
                for(std::size_t i=0; i<small4.shape()[0]; i++) {
                  for(auto it=small4.pointers_begin()[i]; it!=small4.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                ucsr_matrix small5({4,4}, non_zero_per_row, A);
                if(serial || node.rank()==0) small5[3][3] = 1;
                if(serial || node.rank()==0) small5[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small5[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small5[0][1] = 10;
                node.barrier();
		small4.reserve(max_non_zero_per_row);
		small4=std::move(small5);
                assert(small4.num_non_zero_elements() == 4);
                val = small4.non_zero_values_data();
                col = small4.non_zero_indices2_data();
                itv = v_.begin();
                itc = c_.begin();
                for(std::size_t i=0; i<small4.shape()[0]; i++) {
                  for(auto it=small4.pointers_begin()[i]; it!=small4.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                ucsr_matrix small6({4,4}, max_non_zero_per_row, A);
                if(serial || node.rank()==0) small6[3][3] = 1;
                if(serial || node.rank()==0) small6[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small6[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small6[0][1] = 10;
                node.barrier();
                small4.reserve(100);
		small4=std::move(small6);
                assert(small4.num_non_zero_elements() == 4);
                val = small4.non_zero_values_data();
                col = small4.non_zero_indices2_data();
                itv = v_.begin();
                itc = c_.begin();
                for(std::size_t i=0; i<small4.shape()[0]; i++) {
                  for(auto it=small4.pointers_begin()[i]; it!=small4.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                csr_matrix small7({4,4}, max_non_zero_per_row, A);
                if(serial || node.rank()==0) small7[3][3] = 1;
                if(serial || node.rank()==0) small7[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small7[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small7[0][1] = 10;
                node.barrier();
		small7.remove_empty_spaces();
                assert(small7.num_non_zero_elements() == 4);
                val = small7.non_zero_values_data();
                col = small7.non_zero_indices2_data();
                itv = v_.begin();
                itc = c_.begin();
                for(std::size_t i=0; i<small7.shape()[0]; i++) {
		  if(i < small7.shape()[0]-1) 
		    assert(small7.pointers_end()[i]-small7.pointers_begin()[i] == small7.num_elements(i));
                  for(auto it=small7.pointers_begin()[i]; it!=small7.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }
        }

        // ordered
  	v_ = {10,9,3,1};
  	c_ = {1,2,1,3};

        {
                csr_matrix small({4,4}, 2, A);
                if(serial || node.rank()==0) small[3][3] = 1;
                if(serial || node.rank()==0) small[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small[0][1] = 10;
                node.barrier();

                assert(small.num_non_zero_elements() == 4);
                auto val = small.non_zero_values_data();
                auto col = small.non_zero_indices2_data();
		itv = v_.begin();
		itc = c_.begin();
                for(std::size_t i=0; i<small.shape()[0]; i++) {
                  for(auto it=small.pointers_begin()[i]; it!=small.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                node.barrier();

		std::array<int,4> range = {0,2,0,4};
		auto small2 = small[range];
                assert(small2.num_non_zero_elements() == 2);
                val = small2.non_zero_values_data();
                col = small2.non_zero_indices2_data();
                itv = v_.begin();
                itc = c_.begin();
		auto i0 = small2.pointers_begin()[0];
                for(std::size_t i=0; i<small2.shape()[0]; i++) {
                  for(auto it=small2.pointers_begin()[i]; it!=small2.pointers_end()[i]; it++) {
                    assert(val[it-i0] == *(itv++));
                    assert(col[it-i0] == *(itc++));
                  }
                }

		range = {2,4,0,4};
		auto small3 = small[range];
                assert(small3.num_non_zero_elements() == 2);
                val = small3.non_zero_values_data();
                col = small3.non_zero_indices2_data();
                itv = v_.begin()+2;
                itc = c_.begin()+2;
		i0 = small3.pointers_begin()[0];
                for(std::size_t i=0; i<small3.shape()[0]; i++) {
                  for(auto it=small3.pointers_begin()[i]; it!=small3.pointers_end()[i]; it++) {
                    assert(val[it-i0] == *(itv++));
                    assert(col[it-i0] == *(itc++));
                  }
                }

                node.barrier();

        }

        {
                csr_matrix small({4,4}, non_zero_per_row, A);
                if(serial || node.rank()==0) small[3][3] = 1;
                if(serial || node.rank()==0) small[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small[0][1] = 10;
                node.barrier();

                assert(small.num_non_zero_elements() == 4);
                auto val = small.non_zero_values_data();
                auto col = small.non_zero_indices2_data();
                itv = v_.begin();
                itc = c_.begin();
                for(std::size_t i=0; i<small.shape()[0]; i++) {
                  for(auto it=small.pointers_begin()[i]; it!=small.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                node.barrier();

        }

        {
                csr_matrix small({4,4}, max_non_zero_per_row, A);
                if(serial || node.rank()==0) small[3][3] = 1;
                if(serial || node.rank()==0) small[0][2] = 9;
                node.barrier();
                if(serial || node.rank()==node.size()-1) small[2][1] = 3;
                if(serial || node.rank()==node.size()-1) small[0][1] = 10;
                node.barrier();

                assert(small.num_non_zero_elements() == 4);
                auto val = small.non_zero_values_data();
                auto col = small.non_zero_indices2_data();
		itv = v_.begin();
		itc = c_.begin();
                for(std::size_t i=0; i<small.shape()[0]; i++) {
                  for(auto it=small.pointers_begin()[i]; it!=small.pointers_end()[i]; it++) {
                    assert(val[it] == *(itv++));
                    assert(col[it] == *(itc++));
                  }
                }

                node.barrier();

        }

	return 0;
}

#endif
#endif

