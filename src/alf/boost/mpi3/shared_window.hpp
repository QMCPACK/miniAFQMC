#if COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && mpic++ -O3 -std=c++14 -Wall -Wfatal-errors -D_TEST_BOOST_MPI3_SHARED_WINDOW $0x.cpp -o $0x.x && time mpirun -np 3 $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MPI3_SHARED_WINDOW_HPP
#define BOOST_MPI3_SHARED_WINDOW_HPP

#include "../mpi3/shared_communicator.hpp"
#include "../mpi3/dynamic_window.hpp"

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include<mpi.h>

#ifdef __bgq__
#include "Utilities/simple_stack_allocator.hpp"
extern std::unique_ptr<simple_stack_allocator> bgq_dummy_allocator;
extern std::size_t BG_SHM_ALLOCATOR;
#endif

namespace boost{
namespace mpi3{

template<class T /*= void*/>
struct shared_window : window<T>{
	shared_window(shared_communicator& comm, mpi3::size_t n, int disp_unit = sizeof(T)) : 
		window<T>()
	{
		void* base_ptr = nullptr;
		int s = MPI_Win_allocate_shared(n*sizeof(T), disp_unit, MPI_INFO_NULL, comm.impl_, &base_ptr, &this->impl_);
		if(s != MPI_SUCCESS) throw std::runtime_error("cannot create shared window");
	}
	shared_window(shared_communicator& comm, int disp_unit = sizeof(T)) : 
		shared_window(comm, 0, disp_unit)
	{}
	using query_t = std::tuple<mpi3::size_t, int, void*>;
	query_t query(int rank = MPI_PROC_NULL) const{
		query_t ret;
		MPI_Win_shared_query(this->impl_, rank, &std::get<0>(ret), &std::get<1>(ret), &std::get<2>(ret));
		return ret;
	}
	template<class TT = T>
	mpi3::size_t size(int rank = 0) const{
		return std::get<0>(query(rank))/sizeof(TT);
	}
	int disp_unit(int rank = 0) const{
		return std::get<1>(query(rank));
	}
	template<class TT = T>
	TT* base(int rank = 0) const{return static_cast<TT*>(std::get<2>(query(rank)));}
//	template<class T = char>
//	void attach_n(T* base, mpi3::size_t n){MPI_Win_attach(impl_, base, n*sizeof(T));}
};

#if 0
struct managed_shared_memory{
	shared_window<> sw_;
	managed_shared_memory(shared_communicator& c, int s) : sw_(c, c.rank()==0?s:0){}
//	struct segment_manager{};
	using segment_manager = boost::interprocess::segment_manager<char, boost::interprocess::rbtree_best_fit<boost::interprocess::mutex_family>, boost::interprocess::iset_index>;
	segment_manager sm_;
	managed_shared_memory::segment_manager* get_segment_manager(){
		return &sm_;
	}
};
#endif

template<class T /*= char*/> 
shared_window<T> shared_communicator::make_shared_window(
	mpi3::size_t size
){
	return shared_window<T>(*this, size);
}

template<class T /*= char*/>
shared_window<T> shared_communicator::make_shared_window(){
	return shared_window<T>(*this);//, sizeof(T));
}

namespace intranode{

template<class T> struct array_ptr;

// BGQ hack!!!
#ifdef __bgq__

template<>
struct array_ptr<const void>{
        using T = const void;
        T* base;
        std::ptrdiff_t offset = 0; 
        array_ptr(std::nullptr_t = nullptr){}
        array_ptr(array_ptr const& other) = default; 
        array_ptr& operator=(array_ptr const& other) = default;
};

template<>
struct array_ptr<void>{
        using T = void;
        T* base;
        std::ptrdiff_t offset = 0; 
        array_ptr(std::nullptr_t = nullptr){}
        array_ptr(array_ptr const& other) = default; 
        array_ptr& operator=(array_ptr const& other) = default;
};

template<class T>
struct array_ptr{
        T* base = nullptr;
        std::ptrdiff_t offset = 0;
        array_ptr(){}
        array_ptr(std::nullptr_t){}
        array_ptr(array_ptr const& other) = default;
        array_ptr& operator=(array_ptr const& other) = default;
        T& operator*() const{return *(base + offset);}
        T& operator[](mpi3::size_t idx) const{return (base + offset)[idx];}
        T* operator->() const{return base + offset;}
        explicit operator bool() const{return (bool)(base!=nullptr);}//.get();}
        operator array_ptr<void const>() const{
                array_ptr<void const> ret;
                ret.base = base;
                return ret;
        }
        array_ptr operator+(std::ptrdiff_t d) const{
                array_ptr ret(*this);
                ret += d;
                return ret;
        }
        std::ptrdiff_t operator-(array_ptr other) const{
                return offset - other.offset;
        }
        array_ptr& operator--(){--offset; return *this;}
        array_ptr& operator++(){++offset; return *this;}
        array_ptr& operator-=(std::ptrdiff_t d){offset -= d; return *this;}
        array_ptr& operator+=(std::ptrdiff_t d){offset += d; return *this;}
        bool operator==(array_ptr<T> const& other) const{
                if(base == nullptr) return other.base == nullptr;
                if(other.base == nullptr) return base == nullptr;
                return base == other.base and offset == other.offset;
        }
        bool operator!=(array_ptr<T> const& other) const{return not((*this)==other);}
        bool operator<=(array_ptr<T> const& other) const{
                return base + offset <= other.base + other.offset;
        }
};

#else
template<>
struct array_ptr<const void>{
	using T = const void;
	std::shared_ptr<shared_window<>> wSP_;
	std::ptrdiff_t offset = 0;
	array_ptr(std::nullptr_t = nullptr){}
	array_ptr(array_ptr const& other) = default;
	array_ptr& operator=(array_ptr const& other) = default;
};

template<>
struct array_ptr<void>{
	using T = void;
	std::shared_ptr<shared_window<>> wSP_;
	std::ptrdiff_t offset = 0;
	array_ptr(std::nullptr_t = nullptr){}
	array_ptr(array_ptr const& other) = default;
	array_ptr& operator=(array_ptr const& other) = default;
};

template<class T>
struct array_ptr{
	std::shared_ptr<shared_window<T>> wSP_;
	std::ptrdiff_t offset = 0;
	array_ptr(){}
	array_ptr(std::nullptr_t){}
//	array_ptr(std::nullptr_t = nullptr) : offset(0){}
	array_ptr(array_ptr const& other) = default;
//	array_ptr(T* const& other = nullptr) : offset(0){}
//	array_ptr(T* const& other = nullptr) : offset(0){}
	array_ptr& operator=(array_ptr const& other) = default;
	T& operator*() const{return *((T*)(wSP_->base(0)) + offset);}
	T& operator[](mpi3::size_t idx) const{return ((T*)(wSP_->base(0)) + offset)[idx];}
	T* operator->() const{return (T*)(wSP_->base(0)) + offset;}
//	T* get() const{return wSP_->base(0) + offset;}
	explicit operator bool() const{return (bool)wSP_;}//.get();}
//	explicit operator T*() const{return (T*)(wSP_->base(0)) + offset;}//.get();}
//	operator T*() const{return (T*)(wSP_->base(0)) + offset;}//.get();}
	operator array_ptr<void const>() const{
		array_ptr<void const> ret;
		ret.wSP_ = wSP_;
		return ret;
	}
	array_ptr operator+(std::ptrdiff_t d) const{
		array_ptr ret(*this);
		ret += d;
		return ret;
	}
	std::ptrdiff_t operator-(array_ptr other) const{
		return offset - other.offset;
	}
	array_ptr& operator--(){--offset; return *this;}
	array_ptr& operator++(){++offset; return *this;}
	array_ptr& operator-=(std::ptrdiff_t d){offset -= d; return *this;}
	array_ptr& operator+=(std::ptrdiff_t d){offset += d; return *this;}
	bool operator==(array_ptr<T> const& other) const{
                if(wSP_ == nullptr) return other.wSP_ == nullptr;
                if(other.wSP_ == nullptr) return wSP_ == nullptr;
		return wSP_->base(0) == other.wSP_->base(0) and offset == other.offset;
	}
	bool operator!=(array_ptr<T> const& other) const{return not((*this)==other);}
	bool operator<=(array_ptr<T> const& other) const{
		return wSP_->base(0) + offset <= other.wSP_->base(0) + other.offset;
	}
};
#endif

template<class T> struct allocator{
	template<class U> struct rebind{typedef allocator<U> other;};
	using value_type = T;
	using pointer = array_ptr<T>;
	using const_pointer = array_ptr<T const>;
	using reference = T&;
	using const_reference = T const&;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

	mpi3::shared_communicator& comm_;
	allocator(mpi3::shared_communicator& comm) : comm_(comm){}
	allocator() = delete;
	~allocator() = default;
	allocator(allocator const& other) : comm_(other.comm_){}
	template<class U>
	allocator(allocator<U> const& other) : comm_(other.comm_){}

//	template<class ConstVoidPtr = const void*>
	array_ptr<T> allocate(size_type n, const void* hint = 0){
		comm_.barrier();
		array_ptr<T> ret;
		if(n == 0) return ret;
#ifdef __bgq__
                if(bgq_dummy_allocator==nullptr) {
                  // this leaks memory right now!!! needs fixing!!!
                  MPI_Win impl_;
                  void* base_ptr = nullptr;
                  mpi3::size_t size_ = (comm_.root()?BG_SHM_ALLOCATOR*1024*1024:0);
                  int dunit;  
                  int s = MPI_Win_allocate_shared(size_, 1, MPI_INFO_NULL, comm_.impl_, &base_ptr, &impl_);
                  if(s != MPI_SUCCESS) throw std::runtime_error("cannot create shared window");    
                  MPI_Win_shared_query(impl_, 0, &size_, &dunit, &base_ptr); 

                  bgq_dummy_allocator = std::move(std::make_unique<simple_stack_allocator>(base_ptr,size_)); 
                }
                ret.base = bgq_dummy_allocator->allocate<T>(n); 
#else
		ret.wSP_ = std::make_shared<shared_window<T>>(
			comm_.make_shared_window<T>(comm_.root()?n:0)
		);
#endif
		return ret;
	}
	void deallocate(array_ptr<T> ptr, size_type){
#ifndef __bgq__
		ptr.wSP_.reset();
#endif
	}
//	void deallocate(double* const&, std::size_t&){}
	bool operator==(allocator const& other) const{
		return comm_ == other.comm_;
	}
	bool operator!=(allocator const& other) const{
		return not (other == *this);
	}
	template<class U, class... Args>
	void construct(U* p, Args&&... args){
		::new((void*)p) U(std::forward<Args>(args)...);
	}
	template< class U >
	void destroy(U* p){
		p->~U();
	}
};

struct is_root{
	shared_communicator& comm_;
	template<class Alloc>
	is_root(Alloc& a) : comm_(a.comm_){}
	bool root(){return comm_.root();}
        int size() {return comm_.size();}
        int rank() {return comm_.rank();}
        void barrier() {comm_.barrier();}
};


}


}}

#ifdef _TEST_BOOST_MPI3_SHARED_WINDOW

#include "../mpi3/main.hpp"

namespace mpi3 = boost::mpi3; using std::cout;

int mpi3::main(int argc, char* argv[], mpi3::communicator& world){

	mpi3::shared_communicator node = world.split_shared();
	mpi3::shared_window<int> win = node.make_shared_window<int>(node.root()?node.size():0);

	assert(win.base() != nullptr);
	assert(win.size() == node.size());

	win.base()[node.rank()] = node.rank() + 1;
	node.barrier();
	for(int i = 0; i != node.size(); ++i) assert(win.base()[i] == i + 1);


	

	return 0;
}

#endif
#endif

