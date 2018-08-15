#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && clang++ -O3 `#-fconcepts` -std=c++14 -Wall -Wextra `#-fmax-errors=2` -Wfatal-errors -lboost_timer -I${HOME}/prj -D_TEST_BOOST_MULTI_ARRAY_REF $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_ARRAY_REF_HPP
#define BOOST_MULTI_ARRAY_REF_HPP

#include "../multi/ordering.hpp"
#include "../multi/index_range.hpp"

#include<algorithm> // transform
#include<cassert>
#include<iostream> // cerr
#include<memory>

namespace boost{
namespace multi{

using std::cerr;

using index = std::ptrdiff_t;
using size_type = std::ptrdiff_t;

struct strides_t{
	std::ptrdiff_t val = 1;
	strides_t const& next;
	strides_t(std::ptrdiff_t v, strides_t const& n) : val(v), next(n){}
};

template<typename T, std::size_t N>
std::array<T, N-1> tail(std::array<T, N> const& a){
	std::array<T, N-1> ret;
	std::copy(a.begin() + 1, a.end(), ret.begin());
	return ret;
}

namespace detail{
//	template<typename T, typename... Ts>
	template<class Tuple>
	auto head(
		Tuple t
	//	std::tuple<T,Ts...> t
	)
	->decltype(std::get<0>(t)){
		return std::get<0>(t);}

	template<typename Tuple, std::size_t... Ns>//, typename... Ts>
	auto tail_impl(std::index_sequence<Ns...> , Tuple t){
		return std::make_tuple(std::get<Ns+1u>(t)...);
	}
	template<typename... Ts>
	auto tail(std::tuple<Ts...> t){
		return tail_impl( std::make_index_sequence<sizeof...(Ts) - 1u>() , t );
	}
}

template<dimensionality_type D>//, dimensionality_type DM = D>
struct layout_t{
	layout_t<D-1> sub;
	index stride_;
	index offset_;
	index nelems_;
	using extensions_type = std::array<index_extension, D>;
	auto operator()(index i) const{return i*stride_ - offset_;}
	auto origin() const{return -offset_ + sub.origin();}
	layout_t() : sub{}{}
//	template<typename ExtList, typename = std::enable_if_t<!std::is_base_of<layout_t, std::decay_t<ExtList>>{}>>
	constexpr layout_t(extensions_type e) : 
		sub{tail(e)}, 
		stride_{sub.size()*sub.stride()}, offset_{0}, nelems_{std::get<0>(e).size()*sub.num_elements()} 
	{}
	template<class Extensions, typename = decltype(std::get<0>(Extensions{}))>
	constexpr layout_t(Extensions e) : sub{detail::tail(e)}, 
		stride_{sub.size()*sub.stride_}, offset_{0}, nelems_{std::get<0>(e).size()*sub.num_elements()}
	{}
	bool operator==(layout_t const& other) const{
		return sub==other.sub && stride_==other.stride_ && offset_==other.offset_ && nelems_==other.nelems_;
	}
	bool operator!=(layout_t const& other) const{return not(*this==other);}
	friend bool operator!=(layout_t const& self, layout_t const& other){return not(self==other);}
	constexpr size_type num_elements() const{return nelems_;}
	constexpr size_type size() const{
		assert(stride_ != 0);
		assert(nelems_%stride_ == 0);
		return nelems_/stride_;
	}
	layout_t& rotate(){
		using std::swap;
		swap(stride_, sub.stride_);
		swap(offset_, sub.offset_);
		swap(nelems_, sub.nelems_);
		sub.rotate();
		return *this;
	}
	layout_t& rotate(dimensionality_type r){
		if(r >= 0) while(r){rotate(); --r;}
		else return rotate(D - r);
		return *this;
	}
/*	template<dimensionality_type DD = 0> 
	constexpr size_type size() const{
		if constexpr(DD == 0) return size();
		else return sub.template size<DD - 1>();
	}*/
	constexpr size_type size(dimensionality_type d) const{
		return d?sub.size(d-1):size();
	};
	constexpr index stride(dimensionality_type d = 0) const{
		return d?sub.stride(d-1):stride_;
	}
	constexpr index offset() const{return offset_;}
	decltype(auto) shape() const{return sizes();}
	friend decltype(auto) shape(layout_t const& self){return self.shape();}
	auto sizes() const{
		std::array<size_type, D> ret;
		sizes_aux(ret.begin());
		return ret;
	}
	void sizes_aux(size_type* it) const{
		*it = size(); 
		sub.sizes_aux(++it);
	}
	auto strides() const{
		std::array<index, D> ret;
		strides_aux(begin(ret));
		return ret;
	}
	void strides_aux(size_type* it) const{
		*it = stride();
		sub.strides_aux(++it);
	}
	constexpr index_extension extension_aux() const{
		assert(stride_ != 0 and nelems_%stride_ == 0);
		return {offset_/stride_, (offset_ + nelems_)/stride_};
	}
	template<dimensionality_type DD = 0>
	constexpr index_extension extension(dimensionality_type d = DD) const{
		return d?sub.extension(d-1):extension_aux();
	}
	auto extensions() const{
		extensions_type ret;
		extensions_aux(ret.begin());
		return ret;
	}
	void extensions_aux(index_extension* it) const{
		*it = extension();
		++it;
		sub.extensions_aux(it);
	}
};

#if 1
template<>
struct layout_t<1>{
	using extensions_type = std::array<index_extension, 1>;
	index stride_;
	index offset_;
	index nelems_;
	layout_t() = default;
	template<class Extensions, typename = decltype(std::get<0>(Extensions{}))>
	constexpr layout_t(Extensions e) : 
		stride_{1*1}, offset_{0}, nelems_{std::get<0>(e).size()*1}
	{}
	constexpr size_type size(dimensionality_type d = 0) const{
		(void)d;
		assert(d == 0 and stride_ != 0 and nelems_%stride_ == 0);
		return nelems_/stride_;
	}
	auto sizes() const{
		std::array<size_type, 1> ret;
		sizes_aux(ret.begin());
		return ret;
	}
	void sizes_aux(size_type* it) const{*it = size();}
	constexpr index stride(dimensionality_type d = 0) const{assert(d == 0); return stride_;}
	void strides_aux(size_type* it) const{*it = stride();}
	constexpr size_type num_elements() const{return nelems_;}
	constexpr index_extension extension(dimensionality_type d = 0) const{
		(void)d;
		assert(d == 0);
		assert(stride_ != 0 and nelems_%stride_ == 0);
		return {offset_/stride_, (offset_ + nelems_)/stride_};
	}
	constexpr extensions_type extensions() const{return {extension()};}
	void extensions_aux(index_extension* it) const{*it = extension();}
	constexpr auto operator()(index i) const{return i*stride_ - offset_;}
	constexpr auto origin() const{return -offset_;}
	bool operator==(layout_t const& other) const{
		return stride_==other.stride_ and offset_==other.offset_ and nelems_==other.nelems_;
	} bool operator!=(layout_t const& other) const{return not(*this==other);}
	layout_t& rotate(){return *this;}
	decltype(auto) shape() const{return sizes();}
        auto strides() const {return std::array<index, 1>{stride_};}
};
#endif

template<>
struct layout_t<0>{
	using extensions_type = std::array<index_extension, 0>;
	index stride_ = 1;
	index nelems_ = 1;
	index size_ = 1;
	layout_t() = default;
	template<class T>
	layout_t(std::array<T, 0>){}
	void sizes_aux(size_type*) const{}
	void strides_aux(size_type*) const{}
	void extensions_aux(index_extension*) const{};
	constexpr size_type num_elements() const{return 1;}
	template<dimensionality_type DD = 0>
	constexpr size_type size() const{return size_;}
	constexpr index stride(dimensionality_type = 0) const{return stride_;} // assert(d == -1);
	layout_t& rotate(){return *this;}
	bool operator==(layout_t const&) const{return true;}
	bool operator!=(layout_t const&) const{return false;}
	size_type size(dimensionality_type) const{return -1;} // assert(d == -1);
	index_extension extension(dimensionality_type) const{return index_extension{};} // assert(d == -1);
};

template<typename T, dimensionality_type D, class Alloc = std::allocator<T>>
class array;

#if 0
template<class T, dimensionality_type N>
struct recursive_ilist{
	using type = typename std::initializer_list<typename recursive_ilist<T, N-1>::type>;
};
template<class T>
struct recursive_ilist<T, 1>{using type = std::initializer_list<T>;};
template<class T, dimensionality_type N>
using recursive_ilist_t = typename recursive_ilist<T, N>::type;
#endif

template<
	typename T, 
	dimensionality_type D, 
	typename ElementPtr = T const* const, 
	class Layout = layout_t<D>
> 
struct basic_array : Layout{
	static constexpr dimensionality_type dimensionality = D;
	using layout_t = Layout;
	using element = T;
	using element_ptr = ElementPtr;
	using element_const_ptr = typename std::pointer_traits<element_ptr>::template rebind<element const>;
	using value_type      = multi::array<element, D-1>;
	using difference_type = index;
	using const_reference = basic_array<element, D-1, element_const_ptr>; 
	using reference       = basic_array<element, D-1, element_ptr>;
	friend struct basic_array<element, D+1, element_ptr>;
// for at least up to 3D, ...layout const> is faster than ...layout const&>
protected:
	using initializer_list = std::initializer_list<typename basic_array<T, D-1, ElementPtr>::initializer_list>;
	element_ptr data_;
	basic_array() = delete;
	Layout const& layout() const{return *this;}
// MAM
public:
	basic_array(element_ptr data, Layout layout) : Layout(layout), data_(data){}
private:
	basic_array& operator=(basic_array const& other) = default;
	const_reference const_operator_sbracket(index i) const{
		assert( i < this->extension().last() and i >= this->extension().first() );
		return {data_ + Layout::operator()(i), Layout::sub};
	}
        reference operator_sbracket(index i) const{
                assert( i < this->extension().last() and i >= this->extension().first() );
                return {data_ + Layout::operator()(i), Layout::sub};
        }
public:
	element_ptr origin() &{return data_ + Layout::origin();}
	element_ptr origin() &&{return data_ + Layout::origin();}
	element_const_ptr origin() const&{return data_ + Layout::origin();}
	element_const_ptr corigin() const{return data_ + Layout::origin();}
	const_reference operator[](index i) const{return const_operator_sbracket(i);}
	reference operator[](index i){return operator_sbracket(i);}
	auto range(index_range const& ir) const{
		layout_t new_layout = *this; 
		(new_layout.nelems_/=Layout::size())*=ir.size();
		return basic_array<T, D, ElementPtr, multi::layout_t<D>>{
			data_ + Layout::operator()(ir.front()), new_layout
		};
	}
	auto range(index_range const& ir, dimensionality_type n) const{
		return rotated(n).range(ir).rotated(-n);
	}
	auto operator()(index_range a) const{return range(a);}
	auto operator()(index i) const{return operator[](i);}
	decltype(auto) paren_aux() const{return *this;}
	template<class... As>
	auto paren_aux(index_range a, As&&... as) const{return operator()(a).rotated(1).paren_aux(as...);}
	template<class... As>
	auto paren_aux(index i, As&&... as) const{return operator[](i).rotated(1).paren_aux(as...);}
	template<class... As>
	auto operator()(index_range a, As&&... as) const{return paren_aux(a, as...).rotated(-(1 + sizeof...(As)));}
	template<class... As>
	auto operator()(index i, As&&... as) const{return operator[](i).rotated(1)(as...).rotated(-(1 + sizeof...(As)));}
	auto operator()(index_range const& a, index_range const& ir1) const{return paren_aux(a, ir1).rotated(-2);}
	auto operator()(index_range const& a, index       const& i1) const{return paren_aux(a, i1).rotated(-2);}
	auto operator()(index const& a, index_range const& ir1) const{return operator[](a).rotated(1)(ir1).rotated(-2);}
	auto rotated(dimensionality_type i) const{
		layout_t new_layout = *this; 
		new_layout.rotate(i);
		return basic_array<T, D, ElementPtr>{data_, new_layout};
	}
	decltype(auto) front(){return *begin();}
	decltype(auto) back(){return *(begin() + (Layout::size() - 1));}
	class const_iterator : private const_reference{
		index stride_;
		const_iterator(const_reference const& cr, index stride) : 
			const_reference{cr}, stride_{stride}
		{}
		friend struct basic_array;
		explicit operator bool() const{return this->data_;}
	public:
		using difference_type = basic_array::difference_type;
		using value_type = typename basic_array<T, D, ElementPtr>::value_type;
		using pointer = void*;
		using reference = const_reference;
		using iterator_category = std::random_access_iterator_tag;
		const_iterator(std::nullptr_t = nullptr) : const_reference{}{}
		const_iterator(const_iterator const&) = default;
		const_iterator& operator=(const_iterator const& other) = default;
		const_reference const& operator*() const{assert(operator bool()); return *this;}
		const_reference const* operator->() const{return static_cast<const_reference const*>(this);}
		const_reference operator[](index i) const{return {this->data_ + i*stride_, Layout::sub};}
		const_iterator& operator++(){this->data_ += stride_; return *this;}
		const_iterator& operator--(){this->data_ -= stride_; return *this;}
		const_iterator& operator+=(index d){this->data_ += stride_*d; return *this;}
		const_iterator& operator-=(index d){this->data_ -= stride_*d; return *this;}
		const_iterator operator+(index d){const_iterator ret = *this; return ret += d;}
		const_iterator operator-(index d){const_iterator ret = *this; return ret -= d;}
		ptrdiff_t operator-(const_iterator const& other) const{
			assert(stride_ != 0 and (this->data_ - other.data_)%stride_ == 0);
			assert( this->layout() == other.layout() );
			return (this->data_ - other.data_)/stride_;
		}
		bool operator==(const_iterator const& other) const{
			return this->data_ == other.data_ and this->stride_ == other.stride_;
		}
		bool operator!=(const_iterator const& other) const{return not((*this)==other);}
		bool operator<(const_iterator const& other) const{
			if(stride_ < 0) return other.data_ - this->data_ < 0;
			return this->data_ - other.data_ < 0;
		}
	};
	struct iterator : private reference{
		index stride_;
		iterator(basic_array<T, D-1, ElementPtr> const& cr, index stride) : 
			basic_array<T, D-1, ElementPtr>{cr}, stride_{stride}
		{}
		friend struct basic_array;
		explicit operator bool() const{return this->data_;}
	public:
		operator const_iterator() const{
			return const_iterator{static_cast<reference const&>(*this), stride_};
		}
		using difference_type = typename basic_array<T, D, ElementPtr>::difference_type;
		using value_type = typename basic_array<T, D, ElementPtr>::value_type;
		using pointer = void*;
		using reference = typename basic_array<T, D, ElementPtr>::reference; // careful with shadowing reference (there is another one level up)
		iterator(std::nullptr_t = nullptr) : basic_array<T, D-1, ElementPtr>{}{}
		using iterator_category = std::random_access_iterator_tag;
		iterator& operator=(iterator const& other) = default;
		template<class It>
		bool operator==(It const& other) const{
			return this->data_ == other.data_ and this->stride_ == other.stride_;
		}
		template<class It>
		bool operator!=(It const& other) const{return not((*this)==other);}
		reference& operator*() const{return *const_cast<iterator*>(this);}
		reference* operator->(){return static_cast<reference*>(this);}
		iterator& operator++(){this->data_ += stride_; return *this;}
		auto operator++(int){
           iterator ret{*this};   // make a copy for result
           ++(*this);              // Now use the prefix version to do the work
           return ret;      
		}
		iterator& operator--(){this->data_ -= stride_; return *this;}
		iterator operator+(index d){iterator ret = *this; return ret += d;}
		iterator operator-(index d){iterator ret = *this; return ret -= d;}
		iterator& operator+=(index d){this->data_ += stride_*d; return *this;}
		iterator& operator-=(index d){this->data_ -= stride_*d; return *this;}
		ptrdiff_t operator-(iterator const& other) const{
			assert(stride_ != 0 and (this->data_ - other.data_)%stride_ == 0);
			return (this->data_ - other.data_)/stride_;
		}
		bool operator<(iterator const& other) const{
			if(stride_ < 0) return other.data_ - this->data_ < 0;
			return this->data_ - other.data_ < 0;
		}
		friend void iter_swap(iterator const& i1, iterator const& i2){
			i1->swap(*i2);
//			return swap(*i1, *i2);
		}
	};
	friend size_type size(basic_array const& self){return self.size();}
	friend auto sizes(basic_array const& self){return self.sizes();}
	friend auto strides(basic_array const& self){return self.strides();}
	friend auto extension(basic_array const& self){return self.extension();}
	friend auto extensions(basic_array const& self){return self.extensions();}
	const_iterator begin(index i) const{
		Layout new_layout = *this;
		new_layout.rotate(i);
		return const_iterator{
			const_reference{data_ + new_layout(0), new_layout.sub},
			new_layout.stride_
		};
	}
	const_iterator end(index i) const{
		Layout new_layout = *this;
		new_layout.rotate(i);
		return const_iterator{
			const_reference{data_ + new_layout(new_layout.size()), new_layout.sub},
			new_layout.stride_
		};
	}
	iterator begin(index i){
		Layout new_layout = *this;
		new_layout.rotate(i);
		return iterator{
			reference{data_ + new_layout(0), new_layout.sub},
			new_layout.stride_
		};
	}
	iterator end(index i){
		Layout new_layout = *this;
		new_layout.rotate(i);
		return iterator{
			reference{data_ + new_layout(new_layout.size()), new_layout.sub},
			new_layout.stride_
		};
	}
	const_iterator begin()  const{return {const_reference{data_ + Layout::operator()(this->extension().first()), Layout::sub}, Layout::stride_};}
	const_iterator end()    const{return {const_reference{data_ + Layout::operator()(this->extension().last()), Layout::sub}, Layout::stride_};}
	const_iterator cbegin()  const{
		iterator r = begin();
		const_iterator cr = r;
		return r;
	}
	const_iterator cend()    const{return end();}
	iterator begin(){return {reference{data_ + Layout::operator()(this->extension().first()), Layout::sub}, Layout::stride_};}
	iterator end()  {return {reference{data_ + Layout::operator()(this->extension().last()), Layout::sub}, Layout::stride_};}

	friend const_iterator begin(basic_array const& self){return self.begin();}
	friend const_iterator end(basic_array const& self){return self.end();}
	friend iterator begin(basic_array& self){return self.begin();}
	friend iterator end(basic_array& self){return self.end();}
	friend const_iterator cbegin(basic_array const& self){return self.begin();}
	friend const_iterator cend(basic_array const& self){return self.end();}
//	size_type num_elements() const{return layout_.num_elements();}
	friend size_type num_elements(basic_array const& self){return self.num_elements();}
/*	basic_array& operator=(basic_array const&){
		assert(0);
		return *this;
	}*/
protected:
	template<class It>
	void recursive_assign_(It first, It last){
		auto self_first = this->begin();
		while(first != last){
			self_first->recursive_assign_(first->begin(), first->end());
			++first;
			++self_first;
		}
	}
	template<class M>
	void intersection_assign_(M&& other){
		for(
			auto i = std::max(basic_array::extension(0).first(), other.extension(0).first()); 
			i != std::min(basic_array::extension(0).last(), other.extension(0).last()); ++i
		) operator[](i).intersection_assign_(other[i]);
	}
public:
	template<class Array>
	void operator=(Array const& other){
		assert(this->extension() == other.extension());
		for(auto i : this->extension()) operator[](i) = other[i];
	}
	template<class Array>
	bool operator<(Array const& other) const{
		using std::lexicographical_compare;
		using std::begin; using std::end;
		return lexicographical_compare(this->begin(), this->end(), begin(other), end(other)); // needs assignable iterator
	}
	template<class Array>
	bool operator<=(Array const& other) const{return (*this)==other or (*this)<other;}
	template<class Array>
	bool operator>(Array const& other) const{return not ((*this)<=other);}
	template<class Array>
	bool operator>=(Array const& other) const{return not ((*this)<other);}
	template<class Array>
	void swap(Array&& other){
		assert(this->extension() == extension(other));
		using std::swap_ranges; using std::begin; using std::end;
		swap_ranges(this->begin(), this->end(), begin(other)); // for(auto i : self.extension()) swap(self[i], other[i]);		
	}
	friend void swap(basic_array& a1, basic_array& a2){return a1.swap(a2);}
	friend void swap(basic_array& a1, basic_array&& a2){return a1.swap(a2);}
	friend void swap(basic_array&& a1, basic_array& a2){return a1.swap(a2);}
	friend void swap(basic_array&& a1, basic_array&& a2){return a1.swap(a2);}
	template<class Array> 
	bool operator==(Array const& other) const{
		if(this->extension() != other.extension()) return false;
		using std::equal;
		return equal(this->begin(), this->end(), other.begin());
	}
	template<class Array> bool operator!=(Array const& other) const{
		return not((*this) == other);
	}
};

template<class T, std::size_t N> 
index_range extension(T(&)[N]){return {0, N};}

template<typename T, typename ElementPtr, class Layout>
struct basic_array<T, dimensionality_type{1}, ElementPtr, Layout> : Layout{
	constexpr static dimensionality_type dimensionality = 1;
	using value_type = T;
	using element = value_type;
	using element_ptr = ElementPtr;
	using element_const_ptr = typename std::pointer_traits<element_ptr>::template rebind<element const>;
	using layout_t = Layout;
	using const_reference = T const&;
	using reference = decltype(*ElementPtr{});
protected:
//	using initializer_list = recursive_ilist_t<T, dimensionality>;
	using initializer_list = std::initializer_list<T>;
	template<class It>
	void recursive_assign_(It first, It last){
		auto self_first = this->begin();
		while(first != last){
			*self_first = *first;
			++first;
			++self_first;
		}
	}
	template<class M>
	void intersection_assign_(M&& other){
		for(auto i = std::max(basic_array::extension(0).first(), other.extension(0).first()); 
			i != std::min(basic_array::extension(0).last(), other.extension(0).last()); ++i
		) operator[](i) = other[i];
	}
	Layout const& layout() const{return *this;}
	element_ptr data_;
	basic_array() = default;
	basic_array(element_ptr data, layout_t layout) : Layout{layout}, data_{data}{}
	friend struct basic_array<T, dimensionality_type{2}, element_ptr>;
	friend struct basic_array<T, dimensionality_type{2}, typename std::pointer_traits<element_ptr>::template rebind<element>>;
	friend struct basic_array<T, dimensionality_type{1}, element_const_ptr>;
	basic_array& operator=(basic_array const& other) = default;
public:
	template<class Array>
	void operator=(Array const& other){
		assert(this->extension() == other.extension());
		for(auto i : this->extension()) operator[](i) = other[i];
	}
	basic_array(basic_array<element, 1, typename std::pointer_traits<element_ptr>::template rebind<element>> const& other) : Layout(static_cast<Layout const&>(other)), data_(other.data_){}
	const_reference operator[](index i) const{
		assert( i < this->extension().last() and i >= this->extension().first() );
		return data_[Layout::operator()(i)];
	}
	reference operator[](index i){
		assert( i < this->extension().last() and i >= this->extension().first() );
		return data_[Layout::operator()(i)];
	}
	auto range(index_range const& ir) const{
		layout_t new_layout = *this; 
		(new_layout.nelems_/=Layout::size())*=ir.size();
		return basic_array<T, 1, ElementPtr, multi::layout_t<1>>{
			data_ + Layout::operator()(ir.front()), new_layout
		};
	}
	decltype(auto) paren_aux() const{return *this;}
	template<class... As>
	auto paren_aux(index_range a, As&&... as) const{return operator()(a).rotated(1).paren_aux(as...);}
	template<class... As>
	auto paren_aux(index i, As&&... as) const{return operator[](i).rotated(1).paren_aux(as...);}
	auto operator()(index_range const& ir) const{return range(ir);}
	auto operator()(index i) const{return operator[](i);}
	decltype(auto) rotated(dimensionality_type) const{return *this;}
	element_const_ptr origin() const{return data_ + Layout::origin();}
	element_ptr origin(){return data_ + Layout::origin();}
	friend decltype(auto) origin(basic_array const& self){return self.origin();}
	friend decltype(auto) corigin(basic_array const& self){return self.origin();}
	element_const_ptr corigin() const{return origin();}
	class const_iterator{
		friend struct basic_array;
		const_iterator(element_ptr data, index stride) : data_(data), stride_(stride){}
	protected:
		element_ptr data_;
		index stride_;
	public:
		using difference_type = index;
		using value_type = T;
		using pointer = void*;
		using reference = const_reference;
		using iterator_category = std::random_access_iterator_tag;
		const_reference operator*() const{return *data_;}
		element_ptr const* operator->() const{return data_;}
		const_reference operator[](index i) const{return data_[i*stride_];}
		const_iterator& operator++(){data_ += stride_; return *this;}
		const_iterator& operator--(){data_ += stride_; return *this;}
		const_iterator& operator+=(ptrdiff_t d){this->data_ += stride_*d; return *this;}
		const_iterator operator+(ptrdiff_t d) const{return const_iterator(*this)+=d;}
		std::ptrdiff_t operator-(const_iterator const& other) const{
			assert(stride_ != 0 and (data_ - other.data_)%stride_ == 0);
			return (data_ - other.data_)/stride_;
		}
		bool operator==(const_iterator const& other) const{
			return data_ == other.data_ and stride_ == other.stride_;
		}
		bool operator!=(const_iterator const& other) const{return not((*this)==other);}
		bool operator<(const_iterator const& other) const{
			if(stride_ < 0) return other.data_ - this->data_ < 0;
			return this->data_ - other.data_ < 0;
		}
	};
	struct iterator : const_iterator{
		friend struct basic_array;
		using const_iterator::const_iterator;
		reference operator*() const{return *this->data_;}
		element_ptr* operator->() const{return this->data_;}
		iterator& operator++(){const_iterator::operator++(); return *this;}
		iterator& operator+=(ptrdiff_t d){const_iterator::operator+=(d); return *this;}
		iterator operator+(ptrdiff_t d) const{return iterator(*this)+=d;}
	};
//	size_type num_elements() const{return layout_.num_elements();}
	friend size_type num_elements(basic_array const& self){return self.num_elements();}
//	size_type size(dimensionality_type d = 0) const{return layout_.size(d);}
	friend auto size(basic_array const& self){return self.size();}
//	index_range extension() const{return layout_.extension();}
	friend index_range extension(basic_array const& self){return self.extension();}
private:
	iterator _begin() const{return {data_ + Layout::operator()(this->extension().first()), Layout::stride_};}
	iterator _end()   const{return {data_ + Layout::operator()(this->extension().last()), Layout::stride_};}
public:
	const_iterator begin() const{return _begin();}
	const_iterator end()   const{return _end();}
	iterator begin(){return _begin();}
	iterator end()  {return _end();}
	friend auto begin(basic_array const& self){return self.begin();}
	friend auto end(basic_array const& self){return self.end();}
//	friend auto begin(basic_array& self){return self.begin();}
//	friend auto end(basic_array& self){return self.end();}
	template<class Array, typename = decltype(std::declval<basic_array>()[0] = std::declval<Array>()[0])>
	basic_array& operator=(Array&& other){
		assert(this->extension() == other.extension());
		using std::transform; using std::begin; using std::end;
		transform(begin(std::forward<Array>(other)), end(std::forward<Array>(other)), this->begin(), [](auto&& e){return e;});
		return *this;
	}
	template<class Array>
	bool operator==(Array const& other) const{
		using multi::extension;
		if(this->extension() != extension(other)) return false;
		using std::equal; using std::begin; // using std::end;
		return equal(this->begin(), this->end(), begin(other));
	}
	template<class Array>
	bool operator!=(Array const& other) const{return not((*this)==other);}
	template<class Array>
	bool operator<(Array const& other) const{
		using std::lexicographical_compare;
		using std::begin; using std::end;
		return lexicographical_compare(
			begin(*this), end(*this), 
			begin(other), end(other)
		);
	}
	template<class Array>
	bool operator<=(Array const& other) const{return (*this)==other or (*this)<other;}
	template<class A> bool operator>(A const& other) const{return not((*this)<=other);}
	template<class A> bool operator>=(A const& other) const{return not((*this)<other);}
	template<class Array> void swap(Array&& other){
		assert(this->extension() == extension(other));
		using std::swap_ranges; using std::begin; //using std::end;
		swap_ranges(this->begin(), this->end(), begin(other)); // for(auto i : self.extension()) swap(self[i], other[i]);		
	}
	friend void swap(basic_array& a1, basic_array& a2){return a1.swap(a2);}
	friend void swap(basic_array& a1, basic_array&& a2){return a1.swap(a2);}
	friend void swap(basic_array&& a1, basic_array& a2){return a1.swap(a2);}
	friend void swap(basic_array&& a1, basic_array&& a2){return a1.swap(a2);}
};
/*
template<class T, dimensionality_type N, class Ptr>
void swap(basic_array<T, N, Ptr>&& a1, basic_array<T, N, Ptr>&& a2){a1.swap(a2);}
template<class T, dimensionality_type N, class Ptr>
void swap(basic_array<T, N, Ptr>& a1, basic_array<T, N, Ptr>& a2){a1.swap(a2);}
template<class T, dimensionality_type N, class Ptr>
void swap(basic_array<T, N, Ptr>& a1, basic_array<T, N, Ptr>&& a2){a1.swap(a2);}
template<class T, dimensionality_type N, class Ptr>
void swap(basic_array<T, N, Ptr>&& a1, basic_array<T, N, Ptr>& a2){a1.swap(a2);}
*/

template<typename T, dimensionality_type D, typename ElementPtr = T const*>
struct const_array_ref : basic_array<T, D, ElementPtr>{
	using element_const_ptr = typename const_array_ref::element_const_ptr;
	const_array_ref() = delete; // references must be initialized (bound)
	const_array_ref(const_array_ref const&) = default;
	constexpr const_array_ref(
		typename const_array_ref::element_ptr p, 
		typename const_array_ref::extensions_type e
	) noexcept : 
		basic_array<T, D, ElementPtr>{
			p, 
			typename const_array_ref::layout_t{e}
		}
	{}
	template<class Extensions>
	constexpr const_array_ref(
		typename const_array_ref::element_ptr p, Extensions e		
	) noexcept : basic_array<T, D, ElementPtr>{p, typename const_array_ref::layout_t{e}}
	{}
	const_array_ref& operator=(const_array_ref const&) = delete;
	element_const_ptr cdata() const{return const_array_ref::data_;}
	element_const_ptr data() const{return cdata();}
	friend decltype(auto) data(const_array_ref const& self){return self. data();}
	friend decltype(auto) cdata(const_array_ref const& self){return self.cdata();}
};

template<typename T, dimensionality_type D, typename ElementPtr = T const*>
using array_cref = const_array_ref<T, D, ElementPtr>;

template<typename T, dimensionality_type D, typename ElementPtr = T const*>
using carray_cref = array_cref<T, D, ElementPtr>;

}}

namespace boost{
namespace multi{

template<typename T, dimensionality_type D, typename ElementPtr = T*>
struct array_ref : const_array_ref<T, D, ElementPtr>{
	using const_array_ref<T, D, ElementPtr>::const_array_ref;
	template<class Array>
	array_ref& operator=(Array&& other){
		assert(this->extension() == other.extension());
		for(auto i : this->extension()) this->operator[](i) = std::forward<Array>(other)[i];
		return *this;
	}
	array_ref& operator=(array_ref const& o){return operator=<array_ref const&>(o);}
//	typename array_ref::element_ptr 
	ElementPtr const& data(){return this->data_;}
	friend decltype(auto) data(array_ref& self){return self.data();}
};

template<typename... A> using carray_ref = array_ref<A...>;

}}

namespace std{
template<class It, typename = decltype(It{}.stride_)>
void iter_swap(It it1, It it2){it1->swap(*it2);}
}

#if _TEST_BOOST_MULTI_ARRAY_REF

#include<cassert>
#include<numeric> // iota
#include<iostream>

using std::cout;
namespace multi = boost::multi;

int main(){

#if 0
	{
		double const d2D[4][5] = {{1.,2.},{2.,3.}};
		bm::array_ref<double, 2> d2Rce{&d2D[0][0], {4, 5}};
		assert( &d2Rce[2][3] == &d2D[2][3] );
		assert( d2Rce.size() == 4 );
	//	assert( d2Rce.size<0>() == 4);
	//	assert( d2Rce.size<1>() == 5);
		cout << d2Rce.num_elements() << std::endl;
		assert( d2Rce.num_elements() == 20 );
	}
#endif
	{
		double const dc2D[4][5] = {{1.,2.},{2.,3.}};
		multi::array_cref<double, 2> acrd2D{&dc2D[0][0], {4, 5}};
		assert( &acrd2D[2][3] == &dc2D[2][3] );
		assert( acrd2D.size() == 4);
		assert( acrd2D.size(0) == 4);
		assert( acrd2D.size(1) == 5);
		assert( acrd2D.sizes().size() == 2 );
	//	assert( acrd2D.size<0>() == acrd2D.size() );
	//	assert( acrd2D.size<1>() == 5);
		assert( acrd2D.num_elements() == 20 );

		assert( &acrd2D[2][3] == &dc2D[2][3] );
	
		assert( acrd2D.begin() == begin(acrd2D) );
		assert( acrd2D.begin() != acrd2D.end() );
	}
#if 0
	{
		double* d2p = new double[4*5]; std::iota(d2p, d2p + 4*5, 0);

		bm::array_ref<double, 2> d2R{d2p, 4, 5};
		assert(d2R.size()==4);
	}
	cout << "ddd " << d2R[1][1] << '\n';
	for(int i = 0; i != 4 ||!(cout << '\n'); ++i)
		for(int j = 0; j != 5 ||!(cout << '\n'); ++j)
			cout << d2R[i][j] << ' ';

	for(auto it1 = d2R.begin(); it1 != d2R.end() ||!(cout << '\n'); ++it1)
		for(auto it2 = it1->begin(); it2 != it1->end() ||!(cout << '\n'); ++it2)
			cout << *it2 << ' ';

	for(auto&& row : d2R){
		for(auto&& e : row) cout << e << ' '; cout << '\n';
	} cout << '\n';

	for(auto i : d2R.extension()){
		for(auto j : d2R[i].extension())
			cout << d2R[i][j] << ' ';
		cout << '\n';
	}
	cout << '\n';

	for(auto it1 = d2R.begin1(); it1 != d2R.end1() ||!(cout << '\n'); ++it1)
		for(auto it2 = it1->begin(); it2 != it1->end() ||!(cout << '\n'); ++it2)
			cout << *it2 << ' ';

	assert( d2R.begin()[1][1] == 6 );

	assert(d2R.size() == 4);
	auto it = d2R.begin();
	assert((*it)[1] == 1);
	assert( it->operator[](0) == 0 );
	assert( it->operator[](1) == 1 );
	++it;
	assert( it->operator[](0) == 5 );
	assert( it->operator[](1) == 6 );

	assert( *(it->begin()) == 5 );


#if 0
	if(double* d3p = new double[3*4*5]){
		std::iota(d3p, d3p + 3*4*5, 0);
		bm::array_ref<double, 3> d3R{d3p, 3, 4, 5};
		assert(d3R.size() == 3);
		for(int i = 0; i != 3; ++i, cout << '\n')
			for(int j = 0; j != 4; ++j, cout << '\n')
				for(int k = 0; k != 5; ++k)
					cout << d3R[i][j][k] << ' ';
		auto b = d3R.begin();
		
	}
#endif
#endif
}
#endif
#endif

