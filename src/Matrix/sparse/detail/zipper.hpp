// Copyright Alfredo A. Correa 2017.
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef ALF_BOOST_ITERATOR_ZIPPER
#define ALF_BOOST_ITERATOR_ZIPPER

#include<utility>
//#include<boost/iterator/iterator_facade.hpp> // iterator facade is outdate
//#include<boost/iterator/iterator_traits.hpp> // using std iterator traits instead
#include<boost/operators.hpp>
#include<tuple>

namespace boost{ // not an official boost component

namespace detail{
	template<class... T> struct common_category;
	template<class... T> using common_category_t = typename common_category<T...>::type;

	template<class C> struct common_category<C>{using type = C;};
	template<class C> struct common_category<C, C>{using type = C;};

	template<class C1, class C2>
	struct common_category<C1, C2>{using type = common_category_t<C2, C1>;};

	template<class C1, class C2, typename... Cs>
	struct common_category<C1, C2, Cs...>{
		using type = common_category_t<common_category_t<C1, C2>, Cs...>;
	};

	template<>
	struct common_category<std::random_access_iterator_tag, std::bidirectional_iterator_tag>{
		using type = std::bidirectional_iterator_tag;
	};

	template<>
	struct common_category<std::random_access_iterator_tag, std::forward_iterator_tag>{
		using type = std::forward_iterator_tag;
	};

	template<>
	struct common_category<std::random_access_iterator_tag, std::input_iterator_tag>{
		using type = std::input_iterator_tag;
	};

	template<>
	struct common_category<std::bidirectional_iterator_tag, std::forward_iterator_tag>{
		using type = std::forward_iterator_tag;
	};

	template<>
	struct common_category<std::bidirectional_iterator_tag, std::input_iterator_tag>{
		using type = std::input_iterator_tag;
	};

	template<>
	struct common_category<std::forward_iterator_tag, std::input_iterator_tag>{
		using type = std::input_iterator_tag;
	};

	struct take_value{template<class P> decltype(auto) operator()(P&& p) const{auto ret = *p; return ret;}};
	struct derref{template<class P> decltype(auto) operator()(P&& p) const{return *p;}};
//	struct ampersand{template<class V> decltype(auto) operator()(V&& v) const{return &v;}};

	template<typename F, class Tuple, size_t... indices>
	decltype(auto) tuple_apply_impl(F&& f, Tuple&& s, std::index_sequence<indices...>){
		return std::tuple<decltype(std::forward<F>(f)(std::get<indices>(std::forward<Tuple>(s))))...>(
			std::forward<F>(f)(std::get<indices>(std::forward<Tuple>(s)))...
		);
	}

	template <typename F, class Tuple>
	decltype(auto) tuple_apply(F&& f, Tuple&& s){
		return detail::tuple_apply_impl(
			std::forward<F>(f), std::forward<Tuple>(s), 
			std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>()
		);
	}

	template<class Tuple1, class Tuple2, size_t... indices>
	decltype(auto) tuple_zip_impl(Tuple1&& t1, Tuple2&& t2, std::index_sequence<indices...>){
		return 
			std::tuple<
				std::tuple<
					decltype(std::get<indices>(std::forward<Tuple1>(t1))), 
					decltype(std::get<indices>(std::forward<Tuple2>(t2)))
				>...
			>(
				std::tuple<
					decltype(std::get<indices>(std::forward<Tuple1>(t1))), 
					decltype(std::get<indices>(std::forward<Tuple2>(t2)))
				>(
					std::get<indices>(std::forward<Tuple1>(t1)), 
					std::get<indices>(std::forward<Tuple2>(t2))
				)...
			)
		;
	}

	template<class Tuple1, class Tuple2>
	decltype(auto) tuple_zip(Tuple1&& t1, Tuple2&& t2){
		return detail::tuple_zip_impl(
			std::forward<Tuple1>(t1), std::forward<Tuple2>(t2), 
			std::make_index_sequence<std::tuple_size<typename std::decay<Tuple1>::type>::value>()
		);
	}

}

template<class... Iters>
class zipper : 
	boost::totally_ordered<zipper<Iters...>>, // generates operder ops and !=
	boost::additive<zipper<Iters...>, std::common_type_t<typename std::iterator_traits<std::decay_t<Iters>>::difference_type...>>, // generate binary + and -
	boost::unit_steppable<zipper<Iters...>>, // generates postfix ++ and --
	boost::indexable< // generates operator[]
		zipper<Iters...>, 
		std::common_type_t<typename std::iterator_traits<std::decay_t<Iters>>::difference_type...>, 
		decltype(detail::tuple_apply(detail::derref{}, std::declval<std::tuple<Iters...>>()) )
	>
{
	std::tuple<Iters...> impl_ = {};
public:
	using difference_type = std::common_type_t<typename std::iterator_traits<std::decay_t<Iters>>::difference_type...>; 
	using value_type = decltype(detail::tuple_apply(detail::take_value{}, std::declval<std::tuple<Iters...>>()) );
	using pointer = zipper; // = decltype(std::tuple_apply(ampersand{}, std::declval<std::tuple<Iters...>>()) );
	using reference = decltype(detail::tuple_apply(detail::derref{}, std::declval<std::tuple<Iters...>>()) );
	using iterator_category = detail::common_category_t<typename std::iterator_traits<std::decay_t<Iters>>::iterator_category...>;

	zipper() = default;
	zipper(Iters&&... its) : impl_(std::forward<Iters>(its)...){}
	bool operator==(zipper const& other) const{return impl_ == other.impl_;}

	reference operator*() const{return detail::tuple_apply(detail::derref{}, impl_);}
	zipper& operator-=(difference_type n){
		detail::tuple_apply([&](auto&& a){return a -= n;}, impl_);
		return *this;
	}
	zipper& operator+=(std::ptrdiff_t n){
		detail::tuple_apply([&](auto&& a){return a += n;}, impl_);
		return *this;
	}
	zipper& operator++(){
		detail::tuple_apply([&](auto&& a){return ++a;}, impl_);
		return *this;
	}
	zipper& operator--(){
		detail::tuple_apply([&](auto&& a){return --a;}, impl_);
		return *this;
	}
	friend difference_type operator-(zipper const& z1, zipper const& z2){
		return std::get<0>(z1.impl_) - std::get<0>(z2.impl_);
	}
	friend bool operator<(zipper const& z1, zipper const& z2){
		return std::get<0>(z1.impl_) < std::get<0>(z2.impl_);
	}
	friend void iter_swap(zipper const& z1, zipper const& z2){
    	using std::iter_swap;
		detail::tuple_apply(
			[](auto&& pair){iter_swap(std::get<0>(pair), std::get<1>(pair)); return 0;}, 
			detail::tuple_zip(z1.impl_, z2.impl_)
		);
	}
};

template<class... Iters>
zipper<Iters...> zip(Iters&&... vi){
	return zipper<Iters...>(std::forward<Iters>(vi)...);
}

template <class... Iters> 
zipper<Iters...> make_zipper(Iters... vi){return zipper<Iters...>(std::move(vi)...);}

}

// this violates some rules:
namespace std{
template<class... Iters>
void iter_swap(boost::zipper<Iters...> const& a, boost::zipper<Iters...> const& b){
	iter_swap(a, b);
}
}

#endif

