#ifdef COMPILATION_INSTRUCTIONS
time clang++ -O2 -std=c++14 -Wall `#-fmax-errors=2` `#-Wfatal-errors` -I${HOME}/prj $0 -ggdb3 -o $0.x && time $0.x $@ && rm -f $0.x; exit
#endif

#include "../array_ref.hpp"
#include "../array.hpp"

#include<algorithm> // for sort
#include<iostream> // for print
#include<vector>
#include<cmath>

namespace multi = boost::multi;
using std::cout; using std::cerr;

#include<numeric>
#include<vector>

auto f(){
	std::vector<std::vector<double>> v(10, std::vector<double>(3));
	iota(v[5].begin(), v[5].end(), 1);
	return v;
}

int main(){

//	std::pointer_traits<double const*>::rebind<double const> p = 0;//::rebind<double const>::type p;
//	(void)p;

	double d2D[4][5] = {
		{150, 16, 17, 18, 19},
		{ 30,  1,  2,  3,  4}, 
		{100, 11, 12, 13, 14}, 
		{ 50,  6,  7,  8,  9} 
	};
	multi::array_ref<double, 2> d2D_ref{&d2D[0][0], {4, 5}};

	swap(d2D_ref[0], d2D_ref[3]);

	cout << "--\n";
	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1)) 
			cout << d2D_ref[i][j] << ' ';
		cout << '\n';
	}
	cout << '\n';

	multi::array_ref<double, 2>::const_reference crow1 = d2D_ref[1];
	assert( crow1[3] == 3 );
	multi::array_ref<double, 2>::iterator it = begin(d2D_ref);
	multi::array_ref<double, 2>::const_iterator cit = begin(d2D_ref);
	assert(it == cit);

	multi::array_ref<double, 2> d2D_ref2{&d2D[0][0], {4, 5}};
	d2D_ref = d2D_ref2;

	for(auto& r: d2D_ref) for(auto& e: r) e = -e;
	
	for(auto i : d2D_ref.extension(0))
		for(auto j : d2D_ref.extension(1))
			d2D_ref[i][j] = -d2D_ref[i][j];

	for(auto i : d2D_ref.extension()){
		for(auto j : d2D_ref.extension())
			cout << d2D_ref[i][j] <<' ';
		cout <<'\n';
	}

	using std::stable_sort;
	stable_sort( d2D_ref.begin(0), d2D_ref.end(0) );

	cout << "--\n";	
	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1))
			cout << d2D_ref[i][j] << ' ';
		cout << '\n';
	}
	stable_sort( d2D_ref.begin(1), d2D_ref.end(1) );

	cout << "--\n";	
	for(auto i : d2D_ref.extensions()[0]){
		for(auto j : d2D_ref.extensions()[1])
			cout << d2D_ref[i][j] << ' ';
		cout << '\n';
	}
	swap(*begin(d2D_ref), *(begin(d2D_ref) + 1));
//	swap(*d2D_ref.begin(), *(d2D_ref.begin() + 1));

	cout << "--\n";	
	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1))
			cout << d2D_ref[i][j] << ' ';
		cout << '\n';
	}
	std::reverse(d2D_ref.begin(1), d2D_ref.end(1));
	cout << "--\n";	
	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1))
			cout << d2D_ref[i][j] << ' ';
		cout << '\n';
	}

	cout << "--\n";
	auto const& d2D_ref_sv = d2D_ref.range({1, 3});
	for(auto i : d2D_ref_sv.extension(0)){
		for(auto j : d2D_ref_sv.extension(1))
			cout << d2D_ref_sv[i][j] << ' ';
		cout << '\n';
	}
	assert( d2D_ref.range({1,3}).origin() == &d2D_ref_sv[0][0] );
	assert( d2D_ref_sv.origin() == &d2D_ref_sv[0][0] );
	assert( *d2D_ref_sv.origin() == d2D_ref_sv[0][0] );
	assert( d2D_ref.range({1,3}).rotated(1).range({2, 5}).rotated(-1).origin() == &d2D_ref[1][2] );

	return 0;
	multi::array_ref<double, 1> d1D_ref{&d2D[0][0], {5}};

	assert( d2D_ref.cdata() == cdata(d2D_ref) );
	assert( d2D_ref.data() == data(d2D_ref) );
	assert( data(d2D_ref) == &d2D[0][0] );
	*data(d2D_ref) = 1;
	assert(d2D[0][0] == 1);
	for(auto i : d2D_ref.extension(0))
		for(auto j : d2D_ref.extension(1)) 
			d2D_ref[i][j] = 10.*i + j;

	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1))
			cout << d2D_ref[i][j] << ' ';
		cout << '\n';
	}
	cout << '\n';
	multi::array_ref<double, 2> d2D_refref{data(d2D_ref), extensions(d2D_ref)};

	for(auto it1 = begin(d2D_ref); it1 != end(d2D_ref) ||!endl(cout); ++it1)
		for(auto it2 = it1->begin()   ; it2 != it1->end()    ||!endl(cout); ++it2)
			*it2 = 99.;

	assert(d2D_ref[2][2] == 99.);

	double n = 0;
	for(auto it1 = d2D_ref.begin(1); it1 != d2D_ref.end(1) ||!endl(cout); ++it1)
		for(auto it2 = it1->begin()   ; it2 != it1->end()    ||!endl(cout); ++it2)
			*it2 = n++;

	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1)){
			cout << d2D_ref[i][j] << ' ';
		}
		cout << '\n';
	}

	*(d2D_ref.begin()->begin())=88.;
	assert( d2D_ref[0][0] == 88. );

	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1)){
			cout << d2D_ref[i][j] << ' ';
		}
		cout << '\n';
	}

	assert( d2D_ref[0] > d2D_ref[1] );
	using std::swap;
	swap( d2D_ref[2], d2D_ref[3] );

//	using std::stable_sort;
	std::stable_sort( begin(d2D_ref), end(d2D_ref) );
//	swap( d2D_ref[0], d2D_ref[1] );
	
	for(auto i : d2D_ref.extension(0)){
		for(auto j : d2D_ref.extension(1))
			cout << d2D_ref[i][j] << ' ';
		cout << '\n';
	}

/*	using std::for_each;
    using namespace std::string_literals; //""s
	for_each(begin(d2D_ref), end(d2D_ref), [](auto&& row){
		for_each(begin(row), end(row), [](auto&& element){
			element = 99;
		});
	});
*/
#if 0

	for(auto it1 = begin(d2D_cref); it1 != end(d2D_cref) ||!endl(cout); ++it1)
		for(auto it2 = it1->begin()   ; it2 != it1->end()    ||!endl(cout); ++it2)
			cout << *it2 << ' ';

//	d2D_cref[3][1] = 3.; // error const ref not assignable

	double const d2D_prime[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};

	multi::array_cref<double, 2> d2D_prime_cref{&d2D_prime[0][0], {4, 5}};
//	multi::array_cref<double, 2> d2D_prime_cref{&d2D_prime[0][0], extensions(d2D_cref)};
	assert( d2D_cref == d2D_prime_cref ); // deep comparison
//	assert( d2D_cref == d2D_prime );
	assert( not(d2D_cref != d2D_prime_cref) );
	assert( not(d2D_cref < d2D_cref) );
	assert( not(d2D_cref > d2D_cref) );
	assert( d2D_cref <= d2D_cref );
	assert( d2D_cref >= d2D_cref );

	double const d2D_null[4][5] {
		{ 0,  0,  0,  0,  0}, 
		{ 0,  0,  0,  0,  0}, 
		{ 0,  0,  0,  0,  0}, 
		{ 0,  0,  0,  0,  0}
	};
	multi::array_cref<double, 2> d2D_null_cref{&d2D_null[0][0], {4, 5}};

	using std::min;
	assert( &min(d2D_null_cref, d2D_cref) == &d2D_null_cref );
	using std::max;
	assert( &max(d2D_null_cref, d2D_cref) == &d2D_cref );
	
	using std::find;
	auto f = find(begin(d2D_cref), end(d2D_cref), d2D_cref[2]);
	assert( f != end(d2D_cref) );
	assert( (*f)[3] == d2D_cref[2][3] );

	using std::find_if;
	auto fif1 = find_if(begin(d2D_cref), end(d2D_cref), [](auto&& e){return e[3] == 8.111;});
	assert( fif1 == end(d2D_cref) );

	using std::find_if;
	auto fif2 = find_if(begin(d2D_cref), end(d2D_cref), [](auto&& e){return e[3] == 8.;});
	assert( fif2 != end(d2D_cref) );
	assert( fif2->operator[](4) == 9. );

	using std::count;
	assert( count(begin(d2D_cref), end(d2D_cref), d2D_prime_cref[3]) == 1 );
	assert( count(begin(d2D_cref), end(d2D_cref), d2D_prime[3]     ) == 1 );

	using std::min_element;
	using std::max_element;

	assert( min_element(begin(d2D_cref), end(d2D_cref)) == begin(d2D_cref) );
	assert( max_element(begin(d2D_cref), end(d2D_cref)) == begin(d2D_cref) + size(d2D_cref) - 1 );

	using std::minmax_element;
	assert( minmax_element(begin(d2D_cref), end(d2D_cref)).first == min_element(begin(d2D_cref), end(d2D_cref)) );
	assert( minmax_element(begin(d2D_cref), end(d2D_cref)).first == min_element(begin(d2D_cref), end(d2D_cref)) );
	decltype(d2D_cref)::const_iterator it; // it{} == it{0} == it{nullptr} = it(0);
//	assert(not it); // there are not null iterators
	assert( std::addressof(it->operator[](0)) == nullptr);
	it = begin(d2D_cref);
	assert(it == begin(d2D_cref));
	it = decltype(it){};

	std::vector<double>::iterator vit;
	std::list<double>::iterator lit{nullptr};
	assert( std::addressof(*vit) == nullptr );

	std::ptrdiff_t NX = 2;
	std::ptrdiff_t NY = 2;
	std::ptrdiff_t NZ = 2;
	std::vector<double> v(NX*NY*NZ);
	iota(begin(v), end(v), 0.);

	multi::array_cref<double, 3> v3D_cref{v.data(), {NX, NY, NZ}};

	assert( v3D_cref.num_elements() == multi::size_type(v.size()) );
	for(auto i : v3D_cref.extension(0))
		for(auto j : v3D_cref.extension(1))
			for(auto k : v3D_cref.extension(2))
				cout << i << ' ' << j << ' ' << k << ' ' 
					<< v3D_cref[i][j][k] << '\n';

	cout << v3D_cref[9][9][9] << "\n\n";

	assert(d2D_cref.begin() == d2D_cref.begin(0));
	assert(d2D_cref.begin() != d2D_cref.begin(1));
	for(auto it1 = d2D_cref.begin(1); it1 != d2D_cref.end(1)||!endl(cout); ++it1)
		for(auto it2 = it1->begin()   ; it2 != it1->end()   ||!endl(cout); ++it2)
			cout << *it2 << ' ';
#endif
}

