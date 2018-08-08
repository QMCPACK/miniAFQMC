//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2017 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#undef NDEBUG

#include "Message/catch_mpi_main.hpp"

#include "Configuration.h"

// Avoid the need to link with other libraries just to get APP_ABORT
#undef APP_ABORT
#define APP_ABORT(x) {std::cout << x <<std::endl; exit(0);}

#include "OhmmsData/Libxml2Doc.h"
#include "Utilities/RandomGenerator.h"
#include "Utilities/SimpleRandom.h"
#include "Utilities/OhmmsInfo.h"
#include "OhmmsApp/ProjectData.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <complex>

#include "alf/boost/mpi3/communicator.hpp"
#include "alf/boost/mpi3/shared_communicator.hpp"
//#include "alf/boost/mpi3/environment.hpp"

#include "boost/multi_array.hpp"
//#include "AFQMC/Walkers WalkerSetFactory.hpp"
#include "AFQMC/Walkers/SharedWalkerSet.h"

using std::string;
using std::complex;
using std::cout;
using std::endl;

using boost::extents;
using boost::indices;
using range_t = boost::multi_array_types::index_range;

namespace qmcplusplus
{

using namespace afqmc;

void myREQUIRE(const double& a, const double& b)
{
  REQUIRE(a == Approx(b));
}

void myREQUIRE(const std::complex<double>& a, const double& b)
{
  REQUIRE(a.real() == Approx(b));
}

void myREQUIRE(const std::complex<double>& a, const std::complex<double>& b)
{
  REQUIRE(a.real() == Approx(b.real()));
  REQUIRE(a.imag() == Approx(b.imag()));
}

template<class M1, class M2>
void check(M1&& A, M2& B)
{
  REQUIRE(A.shape()[0] == B.shape()[0]);
  REQUIRE(A.shape()[1] == B.shape()[1]);
  for(int i=0; i<A.shape()[0]; i++)
    for(int j=0; j<A.shape()[1]; j++)
      myREQUIRE(A[i][j],B[i][j]);
}

using namespace afqmc;
using communicator = boost::mpi3::communicator;

void test_basic_walker_features(bool serial)
{
  OHMMS::Controller->initialize(0, NULL);
  OhmmsInfo("testlogfile",boost::mpi3::world.rank());

  using Type = std::complex<double>;

  communicator& world = boost::mpi3::world;
  //assert(world.size()%2 == 0); 

  int NMO=8,NAEA=2,NAEB=2, nwalkers=10;

  //auto node = world.split_shared();

  GlobalTaskGroup gTG(world);
  TaskGroup_ TG(gTG,std::string("TaskGroup"),1,serial?1:gTG.getTotalCores()); 
  AFQMCInfo info;
  info.NMO = NMO; 
  info.NAEA = NAEA; 
  info.NAEB = NAEB; 
  info.name = "walker";
  boost::multi_array<Type,2> initA(extents[NMO][NAEA]); 
  boost::multi_array<Type,2> initB(extents[NMO][NAEB]); 
  for(int i=0; i<NAEA; i++) initA[i][i] = Type(1.0);
  for(int i=0; i<NAEB; i++) initB[i][i] = Type(1.0);
  //SimpleRandom<MTRand> rng;
  RandomGenerator_t rng;

const char *xml_block = 
"<WalkerSet name=\"wset0\">  \
  <parameter name=\"min_weight\">0.05</parameter>  \
  <parameter name=\"max_weight\">4</parameter>  \
  <parameter name=\"walker_type\">closed</parameter>  \
  <parameter name=\"load_balance\">async</parameter>  \
  <parameter name=\"pop_control\">pair</parameter>  \
</WalkerSet> \
";
  Libxml2Document doc;
  bool okay = doc.parseFromString(xml_block);
  REQUIRE(okay);

  SharedWalkerSet wset(TG,doc.getRoot(),info,&rng);
  wset.resize(nwalkers,initA,initB);

  REQUIRE( wset.size() == nwalkers );
  int cnt=0;
  double tot_weight=0.0;
  for(SharedWalkerSet::iterator it=wset.begin(); it!=wset.end(); ++it) {
    auto sm = it->SlaterMatrix(Alpha);
    REQUIRE( it->SlaterMatrix(Alpha) == initA );
    it->weight() = cnt*1.0+0.5;
    it->overlap() = cnt*1.0+0.5;
    it->E1() = cnt*1.0+0.5;
    it->EXX() = cnt*1.0+0.5;
    it->EJ() = cnt*1.0+0.5;
    tot_weight+=cnt*1.0+0.5;
    cnt++;
  }
  REQUIRE(cnt==nwalkers);
  cnt=0;
  for(SharedWalkerSet::iterator it=wset.begin(); it!=wset.end(); ++it) {
    Type d_(cnt*1.0+0.5);
    REQUIRE( it->weight() == d_ ); 
    REQUIRE( it->overlap() == cnt*1.0+0.5 );
    REQUIRE( it->E1() == cnt*1.0+0.5 );
    REQUIRE( it->EXX() == cnt*1.0+0.5 );
    REQUIRE( it->EJ() == cnt*1.0+0.5 );
    cnt++;
  }

  wset.reserve(20);
  REQUIRE(wset.capacity() == 20);
  cnt=0;
  for(SharedWalkerSet::iterator it=wset.begin(); it!=wset.end(); ++it) {
    REQUIRE( it->weight() == cnt*1.0+0.5 );
    REQUIRE( it->overlap() == cnt*1.0+0.5 );
    REQUIRE( it->E1() == cnt*1.0+0.5 );
    REQUIRE( it->EXX() == cnt*1.0+0.5 );
    REQUIRE( it->EJ() == cnt*1.0+0.5 );
    cnt++;
  }
/*
cout<<" after -> " <<std::endl;
for(SharedWalkerSet::iterator it=wset.begin(); it!=wset.begin()+1; ++it) {
  REQUIRE( (*it).weight() == cnt*1.0+0.5 );
}
cout<<" after (*it) test  " <<std::endl;
*/
  for(int i=0; i<wset.size(); i++) {
    REQUIRE( wset[i].weight() == i*1.0+0.5 );
    REQUIRE( wset[i].overlap() == i*1.0+0.5 );
    REQUIRE( wset[i].E1() == i*1.0+0.5 );
    REQUIRE( wset[i].EXX() == i*1.0+0.5 );
    REQUIRE( wset[i].EJ() == i*1.0+0.5 );
  }
  for(int i=0; i<wset.size(); i++) {
    auto w = wset[i];
    REQUIRE( w.weight() == i*1.0+0.5 );
    REQUIRE( w.overlap() == i*1.0+0.5 );
    REQUIRE( w.E1() == i*1.0+0.5 );
    REQUIRE( w.EXX() == i*1.0+0.5 );
    REQUIRE( w.EJ() == i*1.0+0.5 );
  }
  REQUIRE(wset.get_TG_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers*TG.getNumberOfTGs());
  REQUIRE(wset.GlobalPopulation() == nwalkers*TG.getNumberOfTGs());
  REQUIRE(wset.GlobalPopulation() == wset.get_global_target_population()); 
  REQUIRE(wset.getNBackProp() == 0);
  REQUIRE(wset.GlobalWeight() == tot_weight*TG.getNumberOfTGs());
  
  wset.scaleWeight(2.0);
  tot_weight*=2.0;
  REQUIRE(wset.GlobalWeight() == tot_weight*TG.getNumberOfTGs());
  
  std::vector<ComplexType> Wdata;
  wset.popControl(Wdata);
  REQUIRE(wset.get_TG_target_population() == nwalkers);
  REQUIRE(wset.get_global_target_population() == nwalkers*TG.getNumberOfTGs());
  REQUIRE(wset.GlobalPopulation() == nwalkers*TG.getNumberOfTGs());
  REQUIRE(wset.GlobalPopulation() == wset.get_global_target_population());  
  REQUIRE(wset.GlobalWeight() == Approx(static_cast<RealType>(wset.get_global_target_population()))); 
  for(int i=0; i<wset.size(); i++) {
    auto w = wset[i];
    REQUIRE( w.overlap() == w.E1()); 
    REQUIRE( w.EXX() == w.E1());
    REQUIRE( w.EJ() == w.E1());
  }

  wset.clean();
  REQUIRE(wset.size() == 0);
  REQUIRE(wset.capacity() == 0);

}

TEST_CASE("swset_test_serial", "[shared_wset]")
{
  test_basic_walker_features(true);
  test_basic_walker_features(false);
}

}
