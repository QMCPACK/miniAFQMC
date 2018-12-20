//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
// Alfredo Correa, correaa@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_INITIALIZE_HPP
#define QMCPLUSPLUS_AFQMC_INITIALIZE_HPP

#include<string>
#include<vector>

#include "Configuration.h"
#include "io/hdf_archive.h"

#include "AFQMC/afqmc_sys.hpp"
#include "Matrix/hdfcsr2ma.hpp"
#include "Numerics/ma_blas.hpp"
#include "Utilities/kp_utilities.hpp"
#include "Utilities/rotate_padded.hpp"
#include "Utilities/rotate.hpp"

//#include "AFQMC/KP3IndexFactorization.hpp"

#include "Numerics/detail/cuda_pointers.hpp"

namespace qmcplusplus
{

namespace afqmc
{

template<class HOps,
         class af_sys
            >
inline HOps Initialize(hdf_archive& dump, const double dt, af_sys& sys, ComplexMatrix<typename af_sys::Alloc>& Propg1)
{
  // the allocator of af_sys must be consistent with Alloc, otherwise LA operations will not work 
  using Alloc = typename af_sys::Alloc;
  using SpAlloc = typename Alloc::template rebind<SPComplexType>::other;
  using IAlloc = typename Alloc::template rebind<int>::other;
  using stdAlloc = std::allocator<ComplexType>; 
  using stdSpAlloc = std::allocator<SPComplexType>; 
  using stdIAlloc = std::allocator<int>; 

  using SpMatrix = SPComplexMatrix<SpAlloc>; 
  using SpTensor = SPComplex3Tensor<SpAlloc>; 
  using CMatrix = ComplexMatrix<Alloc>; 
  using CTensor = Complex3Tensor<Alloc>; 

  using devSpMatrix = SPComplexMatrix<SpAlloc>;                    
  using devSpTensor = SPComplex3Tensor<SpAlloc>;                   
  using devCMatrix = ComplexMatrix<Alloc>;  
  using devCTensor = Complex3Tensor<Alloc>; 

  using pointer = typename Alloc::pointer;
  using sp_pointer = typename SpAlloc::pointer;

  using SpTensor_ref = SPComplex3Tensor_ref<sp_pointer>;
  using Sp4Tensor_ref = SPComplexArray_ref<4,sp_pointer>;

  Alloc& alloc(sys.allocator_); 
  SpAlloc spalloc(alloc); 
  IAlloc ialloc(alloc); 

  Alloc& dev_alloc(sys.allocator_); 
  SpAlloc dev_spalloc(alloc); 
  IAlloc dev_ialloc(alloc); 
  
  WALKER_TYPES type(sys.walker_type);
  int NMO, NAEA, NAEB;
  std::vector<int> dims(7);
  int ndet = 1; 
  int nspins = ((type!=COLLINEAR)?1:2);

  std::cout<<"  Serial hdf5 read. \n";

  // read from HDF
  if(!dump.push("Wavefunction",false)) {
    app_error()<<" Error in initialize: Group Wavefunction not found. \n";
    APP_ABORT("");
  }
  if(!dump.push("NOMSD",false)) {
    app_error()<<" Error in initialize: Group NOMSD not found. \n";
    APP_ABORT("");
  }
  if(!dump.push("HamiltonianOperations",false)) {
    app_error()<<" Error in initialize: Group HamiltonianOperations not found. \n";
    APP_ABORT("");
  }
  if(!dump.push("KP3IndexFactorization",false)) {
    app_error()<<" Error in initialize: Group KP3IndexFactorization not found. \n";
    APP_ABORT("");
  }
  if(!dump.read(dims,"dims")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  assert(dims.size()==7);
  NMO = dims[0];
  NAEA = dims[1];
  NAEB = dims[2];
  if(NAEA!=NAEB) {
    app_error()<<" Error in initialize: NAEA != NAEB. \n"; 
    APP_ABORT("");
  }
  int nkpts = dims[3]; 
  int nsampleQ = dims[5];
  std::vector<ValueType> et;
  if(!dump.read(et,"E0")) {
    app_error()<<" Error in initialize: Problems reading dataset. \n";
    APP_ABORT("");
  }
  ValueType E0=et[0];

  int NEL = (sys.walker_type == COLLINEAR)?2*NAEA:NAEA;

  std::vector<int> nmo_per_kp(nkpts);
  std::vector<int> nchol_per_kp(nkpts);
  std::vector<int> kminus(nkpts);
  std::vector<RealType> gQ;
  boost::multi::array<int,2> QKtok2({nkpts,nkpts});

  {
    if(!dump.read(nmo_per_kp,"NMOPerKP")) {
      app_error()<<" Error in loadKP3IndexFactorization: Problems reading NMOPerKP dataset. \n";
      APP_ABORT("");
    }
    if(!dump.read(nchol_per_kp,"NCholPerKP")) {
      app_error()<<" Error in loadKP3IndexFactorization: Problems reading NCholPerKP dataset. \n";
      APP_ABORT("");
    }
    if(!dump.read(kminus,"MinusK")) {
      app_error()<<" Error in loadKP3IndexFactorization: Problems reading MinusK dataset. \n";
      APP_ABORT("");
    }
    if(!dump.read(QKtok2,"QKTok2")) {
      app_error()<<" Error in loadKP3IndexFactorization: Problems reading QKTok2 dataset. \n";
      APP_ABORT("");
    }
    if(!dump.read(gQ,"gQ")) {
      app_error()<<" Error in loadKP3IndexFactorization: Problems reading gQ dataset. \n";
      APP_ABORT("");
    }
  }

  int Q0=-1;  // stores the index of the Q=(0,0,0) Q-point 
              // this must always exist
  for(int Q=0; Q<nkpts; Q++) {
    if(kminus[Q]==Q) {
      bool found=true;
      for(int KI=0; KI<nkpts; KI++)
        if(KI != QKtok2[Q][KI]) {
          found = false;
          break;
        }
      if(found) {
        if( Q0 > 0 )
          APP_ABORT(" Error: @ Q-points satisfy Q=0 condition.\n");
        Q0=Q;
      } else {
        if( nkpts%2 != 0)
          APP_ABORT(" Error: Unexpected situation: Q==(-Q)!=Q0 and odd Nk. \n");
      }
    }
  }
  if(Q0<0)
    APP_ABORT(" Error: Can not find Q=0 Q-point.\n");

  int nmo_max = *std::max_element(nmo_per_kp.begin(),nmo_per_kp.end());
  int nchol_max = *std::max_element(nchol_per_kp.begin(),nchol_per_kp.end());
  devCTensor H1({nkpts,nmo_max,nmo_max},alloc);
  devCTensor vn0({nkpts,nmo_max,nmo_max},alloc);
  std::vector<devSpMatrix> LQKikn;
  LQKikn.reserve(nkpts);
  for(int Q=0; Q<nkpts; Q++)
    if( Q==Q0 )
      LQKikn.emplace_back( devSpMatrix({nkpts,nmo_max*nmo_max*nchol_max},
                                   dev_spalloc) );
    else if( kminus[Q] == Q )   // only storing half of K points and using symmetry 
      LQKikn.emplace_back( devSpMatrix({nkpts/2,nmo_max*nmo_max*nchol_max},
                                   dev_spalloc) ); 
    else if(Q < kminus[Q])
      LQKikn.emplace_back( devSpMatrix({nkpts,nmo_max*nmo_max*nchol_max},
                                   dev_spalloc) ); 
    else // Q > kminus[Q]
      LQKikn.emplace_back( devSpMatrix({1,1}, dev_spalloc) );

  for( auto& v: LQKikn ) fill_n(v.origin(),v.num_elements(),SPComplexType(0.0));


  ComplexMatrix<stdAlloc> H1_local({nkpts,nmo_max*nmo_max});
  ComplexMatrix<stdAlloc> vn0_local({nkpts,nmo_max*nmo_max});
  {
    {
      if(!dump.read(H1_local,"KPH1")) {
        app_error()<<" Error in loadKP3IndexFactorization: Problems reading dataset. \n";
        APP_ABORT("");
      }
      cuda::copy_n(H1_local.origin(),H1_local.num_elements(),H1.origin());
      if(!dump.read(vn0_local,"KPv0")) {
        app_error()<<" Error in loadKP3IndexFactorization: Problems reading dataset. \n";
        APP_ABORT("");
      }
      cuda::copy_n(vn0_local.origin(),vn0_local.num_elements(),vn0.origin());
    }
    devSpMatrix L_({1,1}, dev_spalloc);
    for(int Q=0; Q<nkpts; Q++) {
      if(Q > kminus[Q]) continue;
      if(!dump.read(L_,std::string("L")+std::to_string(Q))) {
        app_error()<<" Error in loadKP3IndexFactorization: Problems reading dataset. \n";
        APP_ABORT("");
      }
      int nchol = nchol_per_kp[Q];  
      if(Q < kminus[Q] || Q==Q0) {  
        assert(L_.shape()[0] == nkpts);
        Sp4Tensor_ref L2(LQKikn[Q].origin(),{nkpts,nmo_max,nmo_max,nchol_max});
        for(int K=0; K<nkpts; ++K) {        // K is the index of the kpoint pair of (i,k)
          int QK = QKtok2[Q][K];
          int ni = nmo_per_kp[K];
          int nk = nmo_per_kp[QK];
          SpTensor_ref L1(L_[K].origin(),{ni,nk,nchol});
          for(int i=0; i<ni; i++)
            for(int k=0; k<nk; k++)
              copy_n(L1[i][k].origin(),nchol,L2[K][i][k].origin());
        } 
      } else {
        assert(L_.shape()[0] == nkpts/2);
        Sp4Tensor_ref L2(LQKikn[Q].origin(),{nkpts/2,nmo_max,nmo_max,nchol_max});
        for(int K_=0; K_<nkpts/2; ++K_) {        // K is the index of the kpoint pair of (i,k)
          // find the (K,QK) pair on symmetric list
          int cnt(0);
          int K=-1;
          for(int q=0; q<nkpts; q++)
            if(q < QKtok2[Q][q]) {
              if(cnt==K_) {
                K=q;
                break;
              }
              cnt++;
            }
          if(K<0)
            APP_ABORT(" Error: Problems with QK mapping.\n");
          int QK = QKtok2[Q][K];
          int ni = nmo_per_kp[K];
          int nk = nmo_per_kp[QK];
          SpTensor_ref L1(L_[K_].origin(),{ni,nk,nchol});
          for(int i=0; i<ni; i++)
            for(int k=0; k<nk; k++)
              copy_n(L1[i][k].origin(),nchol,L2[K_][i][k].origin());
        }
      }  
    }
  }

  dump.pop();
  dump.pop();

  // set values in sys
  {
    if(!dump.push(std::string("PsiT_0"),false)) {
      app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
      APP_ABORT("");
    }
    {
      ComplexMatrix<stdAlloc> buff(hdfcsr2ma<ComplexMatrix<stdAlloc>,stdAlloc>(dump,NMO,NAEA,stdAlloc{}));
      // can't figure out why std::copy_n is used here, forcing cuda::copy_n for now
      cuda::copy_n(buff.origin(),buff.num_elements(),sys.trialwfn_alpha.origin());
      dump.pop();
    }
    if(sys.walker_type == COLLINEAR) {
      if(!dump.push(std::string("PsiT_1"),false)) {
        app_error()<<" Error in WavefunctionFactory: Group PsiT not found. \n";
        APP_ABORT("");
      }
      ComplexMatrix<stdAlloc> buff(hdfcsr2ma<ComplexMatrix<stdAlloc>,stdAlloc>(dump,NMO,NAEA,stdAlloc{}));
      cuda::copy_n(buff.origin(),buff.num_elements(),sys.trialwfn_beta.origin());
      dump.pop();
    } 
  }

  dump.pop();
  dump.pop();
  // closing it here, be careful!
  dump.close();

  boost::multi::array<int,2> nocc_per_kp({ndet,nspins*nkpts});
  {
    ComplexMatrix<stdAlloc> psiA({NMO,NAEA});
    cuda::copy_n(sys.trialwfn_alpha.origin(),sys.trialwfn_alpha.num_elements(),psiA.origin());
    if(not get_nocc_per_kp(nmo_per_kp,psiA,nocc_per_kp[0]({0,nkpts}))) {
      app_error()<<" Error in KPFactorizedHamiltonian::getHamiltonianOperations():"
             <<" Only wavefunctions in block-diagonal form are accepted. \n";
      APP_ABORT("");
    }
    if(type==COLLINEAR) {
      cuda::copy_n(sys.trialwfn_beta.origin(),sys.trialwfn_beta.num_elements(),psiA.origin());
      if(not get_nocc_per_kp(nmo_per_kp,psiA,nocc_per_kp[0]({nkpts,2*nkpts}))) {
        app_error()<<" Error in KPFactorizedHamiltonian::getHamiltonianOperations():"
               <<" Only wavefunctions in block-diagonal form are accepted. \n";
        APP_ABORT("");
      }
    }
  }
  int nocc_max = *std::max_element(std::addressof(*nocc_per_kp.origin()),
                                   std::addressof(*nocc_per_kp.origin())+nocc_per_kp.num_elements());

  std::vector<devSpMatrix> LQKank;
  LQKank.reserve(ndet*nspins*(nkpts+1));  // storing 2 components for Q=0, since it is not assumed symmetric 
  devCMatrix haj({ndet*nkpts,(type==COLLINEAR?2:1)*nocc_max*nmo_max},dev_alloc);
  cuda::fill_n(haj.origin(),haj.num_elements(),ComplexType(0.0));
  int ank_max = nocc_max*nchol_max*nmo_max;
    for(int Q=0; Q<(nkpts+1); Q++) {
      LQKank.emplace_back(devSpMatrix({nkpts,ank_max},dev_spalloc));
    }
    if(type==COLLINEAR) {
      for(int Q=0; Q<(nkpts+1); Q++) {
        LQKank.emplace_back(devSpMatrix({nkpts,ank_max},dev_spalloc));
      }
    }
  for( auto& v: LQKank ) fill_n(v.origin(),v.num_elements(),SPComplexType(0.0)); 

  std::vector<devSpMatrix> LQKakn;
  LQKakn.reserve(ndet*nspins*(nkpts+1));  
    for(int Q=0; Q<(nkpts+1); Q++) {
      LQKakn.emplace_back(devSpMatrix({nkpts,ank_max},dev_spalloc));
    }
    if(type==COLLINEAR) {
      for(int Q=0; Q<(nkpts+1); Q++) {
        LQKakn.emplace_back(devSpMatrix({nkpts,ank_max},dev_spalloc));
      }
    }
  for( auto& v: LQKakn ) fill_n(v.origin(),v.num_elements(),SPComplexType(0.0)); 

  SPComplexMatrix<SpAlloc> buff({nmo_max,nchol_max},spalloc);
  double scl(2.0);
  if(type==COLLINEAR) scl=1.0;
  for(int K=0, na0=0, nb0=0, ni0=0; K<nkpts; K++) {
    int na = nocc_per_kp[0][K];
    int nb = nocc_per_kp[0][(type==COLLINEAR?nkpts:0)+K];
    int ni = nmo_per_kp[K];
//    auto Psi = get_PsiK<ComplexMatrix<Alloc>>(nmo_per_kp,sys.trialwfn_alpha,K);
//    auto Psib = get_PsiK<ComplexMatrix<Alloc>>(nmo_per_kp,sys.trialwfn_beta,K);
    auto&& Psi(sys.trialwfn_alpha({ni0,ni0+ni},{na0,na0+na}));
    auto* tPsi(std::addressof(sys.trialwfn_alpha));
    if(type==COLLINEAR) tPsi = std::addressof(sys.trialwfn_beta);
    auto&& Psib((*tPsi)({ni0,ni0+ni},{nb0,nb0+nb}));
    //auto&& Psib(sys.trialwfn_beta({ni0,ni0+ni},{nb0,nb0+nb}));
#ifdef MIXED_PRECISION
//    auto spPsi = get_PsiK<ComplexMatrix<SpAlloc>>(nmo_per_kp,sys.trialwfn_alpha,K);
//    auto spPsib = get_PsiK<ComplexMatrix<SpAlloc>>(nmo_per_kp,sys.trialwfn_beta,K);
    auto&& spPsi = Psi;
    auto&& spPsib = Psib;
#else
    auto&& spPsi = Psi;
    auto&& spPsib = Psib;
#endif
    na0+=na;
    nb0+=nb; 
    ni0+=ni;
    for(int Q=0; Q<nkpts; Q++) {
      // haj and add half-transformed right-handed rotation for Q=0 
      int Qm = kminus[Q];
      int QK = QKtok2[Q][K];
      int nk = nmo_per_kp[QK];
      int nchol = nchol_per_kp[Q];
      assert(Psi.shape()[1] == na);
      assert(Psib.shape()[1] == nb);
      if(Q==0) {
        ComplexMatrix_ref<pointer> haj_r(haj[K].origin(),{na,ni});
        ma::product(ComplexType(scl),ma::T(Psi),H1[K]({0,ni},{0,ni}),
                    ComplexType(0.0),haj_r);
        if(type==COLLINEAR) {
          ComplexMatrix_ref<pointer> haj_r(haj[K].origin()+na*ni,{nb,ni});
          ma::product(ComplexType(scl),ma::T(Psib),H1[K]({0,ni},{0,ni}),
                    ComplexType(0.0),haj_r);
        }
      }
      if(type==COLLINEAR) {
        { // Alpha 
          if(Q < Qm || Q==Q0 || ((Q==Qm)&&(K<QK))) {
            int kpos = K;
            if( Q==Qm && Q!=Q0 ) { //find position in symmetric list  
              kpos=0;
              for(int K_=0; K_<K; K_++)
                if(K_ < QKtok2[Q][K_]) kpos++;
            }
            SpTensor_ref Likn(LQKikn[Q][kpos].origin(),{nmo_max,nmo_max,nchol_max});

            SpTensor_ref Lank(LQKank[Q][K].origin(),{na,nchol,nk});
            ma_rotate_padded::getLank(spPsi,Likn,Lank,buff);
            if(Q==Q0) {
              assert(K==QK);
              SpTensor_ref Lank(LQKank[nkpts][K].origin(),{na,nchol,nk});
              ma_rotate_padded::getLank_from_Lkin(spPsi,Likn,Lank,buff);
            }

            SpTensor_ref Lakn(LQKakn[Q][K].origin(),{nocc_max,nmo_max,nchol_max});
            ma_rotate_padded::getLakn(spPsi,Likn,Lakn,buff);
            if(Q==Q0) {
              assert(K==QK);
              SpTensor_ref Lakn(LQKakn[nkpts][K].origin(),{nocc_max,nmo_max,nchol_max});
              ma_rotate_padded::getLakn_from_Lkin(spPsi,Likn,Lakn,buff);
            }
          } else {
            int kpos = QK;
            if( Q==Qm ) { //find position in symmetric list  
              kpos=0;
              for(int K_=0; K_<QK; K_++)
                if(K_ < QKtok2[Q][K_]) kpos++;
            }
            SpTensor_ref Lkin(LQKikn[Qm][QK].origin(),{nmo_max,nmo_max,nchol_max});
            SpTensor_ref Lank(LQKank[Q][K].origin(),{na,nchol,nk});
            ma_rotate_padded::getLank_from_Lkin(spPsi,Lkin,Lank,buff);
            SpTensor_ref Lakn(LQKakn[Q][K].origin(),{nocc_max,nmo_max,nchol_max});
            ma_rotate_padded::getLakn_from_Lkin(spPsi,Lkin,Lakn,buff);
          }
        }
        { // Beta 
          if(Q < Qm || Q==Q0 || ((Q==Qm)&&(K<QK))) {
            int kpos = K;
            if( Q==Qm && Q!=Q0 ) { //find position in symmetric list  
              kpos=0;
              for(int K_=0; K_<K; K_++)
                if(K_ < QKtok2[Q][K_]) kpos++;
            }
            SpTensor_ref Likn(LQKikn[Q][kpos].origin(),{nmo_max,nmo_max,nchol_max});

            SpTensor_ref Lank(LQKank[nkpts+1+Q][K].origin(),{nb,nchol,nk});
            ma_rotate_padded::getLank(spPsib,Likn,Lank,buff);
            if(Q==Q0) {
              assert(K==QK);
              SpTensor_ref Lank(LQKank[nkpts+1+nkpts][K].origin(),{nb,nchol,nk});
              ma_rotate_padded::getLank_from_Lkin(spPsib,Likn,Lank,buff);
            }

            SpTensor_ref Lakn(LQKakn[nkpts+1+Q][K].origin(),{nocc_max,nmo_max,nchol_max});
            ma_rotate_padded::getLakn(spPsib,Likn,Lakn,buff);
            if(Q==Q0) {
              assert(K==QK);
              SpTensor_ref Lakn(LQKakn[nkpts+1+nkpts][K].origin(),{nocc_max,nmo_max,nchol_max});
              ma_rotate_padded::getLakn_from_Lkin(spPsib,Likn,Lakn,buff);
            }

          } else {
            int kpos = QK;
            if( Q==Qm ) { //find position in symmetric list  
              kpos=0;
              for(int K_=0; K_<QK; K_++)
                if(K_ < QKtok2[Q][K_]) kpos++;
            }
            SpTensor_ref Lkin(LQKikn[Qm][QK].origin(),{nmo_max,nmo_max,nchol_max});
            SpTensor_ref Lank(LQKank[nkpts+1+Q][K].origin(),{nb,nchol,nk});
            ma_rotate_padded::getLank_from_Lkin(spPsib,Lkin,Lank,buff);
            SpTensor_ref Lakn(LQKakn[nkpts+1+Q][K].origin(),{nocc_max,nmo_max,nchol_max});
            ma_rotate_padded::getLakn_from_Lkin(spPsib,Lkin,Lakn,buff);
          }
        }
      } else {
        if(Q < Qm || Q==Q0 || ((Q==Qm)&&(K<QK))) {
          int kpos = K;
          if( Q==Qm && Q!=Q0 ) { //find position in symmetric list  
            kpos=0;
            for(int K_=0; K_<K; K_++)
              if(K_ < QKtok2[Q][K_]) kpos++;
          }
          SpTensor_ref Likn(LQKikn[Q][kpos].origin(),{nmo_max,nmo_max,nchol_max});

          SpTensor_ref Lank(LQKank[Q][K].origin(),{na,nchol,nk});
          ma_rotate_padded::getLank(spPsi,Likn,Lank,buff);
          if(Q==Q0) {
            assert(K==QK);
            SpTensor_ref Lank(LQKank[nkpts][K].origin(),{na,nchol,nk});
            ma_rotate_padded::getLank_from_Lkin(spPsi,Likn,Lank,buff);
          }

          SpTensor_ref Lakn(LQKakn[Q][K].origin(),{nocc_max,nmo_max,nchol_max});
          ma_rotate_padded::getLakn(spPsi,Likn,Lakn,buff);
          if(Q==Q0) {
            assert(K==QK);
            SpTensor_ref Lakn(LQKakn[nkpts][K].origin(),{nocc_max,nmo_max,nchol_max});
            ma_rotate_padded::getLakn_from_Lkin(spPsi,Likn,Lakn,buff);
          }

        } else {
          int kpos = QK;
          if( Q==Qm ) { //find position in symmetric list  
            kpos=0;
            for(int K_=0; K_<QK; K_++)
              if(K_ < QKtok2[Q][K_]) kpos++;
          }
          SpTensor_ref Lkin(LQKikn[Qm][kpos].origin(),{nmo_max,nmo_max,nchol_max});
          SpTensor_ref Lank(LQKank[Q][K].origin(),{na,nchol,nk});
          ma_rotate_padded::getLank_from_Lkin(spPsi,Lkin,Lank,buff);
          SpTensor_ref Lakn(LQKakn[Q][K].origin(),{nocc_max,nmo_max,nchol_max});
          ma_rotate_padded::getLakn_from_Lkin(spPsi,Lkin,Lakn,buff);
        }
      }
    }
  }
  
  int global_ncvecs = std::accumulate(nchol_per_kp.begin(),nchol_per_kp.end(),0);

  // propagator
  ComplexMatrix<stdAlloc> P1( {NMO,NMO}, stdAlloc{});
  ComplexMatrix<stdAlloc> v0( {NMO,NMO}, stdAlloc{} );
  for(int K=0, nk0=0; K<nkpts; ++K) {
    for(int i=0, I=nk0; i<nmo_per_kp[K]; i++, I++) {
      P1[I][I] = -0.5*dt*(H1_local[K][i*nmo_max+i] + vn0_local[K][i*nmo_max+i]);
      for(int j=i+1, J=I+1; j<nmo_per_kp[K]; j++, J++) {
        P1[I][J] = H1_local[K][i*nmo_max+j] + vn0_local[K][i*nmo_max+j];
        P1[J][I] = H1_local[K][j*nmo_max+i] + vn0_local[K][j*nmo_max+i];
        P1[I][J] = -0.5*dt*(P1[I][J]+conj(P1[J][I]));
        P1[J][I] = conj(P1[I][J]);
      }
    }
    nk0 += nmo_per_kp[K];
  }
  v0 = ma::exp(P1);
  //Propg1.reextent( {NMO,NMO} );
  // need specialized copy routine
  using std::copy_n;
  cuda::copy_n(v0.origin(),v0.num_elements(),Propg1.origin());

  //return KP3IndexFactorization<Alloc,Alloc>(type,std::move(nmo_per_kp),
  return HOps(type,std::move(nmo_per_kp),
            std::move(nchol_per_kp),std::move(kminus),std::move(nocc_per_kp),
            std::move(QKtok2),std::move(H1),std::move(haj),std::move(LQKikn),
            std::move(LQKank),std::move(LQKakn),std::move(vn0),std::move(gQ),nsampleQ,E0,
            alloc,alloc,global_ncvecs);

} 

}  // afqmc


} // qmcplusplus


#endif
