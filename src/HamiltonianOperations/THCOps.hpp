//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov 
//    Lawrence Livermore National Laboratory 
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_HAMILTONIANOPERATIONS_THCOPS_HPP
#define QMCPLUSPLUS_AFQMC_HAMILTONIANOPERATIONS_THCOPS_HPP

#include<fstream>

#include "Utilities/UtilityFunctions.h"
#include "af_config.h"
#include "Numerics/ma_operations.hpp"

#include "Utilities/taskgroup.h"
#include "alf/boost/mpi3/shared_communicator.hpp"
#include "Matrix/mpi3_SHMBuffer.hpp"
#include "Matrix/mpi3_shared_ma_proxy.hpp"
#include "type_traits/scalar_traits.h"


namespace qmcplusplus
{

namespace afqmc
{

template<class T>
class THCOps
{
#if defined(AFQMC_SP) 
  using SpT = typename to_single_precision<T>::value_type;
  using SpC = typename to_single_precision<ComplexType>::value_type;
#else
  using SpT = T;
  using SpC = ComplexType;  
#endif

  using SpTVector = MArray<SpT,1>;
  using CVector = MArray<ComplexType,1>;
  using CMatrix = MArray<ComplexType,2>;
  using TMatrix = MArray<T,2>;
  using shmCMatrix = mpi3_shared_ma_proxy<ComplexType>;
  using shmVMatrix = mpi3_shared_ma_proxy<T>;
  using communicator = boost::mpi3::shared_communicator;
  using SHM_Buffer = mpi3_SHMBuffer<ComplexType>;

  public:

    THCOps(communicator& c_,
           int nmo_, int naea_, int naeb_,  
           WALKER_TYPES type, 
           CMatrix&& hij_,
           std::vector<SpTVector>&& h1,
           shmCMatrix&& rotmuv_,
           shmCMatrix&& rotpiu_,
           std::vector<shmCMatrix>&& rotpau_,
           shmVMatrix&& luv_,
           shmVMatrix&& piu_,
           std::vector<shmCMatrix>&& pau_,
           TMatrix&& v0_,
           ValueType e0_,
           bool verbose=false ):
                comm(std::addressof(c_)),
                NMO(nmo_),NAEA(naea_),NAEB(naeb_),
                walker_type(type),
                hij(std::move(hij_)),
                haj(std::move(h1)),
                rotMuv(std::move(rotmuv_)),
                rotPiu(std::move(rotpiu_)),
                rotcPua(std::move(rotpau_)),
                Luv(std::move(luv_)),
                Piu(std::move(piu_)),
                cPua(std::move(pau_)),
                v0(std::move(v0_)),
                E0(e0_),
                SM_TMats(nullptr) 
    {
      assert(comm);
      // current partition over 'u' for L/Piu
      assert(Luv.shape()[0] == Piu.shape()[1]);
      for(int i=0; i<rotcPua.size(); i++) {
        // rot Ps are not yet distributed
        assert(rotcPua[i].shape()[0] == rotPiu.shape()[1]);
        if(walker_type==CLOSED)
          assert(rotcPua[i].shape()[1]==NAEA);
        else if(walker_type==COLLINEAR)
          assert(rotcPua[i].shape()[1]==NAEA+NAEB);
        else if(walker_type==NONCOLLINEAR)
          assert(rotcPua[i].shape()[1]==NAEA+NAEB);
      }
      for(int i=0; i<cPua.size(); i++) {
        assert(cPua[i].shape()[0]==Luv.shape()[0]);
        if(walker_type==CLOSED)
          assert(cPua[i].shape()[1]==NAEA);
        else if(walker_type==COLLINEAR)
          assert(cPua[i].shape()[1]==NAEA+NAEB);
        else if(walker_type==NONCOLLINEAR)
          assert(cPua[i].shape()[1]==NAEA+NAEB);
      }
      if(walker_type==NONCOLLINEAR) {
        assert(Piu.shape()[0]==2*NMO);
        assert(rotPiu.shape()[0]==2*NMO);
      } else {
        assert(Piu.shape()[0]==NMO);
        assert(rotPiu.shape()[0]==NMO);
      }
    }

    ~THCOps() {}
    
    THCOps(THCOps const& other) = delete;
    THCOps& operator=(THCOps const& other) = delete;

    THCOps(THCOps&& other) = default;
    THCOps& operator=(THCOps&& other) = default; 

    CMatrix getOneBodyPropagatorMatrix(TaskGroup_& TG, CVector const& vMF) {
      int NMO = hij.shape()[0];
      // in non-collinear case with SO, keep SO matrix here and add it
      // for now, stay collinear
      CMatrix H1({NMO,NMO});
      // add sum_n vMF*Spvn, vMF has local contribution only!
      MArray_ref<ComplexType,1> H1D(H1.origin(),{NMO*NMO});
      std::fill_n(H1D.origin(),H1D.num_elements(),ComplexType(0));
      vHS(vMF, H1D);
      TG.TG().all_reduce_in_place_n(H1D.origin(),H1D.num_elements(),std::plus<>());

      // add hij + v0 and symmetrize
      using std::conj;
      for(int i=0; i<NMO; i++) {
        H1[i][i] += hij[i][i] + v0[i][i];
        for(int j=i+1; j<NMO; j++) {
          H1[i][j] += hij[i][j] + v0[i][j];
          H1[j][i] += hij[j][i] + v0[j][i];
          if( std::abs( H1[i][j] - conj(H1[j][i]) ) > 1e-8 ) {
            app_error()<<" Error in getOneBodyPropagatorMatrix. H1 is not hermitian. \n";
            app_error()<<i <<" " <<j <<" " <<H1[i][j] <<" " <<H1[j][i] <<" "
                       <<H1[i][j]-(hij[i][j] + v0[i][j]) <<" " <<H1[j][i]-(hij[j][i] + v0[j][i]) <<" "
                       <<hij[i][j] <<" " <<hij[j][i] <<" "
                       <<v0[i][j] <<" " <<v0[j][i] <<std::endl;
            APP_ABORT("Error in getOneBodyPropagatorMatrix. H1 is not hermitian. \n");
          }
          H1[i][j] = 0.5*(H1[i][j]+conj(H1[j][i]));
          H1[j][i] = conj(H1[i][j]);
        }
      }
      return H1;
    }

    
    template<class Mat, class MatB>
    void energy(Mat&& E, MatB const& G, int k, bool addH1=true, bool addEJ=true, bool addEXX=true) {
      // G[nel][nmo]
      //static_assert(E.dimensionality==2);  
      //static_assert(G.dimensionality==2);  
      assert(E.shape()[0] == G.shape()[0]);        
      assert(E.shape()[1] == 3);        
      int nwalk = G.shape()[0];
      int nmo_ = rotPiu.shape()[0];
      int nu = rotMuv.shape()[0]; 
      int nu0 = rotMuv.offset()[0]; 
      int nv = rotMuv.shape()[1]; 
      int nel_ = rotcPua[0].shape()[1];
      int nspin = (walker_type==COLLINEAR)?2:1;
      assert(G.shape()[1] == nel_*nmo_);        
      using ma::T;
      int u0,uN;
      std::tie(u0,uN) = FairDivideBoundary(comm->rank(),nu,comm->size());
      int v0,vN;
      std::tie(v0,vN) = FairDivideBoundary(comm->rank(),nv,comm->size());
      // right now the algorithm uses 2 copies of matrices of size nuxnv in COLLINEAR case, 
      // consider moving loop over spin to avoid storing the second copy which is not used  
      // simultaneously
      size_t memory_needs = nspin*nu*nv + nv + nu  + nel_*(nv+nu);
      set_shm_buffer(memory_needs);
      size_t cnt=0;  
      // Guv[nspin][nu][nv]
      MArray_ref<ComplexType,3> Guv(SM_TMats->data(),{nspin,nu,nv});
      cnt+=Guv.num_elements();
      // Guu[u]: summed over spin
      MArray_ref<ComplexType,1> Guu(SM_TMats->data()+cnt,{nv});
      cnt+=Guu.num_elements();
      // T1[nel_][nv]
      MArray_ref<ComplexType,2> T1(SM_TMats->data()+cnt,{nel_,nv});
      cnt+=T1.num_elements();
      // Qub[nu][nel_]: 
      MArray_ref<ComplexType,2> Qub(SM_TMats->data()+cnt,{nu,nel_});
      cnt+=Qub.num_elements();
      MArray_ref<ComplexType,1> Tuu(SM_TMats->data()+cnt,{nu});
      if(Rbk.shape()[0] != nel_ || Rbk.shape()[1] != nmo_)
        Rbk.reextent({nel_,nmo_});
      MArray_ref<ComplexType,1> R1D(Rbk.origin(),{nel_*nmo_});
      
      std::fill_n(E.origin(),E.num_elements(),ComplexType(0.0));  
      if(addH1) { 
//        ma::product(ComplexType(1.0),G,haj[k],ComplexType(0.0),E(E.extension(0),0));
        for(int i=0; i<nwalk; i++) E[i][0] += E0;
      }
      if(walker_type==CLOSED || walker_type==NONCOLLINEAR) {
        for(int wi=0; wi<nwalk; wi++) {
          MArray_cref<ComplexType,2> Gw(G[wi].origin(),{nel_,nmo_});
          MArray_cref<ComplexType,1> G1D(G[wi].origin(),{nel_*nmo_});
          Guv_Guu(Gw,Guv,Guu,T1,k);
          ma::product(rotMuv.get()({u0,uN},{0,rotMuv.shape()[1]}),Guu,
                      Tuu({u0,uN})); 
          E[wi][2] = 0.5*ma::dot(Guu({nu0+u0,nu0+uN}),Tuu({u0,uN})); 
          auto Mptr = rotMuv.get()[u0].origin();  
          auto Gptr = Guv[0][u0].origin();  
          for(size_t k=0, kend=(uN-u0)*nv; k<kend; ++k, ++Gptr, ++Mptr)
            (*Gptr) *= (*Mptr); 
          ma::product(Guv[0]({u0,uN},{0,Guv.shape()[2]}),rotcPua[k].get(),
                      Qub({u0,uN},Qub.extension(1)));
          // using this for now, which should not be much worse
          ma::product(T(Qub({u0,uN},Qub.extension(1))),
                      T(rotPiu.get()({0,rotPiu.shape()[0]},{nu0+u0,nu0+uN})), 
                      Rbk);
          E[wi][1] = -0.5*ma::dot(R1D,G1D);
        }
      } else {
        for(int wi=0; wi<nwalk; wi++) {
          MArray_cref<ComplexType,2> Gw(G[wi].origin(),{nel_,nmo_});
          MArray_cref<ComplexType,1> G1DA(G[wi].origin(),{NAEA*nmo_});
          MArray_cref<ComplexType,1> G1DB(G[wi].origin()+NAEA*nmo_,{NAEB*nmo_});
          Guv_Guu(Gw,Guv,Guu,T1,k);
          // move calculation of Guv/Guu here to avoid storing 2 copies of Guv for alpha/beta
          ma::product(rotMuv.get()({u0,uN},{0,rotMuv.shape()[1]}),Guu,
                      Tuu({u0,uN}));
          E[wi][2] = 0.5*ma::dot(Guu({nu0+u0,nu0+uN}),Tuu({u0,uN}));
          // alpha
          auto Mptr = rotMuv.get()[u0].origin();
          auto Gptr = Guv[0][u0].origin();
          for(size_t k=0, kend=(uN-u0)*nv; k<kend; ++k, ++Gptr, ++Mptr)
            (*Gptr) *= (*Mptr);
          ma::product(Guv[0]({u0,uN},{0,Guv.shape()[2]}),
                      (rotcPua[k].get())({0,rotcPua[k].shape()[0]},{0,NAEA}),
                      Qub({u0,uN},{0,NAEA}));
          // using this for now, which should not be much worse
          ma::product(T(Qub({u0,uN},{0,NAEA})),
                      T(rotPiu.get()({0,rotPiu.shape()[0]},{nu0+u0,nu0+uN})),
                      Rbk({0,NAEA},Rbk.extension(1)));
          E[wi][1] = -0.5*ma::dot(R1D({0,NAEA*nmo_}),G1DA);
          // beta
          Mptr = rotMuv.get()[u0].origin();
          Gptr = Guv[1][u0].origin();
          for(size_t k=0, kend=(uN-u0)*nv; k<kend; ++k, ++Gptr, ++Mptr)
            (*Gptr) *= (*Mptr);
          ma::product(Guv[1]({u0,uN},{0,Guv.shape()[2]}),
                      (rotcPua[k].get())({0,rotcPua[k].shape()[0]},{NAEA,NAEA+NAEB}),
                      Qub({u0,uN},{0,NAEB}));
          // using this for now, which should not be much worse
          ma::product(T(Qub({u0,uN},{0,NAEB})),
                      T(rotPiu.get()({0,rotPiu.shape()[0]},{nu0+u0,nu0+uN})),
                      Rbk({0,NAEB},Rbk.extension(1))); 
          E[wi][1] -= 0.5*ma::dot(R1D({0,NAEB*nmo_}),G1DB);
        }
      }    
      comm->barrier();
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vHS(MatA const& X, MatB&& v, double a=1., double c=0.) {
        MArray_cref<ComplexType,2> X_(X.origin(),{X.shape()[0],1});
        MArray_ref<ComplexType,2> v_(v.origin(),{1,v.shape()[0]});
        vHS(X_,v_,a,c);
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==2)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==2)>
            >
    void vHS(MatA const& X, MatB&& v, double a=1., double c=0.) {
      int nwalk = X.shape()[1];
#if defined(QMC_COMPLEX)
      int nchol = 2*Luv.shape()[1];
#else
      int nchol = Luv.shape()[1];
#endif
      int nmo_ = Piu.shape()[0];
      int nu = Piu.shape()[1];
      assert(Luv.shape()[0]==nu);
      assert(X.shape()[0]==nchol);
      assert(v.shape()[0]==nwalk);
      assert(v.shape()[1]==nmo_*nmo_);
      using ma::T;
      int u0,uN;
      std::tie(u0,uN) = FairDivideBoundary(comm->rank(),nu,comm->size());
      int k0,kN;
      std::tie(k0,kN) = FairDivideBoundary(comm->rank(),nmo_,comm->size());
      int wk0,wkN;
      std::tie(wk0,wkN) = FairDivideBoundary(comm->rank(),nwalk*nmo_,comm->size());
#define LOW_MEMORY
#if defined(LOW_MEMORY)
      size_t memory_needs = nu*nwalk + nu*nmo_;
#else
      size_t memory_needs = nu*nwalk + nwalk*nu*nmo_;
#endif
      set_shm_buffer(memory_needs);
      MArray_ref<ComplexType,2> Tuw(SM_TMats->data(),{nu,nwalk});
      // O[nwalk * nmu * nmu]
#if defined(QMC_COMPLEX)
      // reinterpret as RealType matrices with 2x the columns
      MArray_ref<RealType,2> Luv_R(reinterpret_cast<RealType*>(Luv.origin()),
                                                 {Luv.shape()[0],2*Luv.shape()[1]});
      MArray_cref<RealType,2> X_R(reinterpret_cast<RealType const*>(X.origin()),
                                                 {X.shape()[0],2*X.shape()[1]});
      MArray_ref<RealType,2> Tuw_R(reinterpret_cast<RealType*>(Tuw.origin()),
                                                 {nu,2*nwalk});
      ma::product(Luv_R({u0,uN},Luv_R.extension(1)),X_R,
                  Tuw_R({u0,uN},Tuw_R.extension(1)));
#else
      ma::product(Luv.get()({u0,uN},{0,Luv.shape()[1]}),X,
                  Tuw({u0,uN},Tuw.extension(1)));
#endif
      comm->barrier();
#if defined(LOW_MEMORY)
      MArray_ref<ComplexType,2> Qiu(SM_TMats->data()+nwalk*nu,{nmo_,nu});
      for(int wi=0; wi<nwalk; wi++) {
        // Qiu[i][u] = T[u][wi] * conj(Piu[i][u])
        // v[wi][ik] = sum_u Qiu[i][u] * Piu[k][u]
        // O[nmo * nmu]
        for(int i=k0; i<kN; i++) {
          auto p_ = Piu.get()[i].origin();  
          for(int u=0; u<nu; u++, ++p_)
            Qiu[i][u] = Tuw[u][wi]*conj(*p_);
        }
        MArray_ref<ComplexType,2> v_(v[wi].origin(),{nmo_,nmo_});
        // this can benefit significantly from 2-D partition of work
        // O[nmo * nmo * nmu]
        ma::product(a,Qiu({k0,kN},Qiu.extension(1)),T(Piu.get()),
                    c,v_({k0,kN},v_.extension(1)));
      }
#else
      MArray_ref<ComplexType,2> Qiu(SM_TMats->data()+nwalk*nu,{nwalk*nmo_,nu});
      MArray_ref<ComplexType,3> Qwiu(SM_TMats->data()+nwalk*nu,{nwalk,nmo_,nu});
      // Qiu[i][u] = T[u][wi] * conj(Piu[i][u])
      // v[wi][ik] = sum_u Qiu[i][u] * Piu[k][u]
      // O[nmo * nmu]
      for(int wi=0; wi<nwalk; wi++) 
        for(int i=k0; i<kN; i++) { 
          auto p_ = Piu.get()[i].origin();
          for(int u=0; u<nu; u++, ++p_)
            Qwiu[wi][i][u] = Tuw[u][wi]*conj(*p_);
        }
      MArray_ref<ComplexType,2> v_(v.origin(),{nwalk*nmo_,nmo_});
      // this can benefit significantly from 2-D partition of work
      // O[nmo * nmo * nmu]
      ma::product(a,Qiu({wk0,wkN},Qiu.extension(1)),T(Piu.get()),
                  c,v_({wk0,wkN},v_.extension(1)));
#endif
      comm->barrier();
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vbias(MatA const& G, MatB&& v, double a=1., double c=0.) {
        MArray_cref<ComplexType,2> G_(G.origin(),{1,G.shape()[0]});
        MArray_ref<ComplexType,2> v_(v.origin(),{v.shape()[0],1});
        vbias(G_,v_,a,c);
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==2)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==2)>
            >
    void vbias(MatA const& G, MatB&& v, double a=1., double c=0.) {
      int nwalk = G.shape()[0];
      int nmo_ = Piu.shape()[0];
      int nu = Piu.shape()[1];  
      int nel_ = cPua[0].shape()[1];
#if defined(QMC_COMPLEX)
      int nchol = 2*Luv.shape()[1];  
#else
      int nchol = Luv.shape()[1];  
#endif
      assert(v.shape()[1]==nwalk);
      assert(v.shape()[0]==nchol);  
      using ma::T;
      int c0,cN;
      std::tie(c0,cN) = FairDivideBoundary(comm->rank(),nchol,comm->size());  
      if(haj.size()==1) {
        size_t memory_needs = nwalk*nu + nel_*nu;
        set_shm_buffer(memory_needs);
        MArray_ref<ComplexType,2> Guu(SM_TMats->data(),{nu,nwalk});
        MArray_ref<ComplexType,2> T1(SM_TMats->data()+nwalk*nu,{nu,nel_});
        Guu_from_compact(G,Guu,T1);
#if defined(QMC_COMPLEX)
        // reinterpret as RealType matrices with 2x the columns
        MArray_ref<RealType,2> Luv_R(reinterpret_cast<RealType*>(Luv.origin()),
                                                 {Luv.shape()[0],2*Luv.shape()[1]});
        MArray_ref<RealType,2> Guu_R(reinterpret_cast<RealType*>(Guu.origin()),
                                                 {nu,2*nwalk});
        MArray_ref<RealType,2> v_R(reinterpret_cast<RealType*>(v.origin()),
                                                 {v.shape()[0],2*v.shape()[1]});
        ma::product(a,T(Luv_R(Luv_R.extension(0),{c0,cN})),Guu_R,
                    c,v_R({c0,cN},v_R.extension(1)));
#else
        ma::product(a,T(Luv.get()({0,Luv.shape()[0]},{c0,cN})),Guu,
                    c,v({c0,cN},v.extension(1)));
#endif
      } else {
        size_t memory_needs = nwalk*nu + nmo_*nu;
        set_shm_buffer(memory_needs);
        MArray_ref<ComplexType,2> Guu(SM_TMats->data(),{nu,nwalk});
        MArray_ref<ComplexType,2> T1(SM_TMats->data()+nwalk*nu,{nmo_,nu});
        Guu_from_full(G,Guu,T1);
#if defined(QMC_COMPLEX)
        // reinterpret as RealType matrices with 2x the columns
        MArray_ref<RealType,2> Luv_R(reinterpret_cast<RealType*>(Luv.origin()),
                                                 {Luv.shape()[0],2*Luv.shape()[1]});
        MArray_ref<RealType,2> Guu_R(reinterpret_cast<RealType*>(Guu.origin()),
                                                 {nu,2*nwalk});
        MArray_ref<RealType,2> v_R(reinterpret_cast<RealType*>(v.origin()),
                                                 {v.shape()[0],2*v.shape()[1]});
        ma::product(a,T(Luv_R(Luv_R.extension(0),{c0,cN})),Guu_R,
                    c,v_R({c0,cN},v_R.extension(1))); 
#else
        ma::product(a,T(Luv.get()({0,Luv.shape()[0]},{c0,cN})),Guu,
                    c,v({c0,cN},v.extension(1)));
#endif
      }  
      comm->barrier();
    }

    bool distribution_over_cholesky_vectors() const { return false; }
#if defined(QMC_COMPLEX)
    int local_number_of_cholesky_vectors() const{ return 2*Luv.shape()[1]; }
    int global_number_of_cholesky_vectors() const{return 2*Luv.shape()[1]; }
#else
    int local_number_of_cholesky_vectors() const{ return Luv.shape()[1]; }
    int global_number_of_cholesky_vectors() const{return Luv.shape()[1]; }
#endif

    // transpose=true means G[nwalk][ik], false means G[ik][nwalk]
    bool transposed_G_for_vbias() const {return true;} 
    bool transposed_G_for_E() const {return true;} 
    // transpose=true means vHS[nwalk][ik], false means vHS[ik][nwalk]
    bool transposed_vHS() const {return true;} 

  protected:

    // Guu[nu][nwalk]
    template<class MatA, class MatB, class MatC>
    void Guu_from_compact(MatA const& G, MatB&& Guu, MatC&& T1) {
      int nmo_ = int(Piu.shape()[0]);
      int nu = int(Piu.shape()[1]);
      int nel_ = cPua[0].shape()[1];
      int u0,uN;
      std::tie(u0,uN) = FairDivideBoundary(comm->rank(),nu,comm->size());  
      int nw=G.shape()[0];  

      assert(G.shape()[0] == Guu.shape()[1]);
      assert(G.shape()[1] == nel_*nmo_);
      assert(Guu.shape()[0] == nu); 
      assert(T1.shape()[0] == nu);
      assert(T1.shape()[1] == nel_);

      using ma::transposed;
      comm->barrier();
      ComplexType a = (walker_type==CLOSED)?ComplexType(2.0):ComplexType(1.0);
      for(int iw=0; iw<nw; ++iw) {
        MArray_cref<ComplexType,2> Giw(G[iw].origin(),{nel_,nmo_});
        // transposing inetermediary to make dot products faster in the next step
        ma::product(transposed(Piu.get()({0,Piu.shape()[0]},{u0,uN})),
                  transposed(Giw),
                  T1({u0,uN},T1.extension(1)));
        for(int u=u0; u<uN; ++u)
          Guu[u][iw] = a*ma::dot(cPua[0].get()[u],T1[u]);               
      }
      comm->barrier();
    }  

    // Guu[nu][nwalk]
    template<class MatA, class MatB, class MatC>
    void Guu_from_full(MatA const& G, MatB&& Guu, MatC&& T1) {
      int nmo_ = int(Piu.shape()[0]);
      int nu = int(Piu.shape()[1]);
      int u0,uN;
      std::tie(u0,uN) = FairDivideBoundary(comm->rank(),nu,comm->size());
      int nw=G.shape()[0];

      assert(G.shape()[0] == Guu.shape()[1]);
      assert(Guu.shape()[0] == nu);
      assert(T1.shape()[1] == nu);
      assert(G.shape()[1] == nmo_*nmo_); 
      assert(T1.shape()[0] == nmo_);
      comm->barrier();
      std::fill_n(Guu[u0].origin(),nw*(uN-u0),ComplexType(0.0));  
      ComplexType a = (walker_type==CLOSED)?ComplexType(2.0):ComplexType(1.0);
      for(int iw=0; iw<nw; ++iw) {
        MArray_cref<ComplexType,2> Giw(G[iw].origin(),{nmo_,nmo_});
        ma::product(Giw,Piu.get()({0,Piu.shape()[0]},{u0,uN}), 
                  T1(T1.extension(0),{u0,uN}));
        for(int i=0; i<nmo_; ++i) { 
          auto Ti = T1[i].origin();
          auto Pi = Piu.get()[i].origin();
          for(int u=u0; u<uN; ++u,++Ti,++Pi)
            Guu[u][iw] += a*(*Pi)*(*Ti);
        }
      }
      comm->barrier();
    }  

    // since this is for energy, only compact is accepted
    // Computes Guv and Guu for a single walker
    // As opposed to the other Guu routines, 
    //  this routine expects G for the walker in matrix form
    // rotMuv is partitioned along 'u'
    // G[nel][nmo]
    // Guv[nspin][nu][nu]
    // Guu[u]: summed over spin
    // T1[nel_][nu]
    template<class MatA, class MatB, class MatC, class MatD>
    void Guv_Guu(MatA const& G, MatB&& Guv, MatC&& Guu, MatD&& T1, int k) {

      //static_assert(G.dimensionality == 2);
      //static_assert(T1.dimensionality == 2);
      //static_assert(Guu.dimensionality == 1);
      //static_assert(Guv.dimensionality == 3);
      int nspin = (walker_type==COLLINEAR)?2:1;
      int nmo_ = int(rotPiu.shape()[0]);
      int nu = int(rotMuv.shape()[0]);  // potentially distributed over nodes
      int nv = int(rotMuv.shape()[1]);  // not distributed over nodes
      assert(rotPiu.shape()[1] = nv);
      int v0,vN;
      std::tie(v0,vN) = FairDivideBoundary(comm->rank(),nv,comm->size());
      int nu0 = rotMuv.offset()[0];
      ComplexType zero(0.0,0.0);

      assert(Guu.shape()[0] == nv);
      assert(Guv.shape()[1] == nu);
      assert(Guv.shape()[2] == nv);

      // sync first
      comm->barrier();
      if(walker_type==CLOSED || walker_type==NONCOLLINEAR) {
        int nel_ = (walker_type==CLOSED)?NAEA:(NAEA+NAEB);
        ComplexType scl = (walker_type==CLOSED)?ComplexType(2.0):ComplexType(1.0);
        assert(Guv.shape()[0] == 1);
        assert(G.shape()[0] == size_t(nel_));
        assert(G.shape()[1] == size_t(nmo_));
        assert(T1.shape()[0] == size_t(nel_));
        assert(T1.shape()[1] == size_t(nv));

        using ma::transposed;
        ma::product(G,rotPiu.get()({0,rotPiu.shape()[0]},{v0,vN}),
                    T1(T1.extension(0),{v0,vN}));
        // This operation might benefit from a 2-D work distribution 
        ma::product(scl, rotcPua[k].get()({nu0,nu0+nu},{0,rotcPua[k].shape()[1]}),
                    T1(T1.extension(0),{v0,vN}),
                    zero, Guv[0]({0,Guv.shape()[1]},{v0,vN}));
        for(int v=v0; v<vN; ++v) 
          if( v < nu0 || v >= nu0+nu ) {
            Guu[v] = scl*ma::dot(rotcPua[k].get()[v],T1(T1.extension(0),v)); 
          } else
            Guu[v] = Guv[0][v-nu0][v];  
      } else {
        int nel_ = NAEA+NAEB;
        assert(Guv.shape()[0] == 2);
        assert(G.shape()[0] == nel_);
        assert(G.shape()[1] == nmo_);
        assert(T1.shape()[0] == nel_);
        assert(T1.shape()[1] == nv);

        using ma::transposed;
        ma::product(G,rotPiu.get()({0,rotPiu.shape()[0]},{v0,vN}), 
                    T1(T1.extension(0),{v0,vN})); 
        // This operation might benefit from a 2-D work distribution 
        // Alpha
        ma::product(rotcPua[k].get()({nu0,nu0+nu},{0,NAEA}),
                    T1({0,NAEA},{v0,vN}),
                    Guv[0]({0,Guv.shape()[1]},{v0,vN}));
        ma::product(rotcPua[k].get()({nu0,nu0+nu},{NAEA,NAEA+NAEB}),
                    T1({NAEA,NAEA+NAEB},{v0,vN}),
                    Guv[1]({0,Guv.shape()[1]},{v0,vN}));
        for(int v=v0; v<vN; ++v) 
          if( v < nu0 || v >= nu0+nu ) {
            Guu[v] = ma::dot(rotcPua[k].get()[v],T1(T1.extension(0),v));
          } else
            Guu[v] = Guv[0][v-nu0][v]+Guv[1][v-nu0][v];
      } 
      comm->barrier();
    }

  protected:

    communicator* comm;

    int NMO,NAEA,NAEB;

    WALKER_TYPES walker_type;

    // bare one body hamiltonian
    CMatrix hij;

    // (potentially half rotated) one body hamiltonian
    std::vector<SpTVector> haj;

    /************************************************/
    // Used in the calculation of the energy
    // Coulomb matrix elements of interpolating vectors
    shmVMatrix rotMuv;

    // Orbitals at interpolating points
    shmVMatrix rotPiu;

    // Half-rotated Orbitals at interpolating points
    std::vector<shmCMatrix> rotcPua;
    /************************************************/

    /************************************************/
    // Following 3 used in calculation of vbias and vHS
    // Cholesky factorization of Muv 
    shmVMatrix Luv;

    // Orbitals at interpolating points
    shmVMatrix Piu;
 
    // Half-rotated Orbitals at interpolating points
    std::vector<shmCMatrix> cPua;
    /************************************************/
     
    // one-body piece of Hamiltonian factorization
    TMatrix v0;    

    ValueType E0;

    // shared memory for intermediates
    std::unique_ptr<SHM_Buffer> SM_TMats;

    MArray<ComplexType,2> Rbk;

    void set_shm_buffer(size_t N) {
      if(SM_TMats == nullptr) {
        SM_TMats = std::move(std::make_unique<SHM_Buffer>(*comm,N));
      } else if(SM_TMats->size() < N) {
        SM_TMats->resize(N);
      }
    }

};

}

}

#endif
