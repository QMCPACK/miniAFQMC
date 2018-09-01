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

#include <Utilities/NewTimer.h>
#include "Configuration.h"
#include "type_traits/scalar_traits.h"
#include "Numerics/ma_operations.hpp"
#include "Numerics/ma_blas.hpp"
#include "Utilities/type_conversion.hpp"

#include "Numerics/detail/cuda_pointers.hpp"
#include "Kernels/batchedDot.cuh"

namespace qmcplusplus
{

namespace afqmc
{

template< class Alloc >
class THCOps
{
#if defined(AFQMC_SP) 
  using SpC = typename to_single_precision<ComplexType>::value_type;
#else
  using SpC = ComplexType;  
#endif

  using CVector = boost::multi::array<ComplexType,1,Alloc>;
  using CMatrix = boost::multi::array<ComplexType,2,Alloc>;
  using SpCVector = boost::multi::array<SpC,1,Alloc>;
  using SpCMatrix = boost::multi::array<SpC,2,Alloc>;
  using Allocator = Alloc;  
  using pointer = typename Alloc::pointer;  
  using const_pointer = typename Alloc::const_pointer;  
  using real_type = typename Alloc::template rebind<SPRealType>::other::value_type;
  using const_real_type = typename Alloc::template rebind<SPRealType>::other::value_type const;
  using real_pointer = typename Alloc::template rebind<SPRealType>::other::pointer;
  using real_const_pointer = typename Alloc::template rebind<SPRealType>::other::const_pointer;

  public:

    THCOps(int nmo_, int naea_, int naeb_,  
           WALKER_TYPES type, 
           CMatrix&& h1,
           SpCMatrix&& rotmuv_,
           SpCMatrix&& rotpiu_,
           SpCMatrix&& rotpau_,
           SpCMatrix&& luv_,
           SpCMatrix&& piu_,
           SpCMatrix&& pau_,
           ValueType e0_,
           Allocator alloc_ = Allocator{},  
           bool verbose=false ):
                allocator_(alloc_),
                NMO(nmo_),NAEA(naea_),NAEB(naeb_),
                walker_type(type),
                haj(std::move(h1)),
                rotMuv(std::move(rotmuv_)),
                rotPiu(std::move(rotpiu_)),
                rotcPua(std::move(rotpau_)),
                Luv(std::move(luv_)),
                Piu(std::move(piu_)),
                cPua(std::move(pau_)),
                E0(e0_),
                TMats( {0}, allocator_)
    {
      // simplifying limitation for the miniapp  
      assert(NAEA==NAEB);
      // current partition over 'u' for L/Piu
      assert(Luv.shape()[0] == Piu.shape()[1]);
        // rot Ps are not yet distributed
        assert(rotcPua.shape()[0] == rotPiu.shape()[1]);
        if(walker_type==CLOSED)
          assert(rotcPua.shape()[1]==NAEA);
        else if(walker_type==COLLINEAR)
          assert(rotcPua.shape()[1]==NAEA+NAEB);
        else if(walker_type==NONCOLLINEAR)
          assert(rotcPua.shape()[1]==NAEA+NAEB);
        assert(cPua.shape()[0]==Luv.shape()[0]);
        if(walker_type==CLOSED)
          assert(cPua.shape()[1]==NAEA);
        else if(walker_type==COLLINEAR)
          assert(cPua.shape()[1]==NAEA+NAEB);
        else if(walker_type==NONCOLLINEAR)
          assert(cPua.shape()[1]==NAEA+NAEB);
      if(walker_type==NONCOLLINEAR) {
        assert(Piu.shape()[0]==2*NMO);
        assert(rotPiu.shape()[0]==2*NMO);
      } else {
        assert(Piu.shape()[0]==NMO);
        assert(rotPiu.shape()[0]==NMO);
      }
      // Allocate large array for temporary work
      // energy needs: nu*nv + nv + nu  + nel_*max(nv,nu) + nel_*nmo
      // vHS needs: nu*nwalk + nu*nmo_
      // Sp_G needs: nel_*nmo_
      // vbias needs: nwalk*nu + nel_*nu 
      size_t nu = Luv.shape()[0];
      size_t nv = Luv.shape()[1];
      size_t rnu = rotMuv.shape()[0];
      size_t rnv = rotMuv.shape()[1];
      size_t nel = ((walker_type==COLLINEAR)?(NAEA+NAEB):(NAEA));  
      size_t memory_needs = rnu*rnv + rnu + rnv + size_t(NAEA)*std::max(rnu,rnv) + size_t(2*NAEA*NMO); 
      TMats.reextent( {memory_needs} );

      setup_timers(Timers, THCTimerNames, timer_level_coarse);
    }

    ~THCOps() {}
    
    THCOps(THCOps const& other) = delete;
    THCOps& operator=(THCOps const& other) = delete;

    THCOps(THCOps&& other) = default;
    THCOps& operator=(THCOps&& other) = default; 

    template<class Mat, class MatB>
    RealType energy(Mat&& E, MatB const& G, bool addH1=true) {
      // G[nel][nmo]
      assert(E.shape()[0] == G.shape()[0]);        
      int nwalk = G.shape()[0];
      int nmo_ = rotPiu.shape()[0];
      int nu = rotMuv.shape()[0]; 
      int nv = rotMuv.shape()[1]; 
      int nel_ = rotcPua.shape()[1];
      int nspin = (walker_type==COLLINEAR)?2:1;
      assert(NMO==nmo_);
      assert(G.shape()[1] == nel_*nmo_);        
      using ma::T;
      // right now the algorithm uses 2 copies of matrices of size nuxnv in COLLINEAR case, 
      // consider moving loop over spin to avoid storing the second copy which is not used  
      // simultaneously
      size_t memory_needs = nu*nv + nv + nu  + size_t(NAEA)*std::max(nv,nu) + size_t(2*NMO*NAEA);
      if(TMats.num_elements() < memory_needs) {
       TMats.reextent({memory_needs}); 
       // APP_ABORT(" Error: TMats.num_elements() < memory_needs() \n"); 
      }  
      size_t cnt=0;  
      // Guv[nspin][nu][nv]
      boost::multi::array_ref<ComplexType,2,pointer> Guv(TMats.data(),{nu,nv});
      cnt+=Guv.num_elements();
      // Guu[u]: summed over spin
      boost::multi::array_ref<ComplexType,1,pointer> Guu(TMats.data()+cnt,{nv});
      cnt+=Guu.num_elements();
      // T1[nel_][nv]
      boost::multi::array_ref<ComplexType,2,pointer> T1(TMats.data()+cnt,{NAEA,nv});
      // Qub[nu][nel_]: using same space as T1 
      boost::multi::array_ref<ComplexType,2,pointer> Qub(TMats.data()+cnt,{nu,NAEA});
      cnt+=std::max(Qub.num_elements(), T1.num_elements());
      boost::multi::array_ref<ComplexType,1,pointer> Tuu(TMats.data()+cnt,{nu});
      cnt+=Tuu.num_elements();
      boost::multi::array_ref<ComplexType,2,pointer> Rbk(TMats.data()+cnt,{NAEA,NMO});
      boost::multi::array_ref<ComplexType,1,pointer> R1D(Rbk.origin(),{Rbk.num_elements()});

      cuda::fill_n(E.origin(),E.num_elements(),1,ComplexType(0.0));  
//      cuda::fill_n(E.origin(),E.shape()[0],E.strides()[0],ComplexType(E0));  
      if(addH1) { 
        boost::multi::array_ref<ComplexType,1,pointer> haj1D(haj.origin(),{haj.num_elements()});
        ma::product(ComplexType(1.0),G,haj1D,ComplexType(0.0),E( E.extension(0), 0));
//        for(int i=0; i<nwalk; i++) {
//          E[i][0] += E0;
//        }
      }

      if(walker_type==CLOSED || walker_type==NONCOLLINEAR) {
        for(int wi=0; wi<nwalk; wi++) {
          boost::multi::array_cref<ComplexType,2,const_pointer> Gw(G[wi].origin(),{nel_,nmo_});
          boost::multi::array_cref<ComplexType,1,const_pointer> G1D(G[wi].origin(),{nel_*nmo_});
          Guv_Guu(Gw,Guv,Guu,T1,false);
          ma::product(rotMuv,Guu,Tuu);
          //E[wi][2] = 0.5*ma::dot(Guu,Tuu);
          ma::adotpby(ComplexType(0.5),Guu,Tuu,ComplexType(0.0),E[wi].origin()+2);
/*
          auto Mptr = rotMuv.origin();  
          auto Gptr = Guv.origin();  
          for(size_t k=0, kend=nu*nv; k<kend; ++k, ++Gptr, ++Mptr)
            (*Gptr) *= (*Mptr); 
*/
          ma::axty(ComplexType(1.0),rotMuv,Guv);  
          ma::product(Guv,rotcPua,Qub);
          // using this for now, which should not be much worse
          ma::product(T(Qub),T(rotPiu),Rbk);
          //E[wi][1] = -0.5*ma::dot(R1D,G1D);
          ma::adotpby(ComplexType(-0.5),R1D,G1D,ComplexType(0.0),E[wi].origin()+1);
        }
      } else {
        for(int wi=0; wi<nwalk; wi++) {
          { // Alpha
            boost::multi::array_cref<ComplexType,2,const_pointer> Gw(G[wi].origin(),{NAEA,nmo_});
            boost::multi::array_cref<ComplexType,1,const_pointer> G1DA(Gw.origin(),{Gw.num_elements()});
            cuda::fill_n(Guu.origin(),Guu.num_elements(),1,SpC(0.0));
            Timers[Timer_E1]->start();
            Guv_Guu(Gw,Guv,Guu,T1,false);
            Timers[Timer_E1]->stop();
/*
            auto Mptr = rotMuv.origin();
            auto Gptr = Guv.origin();
            for(size_t k=0, kend=nu*nv; k<kend; ++k, ++Gptr, ++Mptr)
              (*Gptr) *= (*Mptr);
*/
            Timers[Timer_E3]->start();
            ma::axty(ComplexType(1.0),rotMuv,Guv);  
            Timers[Timer_E3]->stop();
            Timers[Timer_E4]->start();
            ma::product(Guv,rotcPua( rotcPua.extension(0), {0,NAEA}), Qub);
            // using this for now, which should not be much worse
            Timers[Timer_E4]->stop();
            Timers[Timer_E5]->start();
            ma::product(T(Qub),T(rotPiu),Rbk);
            //E[wi][1] = -0.5*ma::dot(R1D,G1DA);
            ma::adotpby(ComplexType(-0.5),R1D,G1DA,ComplexType(0.0),E[wi].origin()+1);
            Timers[Timer_E5]->stop();
          }
          {  // beta
            boost::multi::array_cref<ComplexType,2,const_pointer> Gw(G[wi].origin()+NAEA*NMO,{NAEB,nmo_});
            boost::multi::array_cref<ComplexType,1,const_pointer> G1DB(Gw.origin(),{Gw.num_elements()});
            Timers[Timer_E1]->start();
            Guv_Guu(Gw,Guv,Guu,T1,true);
            Timers[Timer_E1]->stop();
            Timers[Timer_E2]->start();
            ma::product(rotMuv,Guu,Tuu);
            //E[wi][2] = 0.5*ma::dot(Guu,Tuu);
            ma::adotpby(ComplexType(0.5),Guu,Tuu,ComplexType(0.0),E[wi].origin()+2);
            Timers[Timer_E2]->stop();
/*
            auto Mptr = rotMuv.origin();
            auto Gptr = Guv.origin();
            for(size_t k=0, kend=nu*nv; k<kend; ++k, ++Gptr, ++Mptr)
              (*Gptr) *= (*Mptr);
*/
            Timers[Timer_E3]->start();
            ma::axty(ComplexType(1.0),rotMuv,Guv);  
            Timers[Timer_E3]->stop();
            Timers[Timer_E4]->start();
            ma::product(Guv,rotcPua( rotcPua.extension(0), {NAEA,NAEA+NAEB} ),Qub);
            Timers[Timer_E4]->stop();
            // using this for now, which should not be much worse
            Timers[Timer_E5]->start();
            ma::product(T(Qub),T(rotPiu),Rbk);
            //E[wi][1] -= 0.5*ma::dot(R1D,G1DB);
            ma::adotpby(ComplexType(-0.5),R1D,G1DB,ComplexType(1.0),E[wi].origin()+1);
            Timers[Timer_E5]->stop();
          }
        }
      }    
      ComplexType Eav = ma::sum(E(E.extension(0),{0,3}));
      using std::real;
      return real(Eav)/nwalk+real(E0);
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vHS(MatA const& X, MatB&& v, double a=1., double c=0.) {
        boost::multi::array_cref<ComplexType,2,const_pointer> X_(X.origin(),{X.shape()[0],1});
        boost::multi::array_ref<ComplexType,2,pointer> v_(v.origin(),{1,v.shape()[0]});
        vHS(X_,v_,a,c);
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==2)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==2)>
            >
    void vHS(MatA const& X, MatB&& v, double a=1., double c=0.) {
      int nwalk = X.shape()[1];
      int nchol = 2*Luv.shape()[1];
      int nmo_ = Piu.shape()[0];
      assert(NMO==nmo_);
      int nu = Piu.shape()[1];
      assert(Luv.shape()[0]==nu);
      assert(X.shape()[0]==nchol);
      assert(v.shape()[0]==nwalk);
      assert(v.shape()[1]==nmo_*nmo_);
      using ma::T;
      using std::conj;
      size_t memory_needs = nu*nwalk + nu*nmo_ + X.num_elements();
      if(TMats.num_elements() < memory_needs) {
       TMats.reextent({memory_needs}); 
       // APP_ABORT(" Error: TMats.num_elements() < memory_needs() \n"); 
      }  
      boost::multi::array_ref<ComplexType,2,pointer> Tuw(TMats.data(),{nu,nwalk});
      // O[nwalk * nmu * nmu]
      // reinterpret as RealType matrices with 2x the columns
      using detail::pointer_cast;
      boost::multi::array_ref<RealType,2,real_pointer> Luv_R(pointer_cast<real_type>(Luv.origin()),
                                                 {Luv.shape()[0],2*Luv.shape()[1]});
      boost::multi::array_cref<RealType,2,real_const_pointer> X_R(pointer_cast<const_real_type>(X.origin()),
                                                 {X.shape()[0],2*X.shape()[1]});
      boost::multi::array_ref<RealType,2,real_pointer> Tuw_R(pointer_cast<real_type>(Tuw.origin()),
                                                 {nu,2*nwalk});
      Timers[Timer_vHS1]->start();
      ma::product(Luv_R,X_R,Tuw_R);
      Timers[Timer_vHS1]->stop();
      boost::multi::array_ref<ComplexType,2,pointer> Qiu(TMats.data()+nwalk*nu,{nmo_,nu});
      for(int wi=0; wi<nwalk; wi++) {
        // Qiu[i][u] = T[u][wi] * conj(Piu[i][u])
        // v[wi][ik] = sum_u Qiu[i][u] * Piu[k][u]
        // O[nmo * nmu]

        // z[i][j] = x[j] * y[i][j]
/*
        for(int i=0; i<nmo_; i++) {
          auto p_ = Piu[i].origin();  
          for(int u=0; u<nu; u++, ++p_)
            Qiu[i][u] = Tuw[u][wi]*conj(*p_);
        }
*/
        Timers[Timer_vHS2]->start();
        ma::acAxpbB(ComplexType(1.0),Piu,Tuw(Tuw.extension(0),wi),ComplexType(0.0),Qiu);
        Timers[Timer_vHS2]->stop();
        boost::multi::array_ref<ComplexType,2,pointer> v_(v[wi].origin(),{nmo_,nmo_});
        // this can benefit significantly from 2-D partition of work
        // O[nmo * nmo * nmu]
        Timers[Timer_vHS3]->start();
        ma::product(a,Qiu,T(Piu),c,v_);
        Timers[Timer_vHS3]->stop();
      }
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vbias(MatA const& G, MatB&& v, double a=1., double c=0.) {
        boost::multi::array_cref<ComplexType,2,const_pointer> G_(G.origin(),{1,G.shape()[0]});
        boost::multi::array_ref<ComplexType,2,pointer> v_(v.origin(),{v.shape()[0],1});
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
      int nel_ = cPua.shape()[1];
      int nchol = 2*Luv.shape()[1];  
      assert(v.shape()[1]==nwalk);
      assert(v.shape()[0]==nchol);  
      using ma::T;
      size_t memory_needs = nwalk*nu + nel_*nu + G.num_elements();
      if(TMats.num_elements() < memory_needs) {
       TMats.reextent({memory_needs});
       // APP_ABORT(" Error: TMats.num_elements() < memory_needs() \n"); 
      }
      boost::multi::array_ref<ComplexType,2,pointer> Guu(TMats.data(),{nu,nwalk});
      boost::multi::array_ref<ComplexType,2,pointer> T1(TMats.data()+nwalk*nu,{nu,nel_});
      Guu_from_compact(G,Guu,T1);
      // reinterpret as RealType matrices with 2x the columns
      using detail::pointer_cast;
      boost::multi::array_ref<RealType,2,real_pointer> Luv_R(pointer_cast<real_type>(Luv.origin()),
                                               {Luv.shape()[0],2*Luv.shape()[1]});
      boost::multi::array_ref<RealType,2,real_pointer> Guu_R(pointer_cast<real_type>(Guu.origin()),
                                               {nu,2*nwalk});
      boost::multi::array_ref<RealType,2,real_pointer> v_R(pointer_cast<real_type>(v.origin()),
                                               {v.shape()[0],2*v.shape()[1]});
      Timers[Timer_vbias3]->start();
      ma::product(a,T(Luv_R),Guu_R,c,v_R);
      Timers[Timer_vbias3]->stop();
    }

    int number_of_cholesky_vectors() const { return 2*Luv.shape()[1]; } 

  protected:

    // Guu[nu][nwalk]
    template<class MatA, class MatB, class MatC>
    void Guu_from_compact(MatA const& G, MatB&& Guu, MatC&& T1) {
      int nmo_ = int(Piu.shape()[0]);
      int nu = int(Piu.shape()[1]);
      int nel_ = cPua.shape()[1];
      int nw=G.shape()[0];  

      assert(G.shape()[0] == Guu.shape()[1]);
      assert(G.shape()[1] == nel_*nmo_);
      assert(Guu.shape()[0] == nu); 
      assert(T1.shape()[0] == nu);
      assert(T1.shape()[1] == nel_);

      using ma::transposed;
      ComplexType a = (walker_type==CLOSED)?ComplexType(2.0):ComplexType(1.0);
      ComplexType zero(0.0);
      boost::multi::array_ref<ComplexType,3,pointer> cPua3D(cPua.origin(),{cPua.shape()[0],1,cPua.shape()[1]});
      boost::multi::array_ref<ComplexType,3,pointer> T13D(T1.origin(),{T1.shape()[0],T1.shape()[1],1});
      for(int iw=0; iw<nw; ++iw) {
        boost::multi::array_cref<ComplexType,2,const_pointer> Giw(G[iw].origin(),{nel_,nmo_});
        // transposing inetermediary to make dot products faster in the next step
        Timers[Timer_vbias1]->start();
        ma::product(transposed(Piu),transposed(Giw),T1);
        Timers[Timer_vbias1]->stop();
/*
        for(int u=0; u<nu; ++u)
          Guu[u][iw] = a*ma::dot(cPua[u],T1[u]);               
*/
        Timers[Timer_vbias2]->start();
/*
        boost::multi::array_ref<ComplexType,3,pointer> Guu3D(Guu.origin()+iw,{Guu.shape()[0],1,1});
        
        using BLAS_CPU::gemmStridedBatched;
        using BLAS_GPU::gemmStridedBatched;
        using detail::to_address;
        gemmStridedBatched(
                'N', 'N',
                1, 1, cPua.shape()[1],
                a,
                cPua.origin(),1,cPua.shape()[1],
                T1.origin(),T1.shape()[1],T1.shape()[1],
                zero,
                Guu.origin()+iw,1,Guu.strides()[0],
                cPua.shape()[0] 
        );
*/
        kernels::batchedDot(cPua.shape()[0], cPua.shape()[1], a, to_address(cPua.origin()), cPua.shape()[1], to_address(T1.origin()), T1.shape()[1], 
                    ComplexType(0.0), to_address(Guu.origin())+iw, Guu.shape()[1]);
        Timers[Timer_vbias2]->stop();
      }
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
    void Guv_Guu(MatA const& G, MatB&& Guv, MatC&& Guu, MatD&& T1, bool beta) {

      int nmo_ = int(rotPiu.shape()[0]);
      int nu = int(rotMuv.shape()[0]);  // potentially distributed over nodes
      int nv = int(rotMuv.shape()[1]);  // not distributed over nodes
      assert(rotPiu.shape()[1] == nv);
      ComplexType zero(0.0,0.0);

      assert(Guu.shape()[0] == nv);
      assert(Guv.shape()[0] == nu);
      assert(Guv.shape()[1] == nv);

      // sync first
      int nel_ = G.shape()[0]; 
      ComplexType scl = (walker_type==CLOSED)?ComplexType(2.0):ComplexType(1.0);
      assert(G.shape()[1] == size_t(nmo_));
      assert(T1.shape()[0] == size_t(nel_));
      assert(T1.shape()[1] == size_t(nv));

      using ma::transposed;
      ma::product(G,rotPiu,T1);
      // This operation might benefit from a 2-D work distribution 
      if(beta)
        ma::product(scl, rotcPua( rotcPua.extension(0), {NAEA,NAEA+NAEB}),
                  T1, zero, Guv);
      else
        ma::product(scl, rotcPua( rotcPua.extension(0), {0,NAEA}),
                  T1, zero, Guv);
//      for(int v=0; v<nv; ++v) 
//        Guu[v] += Guv[v][v];  
      ma::adiagApy(ComplexType(1.0),Guv,Guu);  
    }

  protected:

    Allocator allocator_;

    int NMO,NAEA,NAEB;

    WALKER_TYPES walker_type;

    // (potentially half rotated) one body hamiltonian
    CMatrix haj;

    /************************************************/
    // Used in the calculation of the energy
    // Coulomb matrix elements of interpolating vectors
    SpCMatrix rotMuv;

    // Orbitals at interpolating points
    SpCMatrix rotPiu;

    // Half-rotated Orbitals at interpolating points
    SpCMatrix rotcPua;
    /************************************************/

    /************************************************/
    // Following 3 used in calculation of vbias and vHS
    // Cholesky factorization of Muv 
    SpCMatrix Luv;

    // Orbitals at interpolating points
    SpCMatrix Piu;
 
    // Half-rotated Orbitals at interpolating points
    SpCMatrix cPua;
    /************************************************/
     
    ValueType E0;

    // Allocate large array for temporary work
    // energy needs: nu*nv + nv + nu  + nel_*max(nv,nu) + nel_*nmo
    // vHS needs: nu*nwalk + nu*nmo_
    // Sp_G needs: nel_*nmo_
    // vbias needs: nwalk*nu + nel_*nu 
    SpCVector TMats; 

    enum THCTimers
    {
      Timer_vbias1,
      Timer_vbias2,
      Timer_vbias3,
      Timer_vHS1,
      Timer_vHS2,
      Timer_vHS3,
      Timer_E1,
      Timer_E2,
      Timer_E3,
      Timer_E4,
      Timer_E5
    };

    TimerNameList_t<THCTimers> THCTimerNames = {
      {Timer_vbias1, "vbias_Guu"},
      {Timer_vbias2, "vbias_dot"},
      {Timer_vbias3, "vbias_prod"},
      {Timer_vHS1, "vHS_Tuw"},
      {Timer_vHS2, "vHS_acAxpbB"},
      {Timer_vHS3, "vHS_prod"},
      {Timer_E1, "E_Guv"},
      {Timer_E2, "E_EJ"},
      {Timer_E3, "E_axty"},
      {Timer_E4, "E_Qub"},
      {Timer_E5, "E_Rbk"},
    };
    TimerList_t Timers;

};

}

}

#endif
