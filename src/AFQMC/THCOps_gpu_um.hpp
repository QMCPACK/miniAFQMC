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

#ifndef QMCPLUSPLUS_AFQMC_HAMILTONIANOPERATIONS_THCOPS_GPU_UM_HPP
#define QMCPLUSPLUS_AFQMC_HAMILTONIANOPERATIONS_THCOPS_GPU_UM_HPP

#include<fstream>

#include "Configuration.h"
#include "type_traits/scalar_traits.h"
#include "Numerics/ma_operations.hpp"
#include "Utilities/type_conversion.hpp"

namespace qmcplusplus
{

namespace afqmc
{

namespace gpu_um 
{

class THCOps
{
#if defined(AFQMC_SP) 
  using SpC = typename to_single_precision<ComplexType>::value_type;
#else
  using SpC = ComplexType;  
#endif

  using CVector = boost::multi_array<ComplexType,1>;
  using CMatrix = boost::multi_array<ComplexType,2>;
  using SpCVector = boost::multi_array<SpC,1>;
  using SpCMatrix = boost::multi_array<SpC,2>;

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
           bool verbose=false ):
                NMO(nmo_),NAEA(naea_),NAEB(naeb_),
                walker_type(type),
                haj(std::move(h1)),
                rotMuv(std::move(rotmuv_)),
                rotPiu(std::move(rotpiu_)),
                rotcPua(std::move(rotpau_)),
                Luv(std::move(luv_)),
                Piu(std::move(piu_)),
                cPua(std::move(pau_)),
                E0(e0_)
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
      size_t rnu = rotMuv.shape()[0];
      size_t rnv = rotMuv.shape()[1];
      size_t nel = ((walker_type==COLLINEAR)?(NAEA+NAEB):(NAEA));  
      size_t memory_needs = rnu*rnv + rnu + rnv + size_t(NAEA)*std::max(rnu,rnv) + size_t(2*NAEA*NMO); 
      TMats.resize(extents[memory_needs]);
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
      if(TMats.num_elements() < memory_needs)
        APP_ABORT(" Error: TMats.num_elements() < memory_needs() \n"); 
      size_t cnt=0;  
      // Guv[nspin][nu][nv]
      boost::multi_array_ref<ComplexType,2> Guv(TMats.data(),extents[nu][nv]);
      cnt+=Guv.num_elements();
      // Guu[u]: summed over spin
      boost::multi_array_ref<ComplexType,1> Guu(TMats.data()+cnt,extents[nv]);
      cnt+=Guu.num_elements();
      // T1[nel_][nv]
      boost::multi_array_ref<ComplexType,2> T1(TMats.data()+cnt,extents[NAEA][nv]);
      // Qub[nu][nel_]: using same space as T1 
      boost::multi_array_ref<ComplexType,2> Qub(TMats.data()+cnt,extents[nu][NAEA]);
      cnt+=std::max(Qub.num_elements(), T1.num_elements());
      boost::multi_array_ref<ComplexType,1> Tuu(TMats.data()+cnt,extents[nu]);
      cnt+=Tuu.num_elements();
      boost::multi_array_ref<ComplexType,2> Rbk(TMats.data()+cnt,extents[NAEA][NMO]);
      boost::multi_array_ref<ComplexType,1> R1D(Rbk.origin(),extents[Rbk.num_elements()]);
      
      ComplexType Eav(0);  
      std::fill_n(E.origin(),E.num_elements(),ComplexType(0.0));  
      if(addH1) { 
        boost::multi_array_ref<ComplexType,1> haj1D(haj.origin(),extents[haj.num_elements()]);
        ma::product(ComplexType(1.0),G,haj1D,ComplexType(0.0),E[indices[range_t()][0]]);
        for(int i=0; i<nwalk; i++) {
          E[i][0] += E0;
          Eav+=E[i][0];
        }
      }
      if(walker_type==CLOSED || walker_type==NONCOLLINEAR) {
        for(int wi=0; wi<nwalk; wi++) {
          boost::const_multi_array_ref<ComplexType,2> Gw(G[wi].origin(),extents[nel_][nmo_]);
          boost::const_multi_array_ref<ComplexType,1> G1D(G[wi].origin(),extents[nel_*nmo_]);
          Guv_Guu(Gw,Guv,Guu,T1,false);
          ma::product(rotMuv,Guu,Tuu);
          E[wi][2] = 0.5*ma::dot(Guu,Tuu);
          auto Mptr = rotMuv.origin();  
          auto Gptr = Guv.origin();  
          for(size_t k=0, kend=nu*nv; k<kend; ++k, ++Gptr, ++Mptr)
            (*Gptr) *= (*Mptr); 
          ma::product(Guv,rotcPua,Qub);
          // using this for now, which should not be much worse
          ma::product(T(Qub),T(rotPiu),Rbk);
          E[wi][1] = -0.5*ma::dot(R1D,G1D);
          Eav += (E[wi][1]+E[wi][2]);
        }
      } else {
        for(int wi=0; wi<nwalk; wi++) {
          { // Alpha
            boost::const_multi_array_ref<ComplexType,2> Gw(G[wi].origin(),extents[NAEA][nmo_]);
            boost::const_multi_array_ref<ComplexType,1> G1DA(Gw.origin(),extents[Gw.num_elements()]);
            std::fill_n(Guu.origin(),Guu.num_elements(),SpC(0.0));
            Guv_Guu(Gw,Guv,Guu,T1,false);
            auto Mptr = rotMuv.origin();
            auto Gptr = Guv.origin();
            for(size_t k=0, kend=nu*nv; k<kend; ++k, ++Gptr, ++Mptr)
              (*Gptr) *= (*Mptr);
            ma::product(Guv,rotcPua[indices[range_t()][range_t(0,NAEA)]],Qub);
            // using this for now, which should not be much worse
            ma::product(T(Qub),T(rotPiu),Rbk);
            E[wi][1] = -0.5*ma::dot(R1D,G1DA);
          }
          {  // beta
            boost::const_multi_array_ref<ComplexType,2> Gw(G[wi].origin()+NAEA*NMO,extents[NAEB][nmo_]);
            boost::const_multi_array_ref<ComplexType,1> G1DB(Gw.origin(),extents[Gw.num_elements()]);
            Guv_Guu(Gw,Guv,Guu,T1,true);
            ma::product(rotMuv,Guu,Tuu);
            E[wi][2] = 0.5*ma::dot(Guu,Tuu);
            auto Mptr = rotMuv.origin();
            auto Gptr = Guv.origin();
            for(size_t k=0, kend=nu*nv; k<kend; ++k, ++Gptr, ++Mptr)
              (*Gptr) *= (*Mptr);
            ma::product(Guv,rotcPua[indices[range_t()][range_t(NAEA,NAEA+NAEB)]],Qub);
            // using this for now, which should not be much worse
            ma::product(T(Qub),T(rotPiu),Rbk);
            E[wi][1] -= 0.5*ma::dot(R1D,G1DB);
          }
          Eav += (E[wi][1]+E[wi][2]);
        }
      }    
      using std::real;
      return real(Eav)/nwalk;
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vHS(MatA const& X, MatB&& v, double a=1., double c=0.) {
        boost::const_multi_array_ref<ComplexType,2> X_(X.origin(),extents[X.shape()[0]][1]);
        boost::multi_array_ref<ComplexType,2> v_(v.origin(),extents[1][v.shape()[0]]);
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
      if(TMats.num_elements() < memory_needs)
        APP_ABORT(" Error: TMats.num_elements() < memory_needs() \n");
      boost::multi_array_ref<ComplexType,2> Tuw(TMats.data(),extents[nu][nwalk]);
      // O[nwalk * nmu * nmu]
      // reinterpret as RealType matrices with 2x the columns
      boost::multi_array_ref<RealType,2> Luv_R(reinterpret_cast<RealType*>(Luv.origin()),
                                                 extents[Luv.shape()[0]][2*Luv.shape()[1]]);
      boost::const_multi_array_ref<RealType,2> X_R(reinterpret_cast<RealType const*>(X.origin()),
                                                 extents[X.shape()[0]][2*X.shape()[1]]);
      boost::multi_array_ref<RealType,2> Tuw_R(reinterpret_cast<RealType*>(Tuw.origin()),
                                                 extents[nu][2*nwalk]);
      ma::product(Luv_R,X_R,Tuw_R);
      boost::multi_array_ref<ComplexType,2> Qiu(TMats.data()+nwalk*nu,extents[nmo_][nu]);
      for(int wi=0; wi<nwalk; wi++) {
        // Qiu[i][u] = T[u][wi] * conj(Piu[i][u])
        // v[wi][ik] = sum_u Qiu[i][u] * Piu[k][u]
        // O[nmo * nmu]
        for(int i=0; i<nmo_; i++) {
          auto p_ = Piu[i].origin();  
          for(int u=0; u<nu; u++, ++p_)
            Qiu[i][u] = Tuw[u][wi]*conj(*p_);
        }
        boost::multi_array_ref<ComplexType,2> v_(v[wi].origin(),extents[nmo_][nmo_]);
        // this can benefit significantly from 2-D partition of work
        // O[nmo * nmo * nmu]
        ma::product(a,Qiu,T(Piu),c,v_);
      }
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vbias(MatA const& G, MatB&& v, double a=1., double c=0.) {
        boost::const_multi_array_ref<ComplexType,2> G_(G.origin(),extents[1][G.shape()[0]]);
        boost::multi_array_ref<ComplexType,2> v_(v.origin(),extents[v.shape()[0]][1]);
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
      if(TMats.num_elements() < memory_needs)
        APP_ABORT(" Error: TMats.num_elements() < memory_needs() \n");
      boost::multi_array_ref<ComplexType,2> Guu(TMats.data(),extents[nu][nwalk]);
      boost::multi_array_ref<ComplexType,2> T1(TMats.data()+nwalk*nu,extents[nu][nel_]);
      Guu_from_compact(G,Guu,T1);
      // reinterpret as RealType matrices with 2x the columns
      boost::multi_array_ref<RealType,2> Luv_R(reinterpret_cast<RealType*>(Luv.origin()),
                                               extents[Luv.shape()[0]][2*Luv.shape()[1]]);
      boost::multi_array_ref<RealType,2> Guu_R(reinterpret_cast<RealType*>(Guu.origin()),
                                               extents[nu][2*nwalk]);
      boost::multi_array_ref<RealType,2> v_R(reinterpret_cast<RealType*>(v.origin()),
                                               extents[v.shape()[0]][2*v.shape()[1]]);
      ma::product(a,T(Luv_R),Guu_R,c,v_R);
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
      for(int iw=0; iw<nw; ++iw) {
        boost::const_multi_array_ref<ComplexType,2> Giw(G[iw].origin(),extents[nel_][nmo_]);
        // transposing inetermediary to make dot products faster in the next step
        ma::product(transposed(Piu),transposed(Giw),T1);
        for(int u=0; u<nu; ++u)
          Guu[u][iw] = a*ma::dot(cPua[u],T1[u]);               
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
        ma::product(scl, rotcPua[indices[range_t()][range_t(NAEA,NAEA+NAEB)]], 
                  T1, zero, Guv);
      else
        ma::product(scl, rotcPua[indices[range_t()][range_t(0,NAEA)]], 
                  T1, zero, Guv);
      for(int v=0; v<nv; ++v) 
        Guu[v] += Guv[v][v];  
    }

  protected:

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

};

}

}

}

#endif
