////////////////////////////////////////////////////////////////////////////////
//// This file is distributed under the University of Illinois/NCSA Open Source
//// License.  See LICENSE file in top directory for details.
////
//// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
////
//// File developed by:
////
//// File created by:
//////////////////////////////////////////////////////////////////////////////////

#ifndef  AFQMC_OPS_HPP 
#define  AFQMC_OPS_HPP 

#include "Configuration.h"
#include "Numerics/ma_blas.hpp"
#include "Numerics/ma_operations.hpp"
#include "AFQMC/AFQMCInfo.hpp"
#include "AFQMC/energy.hpp"
#include "AFQMC/vHS.hpp"
#include "AFQMC/mixed_density_matrix.hpp"

namespace qmcplusplus
{

struct afqmc_sys: public AFQMCInfo
{

  public:

    afqmc_sys() {}

    ~afqmc_sys() {}

    ComplexMatrix trialwfn_alpha;
    ComplexMatrix trialwfn_beta;

    void setup(int nmo_, int na) {
      NMO = nmo_;
      NAEA = NAEB = na;
      TWORK1.resize(extents[NAEA][NAEA]);
      TWORK2.resize(extents[NAEA][NMO]);
      TWORK3.resize(extents[NAEA][NMO]);
      IWORK1.resize(extents[2*NMO]);
      S0.resize(extents[NMO][NAEA]);    
      TWORKV1.resize(extents[NAEA*NAEA]); 
      DM.resize(extents[NMO][NMO]);      
      Gcloc.resize(extents[2*NAEA*NMO][16]); // default to 16, resize later if necessary
    } 

    template< class WSet, 
              class ValueMat 
            >
    void calculate_mixed_density_matrix(const WSet& W, ValueMat& W_data, ValueMat& G, bool compact=true)
    {
      int nwalk = W.shape()[0];
      assert(G.num_elements() >= 2*NAEA*NMO*nwalk);
      assert(W_data.shape()[0] >= nwalk);
      assert(W_data.shape()[1] >= 4);
      int N_ = compact?NAEA:NMO;
      boost::multi_array_ref<ComplexType,2> DMr(DM.data(), extents[N_][NMO]); 
      boost::multi_array_ref<ComplexType,4> G_4D(G.data(), extents[2][N_][NMO][nwalk]); 
      for(int n=0; n<nwalk; n++) {
        W_data[n][2] = base::MixedDensityMatrix<ComplexType>(trialwfn_alpha,W[n][0],
                       DMr,IWORK1,TWORK1,TWORK2,TWORKV1,compact);
        G_4D[ indices[0][range_t(0,N_)][range_t(0,NMO)][n] ] = DMr;

        W_data[n][3] = base::MixedDensityMatrix<ComplexType>(trialwfn_beta,W[n][1],
                       DMr,IWORK1,TWORK1,TWORK2,TWORKV1,compact);
        G_4D[ indices[1][range_t(0,N_)][range_t(0,NMO)][n] ] = DMr;
      }
    }

    template<class SpMat,
             class ValueMat
            >
    RealType calculate_energy(ValueMat& W_data, const ValueMat& G, const ValueMat& haj, const SpMat& V) 
    {
      assert(G.shape()[0] == 2*NAEA*NMO);
      if(G.shape()[1] != Gcloc.shape()[1])
        Gcloc.resize(extents[2*NMO*NAEA][G.shape()[1]]);  
      base::calculate_energy(W_data,G,Gcloc,haj,V);
      RealType eav = 0., wgt=0.;
      for(int n=0, nw=G.shape()[1]; n<nw; n++) {
        wgt += W_data[n][1].real();
        eav += W_data[n][0].real()*W_data[n][1].real();
      }
      return eav/wgt;
    }

    template<class WSet, class ValueMat>
    void calculate_overlaps(const WSet& W, ValueMat& W_data)
    {
      assert(W_data.shape()[0] >= W.shape()[0]);
      assert(W_data.shape()[1] >= 4);
      for(int n=0, nw=W.shape()[0]; n<nw; n++) {
        W_data[n][2] = base::Overlap<ComplexType>(trialwfn_alpha,W[n][0],IWORK1,TWORK1);
        W_data[n][3] = base::Overlap<ComplexType>(trialwfn_beta,W[n][1],IWORK1,TWORK1);
      }
    }

    template<class WSet, 
             class ValueMatA,
             class ValueMatB
            >
    void propagate(WSet& W, ValueMatA& Propg, ValueMatB& vHS)
    {

      typedef typename std::decay<ValueMatB>::type::element Type;
      boost::multi_array_ref<Type,3> V(vHS.data(), extents[NMO][NMO][vHS.shape()[1]]);
      boost::multi_array_ref<Type,2> TWORK2r(TWORK2.data(), extents[NMO][NAEA]);
      boost::multi_array_ref<Type,2> TWORK3r(TWORK3.data(), extents[NMO][NAEA]);
      for(int nw=0, nwalk=W.shape()[0]; nw<nwalk; nw++) {

        ma::product(Propg,W[nw][0],S0);
        // need deep-copy, since stride()[1] == nw otherwise
        DM = V[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ];
        base::apply_expM(DM,S0,TWORK2r,TWORK3r,6);
        ma::product(Propg,S0,W[nw][0]);

        ma::product(Propg,W[nw][1],S0);
        base::apply_expM(DM,S0,TWORK2r,TWORK3r,6);
        ma::product(Propg,S0,W[nw][1]);

      }

    }    

    template<class WSet>
    void orthogonalize(WSet& W)
    {
      for(int i=0; i<W.shape()[0]; i++) {

        // QR on the transpose
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            TWORK2[c][r] = W[i][0][r][c];   
        ma::geqrf(TWORK2,TAU,TWORKV2);
        ma::gqr(TWORK2,TAU,TWORKV2);
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            W[i][0][r][c] = TWORK2[c][r];   
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            TWORK2[c][r] = W[i][1][r][c];
        ma::geqrf(TWORK2,TAU,TWORKV2);
        ma::gqr(TWORK2,TAU,TWORKV2);
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            W[i][1][r][c] = TWORK2[c][r];

        // LQ on the direct matrix
        //ma::gelqf(W[i][0],TAU,TWORKV2);
        //ma::glq(W[i][0],TAU,TWORKV2);
        //ma::gelqf(W[i][1],TAU,TWORKV2);
        //ma::glq(W[i][1],TAU,TWORKV2);

      }
    }

  private:

    index_gen indices;

    // work arrays
    ComplexMatrix TWORK1;
    ComplexMatrix TWORK2;
    ComplexMatrix TWORK3;
    IndexVector IWORK1;
    ComplexVector TWORKV1;
    ComplexMatrix S0; 
    ComplexMatrix DM;
    ComplexMatrix Gcloc;
    ComplexVector TWORKV2;  
    ComplexVector TAU;
};
   
}

#endif
