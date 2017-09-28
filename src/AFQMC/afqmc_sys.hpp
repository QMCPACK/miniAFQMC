////////////////////////////////////////////////////////////////////////////
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

/** @file afqmc_sys.hpp
 *  @brief AFQMC global state
 */

#ifndef  AFQMC_OPS_HPP 
#define  AFQMC_OPS_HPP 

#include "Configuration.h"
#include "Numerics/ma_lapack.hpp"
#include "Numerics/ma_operations.hpp"
#include "AFQMC/AFQMCInfo.hpp"
#include "AFQMC/energy.hpp"
#include "AFQMC/vHS.hpp"
#include "AFQMC/mixed_density_matrix.hpp"

namespace qmcplusplus
{

namespace base 
{

/**
 * Quasi-global AFQMC object, contains workspaces 
 * 
 * \todo get rid of it and global state in general
 */
struct afqmc_sys: public AFQMCInfo
{

  public:

    afqmc_sys() {}

    afqmc_sys(int nmo_, int na) 
    {
        setup(nmo_,na);
    }

    ~afqmc_sys() {}

    ComplexMatrix trialwfn_alpha;
    ComplexMatrix trialwfn_beta;

    void setup(int nmo_, int na) {
      NMO = nmo_;
      NAEA = NAEB = na;

      TMat_NM.resize(extents[NAEA][NMO]);
      TMat_MN.resize(extents[NMO][NAEA]);
      TMat_NN.resize(extents[NAEA][NAEA]);
      TMat_MM.resize(extents[NMO][NMO]); 
      TMat_MM2.resize(extents[NMO][NMO]); 

      // reserve enough space in lapack's work array
      // Make sure it is large enough for:
      //  1. getri( TMat_NN )
      WORK.reserve(  ma::getri_optimal_workspace_size(TMat_NN) );
      //  2. geqrf( TMat_NM )
      WORK.reserve(  ma::geqrf_optimal_workspace_size(TMat_NM) );
      //  3. gqr( TMat_NM )
      WORK.reserve(  ma::gqr_optimal_workspace_size(TMat_NM) );
      //  4. gelqf( TMat_MN )
      WORK.reserve(  ma::gelqf_optimal_workspace_size(TMat_MN) );
      //  5. glq( TMat_MN )
      WORK.reserve(  ma::glq_optimal_workspace_size(TMat_MN) );

      // IWORK: integer buffer for getri/getrf  
      IWORK.resize(NMO);

      // TAU: ComplexVector used in QR routines 
      TAU.resize(extents[NMO]);

      // temporary storage for contraction of density matrix with 2-electron integrals  
      Gcloc.resize(extents[1][1]); // force resize later 

    } 

    template< class WSet, 
              class Mat 
            >
    void calculate_mixed_density_matrix(const WSet& W, Mat& W_data, Mat& G, bool compact=true)
    {
      int nwalk = W.shape()[0];
      assert(G.num_elements() >= 2*NAEA*NMO*nwalk);
      assert(W_data.shape()[0] >= nwalk);
      assert(W_data.shape()[1] >= 4);
      int N_ = compact?NAEA:NMO;
      boost::multi_array_ref<ComplexType,2> DM(TMat_MM.data(), extents[N_][NMO]); 
      boost::multi_array_ref<ComplexType,4> G_4D(G.data(), extents[2][N_][NMO][nwalk]); 
      for(int n=0; n<nwalk; n++) {
        W_data[n][2] = base::MixedDensityMatrix<ComplexType>(trialwfn_alpha,W[n][0],
                       DM,TMat_NN,TMat_NM,IWORK,WORK,compact);
        G_4D[ indices[0][range_t(0,N_)][range_t(0,NMO)][n] ] = DM;

        W_data[n][3] = base::MixedDensityMatrix<ComplexType>(trialwfn_beta,W[n][1],
                       DM,TMat_NN,TMat_NM,IWORK,WORK,compact);
        G_4D[ indices[1][range_t(0,N_)][range_t(0,NMO)][n] ] = DM;
      }
    }

    template<class SpMat,
             class Mat
            >
    RealType calculate_energy(Mat& W_data, const Mat& G, const Mat& haj, const SpMat& V) 
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

    template<class WSet, class Mat>
    void calculate_overlaps(const WSet& W, Mat& W_data)
    {
      assert(W_data.shape()[0] >= W.shape()[0]);
      assert(W_data.shape()[1] >= 4);
      for(int n=0, nw=W.shape()[0]; n<nw; n++) {
        W_data[n][2] = base::Overlap<ComplexType>(trialwfn_alpha,W[n][0],TMat_NN,IWORK);
        W_data[n][3] = base::Overlap<ComplexType>(trialwfn_beta,W[n][1],TMat_NN,IWORK);
      }
    }

    template<class WSet, 
             class MatA,
             class MatB
            >
    void propagate(WSet& W, MatA& Propg, MatB& vHS)
    {

      assert(vHS.shape()[0] == NMO*NMO);  
      using Type = typename std::decay<MatB>::type::element;
      boost::multi_array_ref<Type,3> V(vHS.data(), extents[NMO][NMO][vHS.shape()[1]]);
      // re-interpretting matrices to avoid new temporary space  
      boost::multi_array_ref<Type,2> T1(TMat_NM.data(), extents[NMO][NAEA]);
      boost::multi_array_ref<Type,2> T2(TMat_MM2.data(), extents[NMO][NAEA]);
      for(int nw=0, nwalk=W.shape()[0]; nw<nwalk; nw++) {

        // need deep-copy, since stride()[1] == nw otherwise
        TMat_MM = V[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ];

        ma::product(Propg,W[nw][0],TMat_MN);
        base::apply_expM(TMat_MM,TMat_MN,T1,T2,6);
        ma::product(Propg,TMat_MN,W[nw][0]);

        ma::product(Propg,W[nw][1],TMat_MN);
        base::apply_expM(TMat_MM,TMat_MN,T1,T2,6);
        ma::product(Propg,TMat_MN,W[nw][1]);

      }

    }    

    template<class WSet>
    void orthogonalize(WSet& W)
    {
      for(int i=0; i<W.shape()[0]; i++) {

/*
        // QR on the transpose
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            TMat_NM[c][r] = W[i][0][r][c];   
        ma::geqrf(TMat_NM,TAU,WORK);
        ma::gqr(TMat_NM,TAU,WORK);
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            W[i][0][r][c] = TMat_NM[c][r];   
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            TMat_NM[c][r] = W[i][1][r][c];
        ma::geqrf(TMat_NM,TAU,WORK);
        ma::gqr(TMat_NM,TAU,WORK);
        for(int r=0; r<NMO; r++)
          for(int c=0; c<NAEA; c++)
            W[i][1][r][c] = TMat_NM[c][r];
*/

        // LQ on the direct matrix
        ma::gelqf(W[i][0],TAU,WORK);
        ma::glq(W[i][0],TAU,WORK);
        ma::gelqf(W[i][1],TAU,WORK);
        ma::glq(W[i][1],TAU,WORK);

      }
    }

  private:

    //! Buffers using std::vector
    //! Used in QR and invert
    std::vector<ComplexType> WORK; 
    std::vector<int> IWORK; 

    //! Vector used in QR routines 
    ComplexVector TAU;

    //! TMat_AB: Temporary Matrix of dimension [AxB]
    //! N: NAEA
    //! M: NMO
    ComplexMatrix TMat_NN;
    ComplexMatrix TMat_NM;
    ComplexMatrix TMat_MN;
    ComplexMatrix TMat_MM;
    ComplexMatrix TMat_MM2;

    //! storage for contraction of 2-electron integrals with density matrix
    ComplexMatrix Gcloc;
};

}
   
}

#endif
