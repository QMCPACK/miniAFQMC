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
#include "AFQMC/mixed_density_matrix.hpp"
#include "AFQMC/apply_expM.hpp"

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

  ComplexMatrix trialwfn_alpha("trialwfn_alpha", 1, 1);
  ComplexMatrix trialwfn_beta("trialwfn_beta", 1, 1);

    // bad design, but simple for now
  ComplexMatrix rotMuv("rotMuv", 1, 1);
  ComplexMatrix rotPiu("rotPiu", 1, 1);
  ComplexMatrix rotcPua("rotcPua", 1, 1);

    void setup(int nmo_, int na) {
      NMO = nmo_;
      NAEA = NAEB = na;

      Kokkos::resize(TMat_NM, NAEA, NMO);
      Kokkos::resize(TMat_MN, NMO, NAEA);
      Kokkos::resize(TMat_NN, NAEA, NAEA);
      Kokkos::resize(TMat_MM, NMO, NMO); 

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
      Kokkos::resize(Gcloc, 1, 1); // force resize later 

    } 

    // THC expects G[nwalk][ak]
    template< class WSet, 
              class Mat 
            >
    void calculate_mixed_density_matrix(const WSet& W, Mat& W_data, Mat& G, bool compact=true)
    {
      int nwalk = W.dimension_0();
      assert(G.num_elements() >= 2*NAEA*NMO*nwalk);
      assert(G.dimension_0() == nwalk);
      assert(G.dimension_1() >= 2*NAEA*NMO);
      assert(W_data.dimension_0() >= nwalk);
      assert(W_data.dimesnion_1() >= 4);
      int N_ = compact?NAEA:NMO;
      boost::multi_array_ref<ComplexType,4> G_4D(G.data(), extents[nwalk][2][N_][NMO]);
      
      for(int n=0; n<nwalk; n++) {
        W_data(n, 2) = base::MixedDensityMatrix<ComplexType>(trialwfn_alpha,W(n, 0),
							     Kokkos::subview(G_4D, n, 0, Kokkos::ALL(), Kokkos::ALL()),
							     TMat_NN,TMat_NM,IWORK,WORK,compact);

        W_data(n, 3) = base::MixedDensityMatrix<ComplexType>(trialwfn_beta,W(n, 1),
							     Kokkos::subview(G_4D, n, 1, Kokkos::ALL(), Kokkos::ALL()),
							     TMat_NN,TMat_NM,IWORK,WORK,compact);
      }
    }

    template<class WSet, class Mat>
    void calculate_overlaps(const WSet& W, Mat& W_data)
    {
      assert(W_data.simension_0() >= W.dimension_0());
      assert(W_data.dimension_1() >= 4);
      for(int n=0, nw=W.dimension_0(); n<nw; n++) {
        W_data(n, 2) = base::Overlap<ComplexType>(trialwfn_alpha,
						  Kokkos::subview(W, n, 0, Kokkos::ALL(), Kokkos::ALL()),TMat_NN,IWORK);
        W_data(n, 3) = base::Overlap<ComplexType>(trialwfn_beta,
						  Kokkos::subview(W, n, 1, Kokkos::ALL(), Kokkos::ALL()),TMat_NN,IWORK);
      }
    }

    template<class WSet, 
             class MatA,
             class MatB
            >
    void propagate(WSet& W, const MatA& Propg, const MatB& vHS)
    {
      assert(vHS.shape()[1] == NMO*NMO);  
      using Type = typename std::decay<MatB>::type::element;
      boost::const_multi_array_ref<Type,3> V(vHS.data(), extents[vHS.shape()[0]][NMO][NMO]);
      // re-interpretting matrices to avoid new temporary space  
      boost::multi_array_ref<Type,2> T1(TMat_NM.data(), extents[NMO][NAEA]);
      boost::multi_array_ref<Type,2> T2(TMat_MM.data(), extents[NMO][NAEA]);
      for(int nw=0, nwalk=W.shape()[0]; nw<nwalk; nw++) {

        ma::product(Propg,Kokkos::subview(W, nw, 0, Kokkos::ALL(), Kokkos::ALL()),TMat_MN);
        base::apply_expM(Kokkos::subview(V, nw, Kokkso::ALL(), Kokkos::ALL()),TMat_MN,T1,T2,6);
        ma::product(Propg,TMat_MN,Kokkos::subview(W, nw, 0, Kokkos::ALL(), Kokkos::ALL()));

        ma::product(Propg,Kokkos::subview(W, nw, 1, Kokkos::ALL(), Kokkos::ALL()),TMat_MN);
        base::apply_expM(Kokkos::subview(V, nw, Kokkso::ALL(), Kokkos::ALL()),TMat_MN,T1,T2,6);
        ma::product(Propg,TMat_MN,Kokkos::subview(W, nw, 1, Kokkos::ALL(), Kokkos::ALL()));

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
        ma::gelqf(Kokkos::subview(W, i, 0, Kokkos::ALL(), Kokkos::ALL()),TAU,WORK);
        ma::glq(Kokkos::subview(W, i, 0, Kokkos::ALL(), Kokkos::ALL()),TAU,WORK);
        ma::gelqf(Kokkos::subview(W, i, 1, Kokkos::ALL(), Kokkos::ALL()),TAU,WORK);
        ma::glq(Kokkos::subview(W, i, 1, Kokkos::ALL(), Kokkos::ALL()),TAU,WORK);

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

    //! storage for contraction of 2-electron integrals with density matrix
    ComplexMatrix Gcloc;
};

}
   
}

#endif
