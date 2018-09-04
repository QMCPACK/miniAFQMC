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
 * This version works on a single memory space, e.g. CPU, GPU, Managed Memory, etc, based on the allocator. 
 */
template<class AllocType, // = std::allocator<ComplexType>
         class AllocType_ooc = AllocType 
        >
struct afqmc_sys: public AFQMCInfo
{

  public:

    using Alloc = AllocType;
    using pointer = typename Alloc::pointer;
    using const_pointer = typename Alloc::const_pointer;
    using IAlloc = typename Alloc::template rebind<int>::other;;
    using Alloc_ooc = AllocType_ooc;
    using pointer_ooc = typename Alloc_ooc::pointer;
    using const_pointer_ooc = typename Alloc_ooc::const_pointer;
    using IAlloc_ooc = typename Alloc_ooc::template rebind<int>::other;;
    Alloc allocator_;
    IAlloc iallocator_;
    Alloc_ooc allocator_ooc_;
    IAlloc_ooc iallocator_ooc_;

    ComplexMatrix<Alloc> trialwfn_alpha;
    ComplexMatrix<Alloc> trialwfn_beta;
    
    afqmc_sys() = delete;
    afqmc_sys(int nmo_, int na, Alloc alloc_ = Alloc{}, Alloc_ooc alloc_ooc_ = Alloc_ooc{}):
      AFQMCInfo(nmo_,na,na),
      allocator_(alloc_),
      iallocator_(alloc_),
      allocator_ooc_(alloc_ooc_),
      iallocator_ooc_(alloc_ooc_),
      trialwfn_alpha( {nmo_,na}, alloc_ ), 
      trialwfn_beta( {nmo_,na}, alloc_ ), 
      NMO(nmo_),
      NAEA(na),  
      WORK( {0}, alloc_ ),
// hack for status problem in getr?
      IWORK({NMO+1}, iallocator_ ),
      TAU( {NMO}, alloc_ ),
      TMat_NN( {NAEA,NAEA}, alloc_ ),
      TMat_NM( {NAEA,NMO}, alloc_ ),
      TMat_MN( {NMO,NAEA}, alloc_ ),
      TMat_MM( {NMO,NMO}, alloc_ ),
      Gcloc( {0,0}, alloc_ )
    {
      std::size_t buff = ma::getri_optimal_workspace_size(TMat_NN);
      buff = std::max(buff,std::size_t(ma::geqrf_optimal_workspace_size(TMat_NM)));
      buff = std::max(buff,std::size_t(ma::getrf_optimal_workspace_size(TMat_NN)));
      buff = std::max(buff,std::size_t(ma::gqr_optimal_workspace_size(TMat_NM)));
      buff = std::max(buff,std::size_t(ma::gelqf_optimal_workspace_size(TMat_MN)));
      buff = std::max(buff,std::size_t(ma::glq_optimal_workspace_size(TMat_MN)));
      WORK.reextent( {buff} );
    }

    ~afqmc_sys() {}

    // THC expects G[nwalk][ak]
    template< class WSet, 
              class Mat 
            >
    void calculate_mixed_density_matrix(const WSet& W, Mat& W_data, Mat& G, bool compact=true)
    {
      int nwalk = W.shape()[0];
      assert(G.num_elements() >= 2*NAEA*NMO*nwalk);
      assert(G.shape()[0] == nwalk);
      assert(G.shape()[1] >= 2*NAEA*NMO);
      assert(W_data.shape()[0] >= nwalk);
      assert(W_data.shape()[1] >= 5);
      int N_ = compact?NAEA:NMO;
      boost::multi::array_ref<ComplexType,4,pointer> G_4D(G.origin(), {nwalk,2,N_,NMO}); 
      for(int n=0; n<nwalk; n++) {
        base::MixedDensityMatrix<ComplexType>(trialwfn_alpha,W[n][0],
                       G_4D[n][0],TMat_NN,TMat_NM,IWORK,WORK,&W_data[n][3],compact);

        base::MixedDensityMatrix<ComplexType>(trialwfn_beta,W[n][1],
                       G_4D[n][1],TMat_NN,TMat_NM,IWORK,WORK,&W_data[n][4],compact);
      }
    }

    template<class WSet, class Mat>
    void calculate_overlaps(const WSet& W, Mat& W_data)
    {
      assert(W_data.shape()[0] >= W.shape()[0]);
      assert(W_data.shape()[1] >= 5);
      for(int n=0, nw=W.shape()[0]; n<nw; n++) {
        base::Overlap<ComplexType>(trialwfn_alpha,W[n][0],TMat_NN,IWORK,WORK,&W_data[n][3]);
        base::Overlap<ComplexType>(trialwfn_beta,W[n][1],TMat_NN,IWORK,WORK,&W_data[n][4]);
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
      boost::multi::array_cref<Type,3,const_pointer> V(vHS.origin(), {vHS.shape()[0],NMO,NMO});
      // re-interpretting matrices to avoid new temporary space  
      boost::multi::array_ref<Type,2,pointer> T1(TMat_NM.origin(), {NMO,NAEA});
      boost::multi::array_ref<Type,2,pointer> T2(TMat_MM.origin(), {NMO,NAEA});
      for(int nw=0, nwalk=W.shape()[0]; nw<nwalk; nw++) {

        ma::product(Propg,W[nw][0],TMat_MN);
        base::apply_expM(V[nw],TMat_MN,T1,T2,6);
        ma::product(Propg,TMat_MN,W[nw][0]);

        ma::product(Propg,W[nw][1],TMat_MN);
        base::apply_expM(V[nw],TMat_MN,T1,T2,6);
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

// No QR right now
        // LQ on the direct matrix
/*
        ma::gelqf(W[i][0],TAU,WORK);
        ma::glq(W[i][0],TAU,WORK);
        ma::gelqf(W[i][1],TAU,WORK);
        ma::glq(W[i][1],TAU,WORK);
*/

      }
    }

  private:

    int NMO, NAEA;

    //! Buffers using std::vector
    //! Used in QR and invert
    ComplexVector<Alloc> WORK; 
    IntegerVector<IAlloc> IWORK; 

    //! Vector used in QR routines 
    ComplexVector<Alloc> TAU;

    //! TMat_AB: Temporary Matrix of dimension [AxB]
    //! N: NAEA
    //! M: NMO
    ComplexMatrix<Alloc> TMat_NN;
    ComplexMatrix<Alloc> TMat_NM;
    ComplexMatrix<Alloc> TMat_MN;
    ComplexMatrix<Alloc> TMat_MM;

    //! storage for contraction of 2-electron integrals with density matrix
    ComplexMatrix<Alloc> Gcloc;
};

}
   
}

#endif
