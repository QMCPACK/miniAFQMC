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
    using extensions = typename ComplexVector<Alloc>::extensions_type;
    Alloc allocator_;
    IAlloc iallocator_;
    Alloc_ooc allocator_ooc_;
    IAlloc_ooc iallocator_ooc_;

    ComplexMatrix<Alloc> trialwfn_alpha;
    ComplexMatrix<Alloc> trialwfn_beta;
    WALKER_TYPES walker_type;
    
    afqmc_sys() = delete;
    afqmc_sys(int nmo_, int na, WALKER_TYPES wtype, Alloc alloc_ = Alloc{}, Alloc_ooc alloc_ooc_ = Alloc_ooc{}, int nbatch = 32):
      AFQMCInfo(nmo_,na,na),
      walker_type(wtype),
      allocator_(alloc_),
      iallocator_(alloc_),
      allocator_ooc_(alloc_ooc_),
      iallocator_ooc_(alloc_ooc_),
      trialwfn_alpha( {nmo_,na}, alloc_ ), 
      trialwfn_beta( {((wtype==COLLINEAR)?(nmo_):(0)),na}, alloc_ ), 
      NMO(nmo_),
      NAEA(na),  
      WORK( extensions{1}, alloc_ ),
// hack for status problem in getr?
      IWORK(extensions{nbatch*(NMO+1)}, iallocator_ ),
      TAU( extensions{NMO}, alloc_ ),
      TMat3D_NN( {nbatch,NAEA,NAEA}, alloc_ ),
      TMat3D_NM( {nbatch,NAEA,NMO}, alloc_ ),
      TMat3D_MN( {nbatch,NMO,NAEA}, alloc_ ),
      TMat3D_MN2( {nbatch,NMO,NAEA}, alloc_ ),
      TVec( extensions{1}, alloc_ )
    {
      if(nbatch%2==1)
        APP_ABORT(" Error: nbatch%2==1.\n");
      std::size_t buff = nbatch*ma::getri_optimal_workspace_size(TMat3D_NN[0]);
      buff = std::max(buff,nbatch*std::size_t(ma::geqrf_optimal_workspace_size(TMat3D_NM[0])));
      buff = std::max(buff,nbatch*std::size_t(ma::getrf_optimal_workspace_size(TMat3D_NN[0])));
      buff = std::max(buff,nbatch*std::size_t(ma::gqr_optimal_workspace_size(TMat3D_NM[0])));
#ifdef HAVE_LQ_FACTORIZATION
      buff = std::max(buff,nbatch*std::size_t(ma::gelqf_optimal_workspace_size(TMat3D_MN[0])));
      buff = std::max(buff,nbatch*std::size_t(ma::glq_optimal_workspace_size(TMat3D_MN[0])));
#endif
      WORK.reextent( {buff} );
      app_log()<<"\n afqmc_sys allocation per task in MB (nbatch=" <<nbatch <<") \n"
        <<"   - NN: " <<TMat3D_NN.num_elements()*sizeof(ComplexType)/1024.0/1024.0 <<"\n"
        <<"   - NM: " <<TMat3D_NM.num_elements()*sizeof(ComplexType)/1024.0/1024.0 <<"\n"
        <<"   - MN: " <<2*TMat3D_MN.num_elements()*sizeof(ComplexType)/1024.0/1024.0 <<"\n"
        <<"   - WORK: " <<WORK.num_elements()*sizeof(ComplexType)/1024.0/1024.0 <<std::endl;
    }

    ~afqmc_sys() {}

    // THC expects G[nwalk][ak]
    template< class WSet, 
              class Mat 
            >
    void calculate_mixed_density_matrix(WSet& W, Mat& W_data, Mat& G, bool compact=true)
    {
      int nspin = (walker_type==COLLINEAR?2:1);
      int nwalk = W.shape()[0];
      assert(G.num_elements() >= nspin*NAEA*NMO*nwalk);
      assert(W_data.shape()[0] >= nwalk);
      assert(W_data.shape()[1] >= 3+nspin);
      if(TVec.num_elements() < nwalk*nspin*NAEA*NMO) TVec.reextent({nwalk*nspin*NAEA*NMO}); 
      int N_ = NAEA; // compact?NAEA:NMO;
      assert(G.shape()[1] == nwalk);
      assert(G.shape()[0] == nspin*N_*NMO);
      // calculate [nwalk][..] and then tranpose  
      boost::multi::array_ref<ComplexType,4,pointer> G_4D(TVec.origin(), {nwalk,nspin,N_,NMO}); 
      boost::multi::array_ref<ComplexType,2,pointer> G_2D(G_4D.origin(), {nwalk,nspin*N_*NMO}); 
      if(nspin==1) 
        batched::MixedDensityMatrix(trialwfn_alpha,W,G_4D,TMat3D_NN,TMat3D_NM,IWORK,WORK,W_data(W_data.extension(0),{3,5}));
      else
        batched::MixedDensityMatrix(trialwfn_alpha,trialwfn_beta,W,G_4D,TMat3D_NN,TMat3D_NM,IWORK,WORK,W_data(W_data.extension(0),{3,5}));
      ma::transpose(G_2D,G);
    }

    template<class WSet, class Mat>
    void calculate_overlaps(WSet& W, Mat& W_data)
    {
      int nspin = (walker_type==COLLINEAR?2:1);
      assert(W_data.shape()[0] >= W.shape()[0]);
      assert(W_data.shape()[1] >= 3+nspin);
      if(nspin==1) 
        batched::Overlap(trialwfn_alpha,W,TMat3D_NN,IWORK,WORK,W_data(W_data.extension(0),{3,4}));
      else  
        batched::Overlap(trialwfn_alpha,trialwfn_beta,W,TMat3D_NN,IWORK,WORK,W_data(W_data.extension(0),{3,5}));
    }

    template<class WSet, 
             class MatA,
             class MatB
            >
    void propagate(WSet& W, MatA& Propg, MatB& vHS)
    {
      int nspin = (walker_type==COLLINEAR?2:1);
      assert(vHS.shape()[1] == NMO*NMO);  
      using Type = typename std::decay<MatB>::type::element;
      // assumes collinear for now
      int nbatch = TMat3D_NM.shape()[0];
      int nterms = W.shape()[0]*W.shape()[1];
      int nloop = (nterms + nbatch - 1)/nbatch;
      for(int i=0, ndone=0; i<nloop; i++,ndone+=nbatch) {  
        int ni = std::min(nbatch,nterms-ndone);
        boost::multi::array_ref<Type,3,pointer> W3D(W.origin() + ndone * W.strides()[1] , 
                                  {ni,W.shape()[2],W.shape()[3]});
        boost::multi::array_ref<Type,3,pointer> V(vHS.origin() + (ndone/2) * vHS.strides()[0] , 
                                  {ni/2,NMO,NMO});
        boost::multi::array_ref<Type,3,pointer> T1(TMat3D_NM.origin(), 
                                                        {ni,NMO,NAEA});
        
        batched::applyP1(Propg,W3D,T1);
        batched::apply_expM(V,T1,TMat3D_MN,TMat3D_MN2,6); 
        batched::applyP1(Propg,T1,W3D);
      }

    }    

    template<class WSet>
    void orthogonalize(WSet& W)
    {
      int nspin = (walker_type==COLLINEAR?2:1);
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
        if(nspin>1) {
          for(int r=0; r<NMO; r++)
            for(int c=0; c<NAEA; c++)
              TMat_NM[c][r] = W[i][1][r][c];
          ma::geqrf(TMat_NM,TAU,WORK);
          ma::gqr(TMat_NM,TAU,WORK);
          for(int r=0; r<NMO; r++)
            for(int c=0; c<NAEA; c++)
              W[i][1][r][c] = TMat_NM[c][r];
        }    
*/

// No QR right now
        // LQ on the direct matrix
/*
        ma::gelqf(W[i][0],TAU,WORK);
        ma::glq(W[i][0],TAU,WORK);
        if(nspin>1) {
          ma::gelqf(W[i][1],TAU,WORK);
          ma::glq(W[i][1],TAU,WORK);
        }
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
    ComplexArray<3,Alloc> TMat3D_NN;
    ComplexArray<3,Alloc> TMat3D_NM;
    ComplexArray<3,Alloc> TMat3D_MN;
    ComplexArray<3,Alloc> TMat3D_MN2;

    //! storage for contraction of 2-electron integrals with density matrix
    ComplexVector<Alloc> TVec;
};

}
   
}

#endif
