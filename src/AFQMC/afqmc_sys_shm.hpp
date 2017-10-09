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

#ifndef  AFQMC_OPS_SHM_HPP 
#define  AFQMC_OPS_SHM_HPP 

#include "Configuration.h"
#include "Utilities/taskgroup.hpp"
#include "Numerics/ma_blas.hpp"
#include "Numerics/ma_operations.hpp"
#include "AFQMC/AFQMCInfo.hpp"
#include "AFQMC/energy.hpp"
#include "AFQMC/vHS.hpp"
#include "AFQMC/mixed_density_matrix.hpp"
#include "Message/ma_communications.hpp"

namespace qmcplusplus
{

namespace shm
{

struct afqmc_sys: public AFQMCInfo
{

  public:

    afqmc_sys(TaskGroup& tg_, int nw_):TG(tg_) 
    {
      nwalk = nw_;
      node_number = TG.getLocalNodeNumber();  
      core_number = TG.getCoreRank();
      ncores = TG.getNCoresPerTG();  
      nnodes = TG.getNNodesPerTG();  

      one = ComplexType(1.,0.);
      zero = ComplexType(0.,0.);
    }

    afqmc_sys(TaskGroup& tg_, int nw_, int nmo_, int na):TG(tg_)
    {
      nwalk = nw_;
      node_number = TG.getLocalNodeNumber();  
      core_number = TG.getCoreRank();
      ncores = TG.getNCoresPerTG();  
      nnodes = TG.getNNodesPerTG();  

      one = ComplexType(1.,0.);
      zero = ComplexType(0.,0.);

      setup(nmo_,na);  
    }

    ~afqmc_sys() {}

    ComplexMatrix trialwfn_alpha;
    ComplexMatrix trialwfn_beta;
    TaskGroup& TG;

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
      // 1. getri( TMat_NN )
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

      locWlkVec.resize(extents[nwalk]);

      std::tie(NAK_0, NAK_N) = FairDivideBoundary(core_number,2*NAEA*NMO,ncores);

      if(2*nwalk < ncores) {

        // setup boundaries
        for(int w=0,wend=2*nwalk; w<wend; w++) {
          int n0,n1;
          std::tie(n0,n1) = FairDivideBoundary(w,ncores,wend); 
          if(core_number>=n0 && core_number<n1) {
            local_walker_index = w;
            break;
          }
        } 

        MPI_Comm_split(TG.getTGCommLocal(),local_walker_index,core_number,&MPI_local_group);    
        MPI_Comm_rank(MPI_local_group,&MPI_local_group_rank);
        MPI_Comm_size(MPI_local_group,&MPI_local_group_size);

        // enough space for 3 copies of TMat_MN or (M*M + N*N + N*M) 
        int nelm = std::max(3*NMO*NAEA , NMO*NMO + NAEA*NAEA + NMO*NAEA);
        SM_TMats.setup(std::string("SM_TMats_")+std::to_string(local_walker_index),MPI_local_group);
        SM_TMats.resize(nelm);  

        std::tie(M_0,M_N) = FairDivideBoundary(MPI_local_group_rank,NMO,MPI_local_group_size);
        std::tie(N_0,N_N) = FairDivideBoundary(MPI_local_group_rank,NAEA,MPI_local_group_size);

      }

    } 

    template< class WSet, 
              class MatA, 
              class MatB 
            >
    void calculate_mixed_density_matrix(const WSet& W, MatA& W_data, MatB& G, bool compact=true)
    {
      assert( nwalk == W.shape()[0]);
      assert(G.num_elements() >= 2*NAEA*NMO*nwalk);
      assert(W_data.shape()[0] >= nwalk);
      assert(W_data.shape()[1] >= 4);
      int N_ = compact?NAEA:NMO;
      boost::multi_array_ref<ComplexType,4> G_4D(G.data(), extents[2][N_][NMO][nwalk]); 

      if(2*nwalk >= ncores) {

        boost::multi_array_ref<ComplexType,2> DM(TMat_MM.data(), extents[N_][NMO]); 
        for(int n=0, cnt=0; n<nwalk; n++) {

          if(cnt%ncores == core_number) {
            W_data[n][2] = base::MixedDensityMatrix<ComplexType>(trialwfn_alpha,W[n][0],
                         DM,TMat_NN,TMat_NM,IWORK,WORK,compact);
            G_4D[ indices[0][range_t(0,N_)][range_t(0,NMO)][n] ] = DM;
          }
          cnt++;

          if(cnt%ncores == core_number) {
            W_data[n][3] = base::MixedDensityMatrix<ComplexType>(trialwfn_beta,W[n][1],
                         DM,TMat_NN,TMat_NM,IWORK,WORK,compact);
            G_4D[ indices[1][range_t(0,N_)][range_t(0,NMO)][n] ] = DM;
          }
          cnt++;

        }

      } else {

        int nw = local_walker_index/2, sindex=local_walker_index%2;

        boost::multi_array_ref<ComplexType,2> DM(SM_TMats.values(), extents[N_][NMO]);
        boost::multi_array_ref<ComplexType,2> TNN(SM_TMats.values()+N_*NMO, extents[NAEA][NAEA]);
        boost::multi_array_ref<ComplexType,2> TNM(SM_TMats.values()+N_*NMO+NAEA*NAEA, extents[NAEA][NMO]);

        ComplexType ovlp;
        if(sindex==0) {
          ovlp = shm::MixedDensityMatrix<ComplexType>(trialwfn_alpha,W[nw][0],
                         DM,TNN,TNM,IWORK,WORK,
                         M_0,M_N,N_0,N_N,MPI_local_group,compact);
        } else {
          ovlp = shm::MixedDensityMatrix<ComplexType>(trialwfn_beta,W[nw][1],
                         DM,TNN,TNM,IWORK,WORK,
                         M_0,M_N,N_0,N_N,MPI_local_group,compact);
        }

        if(compact)
          G_4D[ indices[sindex][range_t(N_0,N_N)][range_t()][nw] ] = 
            DM[indices[range_t(N_0,N_N)][range_t()]];
        else
          G_4D[ indices[sindex][range_t(M_0,M_N)][range_t()][nw] ] = 
            DM[indices[range_t(M_0,M_N)][range_t()]];
        if(N_0==0)
          W_data[nw][2+sindex] = ovlp;

      }

      TG.local_barrier();  
    }

    template<class SpMat,
             class MatA,
             class MatB,
             class MatC
            >
    RealType calculate_energy(MatA& W_data, const MatB& G, const MatC& haj, const SpMat& V) 
    {
      assert(G.shape()[0] == 2*NAEA*NMO);
      assert(W_data.shape()[0] == G.shape()[1]);
      if(G.shape()[1] != Gcloc.shape()[1] || Gcloc.shape()[0] != V.rows())
        Gcloc.resize(extents[V.rows()][G.shape()[1]]);  
      if(locWlkVec.size() != nwalk)
        locWlkVec.resize(extents[nwalk]);
      if(core_number==0) {
        // zero 
        for(int n=0, ne=W_data.shape()[0]; n<ne; n++) W_data[n][0] = 0.;
      }
      TG.local_barrier();
      shm::calculate_energy(G,Gcloc,haj,V,locWlkVec,TG.getCoreRank()==0);
      {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*((TG.getBuffer())->getMutex()));
        BLAS::axpy(W_data.shape()[0],one,locWlkVec.data(),1,W_data.origin(),W_data.strides()[0]);
      }
      TG.local_barrier();
      RealType eav = 0., wgt=0.;
      for(int n=0, nw=G.shape()[1]; n<nw; n++) {
        wgt += W_data[n][1].real();
        eav += W_data[n][0].real()*W_data[n][1].real();
      }
      return eav/wgt;
    }

    /*
     * Calculates the local energy of a set of walkers with distributed hamiltonians.
     *   v1:
     *   - allgather Gloc -> Gglob
     *   - calculate local component
     *   - Allreduce eloc_glob 
     *   - copy eloc_glob -> eloc_local 
    */
    template< class SpMat,
              class MatA,
              class MatB,
              class MatC
        >
    RealType calculate_distributed_energy_v1(MatA& W_data, const MatA& G, MatC&& Gglob, const MatB& haj, const SpMat& V)
    {
      assert( G.shape()[0] == 2*NAEA*NMO);
      assert( nwalk == G.shape()[1] );
      assert( nwalk == W_data.shape()[0] );
      assert( Gglob.shape()[0] == G.shape()[0]);
      assert( Gglob.shape()[1] == G.shape()[1]*nnodes );
      if(nnodes == 1) 
        return calculate_energy(W_data,G,haj,V);

      ma::allgather_matrix(TG.getTGCommHeads(),
                           G[indices[range_t(NAK_0,NAK_N)][range_t()]],
                           Gglob[indices[range_t(NAK_0,NAK_N)][range_t()]],
                           byCols);

      if( (Gcloc.shape()[0] != V.rows()) || (Gcloc.shape()[1] != Gglob.shape()[1]) ) 
        Gcloc.resize( extents[V.rows()][Gglob.shape()[1]] );    
      TG.local_barrier();

      shm::calculate_energy(Gglob,Gcloc,haj,V,locWlkVec,(TG.getLocalNodeNumber()==0&&TG.getCoreRank()==0));

      // sum over TG
      MPI_Allreduce(MPI_IN_PLACE,locWlkVec.data(),2*nwalk*nnodes,
                        MPI_DOUBLE,MPI_SUM,TG.getTGComm());
      RealType eav[2];
      eav[0] = 0.;
      eav[1] = 0.;
      if(TG.getCoreRank() == 0) {
        int node_number = TG.getLocalNodeNumber();
        for(int i=0; i<nwalk; i++) {
          W_data[i][0] = locWlkVec[node_number*nwalk+i];
          eav[1] += W_data[i][1].real();
          eav[0] += W_data[i][0].real()*W_data[i][1].real();
        }
      }
      TG.local_barrier();

      MPI_Allreduce(MPI_IN_PLACE,eav,2,MPI_DOUBLE,MPI_SUM,TG.getTGComm());
      return eav[0]/eav[1];
    }

    template<class WSet, class Mat>
    void calculate_overlaps(const WSet& W, Mat& W_data)
    {
      assert(W_data.shape()[0] >= W.shape()[0]);
      assert(W_data.shape()[1] >= 4);

      if(2*nwalk >= ncores) {
        for(int n=0, nw=W.shape()[0]; n<nw; n++) {
          if(n%ncores != core_number) continue;
          W_data[n][2] = base::Overlap<ComplexType>(trialwfn_alpha,W[n][0],TMat_NN,IWORK);
          W_data[n][3] = base::Overlap<ComplexType>(trialwfn_beta,W[n][1],TMat_NN,IWORK);
        }
      } else {
        int nw = local_walker_index/2, sindex=local_walker_index%2;
        boost::multi_array_ref<ComplexType,2> T0(SM_TMats.values(), extents[NAEA][NAEA]);
        ComplexType ovlp;
        if(sindex==0) 
          ovlp = shm::Overlap<ComplexType>(trialwfn_alpha,W[nw][0],
                                                     T0,IWORK,N_0,N_N,MPI_local_group); 
        else
          ovlp = shm::Overlap<ComplexType>(trialwfn_beta,W[nw][1],
                                                     T0,IWORK,N_0,N_N,MPI_local_group); 
        if(N_0==0) 
          W_data[nw][2+sindex] = ovlp; 
      }
      TG.local_barrier();  
    }

    template<class WSet, 
             class MatA,
             class MatB
            >
    void propagate(WSet& W, MatA& Propg, MatB& vHS)
    {

      assert( nwalk == W.shape()[0]);  
      assert( nwalk == vHS.shape()[1] ); 
      using Type = typename std::decay<MatB>::type::element;
      boost::multi_array_ref<Type,2> T1(TMat_NM.data(), extents[NMO][NAEA]);
      boost::multi_array_ref<Type,2> T2(TMat_MM2.data(), extents[NMO][NAEA]);
      boost::multi_array_ref<Type,3> V(vHS.data(), extents[NMO][NMO][vHS.shape()[1]]);
      for(int nw=0, cnt=0; nw<nwalk; nw++) {

        if(cnt%ncores == core_number) {
          ma::product(Propg,W[nw][0],TMat_MN);
          // need deep-copy, since stride()[1] == nw otherwise
          TMat_MM = V[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ];
          base::apply_expM(TMat_MM,TMat_MN,T1,T2,6);
          ma::product(Propg,TMat_MN,W[nw][0]);
        }
        cnt++;
        if(cnt%ncores == core_number) {
          ma::product(Propg,W[nw][1],TMat_MN);
          TMat_MM = V[ indices[range_t(0,NMO)][range_t(0,NMO)][nw] ];
          base::apply_expM(TMat_MM,TMat_MN,T1,T2,6);
          ma::product(Propg,TMat_MN,W[nw][1]);
        }
        cnt++;

      }
      TG.local_barrier();  

    }    

    template<class WSet, 
             class MatA,
             class MatB
            >
    void propagate_from_global(WSet& W, MatA& Propg, MatB& vHS)
    {

      assert( nwalk == W.shape()[0]);  
      assert( nnodes*nwalk == vHS.shape()[1] ); 
      using Type = typename std::decay<MatB>::type::element;

      if(2*nwalk >= ncores) {
     
        boost::multi_array_ref<Type,2> T1(TMat_NM.data(), extents[NMO][NAEA]);
        boost::multi_array_ref<Type,2> T2(TMat_MM2.data(), extents[NMO][NAEA]);
        boost::multi_array_ref<Type,3> V(vHS.data(), extents[NMO][NMO][vHS.shape()[1]]);
        for(int nw=0, cnt=0; nw<nwalk; nw++) {

          if(cnt%ncores == core_number) {
            ma::product(Propg,W[nw][0],TMat_MN);
            // need deep-copy, since stride()[1] == nw otherwise
            TMat_MM = V[ indices[range_t(0,NMO)][range_t(0,NMO)][nw+node_number*nwalk] ];
            base::apply_expM(TMat_MM,TMat_MN,T1,T2,6);
            ma::product(Propg,TMat_MN,W[nw][0]);
          }
          cnt++;
          if(cnt%ncores == core_number) {
            ma::product(Propg,W[nw][1],TMat_MN);
            TMat_MM = V[ indices[range_t(0,NMO)][range_t(0,NMO)][nw+node_number*nwalk] ];
            base::apply_expM(TMat_MM,TMat_MN,T1,T2,6);
            ma::product(Propg,TMat_MN,W[nw][1]);
          }
          cnt++;

        }

      } else {

        int nw = local_walker_index/2, sindex=local_walker_index%2;

        boost::multi_array_ref<Type,2> T0(SM_TMats.values(), extents[NMO][NAEA]);
        boost::multi_array_ref<Type,2> T1(SM_TMats.values()+NMO*NAEA, extents[NMO][NAEA]);
        boost::multi_array_ref<Type,2> T2(SM_TMats.values()+2*NMO*NAEA, extents[NMO][NAEA]);

        boost::multi_array_ref<Type,3> V(vHS.data(), extents[NMO][NMO][vHS.shape()[1]]);

        ma::product(Propg[indices[range_t(M_0,M_N)][range_t()]],
                    W[nw][sindex],
                    T0[indices[range_t(M_0,M_N)][range_t()]]);

        TMat_MM[indices[range_t(0,M_N-M_0)][range_t()]] = 
            V[ indices[range_t(M_0,M_N)][range_t()][nw+node_number*nwalk] ];

        MPI_Barrier(MPI_local_group);

        shm::apply_expM(TMat_MM[indices[range_t(0,M_N-M_0)][range_t()]],
                        T0,T1,T2,M_0,M_N,MPI_local_group,6);

        MPI_Barrier(MPI_local_group);

        ma::product(Propg[indices[range_t(M_0,M_N)][range_t()]],
                    T0,
                    W[nw][sindex][indices[range_t(M_0,M_N)][range_t()]]);

      }
      TG.local_barrier();  
    }    

    template<class WSet>
    void orthogonalize(WSet& W)
    {
      for(int i=0, cnt=0; i<W.shape()[0]; i++) {

/*
          // QR on the transpose
        if(cnt%ncores == core_number) {  
          for(int r=0; r<NMO; r++)
            for(int c=0; c<NAEA; c++)
              TWORK2[c][r] = W[i][0][r][c];   
          ma::geqrf(TWORK2,TAU,TWORKV2);
          ma::gqr(TWORK2,TAU,TWORKV2);
          for(int r=0; r<NMO; r++)
            for(int c=0; c<NAEA; c++)
              W[i][0][r][c] = TWORK2[c][r];   
        }
        cnt++;

        if(cnt%ncores == core_number) {
          for(int r=0; r<NMO; r++)
            for(int c=0; c<NAEA; c++)
              TWORK2[c][r] = W[i][1][r][c];
          ma::geqrf(TWORK2,TAU,TWORKV2);
          ma::gqr(TWORK2,TAU,TWORKV2);
          for(int r=0; r<NMO; r++)
            for(int c=0; c<NAEA; c++)
              W[i][1][r][c] = TWORK2[c][r];
        }
        cnt++;
*/

        // LQ on the direct matrix
        if(cnt%ncores == core_number) {
          ma::gelqf(W[i][0],TAU,WORK);
          ma::glq(W[i][0],TAU,WORK);
        }
        cnt++;  
        if(cnt%ncores == core_number) {
          ma::gelqf(W[i][1],TAU,WORK);
          ma::glq(W[i][1],TAU,WORK);
        }
        cnt++;  

      }
      TG.local_barrier();  
    }

  private:

    // Buffers using std::vector
    // Used in QR and invert
    std::vector<ComplexType> WORK;
    std::vector<int> IWORK;

    // Vector used in QR routines 
    ComplexVector TAU;

    // TMat_AB: Temporary Matrix of dimension [AxB]
    // N: NAEA
    // M: NMO
    ComplexMatrix TMat_NN;
    ComplexMatrix TMat_NM;
    ComplexMatrix TMat_MN;
    ComplexMatrix TMat_MM;
    ComplexMatrix TMat_MM2;

    // storage for contraction of 2-electron integrals with density matrix
    ComplexMatrix Gcloc;

    // local array for temporary accumulation of local energies
    ComplexVector locWlkVec;

    int nwalk;
    int nnodes;
    int node_number;
    int core_number;
    int ncores;

    ComplexType one;
    ComplexType zero;

    SMDenseVector<ComplexType> SM_TMats;
    MPI_Comm MPI_local_group;
    int MPI_local_group_size, MPI_local_group_rank; 
    int local_walker_index;
    int M_0, M_N;
    int N_0, N_N;

    int NAK_0, NAK_N;
};

}

}

#endif
