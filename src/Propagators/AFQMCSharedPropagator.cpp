
#include "Configuration.h"
#include "Numerics/OhmmsBlas.h" 
#include "Utilities/UtilityFunctions.h"
#include "Utilities/Utils.hpp"
#include "Propagators/AFQMCSharedPropagator.h"
#include "Walkers/WalkerConfig.hpp"

namespace qmcplusplus 
{

namespace afqmc
{

void AFQMCSharedPropagator::parse()
{
  using qmcplusplus::app_log;

  // defaults
  vbias_bound=50.0;
  nback_prop_steps=0;
  free_projection=false;
  hybrid=true;
  importance_sampling=true;
  apply_constrain=true;
}

void AFQMCSharedPropagator::reset_nextra(int nextra) 
{
  if(nextra==0) return;
  if(last_nextra != nextra ) {
    last_nextra = nextra;
    for(int n=0; n<nextra; n++) {
      int n0,n1;
      std::tie(n0,n1) = FairDivideBoundary(n,TG.getNCoresPerTG(),nextra);
      if(TG.getLocalTGRank()>=n0 && TG.getLocalTGRank()<n1) {
        last_task_index = n;
        break;
      }
    }
#ifndef __bgq__
    local_group_comm = std::move(shared_communicator(TG.TG_local().split(last_task_index)));
#endif
  }
  if(last_task_index < 0 || last_task_index >= nextra)
    APP_ABORT("Error: Problems in AFQMCSharedPropagator::reset_nextra()\n");
}

void AFQMCSharedPropagator::assemble_X(size_t nsteps, size_t nwalk, RealType sqrtdt, 
                          CMatrix_ref& X, CMatrix_ref & vbias, CMatrix_ref& MF, 
                          CMatrix_ref& HWs) 
{
  // remember to call vbi = apply_bound_vbias(*vb);
  // X[m,ni,iw] = rand[m,ni,iw] + im * ( vbias[m,iw] - vMF[m]  )
  // HW[ni,iw] = sum_m [ im * ( vMF[m] - vbias[m,iw] ) * 
  //                     ( rand[m,ni,iw] + halfim * ( vbias[m,iw] - vMF[m] ) ) ] 
  //           = sum_m [ im * ( vMF[m] - vbias[m,iw] ) * 
  //                     ( X[m,ni,iw] - halfim * ( vbias[m,iw] - vMF[m] ) ) ] 
  // MF[ni,iw] = sum_m ( im * X[m,ni,iw] * vMF[m] )   
 
  TG.local_barrier();
  ComplexType im(0.0,1.0);
  ComplexType halfim(0.0,0.5);
  int nCV = int(X.shape()[0]);
  MArray_ref<ComplexType,3> X3D(X.origin(),{X.shape()[0],nsteps,nwalk});
  // generate random numbers
  {
    int i0,iN;
    std::tie(i0,iN) = FairDivideBoundary(TG.TG_local().rank(),int(X.num_elements()),
                                         TG.TG_local().size());  
    sampleGaussianFields_n(X.origin()+i0,iN-i0,*rng);
  }

  // construct X
  std::fill_n(HWs.origin(),HWs.num_elements(),ComplexType(0));  
  std::fill_n(MF.origin(),MF.num_elements(),ComplexType(0));  
  int m0,mN;
  std::tie(m0,mN) = FairDivideBoundary(TG.TG_local().rank(),nCV,TG.TG_local().size());   
  TG.local_barrier();
  for(int m=m0; m<mN; ++m) { 
    auto X_m = X3D[m];
    auto vb_ = vbias[m].origin();
    auto vmf_t = sqrtdt*apply_bound_vbias(vMF[m],1.0);
    auto vmf_ = sqrtdt*vMF[m];
    // apply bounds to vbias
    for(int iw=0; iw<nwalk; iw++) 
      vb_[iw] = apply_bound_vbias(vb_[iw],sqrtdt);
    for(int ni=0; ni<nsteps; ni++) {
      auto X_ = X3D[m][ni].origin();
      auto hws_ = HWs[ni].origin();
      auto mf_ = MF[ni].origin();
      for(int iw=0; iw<nwalk; iw++) {
        ComplexType vdiff = im * ( vb_[iw] - vmf_t );
        X_[iw] += vdiff; 
        hws_[iw] -= vdiff * ( X_[iw] - 0.5*vdiff );
        mf_[iw] += im * X_[iw] * vmf_;
      }
    }
  }
  TG.local_barrier();
}

}


}


