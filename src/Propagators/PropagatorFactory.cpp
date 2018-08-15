
#include "Utilities/UtilityFunctions.h"
#include "Propagators/PropagatorFactory.h"

namespace qmcplusplus
{

namespace afqmc
{

Propagator getPropagator(TaskGroup_& TG, int NMO, int NAEA, Wavefunction& wfn, RandomGenerator_t* rng)
{
  using CVector = MArray<ComplexType,1>; 
  using CMatrix = MArray<ComplexType,2>; 

  RealType vbias_bound=50.0;

  // buld mean field expectation value of the Cholesky matrix
  CVector vMF({size_t(wfn.local_number_of_cholesky_vectors())});
  std::fill_n(vMF.origin(),vMF.num_elements(),ComplexType(0));
  {
    wfn.vMF(vMF);
    // if (not distribution_over_cholesky_vectors()), vMF needs to be reduced over TG
    if(not wfn.distribution_over_cholesky_vectors()) {
      if(not TG.TG_local().root()) 
        std::fill_n(vMF.origin(),vMF.num_elements(),ComplexType(0));
      TG.TG().all_reduce_in_place_n(vMF.origin(),vMF.num_elements(),std::plus<>());  
    }
  }

  // assemble H1(i,j) = h(i,j) + vn0(i,j) + sum_n vMF[n]*Spvn(i,j,n)
  CMatrix H1 = wfn.getOneBodyPropagatorMatrix(TG,vMF);

  AFQMCInfo AFinfo("AFinfo",NMO,NAEA,NAEA);

  if(TG.getNNodesPerTG() == 1) 
    return Propagator(AFQMCSharedPropagator(AFinfo,TG,wfn,std::move(H1),std::move(vMF),rng));
  else { 
    if(wfn.distribution_over_cholesky_vectors()) 
      // use specialized distributed algorithm for case 
      // when vbias doesn't need reduction over TG
      return Propagator(AFQMCDistributedPropagatorDistCV(AFinfo,TG,wfn,std::move(H1),std::move(vMF),rng));
    else
      return Propagator(AFQMCDistributedPropagator(AFinfo,TG,wfn,std::move(H1),std::move(vMF),rng));
  }
}


}

}

