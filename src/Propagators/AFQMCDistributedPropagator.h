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

#ifndef QMCPLUSPLUS_AFQMC_DISTRIBUTEDPROPAGATOR_H
#define QMCPLUSPLUS_AFQMC_DISTRIBUTEDPROPAGATOR_H

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <tuple>

#include "io/hdf_archive.h"
#include "Utilities/RandomGenerator.h"

#include "Utilities/taskgroup.h"
#include "Matrix/mpi3_SHMBuffer.hpp"
#include "alf/boost/mpi3/shm/mutex.hpp"
#include "SlaterDeterminantOperations/SlaterDetOperations.hpp"

#include "Wavefunctions/Wavefunction.hpp"

#include "Propagators/AFQMCSharedPropagator.h"

namespace qmcplusplus
{

namespace afqmc
{

/*
 * AFQMC propagator using distributed Cholesky matrices over nodes in the TG. 
 * Shared memory parallelization is used for on-node concurrency.
 */
class AFQMCDistributedPropagator: public AFQMCSharedPropagator
{
  using base = AFQMCSharedPropagator;
  public:

    AFQMCDistributedPropagator(AFQMCInfo& info, afqmc::TaskGroup_& tg_, 
                          Wavefunction& wfn_, CMatrix&& h1_, CVector&& vmf_, 
                          RandomGenerator_t* r): 
            AFQMCSharedPropagator(info,tg_,wfn_,std::move(h1_),std::move(vmf_),r),
            req_Gsend(MPI_REQUEST_NULL),
            req_Grecv(MPI_REQUEST_NULL),
            req_vsend(MPI_REQUEST_NULL),
            req_vrecv(MPI_REQUEST_NULL)
    {
      assert(TG.getNNodesPerTG() > 1);
    }

    ~AFQMCDistributedPropagator() {
      if(req_Grecv!=MPI_REQUEST_NULL)
          MPI_Request_free(&req_Grecv);
      if(req_Gsend!=MPI_REQUEST_NULL)
          MPI_Request_free(&req_Gsend);
      if(req_vrecv!=MPI_REQUEST_NULL)
          MPI_Request_free(&req_vrecv);
      if(req_vsend!=MPI_REQUEST_NULL)
          MPI_Request_free(&req_vsend);
    }

    AFQMCDistributedPropagator(AFQMCDistributedPropagator const& other) = delete;
    AFQMCDistributedPropagator& operator=(AFQMCDistributedPropagator const& other) = delete;
    AFQMCDistributedPropagator(AFQMCDistributedPropagator&& other) = default;
    AFQMCDistributedPropagator& operator=(AFQMCDistributedPropagator&& other) = default;

    template<class WlkSet>
    void Propagate(int steps, WlkSet& wset, RealType E1,
                   RealType dt, int fix_bias=1) {
      int nblk = steps/fix_bias;
      int nextra = steps%fix_bias;
      for(int i=0; i<nblk; i++)
        step(fix_bias,wset,E1,dt);
      if(nextra>0)
        step(nextra,wset,E1,dt);
      TG.local_barrier();
    }

  protected: 

    MPI_Request req_Gsend, req_Grecv;
    MPI_Request req_vsend, req_vrecv;

    template<class WlkSet>
    void step(int steps, WlkSet& wset, RealType E1, RealType dt);

};

}

}

#include "Propagators/AFQMCDistributedPropagator.icc"

#endif

