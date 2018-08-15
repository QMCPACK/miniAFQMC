//////////////////////////////////////////////////////////////////////////////
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

#ifndef QMCPLUSPLUS_AFQMC_ROTATEHAMILTONIAN_HPP
#define QMCPLUSPLUS_AFQMC_ROTATEHAMILTONIAN_HPP

#include<vector>
#include<tuple>
#include<algorithm>
#include<numeric>

#include "af_config.h"
#include "Numerics/ma_operations.hpp"
#include "SlaterDeterminantOperations/rotate.hpp"

namespace qmcplusplus
{

namespace afqmc
{

inline void check_wavefunction_consistency(WALKER_TYPES type, PsiT_Matrix *A, PsiT_Matrix *B, int NMO, int NAEA, int NAEB) 
{
    if(type == CLOSED) {
      if(A->shape()[1] != NMO || A->shape()[0] != NAEA) {
        app_error()<<" Error: Incorrect Slater Matrix dimensions in check_wavefunction_consistency(): wfn_type=0, NMO, NAEA, A.rows, A.cols: " <<NMO <<" " <<NAEA <<" " <<A->shape()[0] <<" " <<A->shape()[1] <<std::endl; 
        APP_ABORT(" Error: Incorrect Slater Matrix dimensions in check_wavefunction_consistency().\n");
      }
    } else if(type == COLLINEAR) {
      if(A->shape()[1] != NMO || A->shape()[0] != NAEA || B->shape()[1] != NMO || B->shape()[0] != NAEB) {
        app_error()<<" Error: Incorrect Slater Matrix dimensions in check_wavefunction_consistency(): wfn_type=1, NMO, NAEA, NAEB, A.rows, A.cols, B.rows, B.cols: " 
        <<NMO <<" " <<NAEA <<" " <<NAEB <<" " 
        <<A->shape()[0] <<" " <<A->shape()[1] <<" "
        <<B->shape()[0] <<" " <<B->shape()[1] <<std::endl; 
        APP_ABORT(" Error: Incorrect Slater Matrix dimensions in check_wavefunction_consistency().\n");
      }
    } else if(type==NONCOLLINEAR) {
      if(A->shape()[1] != 2*NMO || A->shape()[0] != (NAEB+NAEA)) {
        app_error()<<" Error: Incorrect Slater Matrix dimensions in check_wavefunction_consistency(): wfn_type=1, NMO, NAEA, NAEB, A.rows, A.cols: " <<NMO <<" " <<NAEA <<" " <<NAEB <<" " <<A->shape()[0] <<" " <<A->shape()[1] <<std::endl; 
        APP_ABORT(" Error: Incorrect Slater Matrix dimensions in check_wavefunction_consistency().\n");
      }
    } else {
      app_error()<<" Error: Unacceptable walker_type in check_wavefunction_consistency(): " <<type <<std::endl;
      APP_ABORT(" Error: Unacceptable walker_type in check_wavefunction_consistency(). \n");
    }
}

inline MArray<SPComplexType,1> rotateHij(WALKER_TYPES walker_type, int NMO, int NAEA, int NAEB, PsiT_Matrix *Alpha, PsiT_Matrix *Beta, const MArray<ComplexType,2>& H1)
{
  MArray<SPComplexType,1> N;
  const ComplexType one = ComplexType(1.0);
  const ComplexType zero = ComplexType(0.0);

  // 1-body part 
  if(walker_type == CLOSED) {

    N.reextent({NAEA*NMO});
#if(AFQMC_SP)
    MArray<ComplexType,2> N_({NAEA,NMO});
#else
    MArray_ref<ComplexType,2> N_(N.origin(),{NAEA,NMO});
#endif

    ma::product(*Alpha,H1,N_);
#if(AFQMC_SP)
    std::copy_n(N_.origin(),NAEA*NMO,N.origin());
#endif
    ma::scal(SPComplexType(2.0),N);

  } else if(walker_type == COLLINEAR) {

    N.reextent({(NAEA+NAEB)*NMO});
#if(AFQMC_SP)
    MArray<ComplexType,2> NA_({NAEA,NMO});
    MArray<ComplexType,2> NB_({NAEB,NMO});
#else
    MArray_ref<ComplexType,2> NA_(N.origin(),{NAEA,NMO});
    MArray_ref<ComplexType,2> NB_(N.origin()+NAEA*NMO,{NAEB,NMO});
#endif

    ma::product(*Alpha,H1,NA_);
    ma::product(*Beta,H1,NB_);
#if(AFQMC_SP)
    std::copy_n(NA_.origin(),NAEA*NMO,N.origin());
    std::copy_n(NB_.origin(),NAEB*NMO,N.origin()+NAEA*NMO);
#endif

  } else if(walker_type == NONCOLLINEAR) {

    N.reextent({(NAEA+NAEB)*2*NMO});
#if(AFQMC_SP)
    MArray<ComplexType,2> N_({NAEA+NAEB,2*NMO});
#else
    MArray_ref<ComplexType,2> N_(N.origin(),{NAEA+NAEB,2*NMO});
#endif

    ma::product(*Alpha,H1,N_);
#if(AFQMC_SP)
    std::copy_n(N_.origin(),(NAEA+NAEB)*2*NMO,N.origin());
#endif

  }

  return N;
}

}

}

#endif
