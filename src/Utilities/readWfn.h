#ifndef AFQMC_READWFN_H
#define AFQMC_READWFN_H

#include<cstdlib>
#include<iostream>
#include<vector>
#include<string>
#include<ctype.h>

#include "af_config.h"

namespace qmcplusplus
{

namespace afqmc
{

/*
 * Reads ndets from the ascii file. 
 * If pureSD == false, PsiT contains the Slater Matrices of all the terms in the expansion.
 *      For walker_type==1, PsiT contains 2*ndets terms including both Alpha/Beta components.
 * If pureSD == true, PsiT contains only the reference determinant and excitations contains
 *      the occupation strings of all the determinants in the expansion, including the reference.  
 */ 
void read_wavefunction(std::string filename, int& ndets, std::string& type, WALKER_TYPES walker_type,
        boost::mpi3::shared_communicator& comm, int NMO, int NAEA, int NAEB,
        std::vector<PsiT_Matrix>& PsiT, std::vector<ComplexType>& ci,
        std::vector<int>& excitations); 

// modify for multideterminant case based on type
int readWfn( std::string fileName, MArray<ComplexType,3>& OrbMat, int NMO, int NAEA, int NAEB, int det = 0);

}

}

#endif

