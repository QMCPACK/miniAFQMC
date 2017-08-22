
#ifndef AFQMC_BALANCE_PARTITION_HPP 
#define AFQMC_BALANCE_PARTITION_HPP 

#include <numeric>
#include<iostream>

namespace qmcplusplus { 

// given a list of (N+1) integers, this routine attempts to find a partitioning of n continuous subsets  
// such that the sum of elements in each set is approximately homogeneous
// In other words, the routine will minimize the variance of the difference between the sums in each set
// The number of elements in bucket i are given by indx[i+1]-indx[i]. In other words, tasks from indx[i] through indx[i+1]
// are assigned to bucket i. There are N buckets 
template<typename IType>
void balance_partition_ordered_set(int N, IType* indx, std::vector<IType>& subsets) 
{
    int64_t avg=0;

    // finds optimal position for subsets[i] 
    auto step = [&] (int i) {
      IType i0 = subsets[i];
      subsets[i] = subsets[i-1]+1;
      int64_t vmin = std::abs(*(indx+subsets[i])-*(indx+subsets[i-1])-avg)
                   + std::abs(*(indx+subsets[i+1])-*(indx+subsets[i])-avg);
      for(int k=subsets[i-1]+2 ; k<subsets[i+1]; k++) {
        int64_t v = std::abs(*(indx+k)-*(indx+subsets[i-1])-avg)
                  + std::abs(*(indx+subsets[i+1])-*(indx+k)-avg);
        if( v < vmin ) {
          vmin=v;
          subsets[i] = k;
        }
      }
      return subsets[i]!=i0;
    };

    if(*(indx+N) == 0)
      APP_ABORT("Error in balance_partition_ordered_set(): empty hamiltonian. \n");

    IType nsets = subsets.size()-1;
    IType i0=0;
    IType iN = N;
    while( *(indx + i0) == *(indx + i0 + 1) ) i0++;
    while( *(indx + iN - 1) == *(indx + iN) ) iN--;
    int64_t avNpc = (iN-i0)/nsets;
    int64_t extra = (iN-i0)%nsets;
    for(IType i=0; i<nsets; i++)
      subsets[i]=( i<extra )?(i0+i*(avNpc+1)):(i0+i*avNpc+extra);
    subsets[nsets]=iN;

    for(IType i=0; i<nsets; i++)
      avg += *(indx+subsets[i+1]) - *(indx+subsets[i]);
    avg /= nsets;
    bool changed;
    do {
      changed=false;
      for(IType i=1; i<nsets; i++)
        changed |= step(i);
    } while( changed );

}

}

#endif

