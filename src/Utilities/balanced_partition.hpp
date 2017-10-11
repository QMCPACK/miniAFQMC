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

#ifndef AFQMC_BALANCE_PARTITION_HPP 
#define AFQMC_BALANCE_PARTITION_HPP 

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <list>
#include <stack>
#include <cassert>

// given a list of (N+1) integers, this routine attempts to find a partitioning of n continuous subsets  
// such that the sum of elements in each set is approximately homogeneous
// In other words, the routine will minimize the variance of the difference between the sums in each set
// The number of elements in bucket i are given by indx[i+1]-indx[i]. In other words, tasks from indx[i] through indx[i+1]
// are assigned to bucket i. There are N buckets 
template<typename IType>
void balance_partition_ordered_set(int N, IType* indx, std::vector<IType>& subsets)
{
    int64_t avg=0;

    if(*(indx+N) == 0)
      exit(1);

    IType nsets = subsets.size()-1;
    IType i0=0;
    IType iN = N;
    while( *(indx + i0) == *(indx + i0 + 1) ) i0++;
    while( *(indx + iN - 1) == *(indx + iN) ) iN--;
    int64_t avNpc = (iN-i0)/nsets;
    int64_t extra = (iN-i0)%nsets;
    avg = static_cast<int64_t>(*(indx+iN)) - static_cast<int64_t>(*(indx+i0));
    avg /= nsets;

// no c++14 :-(
//    template<class Iter>
//    auto partition = [=] (IType i0, IType iN, int n, Iter vals) {
    auto partition = [=] (IType i0, IType iN, int n, typename std::list<IType>::iterator vals) {

      // finds optimal position for subsets[i] 
      auto step = [=] (IType i0, IType iN, IType& ik) {
        IType imin = ik;
        ik = i0+1;
        double v1 = double(std::abs(static_cast<int64_t>(*(indx+ik))
                            - static_cast<int64_t>(*(indx+i0))
                            - avg));
        double v2 = double(std::abs(static_cast<int64_t>(*(indx+iN))
                            - static_cast<int64_t>(*(indx+ik))
                            - avg));
        double vmin = v1*v1+v2*v2;
        for(int k=i0+2, kend=iN ; k<kend; k++) {
          v1 = double(std::abs(static_cast<int64_t>(*(indx+k))
                           - static_cast<int64_t>(*(indx+i0))
                           - avg));
          v2 = double(std::abs(static_cast<int64_t>(*(indx+iN))
                           - static_cast<int64_t>(*(indx+k))
                           - avg));
          double v = v1*v1+v2*v2;
          if( v < vmin ) {
            vmin=v;
            ik = k;
          }
        }
        return ik!=imin;
      };

      if(n==2) {
        *vals=i0+1;
        step(i0,iN,*vals);
        return; 
      }
    
      std::vector<IType> set(n+1);
      set[0]=i0;
      set[n]=iN;
      for(int i=n-1; i>=1; i--)
        set[i]=iN+i-n;
      bool changed;
      do {
        changed=false;
        for(IType i=1; i<n; i++)
          changed |= step(set[i-1],set[i+1],set[i]);
      } while( changed );

      std::copy_n(set.begin()+1,n-1,vals);

      return;  
    };

    // dummy factorization
    std::stack<IType> factors;
    IType n0=nsets; 
    for(IType i=2; i<=nsets; i++) {

      while( n0%i == 0 ) {
        factors.push(i);
        n0 /= i;
      } 
      if( n0 == 1 ) break;

    } 
    assert(n0==1);

    std::list<IType> sets;
    sets.push_back(i0);
    sets.push_back(iN);

    while(factors.size() > 0) {

      auto ns = factors.top();
      factors.pop();

      // divide all current partitions into ns sub-partitions
      typename std::list<IType>::iterator it=sets.begin();
      it++;
      for(; it!=sets.end(); it++) {
        typename std::list<IType>::iterator its=it;
        its--;
        auto i0 = *its;
        its = sets.insert(it,std::size_t(ns-1),i0+1);
        partition(i0,*it,ns,its);      
      }

    }

    typename std::list<IType>::iterator it=sets.begin();
    typename std::vector<IType>::iterator itv=subsets.begin();
    for(; itv<subsets.end(); itv++, it++)
      *itv = *it;

    return;
}

#endif

