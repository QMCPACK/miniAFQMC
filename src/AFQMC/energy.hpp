//////////////////////////////////////////////////////////////////////////
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

/** @file energy.hpp
 *  @brief Local energy
 */

#ifndef  AFQMC_ENERGY_HPP 
#define  AFQMC_ENERGY_HPP 

#include <type_traits>
#include "Numerics/ma_operations.hpp"

namespace qmcplusplus
{

// base == Serial implementation
namespace base
{

/** Calculates the local energy from (already evaluated) mixed density matrices.
 *
 *  \f$V_{akbl}(ak, bl) = \sum_{ij} \psi_\text{trial}(i, a)^\dagger \psi_\text{trial}(j, b) (\langle ij|kj\rangle - \langle ij|lk\rangle) \to (\alpha, \beta)\f$
 * 
 *  Vakbl(ak,bl) = sum_i sum_j conj(TrialWfn(i,a)) * conj(TrialWfn(j,b)) * (<ij|kl> - <ij|lk>)
 *    --> (alpha, beta) 
 * 
 * terms, no exchange term is included (e.g. \f$\langle ij|lk\rangle\f$)
 *  The 2-body contribution to the energy is obtained from:
 *  Let \f$G(i,j) {Gmod(i,j)}\f$ be the *modified* "Green's Function" of the walker 
 *            (defined by the Slater Matrix "W"), then:
 *  \f$ G = \psi_\text{trial}^\dagger * [W^T*\psi_\text{trial}^\dagger]^{-1} W^T \f$ 
 *
 *  \f$ G_\text{mod} = [W^T \psi_\text{trial}^\dagger]^{-1}  W^T \f$   
 *
 *  \f$ E_\text{2body} = \sum_{ijkl} G(i,k) (\langle ij|kl \rangle - \left<ij|lk\right>) * G(j,l) + (\alpha/\beta) + (beta/beta)  \f$
 *  \f$         = \sum_{akbl} G_\text{mod}(a,k) * V_{abkl}(ak,jl) G_\text{mod}(jl) = G_\text{mod} * V_{akbl} * G_\text{mod} \f$
 *
 *   The expression can be evaluated with a sparse matrix-dense vector product, 
 *   followed by a dot product between vectors, if we interpret the matrix \f$ G_\text{mod}(a,k)\f$
 *   as a vector with "linearized" index $\mathrm{ak}=a\times \mathrm{NMO} + k$.
 *
 * TODO: handle l-value references properly
 *
 * TODO: add dimensionality information in concept
 *
 * TODO: avoid use of multi_array_ref
 */
template< class Mat,
          class SpMat
        >
inline void calculate_energy(Mat& W_data, const Mat& Gc, Mat& Gcloc, const Mat& haj, const SpMat& Vakbl)
{
  // W[nwalk][2][NMO][NAEA]
 
  assert(W_data.shape()[1] >= 4);
  assert(Gc.shape()[1] == W_data.shape()[0]);
  assert(Gc.shape()[1] == Gcloc.shape()[1]);
  assert(Gc.shape()[0] == Gcloc.shape()[0]);
  assert(Gc.shape()[0] == haj.num_elements());
  assert(Gc.shape()[0] == Vakbl.rows());
  assert(Gc.shape()[0] == Vakbl.cols());

  using Type = typename std::decay<Mat>::type::element;
//  index_gen indices;
  Type zero = Type(0.);
  Type one = Type(1.); 
  Type half = Type(0.5); 

  int nwalk = W_data.shape()[0];
  boost::const_multi_array_ref<Type,1> haj_ref(haj.origin(), extents[haj.num_elements()]);

  for(int n=0; n<nwalk; n++) W_data[n][0] = zero; //< zero

  ma::product(Vakbl, Gc, Gcloc);   //< Vakbl * Gc(bl,nw) = Gcloc(ak,nw)

  //! \f$ E_2(nw) = 0.5 G_c(:,nw)*Gcloc(:,nw)\f$
  // how do I do this through BLAS?
  for(int i=0, iend=Gc.shape()[0]; i<iend; i++) 
    for(int n=0; n<nwalk; n++) 
      W_data[n][0] += Gc[i][n]*Gcloc[i][n];

  for(int n=0; n<nwalk; n++) W_data[n][0] *= half;
    
  //! one-body contribution
  ma::product(one,ma::T(Gc),haj_ref,one,W_data[indices[range_t(0,nwalk)][0]]);

}

}

}

#endif
