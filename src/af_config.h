#ifndef AFQMC_CONFIG_H 
#define AFQMC_CONFIG_H 

#include <string>
#include <algorithm>
#include<cstdlib>
#include<ctype.h>
#include <vector>
#include <map>
#include <complex>
#include <tuple>
#include <fstream>
#include "Configuration.h"

#include "Matrix/csr_matrix.hpp"
#include<boost/multi_array.hpp>

#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "multi/index_range.hpp"
#include "multi/ordering.hpp"
namespace multi = boost::multi;

#include "alf/boost/mpi3/shared_window.hpp"

#define AFQMC_DEBUG 3 
#define AFQMC_TIMER 

#define MAXIMUM_EMPLACE_BUFFER_SIZE 102400 

// maximum size in Bytes for a dataset with walker data on WalkerIO
#define WALKER_HDF_BLOCK_SIZE 100000000 

// maximum size in Bytes for a block of data in CSR matrix HDF IO 
#define CSR_HDF_BLOCK_SIZE 2000000 

#define PsiT_IN_SHM

#define USING_BOOST_MULTI

namespace qmcplusplus
{

  enum WALKER_TYPES {UNDEFINED_WALKER_TYPE, CLOSED, COLLINEAR, NONCOLLINEAR};

  enum SpinTypes {Alpha,Beta};  

  template<typename T>
    using shm_allocator = boost::mpi3::intranode::allocator<T>;

  // Matrix types
  template<typename T, std::size_t N, class Alloc = std::allocator<T>> 
    using MArray = multi::array<T, N, Alloc>;
  template<typename T, std::size_t N>
    using MArray_ref = multi::array_ref<T, N>;
  template<typename T, std::size_t N>
    using MArray_cref = multi::array_cref<T, N>;
  template<typename T, class Alloc = std::allocator<T>> 
    using Matrix = MArray<T,2,Alloc>; 
  template<typename T>
    using Matrix_ref = MArray_ref<T,2>;
  template<typename T, class Alloc = std::allocator<T>> 
    using Vector = MArray<T,1,Alloc>; 
  template<typename T>
    using Vector_ref = MArray_ref<T,1>;

  template<typename T, std::size_t N, class Alloc = shm_allocator<T>> 
    using SHM_MArray = multi::array<T, N, Alloc>;
  template<typename T, class Alloc = shm_allocator<T>>
    using SHM_Matrix = MArray<T,2,Alloc>;
  template<typename T, class Alloc = shm_allocator<T>>
    using SHM_Vector = MArray<T,1,Alloc>;

  // Accelerator structures
  template<typename T, std::size_t N, class Alloc = shm_allocator<T>>
    using device_MArray = multi::array<T, N, Alloc>;
  template<typename T, std::size_t N>
    using device_MArray_ref = multi::array_ref<T,N>;
  template<typename T, class Alloc = shm_allocator<T>>
    using device_Matrix = MArray<T,2,Alloc>;
  template<typename T>
    using device_Matrix_ref = MArray_ref<T,2>;
  template<typename T, class Alloc = shm_allocator<T>>
    using device_Vector = MArray<T,1,Alloc>;
  template<typename T>
    using device_Vector_ref = MArray_ref<T,1>;

  // new types
  using SpCType_shm_csr_matrix = ma::sparse::csr_matrix<SPComplexType,int,std::size_t,
                                boost::mpi3::intranode::allocator<SPComplexType>,
                                boost::mpi3::intranode::is_root>;
  using SpVType_shm_csr_matrix = ma::sparse::csr_matrix<SPValueType,int,std::size_t,
                                boost::mpi3::intranode::allocator<SPValueType>,
                                boost::mpi3::intranode::is_root>;
  using CType_shm_csr_matrix = ma::sparse::csr_matrix<ComplexType,int,std::size_t,
                                boost::mpi3::intranode::allocator<ComplexType>,
                                boost::mpi3::intranode::is_root>;
  using VType_shm_csr_matrix = ma::sparse::csr_matrix<ValueType,int,std::size_t,
                                boost::mpi3::intranode::allocator<ValueType>,
                                boost::mpi3::intranode::is_root>;

  using ComplexSmMat = CType_shm_csr_matrix;

#ifdef PsiT_IN_SHM
  using PsiT_Matrix = ma::sparse::csr_matrix<ComplexType,int,int,
                                boost::mpi3::intranode::allocator<ComplexType>,
                                boost::mpi3::intranode::is_root>;
#else
  using PsiT_Matrix = ma::sparse::csr_matrix<ComplexType,int,int>;
#endif


  using P1Type = ma::sparse::csr_matrix<ComplexType,int,int,
                                boost::mpi3::intranode::allocator<ComplexType>,
                                boost::mpi3::intranode::is_root>;

  enum HamiltonianTypes {Factorized,SymmetricFactorized,s4DInts,THC};

  extern TimerList_t AFQMCTimers;
  enum AFQMCTimerIDs {    
    block_timer,
    pseudo_energy_timer,
    energy_timer,
    vHS_timer,
    vbias_timer,
    G_for_vbias_timer,
    propagate_timer,
    E_comm_overhead_timer,
    vHS_comm_overhead_timer,
    StepPopControl
  };
  //extern TimerNameList_t<AFQMCTimerIDs> AFQMCTimerNames;  

  struct AFQMCInfo 
  {
    public:

    // default constructor
    AFQMCInfo():
        name(""),NMO(-1),NMO_FULL(-1),NAEA(-1),NAEB(-1),NCA(0),NCB(0),NETOT(-1),
        MS2(-99),spinRestricted(true),ISYM(-1)
    {
    }

    AFQMCInfo(std::string nm, int nmo_, int naea_, int naeb_):
        name(nm),NMO(nmo_),NMO_FULL(nmo_),NAEA(naea_),NAEB(naeb_),NCA(0),NCB(0),
        NETOT(-1),MS2(-99),spinRestricted(true),ISYM(-1)
    {
    }

    AFQMCInfo( const AFQMCInfo& other) = default;
    AFQMCInfo& operator=( const AFQMCInfo& other) = default;

    // destructor
    ~AFQMCInfo() {}

    // identifier
    std::string name;

    // number of orbitals
    int NMO_FULL;

    // number of active orbitals
    int NMO;

    // number of active electrons alpha/beta 
    int NAEA, NAEB;

    // number of core electrons alpha/beta
    int NCA,NCB;

    // total number of electrons
    int NETOT; 

    // ms2
    int MS2; 

    // isym
    int ISYM;  

    // if true then RHF calculation, otherwise it is UHF 
    bool spinRestricted;

    // copies values from object
    void copyInfo(const AFQMCInfo& a) {
      name = a.name;
      NMO_FULL=a.NMO_FULL;
      NMO=a.NMO;
      NAEA=a.NAEA;
      NAEB=a.NAEB;
      NCA=a.NCA;
      NCB=a.NCB;
      NETOT=a.NETOT;
      MS2=a.MS2;
      ISYM=a.ISYM;
      spinRestricted=a.spinRestricted;
    }

    // no fully spin polarized yet, not sure what it will break 
    bool checkAFQMCInfoState() {
      if(NMO_FULL<1 || NAEA<1 || NAEB<1 || NCA<0 || NCB<0 ) //|| NETOT!= NCA+NCB+NAEA+NAEB ) //|| MS2<0 )
        return false;
      return true; 
    } 

    void printAFQMCInfoState(std::ostream& out) {
      out<<"AFQMC info: \n"
         <<"name: " <<name <<"\n"
         <<"NMO_FULL: " <<NMO_FULL <<"\n"
         <<"NAEA: " <<NAEA  <<"\n"
         <<"NAEB: " <<NAEB  <<"\n"
         <<"NCA: " <<NCA  <<"\n"
         <<"NCB: " <<NCB  <<"\n"
         <<"NETOT: " <<NETOT  <<"\n"
         <<"MS2: " <<MS2  <<"\n"
         <<"spinRestricted: " <<spinRestricted <<std::endl;
    }

  };

}

#endif
