set(CMAKE_C_COMPILER mpi3clang)
set(CMAKE_CXX_COMPILER mpi3clang++)

set(GNU_OPTS "-O3 -g -ffast-math -fopenmp -fstrict-aliasing -Wno-deprecated -Wno-unused-value -Wno-type-safety -Wno-undefined-var-template")
set(GNU_FLAGS "-Drestrict=__restrict -DADD_ -D__forceinline=inline -D__bgq__")
set(CMAKE_CXX_FLAGS "${GNU_FLAGS} ${GNU_OPTS} -ftemplate-depth-120 -stdlib=libc++")
set(CMAKE_C_FLAGS "${GNU_FLAGS} ${GNU_OPTS} -std=c99" )
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition")

SET(QMC_CUDA 0)
SET(QMC_COMPLEX 1)
SET(ENABLE_OPENMP 1)
SET(HAVE_MPI 1)
set(HAVE_CUDA 0)
SET(QMC_BUILD_STATIC 1)
SET(HAVE_LIBESSL 1)
SET(HAVE_EINSPLINE 0)
SET(HAVE_EINSPLINE_EXT 0)
SET(HAVE_ADIOS 0)
SET(BUILD_QMCTOOLS 0)
SET(BUILD_AFQMC 1)
SET(ENABLE_TIMERS 1)
SET(BGQ 1)

#SET(MPIEXEC "sh")
#SET(MPIEXEC_NUMPROC_FLAG "${qmcpack_SOURCE_DIR}/utils/bgrunjobhelper.sh")
#SET(QE_BIN /soft/applications/quantum_espresso/5.3.0-bgq-omp/bin)

SET(BOOST_ROOT /usr/gapps/qmc/libs/VULCAN_GCC/boost_1_66_0) 
set(CMAKE_FIND_ROOT_PATH
    /usr/local/tools/hdf5/hdf5-1.8.5/serial
    /usr/gapps/qmc/libs/BGQ/libxml2-2.7.4/
    /usr/local/tools/zlib-1.2.6/
)

#include_directories($ENV{IBM_MAIN_DIR}/xlmass/bg/7.3/include)

set(LAPACK_LIBRARY /usr/local/tools/lapack/lib/liblapack.a) 
set(BLAS_LIBRARY /usr/local/tools/essl/5.1/lib/libesslbg.a)
set(FORTRAN_LIBRARIES 
/usr/local/tools/hdf5/hdf5-1.8.5/serial/lib/libhdf5.a
/opt/ibmcmp/xlmass/bg/7.3/bglib64/libmass.a 
/opt/ibmcmp/xlmass/bg/7.3/bglib64/libmassv.a 
/opt/ibmcmp/xlf/bg/14.1/bglib64/libxlf90_r.a
/opt/ibmcmp/xlf/bg/14.1/bglib64/libxlopt.a
/opt/ibmcmp/xlf/bg/14.1/bglib64/libxl.a
)

set(HDF5_LIBRARIES /usr/local/tools/hdf5/hdf5-1.8.5/serial/lib/libhdf5.a)
set(HDF5_INCLUDE_DIR /usr/local/tools/hdf5/hdf5-1.8.5/serial/include)
set(HDF5_INCLUDE_DIRS /usr/local/tools/hdf5/hdf5-1.8.5/serial/include)

FOREACH(type SHARED_LIBRARY SHARED_MODULE EXE)
  SET(CMAKE_${type}_LINK_STATIC_C_FLAGS "-Wl,-Bstatic")
  SET(CMAKE_${type}_LINK_DYNAMIC_C_FLAGS "-Wl,-Bstatic")
  SET(CMAKE_${type}_LINK_STATIC_CXX_FLAGS "-Wl,-Bstatic")
  SET(CMAKE_${type}_LINK_DYNAMIC_CXX_FLAGS "-Wl,-Bstatic")
ENDFOREACH(type)


