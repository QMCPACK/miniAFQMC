#!/bin/tcsh
echo Configuring and building froom inside the build directory.
echo Check the results of the CMake configuration to ensure that the preferred
echo compilers and libraries have been selected. See README and documentation 
echo for guidance.

setenv CXX mpicxx
setenv CC mpicc
setenv BOOST_ROOT /usr/gapps/qmc/libs/QUARTZ_GCC/boost_1_66_0

module load mkl
source /usr/tce/packages/mkl/mkl-2018.0/mkl/bin/mklvars.csh intel64


# MAM:
# Careful with threaded MKL. Only seems to work on LLNL with Intel compilers.
# Can't seem to get -DBLA_VENDOR=Intel10_64lp working in LLNL.

# Debug with 
cmake  \
      -DQMC_EXTRA_LIBS="-L/usr/gapps/qmc/libs/QUARTZ_GCC/boost_1_66_0/lib -lboost_serialization -Wl,-rpath=/usr/gapps/qmc/libs/QUARTZ_GCC/boost_1_66_0/lib " \
      -DLAPACK_LINKER_FLAGS="-Wl,-rpath=/usr/tce/packages/mkl/mkl-2018.0/lib/" \
      -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_PREFIX_PATH=$MKLROOT/lib \
      -DCMAKE_BUILD_TYPE=Release -DENABLE_TIMERS=1 \
      -DBUILD_AFQMC=1 -DQMC_COMPLEX=1 ../../; 
make

