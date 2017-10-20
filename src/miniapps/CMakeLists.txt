PROJECT(miniapps)

if(QMC_BUILD_LEVEL GREATER 4)

ADD_EXECUTABLE(miniafqmc_shm miniafqmc_shm.cpp)
TARGET_LINK_LIBRARIES(miniafqmc_shm qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_EXECUTABLE(miniafqmc_shm2 miniafqmc_shm2.cpp)
TARGET_LINK_LIBRARIES(miniafqmc_shm2 qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_EXECUTABLE(miniafqmc_shm3 miniafqmc_shm3.cpp)
TARGET_LINK_LIBRARIES(miniafqmc_shm3 qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

#ADD_EXECUTABLE(miniafqmc_shm4 miniafqmc_shm4.cpp)
#TARGET_LINK_LIBRARIES(miniafqmc_shm4 qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

#ADD_EXECUTABLE(miniafqmc_shm5 miniafqmc_shm5.cpp)
#TARGET_LINK_LIBRARIES(miniafqmc_shm5 qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_EXECUTABLE(benchmark benchmark.cpp)
TARGET_LINK_LIBRARIES(benchmark qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

#(PETSC_MAT_LIB /usr/tce/packages/petsc/petsc-3.7.2-mvapich2-2.2-gcc-4.8-redhat/include/)
ADD_EXECUTABLE(benchmark_petsc benchmark_petsc.cpp)
TARGET_LINK_LIBRARIES(benchmark_petsc qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()


