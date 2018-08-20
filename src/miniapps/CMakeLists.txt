PROJECT(miniapps)

if(QMC_BUILD_LEVEL GREATER 4)

ADD_EXECUTABLE(miniafqmc_base miniafqmc_base.cpp)
TARGET_LINK_LIBRARIES(miniafqmc_base qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

#ADD_EXECUTABLE(miniafqmc_cuda_managed miniafqmc_cuda_managed.cpp) 
#TARGET_LINK_LIBRARIES(miniafqmc_cuda_managed qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()


