PROJECT(miniapps)

if(QMC_BUILD_LEVEL GREATER 4)

ADD_EXECUTABLE(miniafqmc_shm miniafqmc_shm.cpp)
TARGET_LINK_LIBRARIES(miniafqmc_shm qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()


