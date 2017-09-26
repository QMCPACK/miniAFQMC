PROJECT(miniapps)

if(QMC_BUILD_LEVEL GREATER 4)

ADD_EXECUTABLE(miniafqmc miniafqmc_base.cpp)
TARGET_LINK_LIBRARIES(miniafqmc qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()


