PROJECT(miniapps)

if(QMC_BUILD_LEVEL GREATER 4)

ADD_EXECUTABLE(miniafqmc miniafqmc.cpp)
#TARGET_LINK_LIBRARIES(miniqmc miniafqmc qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
TARGET_LINK_LIBRARIES(miniafqmc qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()


