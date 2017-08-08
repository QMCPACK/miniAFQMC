PROJECT(miniapps)

if(QMC_BUILD_LEVEL GREATER 4)

ADD_EXECUTABLE(miniqmc miniqmc.cpp)
#TARGET_LINK_LIBRARIES(miniqmc miniafqmc qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
TARGET_LINK_LIBRARIES(miniqmc qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()


