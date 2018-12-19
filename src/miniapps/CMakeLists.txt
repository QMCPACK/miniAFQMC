PROJECT(miniapps)

if(QMC_BUILD_LEVEL GREATER 4)

ADD_EXECUTABLE(blas_test blas_test.cpp)
TARGET_LINK_LIBRARIES(blas_test qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_EXECUTABLE(miniafqmc_cuda_gpu miniafqmc_cuda_gpu.cpp)
TARGET_LINK_LIBRARIES(miniafqmc_cuda_gpu qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_EXECUTABLE(miniafqmc_cuda_gpu_batched miniafqmc_cuda_gpu_batched.cpp)
TARGET_LINK_LIBRARIES(miniafqmc_cuda_gpu_batched qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()


