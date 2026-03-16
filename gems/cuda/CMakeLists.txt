add_library(kvlGEMSCUDA STATIC
  cudacontroller.cpp
  atlasmeshvisitcountercuda.cpp
  visitcountersimplecudaimpl.cu
  visitcountertetrahedralmeshcudaimpl.cu
  atlasmeshalphadrawercuda.cpp
  atlasmeshalphadrawercudaimpl.cu
)
target_include_directories(kvlGEMSCUDA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
