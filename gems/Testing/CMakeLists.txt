set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.59.0)

add_executable(kvlAtlasMeshRasterizorTestGPU kvlAtlasMeshRasterizorTestGPU.cxx)
target_link_libraries(kvlAtlasMeshRasterizorTestGPU PRIVATE kvlGEMSCommon)

add_executable(testCompressionLookupTable testCompressionLookupTable.cxx)
target_link_libraries(testCompressionLookupTable PRIVATE kvlGEMSCommon)

add_executable(testdeformMeshPython testdeformMeshPython.cxx)
target_link_libraries(testdeformMeshPython PRIVATE kvlGEMSCommon)

add_executable(testdeformMeshPython.dynmesh testdeformMeshPython.cxx)
target_compile_definitions(testdeformMeshPython.dynmesh PRIVATE USE_DYNAMIC_MESH)
target_link_libraries(testdeformMeshPython.dynmesh PRIVATE kvlGEMSCommon_dynmesh)

if(Boost_FOUND)
  set(GEMS2libs kvlGEMSCommon)
  set(testsrcs
    boosttests.cpp
    testatlasmeshrasterizorbasic.cpp
    atlasmeshvisitcountercpuwrapper.cpp
    atlasmeshalphadrawercpuwrapper.cpp
    testatlasmeshvisitcounter.cpp
    testatlasmeshalphadrawer.cpp
    teststopwatch.cpp
    imageutils.cpp
  )

  if(GEMS_BUILD_CUDA)
    list(APPEND testsrcs
      testcudaimage.cpp
      cudaglobalfixture.cpp
      cudaimagetests.cu
      testcudatetrahedralmesh.cpp
      testdimensioncuda.cpp
    )
    list(APPEND GEMS2libs kvlGEMSCUDA)
  endif()

  add_executable(TestGEMS2 ${testsrcs})
  target_link_libraries(TestGEMS2 PRIVATE ${Boost_LIBRARIES} ${GEMS2libs})
  target_include_directories(TestGEMS2 PRIVATE ${Boost_INCLUDE_DIRS})
endif()

configure_file(test.nii ${GEMS_RUNTIME_PATH}/test.nii COPYONLY)
configure_file(test.txt.gz ${GEMS_RUNTIME_PATH}/test.txt.gz COPYONLY)
configure_file(inp_image_deformmesh.mgz ${GEMS_RUNTIME_PATH}/inp_image_deformmesh.mgz COPYONLY)
configure_file(atlas_level1_deformmesh.txt.gz ${GEMS_RUNTIME_PATH}/atlas_level1_deformmesh.txt.gz COPYONLY)
