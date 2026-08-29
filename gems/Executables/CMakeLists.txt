option(GEMS_MAKE_SPARSE_INITIAL_MESHES "Make sparse initial meshes" ON)

if(GEMS_MAKE_SPARSE_INITIAL_MESHES)
  message(WARNING "Since you're opting to make sparse (i.e., irregular) initial meshes, the TetGen library will be included, which has a restrictive license.")
  if(NOT Tetgen_FOUND)
    find_package(Tetgen REQUIRED)
  endif()
endif()

set(ATLAS_BUILDER_SOURCES
  kvlAtlasMeshBuilder.cxx
  kvlAtlasMeshCollectionFastReferencePositionCost.cxx
  kvlAtlasMeshCollectionModelLikelihoodCalculator.cxx
  kvlMultiResolutionAtlasMesher.cxx
)

add_executable(kvlBuildAtlasMesh kvlBuildAtlasMesh.cxx ${ATLAS_BUILDER_SOURCES})
add_executable(kvlBuildAtlasMesh.tetgen kvlBuildAtlasMesh.cxx ${ATLAS_BUILDER_SOURCES})
add_executable(samseg-atlas samseg-atlas.cxx ${ATLAS_BUILDER_SOURCES})
add_executable(samseg-atlas.tetgen samseg-atlas.cxx ${ATLAS_BUILDER_SOURCES})
add_help(samseg-atlas samseg-atlas.help.xml)

set(ALL_EXECUTABLES kvlBuildAtlasMesh kvlBuildAtlasMesh.tetgen samseg-atlas samseg-atlas.tetgen)
foreach(exe IN LISTS ALL_EXECUTABLES)
  target_compile_definitions(${exe} PRIVATE USE_DYNAMIC_MESH)
  target_link_libraries(${exe} PRIVATE kvlGEMSCommon_dynmesh utils)
endforeach()

if(GEMS_MAKE_SPARSE_INITIAL_MESHES)
  target_compile_definitions(kvlBuildAtlasMesh.tetgen PRIVATE USE_TETGEN)
  target_include_directories(kvlBuildAtlasMesh.tetgen PRIVATE ${Tetgen_INCLUDE_DIR})
  target_link_libraries(kvlBuildAtlasMesh.tetgen PRIVATE ${Tetgen_LIBRARIES})

  target_compile_definitions(samseg-atlas.tetgen PRIVATE USE_TETGEN)
  target_include_directories(samseg-atlas.tetgen PRIVATE ${Tetgen_INCLUDE_DIR})
  target_link_libraries(samseg-atlas.tetgen PRIVATE ${Tetgen_LIBRARIES})
endif()
