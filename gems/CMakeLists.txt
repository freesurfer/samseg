# ITK: try local install first, fetch from source if unavailable.
# FIND_PACKAGE_ARGS is not used here because ITK 5.4.x requires
# include(${ITK_USE_FILE}) for factory registration, which is only
# set by find_package — not by FetchContent's add_subdirectory path.
find_package(ITK 5.4.5 QUIET)
if(NOT ITK_FOUND)
  message(STATUS "ITK not found, fetching from source (v5.4.5)...")
  FetchContent_Declare(ITK
    GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
    GIT_TAG        v5.4.5
    GIT_SHALLOW    TRUE
  )
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(ITK)
  set(ITK_DIR "${itk_BINARY_DIR}" CACHE PATH "" FORCE)
  find_package(ITK REQUIRED)
endif()

# UseITK.cmake handles factory registration and required compiler flags.
# Still required with ITK 5.4.x which does not export namespace targets.
include(${ITK_USE_FILE})

find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  # ITK exports its bundled zlib-ng as a target named "zlib"
  if(TARGET zlib)
    set(ZLIB_LIBRARIES zlib)
    message(STATUS "Using ITK's bundled zlib")
  else()
    message(FATAL_ERROR "ZLIB not found and ITK's bundled zlib is unavailable")
  endif()
endif()

option(GEMS_BUILD_SHARED_LIBS "Build GEMS with shared libraries" OFF)
option(GEMS_BUILD_EXECUTABLES "Build command line executables" OFF)
option(GEMS_BUILD_TESTING "Build tests" OFF)
option(GEMS_BUILD_GUI "Build GUI components (requires FLTK and VTK)" OFF)
option(GEMS_BUILD_MATLAB "Build Matlab wrappers" OFF)
option(GEMS_BUILD_CUDA "Build CUDA support" OFF)
option(GEMS_CROSS_THREAD_REPRODUCIBLE "Enable complete reproducibility" OFF)
option(GEMS_USE_STATIC_ARRAY "Use array in TetrahedronInteriorConstIterator class" ON)

set(GEMS_RUNTIME_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${GEMS_RUNTIME_PATH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${GEMS_RUNTIME_PATH})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GEMS_RUNTIME_PATH})

if(GEMS_BUILD_CUDA)
  enable_language(CUDA)
  add_subdirectory(cuda)
endif()

set(SOURCES
  gzstream.C
  itkMGHImageIO.cxx
  itkMGHImageIOFactory.cxx
  kvlAtlasMeshAlphaDrawer.cxx
  kvlAtlasMeshCollection.cxx
  kvlAtlasMeshCollectionValidator.cxx
  kvlAtlasMeshDeformationConjugateGradientOptimizer.cxx
  kvlAtlasMeshDeformationFixedStepGradientDescentOptimizer.cxx
  kvlAtlasMeshDeformationGradientDescentOptimizer.cxx
  kvlAtlasMeshDeformationLBFGSOptimizer.cxx
  kvlAtlasMeshDeformationOptimizer.cxx
  kvlAtlasMeshLabelImageStatisticsCollector.cxx
  kvlAtlasMeshMultiAlphaDrawer.cxx
  kvlAtlasMeshPositionCostAndGradientCalculator.cxx
  kvlAtlasMeshProbabilityImageStatisticsCollector.cxx
  kvlAtlasMeshRasterizor.cxx
  kvlAtlasMeshSmoother.cxx
  kvlAtlasMeshStatisticsCollector.cxx
  kvlAtlasMeshSummaryDrawer.cxx
  kvlAtlasMeshToIntensityImageCostAndGradientCalculatorBase.cxx
  kvlAtlasMeshToIntensityImageCostAndGradientCalculator.cxx
  kvlAtlasMeshToIntensityImageLogDomainCostAndGradientCalculator.cxx
  kvlAtlasMeshToWishartGaussMixtureCostAndGradientCalculator.cxx
  kvlAtlasMeshToFrobeniusGaussMixtureCostAndGradientCalculator.cxx
  kvlAtlasMeshToDSWbetaGaussMixtureCostAndGradientCalculator.cxx
  kvlAtlasMeshToLabelImageCostAndGradientCalculator.cxx
  kvlAtlasMeshToPointSetCostAndGradientCalculator.cxx
  kvlAtlasMeshVisitCounter.cxx
  kvlAtlasMeshValueDrawer.cxx
  kvlAtlasParameterEstimator.cxx
  kvlAverageAtlasMeshPositionCostAndGradientCalculator.cxx
  kvlCompressionLookupTable.cxx
  kvlConditionalGaussianEntropyCostAndGradientCalculator.cxx
  kvlCroppedImageReader.cxx
  kvlHistogrammer.cxx
  kvlMutualInformationCostAndGradientCalculator.cxx
  kvlAtlasMeshJacobianDeterminantDrawer.cxx
)

function(configure_gems_target target)
  target_include_directories(${target} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces
    ${ITK_INCLUDE_DIRS}
  )
  target_link_libraries(${target}
    PRIVATE samseg_warnings samseg_linkopt ${ZLIB_LIBRARIES}
    PUBLIC ${ITK_LIBRARIES}
  )

  if(DEFINED ENV{GEMS_DEBUG_CXXFLAG})
    message(WARNING "additional gems debug cxxflags: $ENV{GEMS_DEBUG_CXXFLAG}")
    target_compile_options(${target} PRIVATE $ENV{GEMS_DEBUG_CXXFLAG})
  endif()

  if(GEMS_USE_STATIC_ARRAY)
    target_compile_definitions(${target} PRIVATE USING_STATIC_ARRAY MAX_LOADINGS=100)
  endif()

  if(GEMS_BUILD_CUDA)
    target_compile_definitions(${target} PRIVATE CUDA_FOUND)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
    target_link_libraries(${target} PRIVATE kvlGEMSCUDA)
  endif()

  if(GEMS_CROSS_THREAD_REPRODUCIBLE)
    target_compile_definitions(${target} PRIVATE CROSS_THREAD_REPRODUCIBLE)
  endif()
endfunction()

add_library(kvlGEMSCommon ${SOURCES})
configure_gems_target(kvlGEMSCommon)

# Dynamic mesh variant is slower but required by kvlBuildAtlasMesh
add_library(kvlGEMSCommon_dynmesh ${SOURCES})
target_compile_definitions(kvlGEMSCommon_dynmesh PRIVATE USE_DYNAMIC_MESH)
configure_gems_target(kvlGEMSCommon_dynmesh)

if(GEMS_BUILD_EXECUTABLES)
  add_subdirectory(Executables)
endif()

if(GEMS_BUILD_TESTING)
  add_subdirectory(Testing)
endif()

if(GEMS_BUILD_GUI)
  add_subdirectory(GUI)
endif()

if(GEMS_BUILD_MATLAB)
  add_subdirectory(Matlab)
endif()
