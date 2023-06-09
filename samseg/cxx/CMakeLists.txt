include_directories(${CMAKE_SOURCE_DIR}/gems ${CMAKE_CURRENT_SOURCE_DIR})
include(${ITK_USE_FILE})

if(NOT APPLE_ARM64) 
   set(CMAKE_CXX_FLAGS "-fPIC -fpermissive -msse2 -mfpmath=sse")
endif()

# temporary fix so that -g doesn't produce linker errors when binding cxx/python code
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")

# SC 2023/04/04: Commented out as it causes problem with Windows build
# add_compile_options(-Wno-inconsistent-missing-override -Wno-self-assign-field)

pybind11_add_module(gemsbindings
  module.cxx
  pyKvlCalculator.cxx
  pyKvlImage.cxx
  pyKvlMesh.cxx
  pyKvlOptimizer.cxx
  pyKvlTransform.cxx
  pyKvlRigidRegistration.cxx
  pyKvlAffineRegistration.cxx
)

# link utilities
target_link_libraries(gemsbindings PRIVATE kvlGEMSCommon)

# make sure the bindings library gets built into the repository even in out-of-source builds
set_target_properties(gemsbindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/samseg/gems)
