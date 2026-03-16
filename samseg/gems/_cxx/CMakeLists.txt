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

target_include_directories(gemsbindings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ITK_INCLUDE_DIRS})
target_link_libraries(gemsbindings PRIVATE samseg_warnings kvlGEMSCommon ${ITK_LIBRARIES})

# pybind11 debug builds produce objects too large for the linker with -g
target_compile_options(gemsbindings PRIVATE
  $<$<CONFIG:Debug>:-g0>
  $<$<CONFIG:RelWithDebInfo>:-g0>
)

install(TARGETS gemsbindings LIBRARY DESTINATION samseg/gems)
