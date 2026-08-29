set(PYBIND11_FINDPYTHON ON)

FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS QUIET
)
FetchContent_MakeAvailable(pybind11)

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

target_include_directories(gemsbindings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemsbindings PRIVATE samseg_warnings samseg_linkopt kvlGEMSCommon)

# pybind11 debug builds produce objects too large for the linker with -g
if(NOT MSVC)
  target_compile_options(gemsbindings PRIVATE
    $<$<CONFIG:Debug>:-g0>
    $<$<CONFIG:RelWithDebInfo>:-g0>
  )
endif()

install(TARGETS gemsbindings LIBRARY DESTINATION samseg/gems)
