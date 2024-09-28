# project info
project(fs_python)
cmake_minimum_required(VERSION 3.9)
enable_language(C CXX)


# prevent third-party packages from importing as a system, OP: no clue what this is
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)


# set the default build type to 'Release' for optimization purposes
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()


# --------------------------------------------------
#               external dependencies
# --------------------------------------------------
set(pybind11_DIR $ENV{pybind11_DIR})
if(NOT pybind11_DIR)
  set(pydbind11_DIR "pybind11")
endif()

# -------- itk --------
find_package(ITK HINTS ${ITK_DIR} REQUIRED)




# enable std c++11
set(CMAKE_CXX_STANDARD 11)


# --------------------------------------------------
#                    setup python
# --------------------------------------------------

# initialize pybind for python wrapping
add_subdirectory(${pybind11_DIR} pybind11)
message(STATUS "ITK_DIR=${ITK_DIR} pybind11_DIR=${pybind11_DIR}")


# --------------------------------------------------
#             build settings
# --------------------------------------------------
# set -DAPPLE=ON and -DAPPLE_ARM64=ON on cmake command line for arm64 build
# when building with freesurfer, CMAKE_C_COMPILER and CMAKE_CXX_COMPILER will be specified on the cmake command line
if(DEFINED ENV{APPLE})
  MESSAGE(STATUS "ARCH VARIABLE SET TO --[$ENV{APPLE_ARM64}]--")
  if($ENV{APPLE_ARM64} MATCHES "ON")
    MESSAGE(STATUS "SETTING TARGET TO ARM64")
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    add_definitions(-DARM64 -DDarwin -DPNG_ARM_NEON_OPT=0)
  else()
    MESSAGE(STATUS "SETTING TARGET TO X86_64")
    set(CMAKE_OSX_ARCHITECTURES "x86_64")
  endif()
endif()

message(STATUS "CMAKE_C_COMPILER=${CMAKE_C_COMPILER}, CMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID}, CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}, CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")

# warnings
if(NOT DEFINED ENV{APPLE_ARM64} OR $ENV{APPLE_ARM64} MATCHES "OFF")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-absolute-value -Wno-sign-compare -Wno-write-strings -Wno-unused-result -Wno-unused-parameter")
endif()

# clang complains about -Wno-unused-but-set-variable and says to use -Wno-unused-const-variable
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-c-typedef-for-linkage -Wno-unused-const-variable -Wno-inconsistent-missing-override -Wno-self-assign-field -Wno-tautological-overlap-compare -Wno-tautological-compare -Wno-unused-value -Wno-range-loop-analysis -Wno-return-stack-address -Wno-dangling-gsl")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# linker options
if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -dead_strip")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc")
  endif()
else()
  set(STRIP_FLAGS "-fdata-sections -ffunction-sections -Wl,--gc-sections")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STRIP_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${STRIP_FLAGS} -Wl,-Map,ld_map.txt -Wl,--no-demangle")
endif()

if(PROFILING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

message(STATUS "C/C++ standard set to ${CMAKE_CXX_STANDARD}, CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
message(STATUS "For HOST_OS=${HOST_OS} CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "For HOST_OS=${HOST_OS} CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

# --------------------------------------------------
#                 build samseg
# --------------------------------------------------


# build the gems library
add_subdirectory(gems)

#build python
add_subdirectory(samseg/cxx)
