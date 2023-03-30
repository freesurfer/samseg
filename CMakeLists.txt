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


# -------- itk --------
find_package(ITK HINTS ${ITK_DIR} REQUIRED)




# enable std c++11
set(CMAKE_CXX_STANDARD 11)


# --------------------------------------------------
#                    setup python
# --------------------------------------------------

# initialize pybind for python wrapping
add_subdirectory(pybind11)


# --------------------------------------------------
#             build settings
# --------------------------------------------------

# warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-absolute-value -Wno-sign-compare -Wno-write-strings -Wno-unused-result -Wno-unused-parameter")

# clang complains about -Wno-unused-but-set-variable and says to use -Wno-unused-const-variable
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-const-variable -Wno-inconsistent-missing-override")
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

# --------------------------------------------------
#                 build samseg
# --------------------------------------------------


# build the gems library
add_subdirectory(gems)

#build python
add_subdirectory(samseg/cxx)
