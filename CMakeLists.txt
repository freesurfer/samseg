cmake_minimum_required(VERSION 3.15)
project(samseg LANGUAGES C CXX)

include(FetchContent)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, fetching from source...")
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        stable
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endif()

find_package(ITK QUIET)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warning flags applied only to our own targets via INTERFACE library
# so that third-party dependencies (ITK, GDCM, etc.) are not affected.
add_library(samseg_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(samseg_warnings INTERFACE
    -Wall -Wextra
    -Wno-sign-compare -Wno-unused-parameter
  )
endif()

if(APPLE)
  add_link_options(-dead_strip)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_link_options(-static-libgcc)
  endif()
elseif(NOT WIN32)
  add_compile_options(-fdata-sections -ffunction-sections)
  add_link_options(-Wl,--gc-sections)
endif()

if(PROFILING)
  add_compile_options(-pg)
  add_link_options(-pg)
endif()

add_subdirectory(gems)
add_subdirectory(samseg/gems/_cxx)
