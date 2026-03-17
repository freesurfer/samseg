cmake_minimum_required(VERSION 3.24)
project(samseg LANGUAGES C CXX)

include(FetchContent)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PROFILING "Enable profiling with -pg" OFF)

# Warning flags applied only to our own targets via INTERFACE library
# so that third-party dependencies (ITK, GDCM, etc.) are not affected.
add_library(samseg_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(samseg_warnings INTERFACE
    -Wall -Wextra
    -Wno-sign-compare -Wno-unused-parameter
  )
elseif(MSVC)
  target_compile_options(samseg_warnings INTERFACE /W3)
endif()

if(PROFILING AND NOT MSVC)
  target_compile_options(samseg_warnings INTERFACE -pg)
  target_link_options(samseg_warnings INTERFACE -pg)
endif()

# Binary size optimization flags, applied per-target via INTERFACE library
add_library(samseg_linkopt INTERFACE)
if(APPLE)
  target_link_options(samseg_linkopt INTERFACE -Wl,-dead_strip)
elseif(NOT WIN32)
  target_compile_options(samseg_linkopt INTERFACE -fdata-sections -ffunction-sections)
  target_link_options(samseg_linkopt INTERFACE -Wl,--gc-sections)
endif()

add_subdirectory(gems)
add_subdirectory(samseg/gems/_cxx)
