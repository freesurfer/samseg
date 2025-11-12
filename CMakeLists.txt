# project info
project(fs_python)
cmake_minimum_required(VERSION 3.9)
enable_language(C CXX)

# Prevent warnings from headers in third-party libraries
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

# Set the default build type to 'Release' if not specified by the user
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

# --------------------------------------------------
#               External Dependencies
# --------------------------------------------------

# --- pybind11 ---
set(pybind11_DIR "$ENV{pybind11_DIR}")
if(NOT pybind11_DIR)
  set(pybind11_DIR "pybind11")
endif()
add_subdirectory(${pybind11_DIR} pybind11)

# --- ITK ---
add_subdirectory(ITK)
set(ITK_DIR ${ITK_BINARY_DIR})
find_package(ITK REQUIRED)

message(STATUS "Using ITK from: ${ITK_DIR}")
message(STATUS "Using pybind11 from: ${pybind11_DIR}")

# --------------------------------------------------
#                Core Build Settings
# --------------------------------------------------

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# All Python modules must be compiled with Position Independent Code (PIC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# --------------------------------------------------
#     Platform-Specific Settings (Native CMake)
# --------------------------------------------------

# Check if we are building for an Apple OS (macOS, iOS)
# APPLE is a built-in variable, no need to set it with -DAPPLE=ON
if(APPLE)
  message(STATUS "Apple platform detected.")

  # Set Apple-specific linker flags
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -dead_strip")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc")
  endif()

  # NATIVE ARCHITECTURE DETECTION:
  # If the user has NOT specified an architecture, default to the host processor.
  # The user can override this by setting -DCMAKE_OSX_ARCHITECTURES="x86_64" or "arm64"
  if(NOT CMAKE_OSX_ARCHITECTURES)
    message(STATUS "CMAKE_OSX_ARCHITECTURES not set, defaulting to host: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
    set(CMAKE_OSX_ARCHITECTURES "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  endif()

  # Apply arm64-specific definitions ONLY if we are targeting arm64
  # This respects the CMAKE_OSX_ARCHITECTURES variable, not the host processor.
  if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    message(STATUS "Targeting Apple arm64 architecture.")
    add_definitions(-DARM64 -DPNG_ARM_NEON_OPT=0)
  else()
    message(STATUS "Targeting Apple x86_64 architecture.")
  endif()

else()
  # --- Settings for non-Apple platforms (Linux, etc.) ---
  message(STATUS "Non-Apple platform detected.")

  # Set compiler warnings (the original file disabled these for Apple)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -Wno-write-strings -Wno-unused-result -Wno-unused-parameter")

  # Set linker flags for code stripping
  set(STRIP_FLAGS "-fdata-sections -ffunction-sections -Wl,--gc-sections")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STRIP_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${STRIP_FLAGS} -Wl,--no-demangle")
endif()

# --- Compiler-specific warnings (for all platforms) ---
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-c-typedef-for-linkage -Wno-unused-const-variable -Wno-inconsistent-missing-override -Wno-self-assign-field -Wno-tautological-overlap-compare -Wno-tautological-compare -Wno-unused-value -Wno-range-loop-analysis -Wno-return-stack-address -Wno-dangling-gsl")
else()
  # Assume GCC-like
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
endif()

# --- Profiling Option ---
# Enable with -DPROFILING=ON
if(PROFILING)
  message(STATUS "Profiling enabled (-pg)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

message(STATUS "C++ standard set to: ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")

# --------------------------------------------------
#                 Build Targets
# --------------------------------------------------

# Build the gems library
add_subdirectory(gems)

# Build the samseg python module
add_subdirectory(samseg/cxx)
