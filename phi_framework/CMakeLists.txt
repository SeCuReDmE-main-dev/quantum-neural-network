cmake_minimum_required(VERSION 3.10)
project(quantum_neural_phi C CXX)

# Add module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# Find dependencies
find_package(OpenBLAS REQUIRED)
find_package(XCFun)

# Configure XCFun if not found
if(NOT XCFun_FOUND)
    include(ExternalProject)
    ExternalProject_Add(libxcfun
        GIT_REPOSITORY https://github.com/dftlibs/xcfun.git
        GIT_TAG v2.1.1
        CMAKE_ARGS 
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external
            -DCMAKE_BUILD_TYPE=Release
            -DXCFUN_PYTHON_INTERFACE=OFF
    )
endif()

# Set compiler flags
if(MSVC)
    add_compile_options(/O2 /W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
