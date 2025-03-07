# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src/libxcfun")
  file(MAKE_DIRECTORY "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src/libxcfun")
endif()
file(MAKE_DIRECTORY
  "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src/libxcfun-build"
  "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix"
  "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/tmp"
  "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src/libxcfun-stamp"
  "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src"
  "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src/libxcfun-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src/libxcfun-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/server-database-prefab/quantum-neural-network/phi_framework/build/libxcfun-prefix/src/libxcfun-stamp${cfgdir}") # cfgdir has leading slash
endif()
