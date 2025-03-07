set(OpenBLAS_ROOT $ENV{CONDA_PREFIX}/Library)

find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATHS ${OpenBLAS_ROOT}/include
)

find_library(OpenBLAS_LIBRARY
    NAMES openblas
    PATHS ${OpenBLAS_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
)

if(OpenBLAS_FOUND)
    set(BLAS_LIBRARIES ${OpenBLAS_LIBRARY})
    set(BLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
endif()
