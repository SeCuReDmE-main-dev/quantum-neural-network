find_path(XCFun_INCLUDE_DIR
    NAMES xcfun.h
    PATHS ${CMAKE_BINARY_DIR}/external/include
)

find_library(XCFun_LIBRARY
    NAMES xcfun
    PATHS ${CMAKE_BINARY_DIR}/external/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XCFun
    REQUIRED_VARS XCFun_LIBRARY XCFun_INCLUDE_DIR
)

if(XCFun_FOUND)
    set(XCFun_LIBRARIES ${XCFun_LIBRARY})
    set(XCFun_INCLUDE_DIRS ${XCFun_INCLUDE_DIR})
endif()
