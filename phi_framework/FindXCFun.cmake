find_path(XCFun_INCLUDE_DIR
    NAMES xcfun.h
    PATHS 
        ${CMAKE_BINARY_DIR}/external/include
        ${CMAKE_INSTALL_PREFIX}/include
        $ENV{XCFUN_ROOT}/include
        $ENV{PROGRAMFILES}/XCFun/include
        $ENV{PROGRAMFILES\(X86\)}/XCFun/include
        $ENV{USERPROFILE}/AppData/Local/Programs/XCFun/include
    PATH_SUFFIXES xcfun
)

find_library(XCFun_LIBRARY
    NAMES xcfun libxcfun
    PATHS 
        ${CMAKE_BINARY_DIR}/external/lib
        ${CMAKE_INSTALL_PREFIX}/lib
        $ENV{XCFUN_ROOT}/lib
        $ENV{PROGRAMFILES}/XCFun/lib
        $ENV{PROGRAMFILES\(X86\)}/XCFun/lib
        $ENV{USERPROFILE}/AppData/Local/Programs/XCFun/lib
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XCFun
    REQUIRED_VARS XCFun_LIBRARY XCFun_INCLUDE_DIR
)

if(XCFun_FOUND)
    set(XCFun_LIBRARIES ${XCFun_LIBRARY})
    set(XCFun_INCLUDE_DIRS ${XCFun_INCLUDE_DIR})
    
    if(NOT TARGET XCFun::XCFun)
        add_library(XCFun::XCFun UNKNOWN IMPORTED)
        set_target_properties(XCFun::XCFun PROPERTIES
            IMPORTED_LOCATION "${XCFun_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${XCFun_INCLUDE_DIR}"
        )
    endif()
endif()
