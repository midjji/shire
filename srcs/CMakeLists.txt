CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(gpl-shire CXX CUDA) # defines ${CMAKE_PROJECT_NAME}(top level) PROJECT_NAME(most recent)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "
                         run in terminal and rerun: export CC=/usr/bin/clang;export CXX=/usr/bin/clang++;
 you must use clang to compile this, mostly because of the sqlitedatabase interface. ")
endif()

add_subdirectory(mlib)
add_library(imo INTERFACE)
target_link_libraries(imo INTERFACE mlib)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/clemens-ip/CMakeLists.txt")
    add_subdirectory(clemens-ip)
    set(HAS_KLT ON)    
    add_subdirectory(utils)
    add_subdirectory(shire) # depend on utils

    target_link_libraries(imo INTERFACE imo-shire)
else()
    set(HAS_KLT OFF)
    message(WARNING "the klt tracker and parameter gui used is unfortunately proprietary.
        Only building the annotation tool and show result tool, however the rest of the code is GPL so feel free to plugin your own replacements.  ")
endif()

add_subdirectory(sfm)
target_link_libraries(imo INTERFACE imo-sfm)





add_subdirectory(apps)








