cmake_minimum_required(VERSION 3.16)
project(imo-sfm VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

file(GLOB srcs  "*.h" "*.cpp" )

add_library(${PROJECT_NAME} ${srcs})
target_link_libraries(${PROJECT_NAME} PUBLIC mlib)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_configure_warnings(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

