cmake_minimum_required(VERSION 3.16)
project(imo-shire VERSION 1.0
    DESCRIPTION "Very nice project"
    LANGUAGES CXX)

file(GLOB srcs  "*.h" "*.cpp" )

add_library(${PROJECT_NAME} ${srcs})
target_link_libraries(${PROJECT_NAME} PUBLIC mlib clemens-ip imo-utils imo-sfm)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_configure_warnings(${PROJECT_NAME})



