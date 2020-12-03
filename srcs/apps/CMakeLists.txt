cmake_minimum_required(VERSION 3.5)
file(GLOB   apps  "*.cpp")

# be specific to simplify...
set(apps "")
if(HAS_KLT)
    list(APPEND apps test_parameditor.cpp)
    list(APPEND apps daimler-hir.cpp)
    list(APPEND apps kitti-hir.cpp)
    if(WITH_MLIB_CUDA)
        list(APPEND apps mots-hir.cpp)
    endif()
endif()


list(APPEND apps annotation_tool_main.cpp)
list(APPEND apps show_result.cpp)










message("making binaries: ")
foreach(item ${apps})
    message("    ${item}")
    get_filename_component(name ${item} NAME_WE)
    option(build_${name} "compile and link this app" ON)
endforeach()



# add an option for each
foreach(item ${apps})
    get_filename_component(name ${item} NAME_WE)
    if(build_${name})
        add_executable(${name} ${item})
        target_compile_features(${name} PUBLIC cxx_std_17)
    endif()
endforeach()





foreach(item ${apps})
    get_filename_component(name ${item} NAME_WE)
    if(build_${name})
        message("linking: ${name} to dlib")

        target_link_libraries(${name} imo)
        WarningConfig()
        target_compile_options(${name} PRIVATE ${WARNINGS})
    endif()
endforeach()

