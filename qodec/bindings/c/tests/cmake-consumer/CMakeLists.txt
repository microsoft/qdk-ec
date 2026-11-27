# A downstream project that finds qodec only through its installed CMake
# package. Run by tools/check_cmake_package.py, or by hand with
# -DCMAKE_PREFIX_PATH=<qodec install prefix>.
cmake_minimum_required(VERSION 3.22)
project(qodec_cmake_consumer LANGUAGES C CXX)

find_package(qodec CONFIG REQUIRED)

get_target_property(qodec_type qodec::qodec TYPE)
if(qodec_type STREQUAL "STATIC_LIBRARY" AND NOT WIN32)
    # Linking the static library into a shared one checks that it is
    # position-independent.
    add_library(consumer_module SHARED consumer_module.c)
else()
    add_library(consumer_module STATIC consumer_module.c)
endif()
target_link_libraries(consumer_module PRIVATE qodec::qodec)

# The same source, compiled as C++, checks the header's C++ compatibility.
configure_file(consumer.c consumer.cpp COPYONLY)
add_executable(consumer_c consumer.c)
add_executable(consumer_cxx "${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp")
foreach(consumer IN ITEMS consumer_c consumer_cxx)
    target_include_directories(${consumer} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${consumer} PRIVATE consumer_module qodec::qodec)
    if(WIN32 AND qodec_type STREQUAL "SHARED_LIBRARY")
        add_custom_command(TARGET ${consumer} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_RUNTIME_DLLS:${consumer}>" "$<TARGET_FILE_DIR:${consumer}>"
            COMMAND_EXPAND_LISTS)
    endif()
endforeach()
if(MSVC)
    set(warning_options /W4 /WX)
else()
    set(warning_options -Wall -Wextra -Werror -pedantic)
endif()
foreach(consumer IN ITEMS consumer_module consumer_c consumer_cxx)
    target_compile_options(${consumer} PRIVATE ${warning_options})
endforeach()
set_target_properties(consumer_c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
set_target_properties(consumer_cxx PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
