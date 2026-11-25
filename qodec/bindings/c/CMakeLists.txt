# Builds and installs the qodec C library with Cargo. Cargo remains the Rust
# build tool; this layer selects the linkage, configuration and target that
# CMake asks for, then installs the result as the `qodec` CMake package.
cmake_minimum_required(VERSION 3.22)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml" qodec_manifest_version REGEX "^version = ")
string(REGEX REPLACE "^version = \"([^\"]+)\"$" "\\1" qodec_version "${qodec_manifest_version}")
project(qodec VERSION "${qodec_version}" DESCRIPTION "C bindings for qodec" LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/QodecRust.cmake)

option(BUILD_SHARED_LIBS "Build qodec_c as a shared library" OFF)
if(BUILD_SHARED_LIBS)
    set(qodec_library_type SHARED)
else()
    set(qodec_library_type STATIC)
endif()

qodec_find_rust_toolchain()
qodec_rust_target(qodec_rust_target)
qodec_cargo_library(qodec_c_cargo TARGET "${qodec_rust_target}" OUTPUT_DIRECTORY qodec_output_directory)

set(qodec_cmake_install_dir "${CMAKE_INSTALL_DATADIR}/qodec")
install(FILES include/qodec.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
qodec_install_library("${qodec_output_directory}")

file(RELATIVE_PATH qodec_config_to_prefix "/prefix/${qodec_cmake_install_dir}" "/prefix")
string(REGEX REPLACE "/$" "" qodec_config_to_prefix "${qodec_config_to_prefix}")
configure_file(cmake/qodecTargets.cmake.in qodecTargets.cmake @ONLY)
# qodec is pre-1.0: a minor release may change the C ABI, so only patch
# releases are compatible.
write_basic_package_version_file(qodecConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES
        cmake/qodecConfig.cmake
        "${CMAKE_CURRENT_BINARY_DIR}/qodecConfigVersion.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/qodecTargets.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/qodecTargets-$<LOWER_CASE:$<CONFIG>>.cmake"
    DESTINATION "${qodec_cmake_install_dir}")
