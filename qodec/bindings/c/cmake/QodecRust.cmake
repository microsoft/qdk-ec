# Cargo integration for the qodec C library.

# Let chains, used by the qodec core, need Rust 1.88 with edition 2024.
set(QODEC_RUST_MINIMUM_VERSION 1.88)

function(qodec_find_rust_toolchain)
    set(cargo_bin_hints "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" "$ENV{USERPROFILE}/.cargo/bin")
    find_program(QODEC_CARGO cargo HINTS ${cargo_bin_hints} DOC "Cargo used to build the qodec C library")
    find_program(QODEC_RUSTC rustc HINTS ${cargo_bin_hints} DOC "rustc used by Cargo for the qodec C library")
    if(NOT QODEC_CARGO OR NOT QODEC_RUSTC)
        message(FATAL_ERROR
            "qodec needs Rust ${QODEC_RUST_MINIMUM_VERSION} or later with Cargo. "
            "Install it from https://rustup.rs, or set QODEC_CARGO and QODEC_RUSTC.")
    endif()
    execute_process(COMMAND "${QODEC_RUSTC}" -vV
        OUTPUT_VARIABLE rustc_description ERROR_VARIABLE rustc_error RESULT_VARIABLE rustc_result)
    if(rustc_result)
        message(FATAL_ERROR "Cannot run ${QODEC_RUSTC}: ${rustc_error}")
    endif()
    string(REGEX MATCH "release: ([0-9]+\\.[0-9]+\\.[0-9]+)" _ "${rustc_description}")
    set(rustc_version "${CMAKE_MATCH_1}")
    if(rustc_version VERSION_LESS QODEC_RUST_MINIMUM_VERSION)
        message(FATAL_ERROR
            "qodec needs Rust ${QODEC_RUST_MINIMUM_VERSION} or later; ${QODEC_RUSTC} is ${rustc_version}.")
    endif()
    message(STATUS "qodec: using rustc ${rustc_version} (${QODEC_RUSTC})")
endfunction()

# Selects the Rust target that matches the CMake toolchain, so Cargo never
# silently builds for the host instead. Only the qualified native x64 targets
# are inferred; set QODEC_RUST_TARGET to build any other target.
function(qodec_rust_target out_target)
    set(QODEC_RUST_TARGET "" CACHE STRING "Rust target triple; empty selects the triple matching the CMake toolchain")
    if(QODEC_RUST_TARGET)
        set(${out_target} "${QODEC_RUST_TARGET}" PARENT_SCOPE)
        return()
    endif()
    set(processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
        list(LENGTH CMAKE_OSX_ARCHITECTURES architecture_count)
        if(architecture_count GREATER 1)
            message(FATAL_ERROR "qodec builds one architecture at a time; CMAKE_OSX_ARCHITECTURES is ${CMAKE_OSX_ARCHITECTURES}.")
        endif()
        set(processor "${CMAKE_OSX_ARCHITECTURES}")
    endif()
    string(TOLOWER "${processor}" processor)
    if(NOT processor MATCHES "^(x86_64|amd64|x64)$")
        set(triple "")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(triple "x86_64-unknown-linux-gnu")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(triple "x86_64-apple-darwin")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND MSVC)
        set(triple "x86_64-pc-windows-msvc")
    endif()
    if(NOT triple)
        message(FATAL_ERROR
            "No Rust target is known for ${CMAKE_SYSTEM_NAME} on '${processor}' with ${CMAKE_C_COMPILER_ID}. "
            "Set QODEC_RUST_TARGET to the Rust target triple.")
    endif()
    set(${out_target} "${triple}" PARENT_SCOPE)
endfunction()

# Cargo's output file names and their install locations, in matching order.
function(qodec_library_files out_staged out_installed)
    set(name qodec_c)
    if(NOT BUILD_SHARED_LIBS)
        set(file "${CMAKE_STATIC_LIBRARY_PREFIX}${name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(${out_staged} "${file}" PARENT_SCOPE)
        set(${out_installed} "${CMAKE_INSTALL_LIBDIR}/${file}" PARENT_SCOPE)
    elseif(WIN32)
        # Cargo names the import library qodec_c.dll.lib; it installs as qodec_c.lib.
        set(${out_staged} "${name}.dll" "${name}.dll.lib" PARENT_SCOPE)
        set(${out_installed} "${CMAKE_INSTALL_BINDIR}/${name}.dll" "${CMAKE_INSTALL_LIBDIR}/${name}.lib" PARENT_SCOPE)
    else()
        set(file "${CMAKE_SHARED_LIBRARY_PREFIX}${name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(${out_staged} "${file}" PARENT_SCOPE)
        set(${out_installed} "${CMAKE_INSTALL_LIBDIR}/${file}" PARENT_SCOPE)
    endif()
endfunction()

# Cargo's environment: the selected rustc and linker, and on MSVC the CRT
# chosen by CMAKE_MSVC_RUNTIME_LIBRARY for the current configuration.
function(qodec_cargo_environment out_environment target)
    string(TOUPPER "${target}" target_variable)
    string(REPLACE "-" "_" target_variable "${target_variable}")
    set(environment "RUSTC=${QODEC_RUSTC}")
    if(NOT MSVC)
        list(APPEND environment "CARGO_TARGET_${target_variable}_LINKER=${CMAKE_C_COMPILER}")
        set(${out_environment} "${environment}" PARENT_SCOPE)
        return()
    endif()
    list(APPEND environment "CARGO_TARGET_${target_variable}_LINKER=${CMAKE_LINKER}")
    if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        set(runtime "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    elseif(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "")
        message(FATAL_ERROR "qodec needs CMAKE_MSVC_RUNTIME_LIBRARY to select the CRT; it is empty.")
    else()
        set(runtime "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    endif()
    set(runtime "$<GENEX_EVAL:${runtime}>")
    set(static_runtime "$<OR:$<STREQUAL:${runtime},MultiThreaded>,$<STREQUAL:${runtime},MultiThreadedDebug>>")
    # Rust always uses the release CRT; it does not link the debug CRTs.
    set(flags "$ENV{RUSTFLAGS} -Ctarget-feature=+crt-static")
    string(STRIP "${flags}" flags)
    string(REPLACE ">" "$<ANGLE-R>" flags "${flags}")
    string(REPLACE "," "$<COMMA>" flags "${flags}")
    # COMMAND_EXPAND_LISTS must preserve semicolons within RUSTFLAGS.
    string(REPLACE ";" "\\$<SEMICOLON>" flags "${flags}")
    list(APPEND environment "$<${static_runtime}:RUSTFLAGS=${flags}>")
    set(${out_environment} "${environment}" PARENT_SCOPE)
endfunction()

# Builds only the requested crate type of qodec-c, so no rlib or unused native
# library is produced. Debug uses Cargo's dev profile; every other
# configuration uses the workspace's release profile.
function(qodec_cargo_library name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "TARGET;OUTPUT_DIRECTORY" "")
    get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT multi_config AND NOT CMAKE_BUILD_TYPE)
        message(FATAL_ERROR "Set CMAKE_BUILD_TYPE (for example Release or Debug) to build qodec.")
    endif()
    set(target_directory "${CMAKE_CURRENT_BINARY_DIR}/cargo")
    set(output_directory "${target_directory}/${arg_TARGET}/$<IF:$<CONFIG:Debug>,debug,release>")
    set(native_libraries_file "${output_directory}/qodec_c-native-static-libs.txt")
    set(install_name "")
    if(BUILD_SHARED_LIBS)
        set(crate_type cdylib)
        set(rustc_arguments "")
        set(native_libraries_file "")
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            # Without a SONAME, consumers would record the build-time path.
            set(rustc_arguments "-Clink-arg=-Wl,-soname,${CMAKE_SHARED_LIBRARY_PREFIX}qodec_c${CMAKE_SHARED_LIBRARY_SUFFIX}")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            set(install_name "@rpath/${CMAKE_SHARED_LIBRARY_PREFIX}qodec_c${CMAKE_SHARED_LIBRARY_SUFFIX}")
            set(rustc_arguments "-Clink-arg=-Wl,-install_name,${install_name}")
        endif()
    else()
        set(crate_type staticlib)
        set(rustc_arguments "--print=native-static-libs=${native_libraries_file}")
    endif()
    qodec_library_files(staged installed)
    # The target writer expects the paths in a single list argument.
    string(REPLACE ";" "\\;" installed_for_command "${installed}")
    list(TRANSFORM staged PREPEND "${output_directory}/" OUTPUT_VARIABLE byproducts)
    qodec_cargo_environment(environment "${arg_TARGET}")
    add_custom_target(${name} ALL
        COMMAND "${CMAKE_COMMAND}" -E env ${environment}
            "${QODEC_CARGO}" rustc
            --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
            --lib --crate-type ${crate_type}
            --target "${arg_TARGET}"
            --target-dir "${target_directory}"
            --profile "$<IF:$<CONFIG:Debug>,dev,release>"
            -- ${rustc_arguments}
        COMMAND "${CMAKE_COMMAND}"
            "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/qodecTargets-$<LOWER_CASE:$<CONFIG>>.cmake"
            "-DCONFIG=$<UPPER_CASE:$<CONFIG>>"
            "-DLIBRARY_FILES=${installed_for_command}"
            "-DNATIVE_LIBRARIES_FILE=${native_libraries_file}"
            "-DINSTALL_NAME=${install_name}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/QodecWriteTargets.cmake"
        BYPRODUCTS ${byproducts}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Building qodec_c with Cargo"
        USES_TERMINAL
        # Drop the empty MSVC environment argument for the dynamic CRT.
        COMMAND_EXPAND_LISTS
        VERBATIM)
    set(${arg_OUTPUT_DIRECTORY} "${output_directory}" PARENT_SCOPE)
endfunction()

function(qodec_install_library output_directory)
    qodec_library_files(staged installed)
    foreach(file destination IN ZIP_LISTS staged installed)
        get_filename_component(directory "${destination}" DIRECTORY)
        get_filename_component(installed_name "${destination}" NAME)
        install(FILES "${output_directory}/${file}" DESTINATION "${directory}" RENAME "${installed_name}")
    endforeach()
    if(BUILD_SHARED_LIBS AND WIN32)
        install(FILES "${output_directory}/qodec_c.pdb" DESTINATION "${CMAKE_INSTALL_BINDIR}" OPTIONAL)
    endif()
endfunction()
