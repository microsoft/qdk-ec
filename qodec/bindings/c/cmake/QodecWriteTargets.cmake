# Writes qodecTargets-<config>.cmake after Cargo builds qodec_c.
#
# Inputs: OUTPUT, CONFIG (upper case), LIBRARY_FILES (install-relative paths:
# the library, then the Windows import library for a DLL),
# NATIVE_LIBRARIES_FILE (rustc's native-static-libs output; empty when shared), and
# INSTALL_NAME (macOS shared-library install name; empty otherwise).

list(GET LIBRARY_FILES 0 location)
set(contents "set_property(TARGET qodec::qodec APPEND PROPERTY IMPORTED_CONFIGURATIONS ${CONFIG})\n")
string(APPEND contents "set_target_properties(qodec::qodec PROPERTIES\n")
string(APPEND contents "    IMPORTED_LOCATION_${CONFIG} \"\${_IMPORT_PREFIX}/${location}\"")
list(LENGTH LIBRARY_FILES file_count)
if(file_count GREATER 1)
    list(GET LIBRARY_FILES 1 import_library)
    string(APPEND contents "\n    IMPORTED_IMPLIB_${CONFIG} \"\${_IMPORT_PREFIX}/${import_library}\"")
endif()
if(INSTALL_NAME)
    # CMake needs this to add the library's directory to consumer rpaths.
    string(APPEND contents "\n    IMPORTED_SONAME_${CONFIG} \"${INSTALL_NAME}\"")
endif()
string(APPEND contents ")\n")

if(NATIVE_LIBRARIES_FILE)
    file(READ "${NATIVE_LIBRARIES_FILE}" native_arguments)
    separate_arguments(native_arguments NATIVE_COMMAND "${native_arguments}")
    set(native_libraries "")
    set(framework FALSE)
    foreach(argument IN LISTS native_arguments)
        if(framework)
            list(APPEND native_libraries "-framework ${argument}")
            set(framework FALSE)
        elseif(argument STREQUAL "-framework")
            set(framework TRUE)
        elseif(argument MATCHES "^-l(.+)$")
            list(APPEND native_libraries "${CMAKE_MATCH_1}")
        elseif(argument MATCHES "^/defaultlib:")
            # The consumer's CMAKE_MSVC_RUNTIME_LIBRARY selects the CRT.
        else()
            list(APPEND native_libraries "${argument}")
        endif()
    endforeach()
    # The libraries are those of the Rust standard library and do not vary by
    # configuration, so each configuration's file sets the same list.
    string(APPEND contents "set_property(TARGET qodec::qodec PROPERTY INTERFACE_LINK_LIBRARIES \"${native_libraries}\")\n")
endif()

file(WRITE "${OUTPUT}" "${contents}")
