if(NOT HOST_TRIPLET MATCHES "^x64-"
   OR NOT "${VCPKG_TARGET_IS_LINUX}" STREQUAL "${VCPKG_HOST_IS_LINUX}"
   OR NOT "${VCPKG_TARGET_IS_OSX}" STREQUAL "${VCPKG_HOST_IS_OSX}"
   OR NOT "${VCPKG_TARGET_IS_WINDOWS}" STREQUAL "${VCPKG_HOST_IS_WINDOWS}")
    message(FATAL_ERROR "qodec builds only natively on x64 hosts")
endif()

vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL "https://github.com/microsoft/qdk-ec.git"
    REF e3a31e17651668c74f27942dd6f11c0b1b7246c9
    HEAD_REF apaetz/qodec-vcpkg
)

# Cargo resolves the whole workspace, so build only qodec and its C bindings.
file(READ "${SOURCE_PATH}/Cargo.toml" workspace)
string(REGEX REPLACE "\n(default-)?members = \\[[^]]*\\]" "" workspace "${workspace}")
string(REPLACE "[workspace]\n" "[workspace]\nmembers = [\"qodec\", \"qodec/bindings/c\"]\n" workspace "${workspace}")
file(WRITE "${SOURCE_PATH}/Cargo.toml" "${workspace}")
foreach(manifest IN ITEMS "qodec/Cargo.toml" "qodec/bindings/c/Cargo.toml")
    file(READ "${SOURCE_PATH}/${manifest}" contents)
    string(FIND "${contents}" "\n[dev-dependencies]" dev_start)
    string(FIND "${contents}" "\n[" last_section REVERSE)
    if(dev_start EQUAL -1 OR NOT dev_start EQUAL last_section)
        message(FATAL_ERROR "${manifest} no longer ends with its [dev-dependencies] section")
    endif()
    string(SUBSTRING "${contents}" 0 ${dev_start} contents)
    file(WRITE "${SOURCE_PATH}/${manifest}" "${contents}\n")
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/crates.cmake")
set(vendor "${CURRENT_BUILDTREES_DIR}/cargo-vendor")
set(notices "${CURRENT_BUILDTREES_DIR}/THIRD-PARTY-NOTICES")
file(REMOVE_RECURSE "${vendor}")
file(WRITE "${notices}" "qodec statically links the following Rust crates and the Rust standard library.\n")
while(QODEC_CRATES)
    list(POP_FRONT QODEC_CRATES crate_name crate_version crate_sha512)
    vcpkg_download_distfile(crate_archive
        URLS "https://static.crates.io/crates/${crate_name}/${crate_name}-${crate_version}.crate"
        FILENAME "rust-crate-${crate_name}-${crate_version}.crate"
        SHA512 ${crate_sha512}
    )
    file(ARCHIVE_EXTRACT INPUT "${crate_archive}" DESTINATION "${vendor}")
    file(SHA256 "${crate_archive}" crate_sha256)
    file(WRITE "${vendor}/${crate_name}-${crate_version}/.cargo-checksum.json"
        "{\"files\":{},\"package\":\"${crate_sha256}\"}")
    file(GLOB crate_licenses "${vendor}/${crate_name}-${crate_version}/LICENSE*" "${vendor}/${crate_name}-${crate_version}/COPYING*")
    if(NOT crate_licenses)
        message(FATAL_ERROR "Crate ${crate_name} ${crate_version} has no license file")
    endif()
    foreach(license IN LISTS crate_licenses)
        cmake_path(GET license FILENAME license_name)
        file(READ "${license}" license_text)
        file(APPEND "${notices}" "\n${crate_name} ${crate_version}/${license_name}:\n\n${license_text}\n")
    endforeach()
endwhile()

set(rust_version 1.97.0)
if(VCPKG_TARGET_IS_LINUX)
    set(rust_target x86_64-unknown-linux-gnu)
    set(rust_sha512_rustc 20418278c5b71c1809df12a65d0cd32c42b490ce61a31c178756fbc911cfc61ca681581b7e5848e99fa88c7ecd0a93cca4a9039cd5b79f6b631963d6829bf5e3)
    set(rust_sha512_cargo e9d0f861ef3f65e8f3ff13f0f6c43682fbc3e288d36dbcb9c5d36384b7ae6f7e36a22b63398fab28cb4419ffe07b3600aef191fc4f4ba821d2e5177bb731aae6)
    set(rust_sha512_rust-std ff7d58ac5b2b41e8181f95180666d9782f1d190a140a1e8fa39852519d2fd9a5eaa82dc5644f0e9c6ab71b8fb9264d3dc0d31327ca3350d8388ec99bea36f55d)
elseif(VCPKG_TARGET_IS_OSX)
    set(rust_target x86_64-apple-darwin)
    set(rust_sha512_rustc 203ab895af93ee1a83eb0cb6470a6db33921e5f41d2a5e9f790170d2753ba16b85fc409f221460334b9ce0e588bf6bb31b4222ebbf9139c2967f06ea58fd4be3)
    set(rust_sha512_cargo 23141798a4d6c3b490ca833b172a721594947e13e86cfe9a076d0fba077b39035a285418663409ad52f35f526efb5321bac31052a246b8f02cc610d79a879aa2)
    set(rust_sha512_rust-std 6a43076c5c267f972996539cba8f180b50798c08c0863c8067c317530bdedaf0da87f5c5bbcd55f0128ab9b56da0bf1b76bf423434dd1d8512a68f2438021dec)
elseif(VCPKG_TARGET_IS_WINDOWS)
    set(rust_target x86_64-pc-windows-msvc)
    set(rust_sha512_rustc 229c7a82358579cb5a074ba6a45c775efa68fc5e0abd8aaa1785f7370a259af6effab111a9a130f85eef21b0a0fdba004e5f42c458d76cee989128efb7a80556)
    set(rust_sha512_cargo 6c835349aa89ba47b51330d0b53745522587fde58ed5111c7f35267efd91fb66ecba6d2cf2bd13cc3158334db49b7b25a2f1af4db480049ab71969ba5409edef)
    set(rust_sha512_rust-std f789b970c4327fbd16bd8446cd36e453ee4f53f6bcde2a50d5655cdcda0e0d7931ebc79deab6ee6a5c6a89a4cc2293a9d82ca1f63b40489a56d5069965763759)
endif()
set(rust_home "${CURRENT_BUILDTREES_DIR}/rust-${rust_version}")
file(REMOVE_RECURSE "${rust_home}")
foreach(component IN ITEMS rustc cargo rust-std)
    set(rust_package "${component}-${rust_version}-${rust_target}")
    vcpkg_download_distfile(rust_archive
        URLS "https://static.rust-lang.org/dist/${rust_package}.tar.xz"
        FILENAME "${rust_package}.tar.xz"
        SHA512 ${rust_sha512_${component}}
    )
    file(ARCHIVE_EXTRACT INPUT "${rust_archive}" DESTINATION "${rust_home}/archives")
    file(STRINGS "${rust_home}/archives/${rust_package}/components" installed_components)
    foreach(installed IN LISTS installed_components)
        file(COPY "${rust_home}/archives/${rust_package}/${installed}/" DESTINATION "${rust_home}/toolchain")
    endforeach()
endforeach()
foreach(license IN ITEMS COPYRIGHT LICENSE-APACHE LICENSE-MIT)
    file(READ "${rust_home}/archives/rust-std-${rust_version}-${rust_target}/${license}" license_text)
    file(APPEND "${notices}" "\nRust standard library ${rust_version}/${license}:\n\n${license_text}\n")
endforeach()

set(ENV{CARGO_HOME} "${CURRENT_BUILDTREES_DIR}/cargo-home")
file(MAKE_DIRECTORY "$ENV{CARGO_HOME}")
file(WRITE "$ENV{CARGO_HOME}/config.toml" "\
[source.crates-io]
replace-with = \"vcpkg-vendor\"

[source.vcpkg-vendor]
directory = \"${vendor}\"

[net]
offline = true
")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/qodec/bindings/c"
    OPTIONS
        "-DQODEC_CARGO=${rust_home}/toolchain/bin/cargo${VCPKG_HOST_EXECUTABLE_SUFFIX}"
        "-DQODEC_RUSTC=${rust_home}/toolchain/bin/rustc${VCPKG_HOST_EXECUTABLE_SUFFIX}"
        "-DQODEC_RUST_TARGET=${rust_target}"
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME qodec)
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/qodec/LICENSE" "${notices}")
