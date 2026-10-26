# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Installation rules for headers, libraries, and binaries

# The installed CLI is native and deliberately does not link the simulation runtime.
add_subdirectory(tools/openpfc)
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/share/openpfc/version
    CONTENT "${PROJECT_VERSION}${OpenPFC_VERSION_SUFFIX}\n")
install(FILES ${CMAKE_BINARY_DIR}/share/openpfc/version DESTINATION share/openpfc)

# Keep source, build-tree and relocatable installed CLI presets identical.
foreach(_app aluminumNew cahn_hilliard thin_film surface_diffusion kawahara ehd_film
            gradient_elasticity higher_order_pfc)
    file(GLOB _presets CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/apps/${_app}/inputs_json/*)
    foreach(_preset IN LISTS _presets)
        get_filename_component(_name ${_preset} NAME)
        configure_file(${_preset}
            ${CMAKE_BINARY_DIR}/share/openpfc/cases/${_app}/inputs_json/${_name} COPYONLY)
    endforeach()
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/${_app}/inputs_json
        DESTINATION share/openpfc/cases/${_app})
endforeach()

if(OpenPFC_BUILD_TESTS)
    add_test(NAME NativeCLI_Cases
      COMMAND ${CMAKE_COMMAND} "-Dcli=$<TARGET_FILE:openpfc_cli>" "-Dbuild=${CMAKE_BINARY_DIR}"
        -P ${CMAKE_SOURCE_DIR}/tests/cli/check_cases.cmake)
    set_tests_properties(NativeCLI_Cases PROPERTIES TIMEOUT 120)
    openpfc_installed_consumer_coverage_arguments(_native_cli_coverage)
    add_test(NAME NativeCLI_InstalledModels
      COMMAND ${CMAKE_COMMAND} "-Dbuild=${CMAKE_BINARY_DIR}" "-Dsource=${CMAKE_SOURCE_DIR}"
        "-Dcc=${CMAKE_C_COMPILER}" "-Dcxx=${CMAKE_CXX_COMPILER}" "-Dconfiguration=$<CONFIG>"
        "-Dgpu_sdk=$<OR:$<BOOL:${OpenPFC_CUDA_AVAILABLE}>,$<BOOL:${OpenPFC_HIP_AVAILABLE}>>"
        "${_native_cli_coverage}" -P ${CMAKE_SOURCE_DIR}/tests/cli/check_models.cmake)
    set_tests_properties(NativeCLI_InstalledModels PROPERTIES TIMEOUT 600 RUN_SERIAL TRUE)
    foreach(_pair "tungsten,tungsten" "aluminum,aluminum_etd" "cahn_hilliard,cahn_hilliard"
                  "thin_film,thin_film" "surface_diffusion,surface_diffusion"
                  "kawahara,kawahara" "ehd_film,ehd_film"
                  "gradient_elasticity,gradient_elasticity" "higher_order_pfc,higher_order_pfc")
        string(REPLACE "," ";" _entry ${_pair})
        list(GET _entry 0 _app)
        list(GET _entry 1 _binary)
        if(TARGET ${_binary})
            add_test(NAME openpfc-cli-${_app}-smoke
              COMMAND ${CMAKE_COMMAND} "-Dcli=$<TARGET_FILE:openpfc_cli>" "-Dbinary=$<TARGET_FILE:${_binary}>"
                "-Dapp=${_app}" "-Dbuild=${CMAKE_BINARY_DIR}"
                -P ${CMAKE_SOURCE_DIR}/tests/cli/smoke_case.cmake)
            set_tests_properties(openpfc-cli-${_app}-smoke PROPERTIES TIMEOUT 120)
        endif()
    endforeach()
    if(TARGET cahn_hilliard AND OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
       (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
        find_package(Python3 3.8 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME cahn-hilliard-diagnostics-workflow
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tests/cahn_hilliard_diagnostics_smoke.py
                    --binary $<TARGET_FILE:cahn_hilliard> --mpiexec ${MPIEXEC_EXECUTABLE})
            set_tests_properties(cahn-hilliard-diagnostics-workflow PROPERTIES TIMEOUT 180 PROCESSORS 2)
        endif()
    endif()
endif()

# Install public headers only. Device TUs live under src/openpfc/runtime/gpu/;
# kernel .inc files live next to those TUs under src/ and are not installed.
# Stray .md under include/ must not ship.
# FetchContent nlohmann_json is a build-time dependency only — do not dump its
# headers into the prefix. Consumers that include JSON-using public headers
# get nlohmann_json via find_dependency in OpenPFCConfig.cmake.
# Implementation includes ship too: simulation_state.hpp includes
# simulation_state.ipp and solvers/interface_diffusion.hpp includes
# interface_diffusion.inl.
install(DIRECTORY include/openpfc DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.ipp" PATTERN "*.inl")

# Upstream toml++ installs itself only as a top-level project. When fetched
# as our subproject, deliver its header-only target as a separate relocatable
# dependency package. System packages remain external find_dependency inputs.
if(OpenPFC_TOML_FETCHED)
    include(CMakePackageConfigHelpers)
    install(DIRECTORY "${tomlplusplus_SOURCE_DIR}/include/" DESTINATION include)
    install(FILES "${tomlplusplus_SOURCE_DIR}/LICENSE"
            DESTINATION share/licenses/tomlplusplus)
    install(TARGETS tomlplusplus_tomlplusplus
            EXPORT OpenPFCTomlPlusPlusTargets INCLUDES DESTINATION include)
    install(EXPORT OpenPFCTomlPlusPlusTargets
            NAMESPACE tomlplusplus:: FILE tomlplusplusTargets.cmake
            DESTINATION lib/cmake/tomlplusplus)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/tomlplusplusConfigVersion.cmake"
        VERSION "${tomlplusplus_FIND_VERSION}" COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
    install(FILES "${tomlplusplus_SOURCE_DIR}/cmake/tomlplusplusConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/tomlplusplusConfigVersion.cmake"
            DESTINATION lib/cmake/tomlplusplus)
endif()

# Install library binary
install(TARGETS openpfc
    EXPORT OpenPFCTargets
    ARCHIVE DESTINATION lib   # .a files
    LIBRARY DESTINATION lib   # .so files
    RUNTIME DESTINATION bin   # executable files (not needed now but future proof)
)

# Install GPU kernel library if CUDA is enabled
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
    install(TARGETS openpfc_gpu_kernels
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

# Install HIP kernel library if HIP is enabled (audit 11 / PM: this block was
# missing, so install(EXPORT OpenPFCTargets) failed or HIP installs shipped
# without libopenpfc_hip_kernels -- mirror the CUDA block above).
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE)
    install(TARGETS openpfc_hip_kernels
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
    install(TARGETS openpfc_microelasticity_hip
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE AND OpenPFC_ENABLE_CUDA_SPECTRAL)
    install(TARGETS openpfc_microelasticity_cuda
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

# The device step is not part of openpfc_hip_kernels / openpfc_gpu_kernels.
# Install the library built from the runtime/gpu translation unit so an
# installed consumer of the header can link it. One spectral backend per build.
if(OpenPFC_ENABLE_HIP AND OpenPFC_HIP_AVAILABLE AND OpenPFC_ENABLE_HIP_SPECTRAL)
    install(TARGETS openpfc_incompressible_hip
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE AND OpenPFC_ENABLE_CUDA_SPECTRAL)
    install(TARGETS openpfc_incompressible_cuda
        EXPORT OpenPFCTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()
