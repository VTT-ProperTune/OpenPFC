# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Installation rules for headers, libraries, and binaries

# Python CLI is usable from the build tree and relocatable after installation.
configure_file(${CMAKE_SOURCE_DIR}/scripts/openpfc ${CMAKE_BINARY_DIR}/bin/openpfc
    COPYONLY FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/share/openpfc/version
    CONTENT "${PROJECT_VERSION}${OpenPFC_VERSION_SUFFIX}\n")
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/openpfc DESTINATION bin)
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

if(OpenPFC_BUILD_TESTS AND TARGET tungsten)
    find_package(Python3 3.8 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME openpfc-cli-smoke
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tests/cli_smoke.py
                --cli ${CMAKE_BINARY_DIR}/bin/openpfc
                --binary $<TARGET_FILE:tungsten>)
        set_tests_properties(openpfc-cli-smoke PROPERTIES TIMEOUT 120)
        foreach(_pair "aluminum,aluminum_etd" "cahn_hilliard,cahn_hilliard"
                      "thin_film,thin_film" "surface_diffusion,surface_diffusion"
                      "kawahara,kawahara" "ehd_film,ehd_film"
                      "gradient_elasticity,gradient_elasticity"
                      "higher_order_pfc,higher_order_pfc")
            string(REPLACE "," ";" _entry ${_pair})
            list(GET _entry 0 _app)
            list(GET _entry 1 _binary)
            if(TARGET ${_binary})
                add_test(NAME openpfc-cli-${_app}-smoke
                    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tests/cli_smoke.py
                        --cli ${CMAKE_BINARY_DIR}/bin/openpfc
                        --binary $<TARGET_FILE:${_binary}> --app ${_app})
                set_tests_properties(openpfc-cli-${_app}-smoke PROPERTIES TIMEOUT 120)
            endif()
        endforeach()
        if(TARGET cahn_hilliard AND OpenPFC_RUN_MPI_SUITES AND MPIEXEC_EXECUTABLE AND
           (OpenPFC_MPI_TEST_MAX_WORLD_SIZE EQUAL 0 OR OpenPFC_MPI_TEST_MAX_WORLD_SIZE GREATER_EQUAL 2))
            add_test(NAME cahn-hilliard-diagnostics-workflow
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/scripts/tests/cahn_hilliard_diagnostics_smoke.py
                    --binary $<TARGET_FILE:cahn_hilliard> --mpiexec ${MPIEXEC_EXECUTABLE})
            set_tests_properties(cahn-hilliard-diagnostics-workflow PROPERTIES
                TIMEOUT 180 PROCESSORS 2)
        endif()
    endif()
endif()

# Install public headers only. Device TUs live under src/openpfc/runtime/gpu/;
# kernel .inc files live next to those TUs under src/ and are not installed.
# Stray .md under include/ must not ship.
# FetchContent nlohmann_json is a build-time dependency only — do not dump its
# headers into the prefix. Consumers that include JSON-using public headers
# get nlohmann_json via find_dependency in OpenPFCConfig.cmake.
install(DIRECTORY include/openpfc DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

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
