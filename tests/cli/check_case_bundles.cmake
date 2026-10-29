# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Included by NativeCLI_InstalledModels with its no-Python PATH and MPI tools.
# Logical cases: compiled-case-runtime, case-portability, case-cache-integrity,
# and cray-host-library-classification. Assertions/variants are not test counts.
set(case_project "${root}/compiled case with spaces ü")
cli_call(0 init "${case_project}" --app=tungsten)
file(READ "${case_project}/openpfc.json" case_manifest)
string(JSON case_manifest SET "${case_manifest}" profile "\"local\"")
string(JSON case_manifest SET "${case_manifest}" executable "\"${starter}/model\"")
string(JSON case_manifest SET "${case_manifest}" build_dir "\"${build}\"")
file(WRITE "${case_project}/openpfc.json" "${case_manifest}\n")
file(READ "${starter}/input.json" frozen_input)
file(WRITE "${case_project}/input.json" "${frozen_input}")
set(case_image "${root}/compiled case.sif")
set(case_args --prepare-only --output "${case_image}" ${host_args})
cli_call(0 bundle "${case_project}" ${case_args})
if(NOT output MATCHES "Bundle: ([^\n]+)")
  message(FATAL_ERROR "Case bundle did not report recipe: ${output}")
endif()
set(case_recipe "${CMAKE_MATCH_1}")
set(payload "${case_recipe}/payload")
file(READ "${payload}/provenance.json" case_provenance)
string(JSON kind GET "${case_provenance}" kind)
string(JSON recorded_input GET "${case_provenance}" input_sha256)
string(JSON case_build_type TYPE "${case_provenance}" build_id)
file(SHA256 "${case_project}/input.json" input_digest)
file(SHA256 "${starter}/model" model_digest)
file(SHA256 "${payload}/bin/model" bundled_model_digest)
if(NOT kind STREQUAL "openpfc.case-bundle" OR NOT case_build_type STREQUAL NULL OR
   NOT recorded_input STREQUAL input_digest OR NOT model_digest STREQUAL bundled_model_digest)
  message(FATAL_ERROR "Case provenance lost input/binary identity or invented build provenance")
endif()
file(READ "${payload}/case/openpfc.json" portable_case)
string(JSON portable_binary GET "${portable_case}" executable)
if(NOT portable_binary STREQUAL "/opt/openpfc/bin/model")
  message(FATAL_ERROR "Portable case retained a build-host binary")
endif()
file(READ "${case_recipe}/bundle.def" case_definition)
if(NOT case_definition MATCHES "exec /opt/openpfc/bin/openpfc image" OR
   case_definition MATCHES "python|baked-case")
  message(FATAL_ERROR "Case runscript is not native: ${case_definition}")
endif()
file(GLOB_RECURSE case_files "${payload}/*")
foreach(path IN LISTS case_files)
  if(path MATCHES "\.(py|hpp|h)$" OR path MATCHES "/lib/lib(mpi|fabric|ibverbs|cxi)")
    message(FATAL_ERROR "Case payload contains source, Python or host MPI/fabric: ${path}")
  endif()
endforeach()
set(image_cli "${payload}/bin/openpfc")
execute_process(COMMAND "${image_cli}" image --version
  RESULT_VARIABLE code OUTPUT_VARIABLE image_version ERROR_VARIABLE err)
file(READ "${build}/share/openpfc/version" build_version)
string(STRIP "${build_version}" build_version)
if(NOT code EQUAL 0 OR NOT image_version STREQUAL "OpenPFC ${build_version}\n")
  message(FATAL_ERROR "Image version differs from the case build: ${image_version}\n${err}")
endif()
execute_process(COMMAND "${image_cli}" image --provenance
  RESULT_VARIABLE code OUTPUT_VARIABLE image_provenance ERROR_VARIABLE err)
if(NOT code EQUAL 0 OR NOT image_provenance STREQUAL case_provenance)
  message(FATAL_ERROR "Image provenance differs from the frozen payload: ${err}")
endif()
set(copied_case "${root}/copied case ü with spaces")
check("${image_cli}" image init "${copied_case}")
file(READ "${copied_case}/input.json" copied_input)
if(NOT copied_input STREQUAL frozen_input)
  message(FATAL_ERROR "Native image init changed the frozen input")
endif()
execute_process(COMMAND "${image_cli}" image init "${copied_case}"
  RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT code EQUAL 2 OR NOT err MATCHES "refusing to overwrite")
  message(FATAL_ERROR "Image init overwrote an existing case: ${out}\n${err}")
endif()
# Every outer rank is replaced by the app, with the copied case as cwd.
check("${direct_mpiexec}" -n 2 "${image_cli}" image run "${copied_case}")
file(READ "${copied_case}/results/summary.json" case_result)
if(NOT case_result STREQUAL mpi_result)
  message(FATAL_ERROR "Native case runtime differs from the direct MPI result: ${case_result}")
endif()
# Executable failures propagate through exec, rather than being hidden by a wrapper.
file(WRITE "${copied_case}/input.json" "{\"cells\":16,\"steps\":10,\"dt\":1,\"diffusivity\":1}")
execute_process(COMMAND "${image_cli}" image run "${copied_case}"
  RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(code EQUAL 0)
  message(FATAL_ERROR "Native image runtime hid a model failure")
endif()
# Same native packager supports another catalogue entry, not only tungsten.
string(JSON other_manifest SET "${case_manifest}" app "\"thin_film\"")
file(WRITE "${case_project}/openpfc.json" "${other_manifest}\n")
cli_call(0 bundle "${case_project}" ${case_args})
file(WRITE "${case_project}/openpfc.json" "${case_manifest}\n")
# Portability and target checks happen before any bundle is published.
foreach(bad_input IN ITEMS
    "{\"restart_from\":\"old-checkpoint\"}"
    "{\"initial_conditions\":[{\"type\":\"from_file\",\"filename\":\"missing.bin\"}]}"
    "{\"initial_conditions\":[{\"type\":\"unknown\"}]}"
    "{\"boundary_conditions\":[{\"type\":\"from_file\",\"filename\":\"missing.bin\"}]}"
    "{\"fields\":[{\"data\":\"/outside/psi.bin\"}]}"
    "{\"checkpoint\":{\"directory\":\"results/../../outside\"}}"
    "{\"diagnostics\":{\"csv\":\"outside.csv\"}}")
  file(WRITE "${case_project}/input.json" "${bad_input}\n")
  cli_call(2 bundle "${case_project}" ${case_args})
  file(WRITE "${copied_case}/input.json" "${bad_input}\n")
  execute_process(COMMAND "${image_cli}" image run "${copied_case}"
    RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT code EQUAL 2)
    message(FATAL_ERROR "Image run accepted nonportable edited input: ${bad_input}")
  endif()
endforeach()
file(WRITE "${case_project}/input.json" "${frozen_input}")
cli_call(2 bundle "${case_project}" ${case_args} --input "${case_project}/input.json")
cli_call(2 bundle "${case_project}" ${case_args} --sdk "${sdk}")
foreach(profile tohtori lumi)
  string(JSON bad_manifest SET "${case_manifest}" profile "\"${profile}\"")
  file(WRITE "${case_project}/openpfc.json" "${bad_manifest}\n")
  cli_call(2 bundle "${case_project}" ${case_args})
endforeach()
file(WRITE "${case_project}/openpfc.json" "${case_manifest}\n")
file(WRITE "${case_image}.sha256" "existing sidecar\n")
cli_call(2 bundle "${case_project}" ${case_args})
file(REMOVE "${case_image}.sha256")
cli_call(0 bundle "${case_project}" ${case_args})
if(NOT output MATCHES "Bundle: ${case_recipe}")
  message(FATAL_ERROR "Unchanged case did not reuse its recipe")
endif()
# Frozen input bytes, including whitespace, participate in the case identity.
file(APPEND "${case_project}/input.json" "\n")
cli_call(0 bundle "${case_project}" ${case_args})
if(output MATCHES "Bundle: ${case_recipe}\n")
  message(FATAL_ERROR "Changed frozen input reused an old case recipe")
endif()
file(WRITE "${case_project}/input.json" "${frozen_input}")
# Frozen input bytes and CLI/build manifests are checked on a cache hit.
file(CHMOD "${payload}/case/input.json" PERMISSIONS OWNER_READ OWNER_WRITE)
file(APPEND "${payload}/case/input.json" "corruption")
cli_call(2 bundle "${case_project}" ${case_args})
if(NOT error MATCHES "cached bundle payload changed")
  message(FATAL_ERROR "Cached case input corruption went undetected: ${error}")
endif()

# Real ELF fixture: Cray-style libraries plus Slingshot next to ordinary libs.
# Unlike the retired mocked ldd tests, this exercises recursive ELF resolution.
set(fixture "${build}/cli-cray-${label}")
file(MAKE_DIRECTORY "${fixture}/host" "${fixture}/system")
file(WRITE "${fixture}/library.c" "int bundle_fixture(void) { return 0; }\n")
foreach(name libmpi_gnu_123.so.12 libfabric.so.1 libpmi.so.0)
  check("${cc}" -shared -fPIC "${fixture}/library.c" "-Wl,-soname,${name}"
    -o "${fixture}/host/${name}")
endforeach()
foreach(name libcxi.so.1 libordinary.so.1)
  check("${cc}" -shared -fPIC "${fixture}/library.c" "-Wl,-soname,${name}"
    -o "${fixture}/system/${name}")
endforeach()
file(WRITE "${fixture}/main.c" "int main(void) { return 0; }\n")
check("${cc}" "${fixture}/main.c" -o "${fixture}/model"
  "-L${fixture}/host" "-L${fixture}/system"
  "-Wl,-rpath,${fixture}/host:${fixture}/system" -Wl,--no-as-needed
  -l:libmpi_gnu_123.so.12 -l:libfabric.so.1 -l:libpmi.so.0
  -l:libcxi.so.1 -l:libordinary.so.1)
string(JSON fixture_manifest SET "${case_manifest}" executable "\"${fixture}/model\"")
file(WRITE "${case_project}/openpfc.json" "${fixture_manifest}\n")
set(cray_args --prepare-only --output "${case_image}" --site=cray
  --host-prefix "${fixture}/host")
cli_call(2 bundle "${case_project}" ${cray_args})
if(NOT error MATCHES "libcxi.so.1" OR NOT error MATCHES "--host-lib")
  message(FATAL_ERROR "Slingshot was not classified as a host library: ${error}")
endif()
cli_call(0 bundle "${case_project}" ${cray_args}
  --host-lib "${fixture}/system/libcxi.so.1")
if(NOT output MATCHES "Bundle: ([^\n]+)")
  message(FATAL_ERROR "Cray fixture recipe missing")
endif()
set(cray_recipe "${CMAKE_MATCH_1}")
file(READ "${cray_recipe}/payload/provenance.json" cray_provenance)
string(JSON host_mpi GET "${cray_provenance}" host_libraries libmpi_gnu_123.so.12 path)
string(JSON host_cxi GET "${cray_provenance}" host_libraries libcxi.so.1 path)
if(NOT EXISTS "${cray_recipe}/payload/lib/libordinary.so.1" OR
   EXISTS "${cray_recipe}/payload/lib/libcxi.so.1")
  message(FATAL_ERROR "Cray fixture lost the host/bundled split")
endif()
file(READ "${cray_recipe}/run-host.sh" cray_helper)
if(NOT cray_helper MATCHES "/var/spool/slurmd:/var/spool/slurmd")
  message(FATAL_ERROR "Cray helper omitted the PMI socket directory")
endif()
string(FIND "${cray_helper}" "${fixture}/system:${fixture}/system" broad_bind)
if(NOT broad_bind LESS 0)
  message(FATAL_ERROR "Cray helper binds a shared system-library directory")
endif()

# A missing transitive dependency is diagnosed without executing the app.
file(RENAME "${fixture}/system/libordinary.so.1" "${fixture}/system/libordinary.hidden")
cli_call(2 bundle "${case_project}" ${cray_args}
  --host-lib "${fixture}/system/libcxi.so.1")
if(NOT error MATCHES "libordinary.so.1")
  message(FATAL_ERROR "Unresolved case dependency was not diagnosed: ${error}")
endif()
file(RENAME "${fixture}/system/libordinary.hidden" "${fixture}/system/libordinary.so.1")
