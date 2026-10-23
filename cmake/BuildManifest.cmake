# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

include("${CMAKE_CURRENT_LIST_DIR}/OpenPFCCompatibility.cmake")

function(_openpfc_manifest_variables out)
  set(_object "{}")
  foreach(_name IN LISTS ARGN)
    if(DEFINED ${_name})
      _openpfc_json_field(_object "${_name}" "${${_name}}")
    else()
      _openpfc_json_field(_object "${_name}" unknown)
    endif()
  endforeach()
  set(${out} "${_object}" PARENT_SCOPE)
endfunction()

function(_openpfc_manifest_target out target)
  set(_object "{}")
  foreach(_property TYPE IMPORTED_CONFIGURATIONS IMPORTED_LOCATION IMPORTED_SONAME
      INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS
      INTERFACE_COMPILE_FEATURES INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES
      INTERFACE_LINK_OPTIONS COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_OPTIONS
      LINK_LIBRARIES POSITION_INDEPENDENT_CODE CXX_STANDARD CUDA_STANDARD HIP_STANDARD
      CUDA_ARCHITECTURES HIP_ARCHITECTURES CUDA_SEPARABLE_COMPILATION
      CUDA_RESOLVE_DEVICE_SYMBOLS CUDA_RUNTIME_LIBRARY MSVC_RUNTIME_LIBRARY)
    get_target_property(_value "${target}" "${_property}")
    if(_value STREQUAL "_value-NOTFOUND")
      set(_value unknown)
    endif()
    _openpfc_json_field(_object "${_property}" "${_value}")
  endforeach()
  get_target_property(_configs "${target}" IMPORTED_CONFIGURATIONS)
  foreach(_config IN LISTS _configs)
    foreach(_base IMPORTED_LOCATION IMPORTED_SONAME IMPORTED_IMPLIB)
      get_target_property(_value "${target}" "${_base}_${_config}")
      if(NOT _value STREQUAL "_value-NOTFOUND")
        _openpfc_json_field(_object "${_base}_${_config}" "${_value}")
      endif()
    endforeach()
  endforeach()
  set(_sources "{}")
  get_target_property(_source_list "${target}" SOURCES)
  foreach(_source IN LISTS _source_list)
    if(_source MATCHES "\\.(cu|hip)$")
      set(_entry "{}")
      foreach(_property COMPILE_FLAGS COMPILE_OPTIONS COMPILE_DEFINITIONS LANGUAGE)
        get_source_file_property(_value "${_source}" "${_property}")
        if(_value STREQUAL NOTFOUND)
          set(_value "")
        endif()
        _openpfc_json_field(_entry "${_property}" "${_value}")
      endforeach()
      string(JSON _sources SET "${_sources}" "${_source}" "${_entry}")
    endif()
  endforeach()
  string(JSON _object SET "${_object}" device_sources "${_sources}")
  set(${out} "${_object}" PARENT_SCOPE)
endfunction()

set(OpenPFC_SDK_CUDA FALSE)
set(OpenPFC_SDK_HIP FALSE)
# Use the compiled targets, not stale toolkit cache variables, as the authority.
if(TARGET openpfc_gpu_kernels)
  set(OpenPFC_SDK_CUDA TRUE)
endif()
if(TARGET openpfc_hip_kernels)
  set(OpenPFC_SDK_HIP TRUE)
endif()
set(_capabilities "{\"cpu\":true,\"mpi\":true}")
foreach(_pair "cuda;OpenPFC_SDK_CUDA" "hip;OpenPFC_SDK_HIP"
    "heffte;OpenPFC_ENABLE_HEFFTE" "fftw;OpenPFC_HAS_FFTW3"
    "hdf5;OpenPFC_ENABLE_HDF5" "cuda_spectral;OpenPFC_ENABLE_CUDA_SPECTRAL"
    "hip_spectral;OpenPFC_ENABLE_HIP_SPECTRAL")
  list(GET _pair 0 _key)
  list(GET _pair 1 _variable)
  if(${_variable})
    string(JSON _capabilities SET "${_capabilities}" "${_key}" true)
  else()
    string(JSON _capabilities SET "${_capabilities}" "${_key}" false)
  endif()
endforeach()

set(_targets "{}")
set(_queue openpfc openpfc_kernel_obj openpfc_frontend_obj MPI::MPI_C MPI::MPI_CXX)
foreach(_target openpfc_gpu_kernels openpfc_hip_kernels openpfc_microelasticity_cuda
    openpfc_microelasticity_hip openpfc_incompressible_cuda openpfc_incompressible_hip)
  if(TARGET "${_target}")
    list(APPEND _queue "${_target}")
  endif()
endforeach()
# Walk actual target dependencies; optional packages left in a cache are not SDK
# facilities. Preserve raw CMake expressions rather than pretending to resolve
# them outside their language/configuration context.
set(_visited)
while(_queue)
  list(POP_FRONT _queue _target)
  if(NOT TARGET "${_target}" OR _target IN_LIST _visited)
    continue()
  endif()
  list(APPEND _visited "${_target}")
  _openpfc_manifest_target(_record "${_target}")
  string(JSON _targets SET "${_targets}" "${_target}" "${_record}")
  foreach(_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
    get_target_property(_links "${_target}" "${_property}")
    foreach(_link IN LISTS _links)
      # Static archives wrap private dependencies in LINK_ONLY.
      if(_link MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
        set(_link "${CMAKE_MATCH_1}")
      endif()
      if(TARGET "${_link}")
        list(APPEND _queue "${_link}")
      endif()
    endforeach()
  endforeach()
endwhile()

_openpfc_mpi_library_evidence(_mpi_libraries)
_openpfc_manifest_variables(_mpi MPI_C_COMPILER MPI_CXX_COMPILER MPI_C_VERSION
  MPI_CXX_VERSION MPI_C_COMPILE_OPTIONS MPI_CXX_COMPILE_OPTIONS
  MPI_C_COMPILE_DEFINITIONS MPI_CXX_COMPILE_DEFINITIONS MPI_C_INCLUDE_DIRS
  MPI_CXX_INCLUDE_DIRS MPI_C_LINK_FLAGS MPI_CXX_LINK_FLAGS MPI_C_LIBRARIES
  MPI_CXX_LIBRARIES OpenPFC_MPI_CUDA_AWARE OpenPFC_MPI_HIP_AWARE
  OPENPFC_OMPI_CUDA_MPI_AWARE_OK OPENPFC_GPU_AWARE_MPI_CUDA_PROBE_SKIPPED
  OPENPFC_MPIX_QUERY_CUDA_SUPPORT_COMPILES)
string(JSON _mpi SET "${_mpi}" libraries "${_mpi_libraries}")
foreach(_backend CUDA HIP)
  if(OpenPFC_SDK_${_backend} AND OpenPFC_MPI_${_backend}_AWARE)
    string(JSON _mpi SET "${_mpi}" "${_backend}_device_buffers" true)
  else()
    string(JSON _mpi SET "${_mpi}" "${_backend}_device_buffers" false)
  endif()
endforeach()
_openpfc_json_field(_mpi abi_verification "unverified: file identity and library version are evidence, not a cross-vendor ABI guarantee")
_openpfc_json_field(_mpi gpu_aware_runtime "unknown: no device-buffer communication probe performed by SDK packaging")

_openpfc_manifest_variables(_stack Heffte_VERSION Heffte_DIR Heffte_CUDA_FOUND
  Heffte_ROCM_FOUND Heffte_FFTW_FOUND FFTW3_VERSION FFTW3_LIBRARY
  HDF5_VERSION HDF5_IS_PARALLEL hip_VERSION HIP_VERSION HIP_VERSION_STRING
  ROCM_VERSION ROCM_VERSION_STRING CMAKE_HIP_COMPILER_ROCM_ROOT
  CUDAToolkit_VERSION nlohmann_json_VERSION)
_openpfc_manifest_variables(_components OpenPFC_BUILD_APPS OpenPFC_BUILD_EXAMPLES
  OpenPFC_BUILD_TESTS OpenPFC_BUILD_BENCHMARKS OpenPFC_BUILD_DOCUMENTATION
  OpenPFC_ENABLE_GPU_AUTOTUNING OpenPFC_PROFILING_LEVEL OpenPFC_NAN_CHECK_ACTIVE
  BUILD_SHARED_LIBS)

set(_toolchains "{}")
foreach(_language C CXX CUDA HIP)
  if(_language STREQUAL CUDA AND NOT OpenPFC_SDK_CUDA OR
     _language STREQUAL HIP AND NOT OpenPFC_SDK_HIP)
    # Explicitly ignore stale compiler/toolkit values from disabled backends.
    set(_toolchain "{\"enabled\":false,\"architectures\":\"none\"}")
  else()
    _openpfc_manifest_variables(_toolchain
      CMAKE_${_language}_COMPILER CMAKE_${_language}_COMPILER_ID
      CMAKE_${_language}_COMPILER_VERSION CMAKE_${_language}_COMPILER_TARGET
      CMAKE_${_language}_COMPILER_EXTERNAL_TOOLCHAIN CMAKE_${_language}_COMPILER_ARG1
      CMAKE_${_language}_COMPILER_LAUNCHER CMAKE_${_language}_HOST_COMPILER
      CMAKE_${_language}_HOST_COMPILER_ID CMAKE_${_language}_HOST_COMPILER_VERSION
      CMAKE_${_language}_HOST_LINK_LAUNCHER CMAKE_${_language}_SIMULATE_ID
      CMAKE_${_language}_SIMULATE_VERSION CMAKE_${_language}_COMPILER_ABI
      CMAKE_${_language}_COMPILER_FRONTEND_VARIANT CMAKE_${_language}_RUNTIME_LIBRARY
      CMAKE_${_language}_RUNTIME_LIBRARY_DEFAULT
      CMAKE_${_language}_STANDARD CMAKE_${_language}_EXTENSIONS
      CMAKE_${_language}_IMPLICIT_LINK_LIBRARIES CMAKE_${_language}_IMPLICIT_LINK_DIRECTORIES)
    string(JSON _toolchain SET "${_toolchain}" enabled true)
    if(_language MATCHES "^(CUDA|HIP)$")
      _openpfc_json_field(_toolchain architectures "${CMAKE_${_language}_ARCHITECTURES}")
      set(_host unknown)
      set(_host_path "${CMAKE_${_language}_HOST_COMPILER}")
      set(_host_evidence "explicit CMake host compiler")
      if(NOT _host_path AND CMAKE_${_language}_HOST_LINK_LAUNCHER)
        set(_host_path "${CMAKE_${_language}_HOST_LINK_LAUNCHER}")
        set(_host_evidence "host link launcher detected by CMake; device flags may override host compilation")
      endif()
      if(_host_path)
        execute_process(COMMAND "${_host_path}" --version
          OUTPUT_VARIABLE _host ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
          RESULT_VARIABLE _host_result)
        if(NOT _host_result EQUAL 0 OR NOT _host)
          set(_host unknown)
        endif()
      else()
        set(_host_evidence "unknown: host compilation is selected by the device toolchain; see CXX and device flags")
      endif()
      _openpfc_json_field(_toolchain host_compiler_diagnostic "${_host}")
      _openpfc_json_field(_toolchain host_compiler_evidence "${_host_evidence}")
    endif()
  endif()
  string(JSON _toolchains SET "${_toolchains}" "${_language}" "${_toolchain}")
endforeach()
set(_triple "${CMAKE_CXX_COMPILER_TARGET}")
set(_triple_evidence "CMAKE_CXX_COMPILER_TARGET")
if(NOT _triple)
  set(_triple_evidence "compiler -dumpmachine default target; per-configuration flags may override the ABI")
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -dumpmachine
    OUTPUT_VARIABLE _triple RESULT_VARIABLE _result ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _result EQUAL 0)
    set(_triple unknown)
  endif()
endif()
_openpfc_manifest_variables(_platform CMAKE_SYSTEM_NAME CMAKE_SYSTEM_PROCESSOR
  CMAKE_SIZEOF_VOID_P CMAKE_CXX_BYTE_ORDER CMAKE_TOOLCHAIN_FILE CMAKE_LINKER
  CMAKE_AR CMAKE_RANLIB CMAKE_GENERATOR CMAKE_VERSION CMAKE_SYSROOT)
_openpfc_json_field(_platform target_triple "${_triple}")
_openpfc_json_field(_platform target_triple_evidence "${_triple_evidence}")
set(_runtime_libraries "{}")
foreach(_runtime libstdc++.so libc++.so libgcc_s.so.1)
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${_runtime}"
    OUTPUT_VARIABLE _path ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(IS_ABSOLUTE "${_path}" AND EXISTS "${_path}")
    file(REAL_PATH "${_path}" _resolved)
    file(SHA256 "${_resolved}" _sha)
    set(_runtime_record "{}")
    _openpfc_json_field(_runtime_record path "${_path}")
    _openpfc_json_field(_runtime_record resolved_path "${_resolved}")
    _openpfc_json_field(_runtime_record sha256 "${_sha}")
    string(JSON _runtime_libraries SET "${_runtime_libraries}" "${_runtime}" "${_runtime_record}")
  else()
    _openpfc_json_field(_runtime_libraries "${_runtime}" unknown)
  endif()
endforeach()
string(JSON _platform SET "${_platform}" compiler_runtime_candidates "${_runtime_libraries}")
set(_linker_version unknown)
if(CMAKE_LINKER)
  execute_process(COMMAND "${CMAKE_LINKER}" --version
    OUTPUT_VARIABLE _linker_version ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
_openpfc_json_field(_platform linker_version "${_linker_version}")

# Probe with the definitions/options of the objects that form the SDK, including
# directory settings captured by those targets. This context is never exported.
add_library(_OpenPFC_SDKHostContext INTERFACE IMPORTED)
set_property(TARGET _OpenPFC_SDKHostContext PROPERTY
  INTERFACE_LINK_LIBRARIES MPI::MPI_CXX)
foreach(_property COMPILE_DEFINITIONS COMPILE_OPTIONS)
  get_target_property(_value openpfc_kernel_obj "${_property}")
  if(NOT _value STREQUAL "_value-NOTFOUND")
    set_property(TARGET _OpenPFC_SDKHostContext PROPERTY "INTERFACE_${_property}" "${_value}")
  endif()
endforeach()

set(OpenPFC_BUILD_CONFIGURATIONS "${CMAKE_CONFIGURATION_TYPES}")
if(NOT OpenPFC_BUILD_CONFIGURATIONS)
  set(OpenPFC_BUILD_CONFIGURATIONS "${CMAKE_BUILD_TYPE}")
endif()
foreach(_configuration IN LISTS OpenPFC_BUILD_CONFIGURATIONS)
  string(TOUPPER "${_configuration}" _upper)
  _openpfc_sdk_probe(_host_probe _probe_status "${_configuration}" _OpenPFC_SDKHostContext)
  if(_probe_status STREQUAL "compiled-linked-ran")
    string(JSON _stdlib GET "${_host_probe}" stdlib)
    if(_stdlib STREQUAL "libstdc++")
      string(JSON _abi GET "${_host_probe}" stdlib_abi)
      set(_abi_definition "_GLIBCXX_USE_CXX11_ABI=${_abi}")
      if(CMAKE_CONFIGURATION_TYPES)
        set(_abi_definition "$<$<CONFIG:${_configuration}>:${_abi_definition}>")
      endif()
      set_property(TARGET openpfc APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
        "${_abi_definition}")
      string(JSON _debug GET "${_host_probe}" stdlib_debug)
      if(_debug STREQUAL "ON")
        set(_debug_definition _GLIBCXX_DEBUG)
        if(CMAKE_CONFIGURATION_TYPES)
          set(_debug_definition "$<$<CONFIG:${_configuration}>:_GLIBCXX_DEBUG>")
        endif()
        set_property(TARGET openpfc APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
          "${_debug_definition}")
      endif()
      _openpfc_manifest_target(_record openpfc)
      string(JSON _targets SET "${_targets}" openpfc "${_record}")
    endif()
  endif()
  set(_flags "{}")
  foreach(_language C CXX CUDA HIP)
    if(_language STREQUAL CUDA AND NOT OpenPFC_SDK_CUDA OR
       _language STREQUAL HIP AND NOT OpenPFC_SDK_HIP)
      continue()
    endif()
    _openpfc_manifest_variables(_language_flags CMAKE_${_language}_FLAGS
      CMAKE_${_language}_FLAGS_${_upper})
    string(JSON _flags SET "${_flags}" "${_language}" "${_language_flags}")
  endforeach()
  foreach(_kind EXE SHARED MODULE STATIC)
    _openpfc_manifest_variables(_link_flags CMAKE_${_kind}_LINKER_FLAGS
      CMAKE_${_kind}_LINKER_FLAGS_${_upper})
    string(JSON _flags SET "${_flags}" "${_kind}" "${_link_flags}")
  endforeach()
  get_directory_property(_directory_options COMPILE_OPTIONS)
  get_directory_property(_directory_definitions COMPILE_DEFINITIONS)
  get_directory_property(_directory_link LINK_OPTIONS)
  _openpfc_json_field(_flags directory_compile_options "${_directory_options}")
  _openpfc_json_field(_flags directory_compile_definitions "${_directory_definitions}")
  _openpfc_json_field(_flags directory_link_options "${_directory_link}")
  set(_manifest "{\"schema_version\":1}")
  _openpfc_json_field(_manifest version "${PROJECT_VERSION}${OpenPFC_VERSION_SUFFIX}")
  _openpfc_json_field(_manifest source_revision "${OpenPFC_SOURCE_REVISION}")
  _openpfc_json_field(_manifest source_state "${OpenPFC_SOURCE_STATE}")
  _openpfc_json_field(_manifest configuration "${_configuration}")
  _openpfc_json_field(_manifest probe_status "${_probe_status}")
  foreach(_key capabilities targets toolchains platform mpi stack components flags)
    string(JSON _manifest SET "${_manifest}" "${_key}" "${_${_key}}")
  endforeach()
  string(JSON _manifest SET "${_manifest}" host_probe "${_host_probe}")
  set(_manifest_dir "${CMAKE_CURRENT_BINARY_DIR}/share/openpfc/${_configuration}")
  file(MAKE_DIRECTORY "${_manifest_dir}")
  file(WRITE "${_manifest_dir}/build-manifest.json" "${_manifest}\n")
endforeach()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/share/openpfc/$<CONFIG>/build-manifest.json"
  DESTINATION share/openpfc)
