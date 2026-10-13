# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Capture the sources that produced this build, not the consumer checkout.
# Non-Git source archives remain usable but cannot claim a Git identity.
set(OpenPFC_SOURCE_REVISION "unknown")
set(OpenPFC_SOURCE_STATE "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}"
    rev-parse --show-toplevel OUTPUT_VARIABLE _openpfc_git_root
    OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _openpfc_root_result
    ERROR_QUIET)
  file(REAL_PATH "${PROJECT_SOURCE_DIR}" _openpfc_source_root)
  if(_openpfc_root_result EQUAL 0)
    file(REAL_PATH "${_openpfc_git_root}" _openpfc_git_root)
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}"
    rev-parse HEAD OUTPUT_VARIABLE _openpfc_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _openpfc_git_result
    ERROR_QUIET)
  if(_openpfc_root_result EQUAL 0 AND
      _openpfc_git_root STREQUAL _openpfc_source_root AND
      _openpfc_git_result EQUAL 0 AND _openpfc_revision MATCHES "^[0-9a-f]+$"
      AND NOT _openpfc_revision STREQUAL "")
    string(LENGTH "${_openpfc_revision}" _openpfc_revision_length)
    if(_openpfc_revision_length EQUAL 40)
      set(OpenPFC_SOURCE_REVISION "${_openpfc_revision}")
      execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}"
        status --porcelain --untracked-files=normal
        OUTPUT_VARIABLE _openpfc_dirty RESULT_VARIABLE _openpfc_status_result
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(_openpfc_status_result EQUAL 0)
        if(_openpfc_dirty STREQUAL "")
          set(OpenPFC_SOURCE_STATE "clean")
        else()
          set(OpenPFC_SOURCE_STATE "dirty")
        endif()
      endif()
    endif()
  endif()
endif()

