# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise manifest serialization/config selection without a GPU toolchain.

These compilerless fixtures do not qualify real CUDA/HIP installations; the
installed consumer CTest covers the actual SDK built by scripts/build.sh.
"""
import json
import subprocess
from pathlib import Path

import pytest

CMAKE = Path(__file__).resolve().parents[2] / "cmake"


@pytest.mark.parametrize("backend", ["cpu", "cuda", "hip"])
def test_manifest_targets_and_configurations(tmp_path, backend):
    source = tmp_path / "source"
    source.mkdir()
    build = tmp_path / "build"
    target = {"cpu": "", "cuda": "openpfc_gpu_kernels", "hip": "openpfc_hip_kernels"}[
        backend
    ]
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.21)\n"
        "project(manifest_fixture VERSION 0.5.0 LANGUAGES NONE)\n"
        'set(CMAKE_CONFIGURATION_TYPES "Debug;Release")\n'
        "set(CMAKE_CXX_FLAGS [==[-DQUOTED=\"a\\b\";list\nsecond line]==])\n"
        'set(CMAKE_CXX_FLAGS_DEBUG "-g")\n'
        'set(CMAKE_CXX_FLAGS_RELEASE "-O3")\n'
        'set(CMAKE_CXX_COMPILER_LAUNCHER "env;API_TOKEN=not-for-install;ccache")\n'
        "set(CMAKE_CUDA_ARCHITECTURES 90)\n"
        "set(CMAKE_HIP_ARCHITECTURES gfx90a)\n"
        'set(CMAKE_CUDA_FLAGS "--fmad=false --generate-code=arch=compute_90,code=sm_90")\n'
        'set(CMAKE_HIP_FLAGS "-ffp-contract=off")\n'
        # Stale cached enable/toolkit settings must not override actual targets.
        "set(OpenPFC_ENABLE_CUDA ON)\n"
        "set(OpenPFC_ENABLE_HIP ON)\n"
        "set(OpenPFC_ENABLE_HEFFTE OFF)\n"
        "set(OpenPFC_HAS_FFTW3 OFF)\n"
        "set(OpenPFC_ENABLE_HDF5 OFF)\n"
        'set(OpenPFC_SOURCE_REVISION "unknown")\n'
        'set(OpenPFC_SOURCE_STATE "unknown")\n'
        'set(OpenPFC_VERSION_SUFFIX "")\n'
        "add_library(openpfc INTERFACE)\n"
        "add_library(openpfc_kernel_obj INTERFACE)\n"
        "add_library(openpfc_frontend_obj INTERFACE)\n"
        "add_library(MPI::MPI_CXX INTERFACE IMPORTED)\n"
        "add_library(MPI::MPI_C INTERFACE IMPORTED)\n"
        'target_compile_options(openpfc INTERFACE "$<$<CONFIG:Debug>:-g>")\n'
        + (f"add_library({target} INTERFACE)\n" if target else "")
        + f'include("{CMAKE / "OpenPFCCompatibility.cmake"}")\n'
        "function(_openpfc_sdk_probe out status configuration link_target)\n"
        '  set(${out} "{}" PARENT_SCOPE)\n'
        '  set(${status} "unknown: compilerless serialization fixture" PARENT_SCOPE)\n'
        "endfunction()\n"
        f'include("{CMAKE / "BuildManifest.cmake"}")\n'
    )
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja Multi-Config"],
        check=True, capture_output=True, text=True,
    )
    for config, flag in [("Debug", "-g"), ("Release", "-O3")]:
        path = build / "share/openpfc" / config / "build-manifest.json"
        manifest = json.loads(path.read_text())
        assert "not-for-install" not in path.read_text()
        assert "credential-like diagnostic redacted" in manifest["toolchains"]["CXX"]["CMAKE_CXX_COMPILER_LAUNCHER"]
        assert manifest["schema_version"] == 1
        assert manifest["configuration"] == config
        assert manifest["capabilities"]["cuda"] == (backend == "cuda")
        assert manifest["capabilities"]["hip"] == (backend == "hip")
        assert not manifest["capabilities"]["heffte"]
        assert not manifest["capabilities"]["fftw"]
        assert not manifest["capabilities"]["hdf5"]
        assert manifest["flags"]["CXX"][f"CMAKE_CXX_FLAGS_{config.upper()}"] == flag
        assert manifest["flags"]["CXX"]["CMAKE_CXX_FLAGS"] == '-DQUOTED="a\\b";list\nsecond line'
        assert manifest["targets"]["openpfc"]["INTERFACE_COMPILE_OPTIONS"] == "$<$<CONFIG:Debug>:-g>"
        for language in ["CUDA", "HIP"]:
            enabled = backend == language.lower()
            assert manifest["toolchains"][language]["enabled"] == enabled
            if enabled:
                assert manifest["toolchains"][language]["architectures"] == {
                    "CUDA": "90", "HIP": "gfx90a"
                }[language]
                assert f"CMAKE_{language}_FLAGS" in manifest["flags"][language]
            else:
                assert manifest["toolchains"][language]["architectures"] == "none"
                assert language not in manifest["flags"]
        prefix = tmp_path / config
        subprocess.run(
            ["cmake", "--install", str(build), "--config", config, "--prefix", str(prefix)],
            check=True, capture_output=True, text=True,
        )
        installed = prefix / "share/openpfc/build-manifest.json"
        assert json.loads(installed.read_text()) == manifest
