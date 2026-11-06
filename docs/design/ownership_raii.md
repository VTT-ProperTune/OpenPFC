<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Ownership, views, and the C++20 baseline

This is the ownership classification for issue #171. A raw pointer is not
a bug. Smart pointers express ownership. A contiguous non-owning sequence
is a `std::span` when that does not hide geometry or an asynchronous
lifetime. No hot loop was rewritten, and clang-tidy was not bulk-applied.

## Language decision

Option A: stay on C++20.

`cmake/CompilerSettings.cmake` and `tools/openpfc/model.cmake` set
`CMAKE_CXX_STANDARD` to 20 and require it.
`cmake/OpenPFCCompatibility.cmake` probes consumers with the same flag.
The compatibility workflow still builds GCC 11, GCC 12, and GCC 14.
Pull-request CI builds GCC 13. Coverage builds GCC 11. The CUDA
compile-only job uses `nvidia/cuda:12.6.0-devel-ubuntu24.04`. The HIP
compile-only job uses `rocm/dev-ubuntu-24.04:6.2-complete`. Those floors
are the reason not to move the supported baseline to C++23. Option B, a
host C++23 dialect with a lower device dialect, would be a mixed dialect
this tree does not have. Option C would drop GCC 11. LUMI was not
measured for this decision. The Tohtori CPU configure for this change
printed GNU 15.2.0 and C++ standard 20. That observation is not a LUMI
result and does not raise the floor. No HIP target in this tree pins
`CMAKE_CXX_STANDARD` to 17. Device translation units were not
recompiled for this note.

## Views

`SparseExchange::bind` and its stored field are `std::span`. `start`
keeps that span until `finish`. The caller must keep the buffer alive
across that pair. The test is `SparseExchange HostSpace: start/finish
and unbound start throw` in
`tests/integration/scenarios/parallel_scaling/test_comm_sparse_exchange.cpp`.

`face_flux` takes `std::span` for the owned coefficient, potential, and
right-hand side. The six halo faces stay `std::array<const double *, 6>`.
They are six separate buffers, not one contiguous sequence.

`FieldView` and `FieldOutput` keep pointer, size, and, for the view,
extents, spacing, and origin. Replacing the type with `std::span` would
drop that geometry. `span()` returns the storage portion and does not
own it. CPU `DataBuffer::copy_from_host` and `copy_to_host` accept
`std::span`. The pointer-and-size overloads stay for a caller that
already holds a C pointer. The GPU buffer has the same span overloads.
Its destructor was not changed.

`IHostFFT::forward` and `backward` still take `RealVector` and
`ComplexVector`. Every backend checks those sizes against the inbox and
outbox, and the call is collective. A second virtual span contract is
not practical in this pass. Callers that own a vector already pass it.
No FFT loop was rewritten.

## MPI owners

`MPI_File_guard` and `MPI_Type_guard` are `noexcept`. A close or free
failure calls `abort_on_mpi_error`. `environment::~environment` does the
same for `MPI_Finalize`.

`BinaryWriter` and `BinaryReader` used to throw from `noexcept(false)`
destructors. `close()` now reports `MPI_Type_free`. The destructor is
`noexcept` and aborts on that failure. A second `close()` does nothing.
The flag stays set when the free fails, so the destructor can retry.
`close()` is local, not collective. `set_domain` and the filetype
rebuild use the reporting path. `ResultsWriter::close` is the virtual
hook. Its destructor stays `noexcept(false)` so a user type whose
destructor calls a potentially throwing function, such as
`std::ofstream::close` in the results-writer example, still overrides
it. Library writers do not throw from destructors. The snapshot series
calls `writer->close()` on every rank, agrees, and then writes the
manifest.

`communicator` owns the `MPI_Comm` with `std::shared_ptr` and a deleter.
The `new MPI_Comm` does not escape the constructor. `MPI_COMM_WORLD`,
`MPI_COMM_SELF`, and `MPI_COMM_NULL` are not freed. Any other
`MPI_Comm_free` failure aborts. The deleter does not throw.

`HostPersistentFaces` owns persistent `MPI_Request` values.
`free_all_requests` is `noexcept` and aborts when `MPI_Request_free`
fails. That function runs from the destructor, from noexcept
move-assignment, and from the constructor's failure path. `m_buf` is
not owned. It is the caller's field.

`create_face_type` frees a datatype if `MPI_Type_commit` fails. A free
failure aborts before the commit error is thrown. The returned type is
an `MPI_Type_guard`.

`HostFacesHalo::m_pending_field` and `m_bound_buf` are non-owning. The
bound field must outlive `start` until the matching wait.

`DiagnosticsSeries` owns a `FILE *`. `close()` reports `fclose` failure.
The destructor calls `close()` and does not throw.

`HidGuard` in the HDF5 writer owns an `hid_t` and calls the close
function from its destructor without throwing. The close status is
discarded. That is the existing RAII wrapper. It was not given a
`unique_ptr` solely to change the spelling.

## Device owners

`GPUDataBuffer` owns `m_device_ptr`. The destructor and noexcept
move-assignment free it. A free failure is `failed_release()`, not an
exception. A `unique_ptr` would hide that status, so the member stays a
pointer inside this RAII type.

`CudaComplexLineFFT::Buffer` and the HIP `DeviceBuffer` free with
`cudaFree` or `hipFree` and discard the status. The destructor does not
throw. Copy is deleted. The same pattern covers the rocFFT work buffer.
This commit does not rebuild those translation units, so the free
status is not newly reported.

`SpectralETDSystem::release_graph` is `noexcept`. The destructor calls
it. The graph pointer is cleared after destroy.

Vlasov `device_free` is `noexcept` and discards `hipFree`.
`~DeviceBrick` uses it. Those pointers are application scratch owned by
the brick, not a new library handle.

Padded device halo exchange frees scratch with `cudaFree` or `hipFree`
and discards the status inside the exchange object. The pointer does
not escape as a bare owner.

Test translation units that call `cudaMalloc` or `hipMalloc` free that
scratch before return. They are not library owners.

## What this pass did not do

CPU `DataBuffer` still defaults its special members. The defaults state
that the host buffer is copyable. The GPU specialization deletes copy.
Removing the defaults would not make that difference easier to see.

No performance-sensitive loop was rewritten. No algorithm changed.
CUDA and HIP sources other than the `copy_to_host(std::span)` forwarder
in `databuffer_gpu.hpp` were not edited, and those device targets were
not recompiled here. The installed consumer is
`tests/packaging/consumer`, configured with `find_package(OpenPFC)`
after this library is installed. That build is host-only.
