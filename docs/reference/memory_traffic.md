<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Allocation and copy accounting

`#include <openpfc/kernel/diagnostics/memory_traffic.hpp>` provides optional,
caller-owned accounting of successful allocations and explicit host/device
copies in namespace `pfc::diagnostics`. It answers questions such as "did a
warmed step allocate scratch?" or "did this retry upload anything again?". It
does not measure cache, memory-bus or HBM traffic.

## Observing a region

```cpp
namespace dg = pfc::diagnostics;
dg::Traffic traffic;
{
  dg::Scope observing(&traffic); // current thread only
  stepper.step(t, u);
}
auto device = traffic.allocations(dg::Space::Device);
auto h2d = static_cast<std::size_t>(dg::Direction::HostToDevice);
std::uint64_t uploaded = traffic.copies.bytes[h2d];
```

`Scope` makes an observation `dg::current` for the calling thread and restores
the previous one on exit, so scopes nest. `Scope(nullptr)` and
`PauseObservation` disable observation, including an enclosing one, until they
end. The observation must outlive every scope that names it.

## Allocations

`allocations(Space::Host)` and `allocations(Space::Device)` return an
`AllocationTotals` snapshot: successful request/free counts, requested/freed
bytes, initial live bytes, current live bytes and the peak of simultaneously
live requested payload bytes. A failed request is not an allocation. A runtime
free failure (`AllocationToken::failed_release()`) leaves its payload live and
sets `complete=false`, which stays set across interval resets.

A successful allocation returns a move-only `AllocationToken` that shares the
ledger, so it may outlive the `Traffic` object. Release it only after the
matching free succeeded. `reset_interval()` clears request, free and copy
counters at a quiescent point and keeps all live allocations as the new
initial and peak baseline; it never erases ownership.

GPU `DataBuffer` (CUDA and HIP) reports every device allocation, resize and
free. Resize records the new allocation before freeing the old one, so their
overlap appears in the peak. Buffers created outside an observation do not
acquire tokens retroactively.

Host allocations are counted only when the consumer installs its own global
`new`/`delete` provider that calls
`dg::successful_allocation(Space::Host, bytes)` after each successful
allocation, stores the token with it and calls `release()` after the actual
free. The library never replaces the global allocation functions. The
[unit-test provider](../../tests/fixtures/allocation_provider.hpp) shows this
ABI; link such a provider only into an isolated executable. It counts requested
object bytes, not allocator headers, padding, `malloc`, runtime pools or RSS.

## Copies

`copies.calls[direction]` and `copies.bytes[direction]`, indexed by
`Direction` (`HostToHost`, `HostToDevice`, `DeviceToHost`, `DeviceToDevice`),
count successful explicit transfers recorded with `dg::copy(direction,
bytes)`. GPU `DataBuffer::copy_from_host`, `copy_to_host` and `to_host` record
their uploads and downloads. Zero-length copies are not counted.

Users of the counters include the RK stage allocation test
(`tests/unit/kernel/simulation/steppers/test_rk_stage_allocations.cpp`) and the
resident GPU RK integration check (`tests/integration/rk_coupled/`).
