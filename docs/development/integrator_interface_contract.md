<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Integrator interface contract

This document defines the semantic requirements an implementer must satisfy to
build a valid time-integration `Integrator` for OpenPFC. It exists as a
foundational specification: before any concrete interface extraction work
proceeds, this documents what "a valid integrator" actually means in terms an
implementer can check their code against.

## How this differs from the time integration architecture contract

[`docs/concepts/time_integration_contract.md`](../concepts/time_integration_contract.md)
is the conceptual, architect-facing overview: it describes the `Simulator`
class's `begin_integrator_step()`/`end_integrator_step()` hooks, when to use
`step_with_physics()` versus calling the hooks directly, and the overall
relationship between integrators, models, and the simulator framework. This
document is narrower and implementer-facing: it enumerates the concrete
semantic requirements (ownership, timing, thread-safety, serialization) an
integrator implementation must satisfy, grounded in the existing behavior of
`Simulator` and the `EulerStepper`/`Heat3D` stepping patterns, so an
implementer has a checklist to validate a new integrator against rather than
an architectural narrative to interpret.

## 1. Time state ownership requirements

- The integrator must not own the simulation's current time or step size directly; these are owned by `Time` (see [`include/openpfc/kernel/simulation/simulator.hpp`](../../include/openpfc/kernel/simulation/simulator.hpp)) and passed to the integrator's step function as arguments.
- On a successful step, time must advance from `current` to `current + dt`; the integrator's step function must return this new time value, matching the pattern in [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp) (`return t + m_dt;`).
- If a step is rejected (future adaptive-stepping scope), time must not change: the caller is responsible for not advancing `Time` when the integrator reports rejection (see [Error control API requirements](#5-error-control-api-requirements)).
- The integrator must remain compatible with the time state orchestration pattern in [`include/openpfc/kernel/simulation/simulator.hpp`](../../include/openpfc/kernel/simulation/simulator.hpp): it advances state via `Simulator::step_with_physics()` or the explicit `begin_integrator_step()`/`end_integrator_step()` hooks, never by mutating `Time` itself.

## 2. Stage evaluation contract requirements

- Stage evaluation must only read the model's owned region (interior cells) plus the already-exchanged halo/ghost region; it must not assume access to any other rank's data.
- Ghost zones must be fully exchanged before a stage evaluation that reads them begins -- an integrator must not read halo cells until the halo exchange covering them has completed.
- Interior stage evaluations that do not depend on halo data must be safe to run concurrently with an in-flight (non-blocking) halo exchange, matching the overlap pattern demonstrated in [`apps/heat3d/src/cpu/heat3d_fd_manual.cpp`](../../apps/heat3d/src/cpu/heat3d_fd_manual.cpp) (`start_halo_exchange()` before the interior stencil pass, `finish_halo_exchange()` only before the pass that reads the halo ring).
- The integrator must be compatible with the communication coordination pattern in [`apps/heat3d/src/cpu/heat3d_fd_manual.cpp`](../../apps/heat3d/src/cpu/heat3d_fd_manual.cpp): it must not introduce its own ad hoc communication that bypasses the model/gradient evaluator's halo exchanger.

## 3. Workspace management requirements

- Scratch buffers (RHS evaluation buffers, per-stage buffers) must be allocated once, in the integrator's constructor, and deallocated in its destructor -- no per-step allocation is allowed, matching `EulerStepper`'s `m_du` buffer in [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp), pre-allocated in the constructor and reused across every call to `step()`.
- Buffers must be reused across steps: an integrator that reallocates scratch storage inside `step()` does not satisfy this contract.
- Multi-stage methods must pre-allocate one scratch buffer per stage at construction time (see `ExplicitRKStepper`'s per-stage `m_k` buffers), not allocate them lazily on first use.
- The integrator must follow the persistent scratch buffer member pattern used by `EulerStepper` in [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp): scratch state lives in the integrator instance, not in static or thread-local storage.

## 4. Halo/boundary preparation requirements

- The integrator itself does not perform halo exchange; the model/gradient evaluator it is composed with owns that responsibility, and the integrator must call into it (or rely on the caller having done so) before accessing ghost data.
- Halo exchange must complete before the first stage evaluation that reads ghost data executes; an integrator must not reorder its stages such that a ghost-dependent evaluation runs ahead of the exchange that supplies it.
- Overlapping communication with interior-region computation is supported and expected where the discretization allows it, as demonstrated in [`apps/heat3d/src/cpu/heat3d_fd_manual.cpp`](../../apps/heat3d/src/cpu/heat3d_fd_manual.cpp): start the halo exchange, compute the interior (halo-independent) region, then finish the exchange before computing the border region.
- Boundary condition application must happen after the time update for that step, never before -- an integrator must not apply boundary conditions to a state that has not yet been advanced to the new time.

## 5. Error control API requirements

- Current stepper implementations (see [`include/openpfc/kernel/simulation/steppers/euler.hpp`](../../include/openpfc/kernel/simulation/steppers/euler.hpp)) do not implement adaptive error estimation; a valid integrator today is not required to reject steps.
- Where a step-rejection protocol does exist, the *caller* is responsible for restoring state and not advancing time on rejection or on `NoDecision` -- the integrator's role is limited to producing evidence / reporting a verdict, not performing the rollback itself.
- Any error-control API an integrator exposes must remain compatible with the plain `t = stepper.step(t, u)` time-advance pattern used throughout the current stepper classes; it must not require the caller to change how it invokes `step()` on the success path.
- Method-independent error evidence and controller normalization live in [`include/openpfc/kernel/integrator/error_evidence.hpp`](../../include/openpfc/kernel/integrator/error_evidence.hpp): integrators produce `ErrorEvidence` (embedded-pair, residual / a-posteriori, or a documented method-specific extension hook); optional `reduce_error_evidence` promotes `AggregationScope::RankLocal` to `AlreadyReduced` (single-rank identity; already-reduced is not double-reduced); `normalize_error_evidence` returns a dimensionless metric plus `StepAttemptVerdict` (`Accept` / `Reject` / `NoDecision`) without computing a next `dt`.
- Transient per-step `ErrorEvidence` is **not** checkpointed. Only future controller history that affects subsequent decisions may persist (restore is not implemented in this seam).
- Still out of scope for this seam: the embedded RK stepper body (#162), adaptive tolerance / step-bound JSON schema (#163), full Simulator accept/reject/output orchestration, and on-hold `IntegratorResult` (#141).

## 6. Checkpoint/restart semantics requirements

- The current simulation time and step size (owned by `Time`, not the integrator -- see [Time state ownership requirements](#1-time-state-ownership-requirements)) must be serializable as part of any checkpoint.
- Any integrator-specific workspace contents that affect the *next* step's result (e.g. multi-step method history, not the per-step scratch buffers described in [Workspace management requirements](#3-workspace-management-requirements), which are pure scratch and need not be preserved) must be serializable.
- Serialization must round-trip: restoring a checkpoint and resuming stepping must produce the same subsequent state as if the run had not been interrupted, for any integrator that carries state across steps beyond the current `Time`.
- Integrators that carry no cross-step state beyond `Time` (e.g. `EulerStepper`, which recomputes everything from `t` and `u` each step) have nothing additional to serialize and trivially satisfy this requirement.
