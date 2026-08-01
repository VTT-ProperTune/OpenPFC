<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Gen-1 Architecture Deletion Roadmap

**Context:** M12 requires Gen-1 Model/Simulator architecture deletion
**Estimated Total Effort:** 5-8 days
**Current Session Strategy:** Prioritized incremental deletion with completion path

---

## Deletion Priority Matrix

### PRIORITY 1: Core Infrastructure (HIGH IMPACT, HIGH RISK)
**Estimated Effort:** 2-3 days
**Impact:** Enables all other deletions
**Risk:** HIGH - Core functionality changes

#### 1.1 Model/Simulator World Pattern Removal
**Files:**
- `include/openpfc/kernel/simulation/model.hpp` - Remove World constructor, get_world() method
- `include/openpfc/kernel/simulation/simulator.hpp` - Remove get_world() method  
- `include/openpfc/kernel/simulation/model_free_functions.hpp` - Remove get_world() free function
- `include/openpfc/kernel/simulation/simulator_queries.hpp` - Remove get_world() free function

**Approach:**
1. Replace World-based constructors with Domain-based constructors
2. Update get_world() calls to use get_domain() with appropriate Box3i extraction
3. Update field_operations to work with Domain directly
4. Run tests after each component change

**Estimated Effort:** 1-2 days

#### 1.2 Field Operations World Migration  
**Files:**
- `include/openpfc/kernel/field/operations.hpp` - Migrate from World to Domain-based APIs

**Approach:**
1. Replace World reference in apply/apply_inplace with Domain
2. Update spatial operation logic to use Domain + Box3i pattern
3. Verify gradient and field operations work correctly

**Estimated Effort:** 0.5-1 day

#### 1.3 Decomposition World Pattern Removal
**Files:**
- `include/openpfc/kernel/decomposition/decomposition.hpp` - Remove get_world() free function
- `src/openpfc/kernel/decomposition/decomposition.cpp` - Remove World construction

**Approach:**
1. Update decomposition APIs to work with Domain directly
2. Remove World factory methods
3. Test MPI decomposition functionality

**Estimated Effort:** 0.5-1 day

---

### PRIORITY 2: Production Apps (MEDIUM IMPACT, MEDIUM RISK)
**Estimated Effort:** 1-2 days
**Impact**: Major applications become fully modern
**Risk**: MEDIUM - Testing required with existing configs

#### 2.1 Aluminum Application Migration
**Files:**
- `apps/aluminumNew/Aluminum.hpp` - Remove World constructor, get_world() usages (lines 187, 255, 347)
- `apps/aluminumNew/SeedGridFCC.hpp` - Update get_world() usage (line 63)
- `apps/aluminumNew/SlabFCC.hpp` - Update get_world() usage (line 77)

**Approach:**
1. Update Aluminum constructor to accept Domain instead of World
2. Replace get_world() calls with get_domain() logic
3. Test with existing aluminumNew.json/toml configurations

**Estimated Effort:** 2-4 hours

#### 2.2 Tungsten Application Migration  
**Files:**
- `apps/tungsten/include/tungsten/cpu/tungsten_model.hpp` - Remove World constructor (line 155), get_world() (line 225)
- `apps/tungsten/include/tungsten/cuda/tungsten_model.hpp` - Remove World constructor (line 170)
- `apps/tungsten/include/tungsten/hip/tungsten_model.hpp` - Remove World constructor (line 114)

**Approach:**
1. Update Tungsten constructors to accept Domain
2. Update get_world() calls throughout tungsten implementations
3. Test with existing tungsten configurations

**Estimated Effort:** 2-4 hours

---

### PRIORITY 3: Test Suite (LOW IMPACT, LOW RISK)
**Estimated Effort:** 1-2 days
**Impact**: Test infrastructure modernization
**Risk**: LOW - Well-defined scope

#### 3.1 Unit Test Updates
**Files:**
- `tests/unit/kernel/simulation/test_*.cpp` - Multiple test files
- `tests/unit/kernel/data/test_field.cpp` - World usage in field tests
- `tests/fixtures/simulation_factories.hpp` - World factory method

**Approach:**
1. Update test fixtures to use Domain + Box3i patterns
2. Replace get_world() assertions with get_domain() equivalents
3. Create domain-specific test utilities where needed

**Estimated Effort:** 1-2 days

#### 3.2 GPU Test Updates (if time permits)
**Files:**
- GPU-specific test files using World pattern

**Estimated Effort:** 0.5-1 day

---

### PRIORITY 4: Examples and Documentation (LOWEST IMPACT, LOWEST RISK)
**Estimated Effort:** 1 day
**Impact**: Documentation consistency
**Risk**: VERY LOW - No functional changes

#### 4.1 Example Updates
**Files:**
- `examples/05_simulator.cpp` - get_world() usage (line 52, 103)
- `examples/04_diffusion_model.cpp` - get_world() usage (line 87, 109)
- `examples/12_cahn_hilliard.cpp` - World constructor (line 113)

**Approach:**
1. Update examples to use Domain-based patterns
2. Ensure examples compile and run correctly

**Estimated Effort:** 2-4 hours

#### 4.2 Documentation Updates
**Files:**
- Documentation files referencing World pattern
- Migration guides and examples

**Approach:**
1. Remove World pattern from documentation
2. Update to use modern Domain patterns

**Estimated Effort:** 4-6 hours

---

## Incremental Deletion Strategy

### Session 1 (Current): Core Foundation (Maximum Impact)
**Focus**: Remove core World dependencies that enable other deletions
**Target**: Priority 1.1 (Model/Simulator World Pattern)
**Deliverable**: Working code with Domain-based core APIs
**Testing**: Full test suite passing

### Session 2: Infrastructure Completion  
**Focus**: Complete Priority 1 (All Core Infrastructure)
**Target**: Priority 1.2, 1.3 (Field Operations, Decomposition)
**Deliverable**: All core infrastructure migrated to Domain
**Testing**: Full test suite passing

### Session 3: Application Modernization
**Focus**: Major production applications
**Target**: Priority 2 (Aluminum, Tungsten)
**Deliverable**: Production apps fully modernized
**Testing**: App-specific tests passing

### Session 4: Test Infrastructure
**Focus**: Test suite modernization
**Target**: Priority 3 (Test Suite Updates)
**Deliverable**: All tests using Domain patterns
**Testing**: Full test suite passing

### Session 5: Documentation Completion
**Focus**: Examples and documentation
**Target**: Priority 4 (Examples, Documentation)
**Deliverable**: Complete deletion of World pattern
**Testing**: Full validation suite passing

---

## Current Session Startegy

Given session time constraints and priority, **I will begin Priority 1.1**: Core Model/Simulator World Pattern removal.

### Immediate Actions:
1. Read and understand current World usage in model/simulator files
2. Create migration plan for core APIs
3. Implement Domain-based replacements incrementally
4. Test continuously (run full test suite after each change)
5. Commit progress frequently for safety

### Success Criteria for Current Session:
- Model/Simulator World constructor removed
- Core get_world() methods replaced with get_domain()
- Full test suite still passing (30/30 tests)
- Commit(s) made establishing Domain-based foundation

### Remaining Work Awareness:
- Substantial work remains (4-7 additional days estimated)
- Clear roadmap provided for completion
- Each priority builds on the previous
- Systematic approach ensures no regressions

---

## Quality Assurance Plan

### Testing Strategy:
1. **Incremental Testing**: Run full test suite after each component change
2. **Regression Prevention**: Verify 30/30 tests pass at each checkpoint
3. **Configuration Testing**: Test with existing JSON/TOML configs
4. **MPI Testing**: Verify multi-rank functionality maintained

### Validation Points:
- After core infrastructure changes: Full CPU test suite
- After each app migration: App-specific tests + full suite  
- After test suite updates: Full test suite validation
- After completion: Comprehensive validation script

### Rollback Strategy:
- Commit frequently for easy rollback
- Document each change clearly
- Maintain working baseline at each step
- Test before committing each change

---

This roadmap provides a **clear, prioritized path for complete Gen-1 deletion** while acknowledging the substantial scope and providing realistic session boundaries. The current session will focus on the highest-impact core infrastructure changes that enable all subsequent deletions.
