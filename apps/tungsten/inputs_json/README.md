<!--
SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Tungsten PFC Model - JSON Input Files

This directory contains JSON input files for the Tungsten PFC model. These files follow the same structure as the TOML files in the `inputs_toml` directory, with the following sections:

## Structure

Each JSON file follows this structure:

```json
{
  "model": {
    "name": "tungsten",
    "params": {
      // Model parameters
    }
  },
  "domain": {
    // Domain configuration
  },
  "timestepping": {
    // Time stepping parameters
  },
  "fields": [
    // Field definitions
  ],
  "initial_conditions": [
    // Initial condition definitions
  ],
  "boundary_conditions": [
    // Boundary condition definitions
  ]
}
```

## Available Input Files

1. `tungsten_fixed_bc.json` - Fixed boundary conditions with seed grid
2. `tungsten_moving_bc.json` - Moving boundary conditions with seed grid
3. `tungsten_moving_bc_options.json` - Moving boundary with initial position options
4. `tungsten_performance.json` - Performance testing configuration
5. `tungsten_restart.json` - Restart from checkpoint
6. `tungsten_single_seed.json` - Single seed initial condition

## Model Parameters

The model parameters are identical to those in the TOML files. See the TOML files for detailed descriptions of each parameter.

## Domain Configuration

- `Lx`, `Ly`, `Lz`: Grid size in each dimension
- `dx`, `dy`, `dz`: Grid spacing in each dimension
- `origin`: Coordinate system origin ("center" or "corner")

## Time Stepping

- `t0`: Initial time
- `t1`: Final time
- `dt`: Time step size
- `saveat`: Save interval

## Field Definitions

Each field definition has:
- `name`: Field name
- `data`: Output file path pattern

## Initial Conditions

Supported initial condition types:
- `constant`: Fill entire field with constant value
- `single_seed`: Create a single crystal seed
- `seed_grid`: Create a grid of crystal seeds
- `random_seeds`: Create random crystal seeds
- `from_file`: Load field from file

## Boundary Conditions

Supported boundary condition types:
- `fixed`: Maintain field values at boundaries
- `moving`: Allow boundaries to move over time
