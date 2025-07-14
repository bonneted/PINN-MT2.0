# Deep Notched Plate PINN Implementation

## Overview
This file documents the modifications made to `pinn_inverse.py` to implement the deep notched plate problem using Physics-Informed Neural Networks (PINNs) with coordinate mapping and adaptive sampling.

## Key Changes Made

### 1. Problem Configuration
- Changed from side-loaded plate to deep notched plate geometry
- Added support for both isotropic and orthotropic material laws
- Integrated coordinate mapping for complex geometry handling
- Added notch-specific boundary conditions

### 2. Coordinate Mapping
- Implemented `coordMap()` function for transforming computational coordinates to physical coordinates
- Added `inv_coord_map()` function for inverse coordinate mapping using Gauss-Newton solver
- Implemented `GaussNewton` class for optimization
- Added tensor mapping functions (`tensMap`, `calcNormal`) for stress transformations

### 3. Geometry and Boundary Conditions
- Replaced rectangular geometry with `ListPointCloud` for notched geometry
- Implemented free surface boundary conditions for notch edges
- Added integral stress boundary conditions
- Implemented hard boundary conditions with coordinate mapping

### 4. PDE Formulation
- Modified PDE to work with coordinate-mapped derivatives
- Added constitutive stress relationships for isotropic/orthotropic materials
- Implemented momentum balance equations with mapped coordinates

### 5. Adaptive Sampling
- Added `adaptive_sampling_grid()` function for loss-based sampling
- Implemented iterative training with adaptive point addition
- Added loss gradient-based weight calculation

### 6. Network Architecture
- Configured for SPINN (Separable PINN) architecture
- Added utility functions for parameter counting
- Simplified network configuration for deep notched problem

## File Dependencies
- `deep_notched_60x88.txt` - Coordinate mapping points
- `data_fem/isotropic_100x100.dat` - FEM reference solution (isotropic)
- `data_fem/orthotropic_100x100.dat` - FEM reference solution (orthotropic)

## Usage
The modified script supports the following key arguments:
- `--material_law`: Choose between 'isotropic' or 'orthotropic'
- `--stress_bc`: Enable stress boundary conditions
- `--n_DIC`: Number of DIC measurement points
- `--net_type`: Network type (currently supports 'spinn')
- `--bc_type`: Boundary condition type ('hard')

## Training Process
1. Initial training with fixed sampling points
2. Adaptive sampling based on PDE loss distribution
3. Iterative refinement with loss-weighted training
4. Multiple cycles of sampling and training for convergence

## Key Features
- Coordinate mapping for complex geometries
- Adaptive sampling for efficient training
- Support for multiple material laws
- Free surface and integral boundary conditions
- JAX-based automatic differentiation with coordinate transformations
