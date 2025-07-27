# Truss Peak Power Optimization

This repository provides a modular and well-documented implementation for truss topology optimization with respect to peak power, including penalization strategies. The code is organized for clarity and ease of extension, and is inspired by the example at https://gitlab.com/ma.shenyuan/truss_exp/-/blob/main/example/peak_power.ipynb.

## Structure

- `run_truss_peak_power_optimization.py`: **Main script** for running all truss optimization and plotting experiments. This is the entry point for most users.
- `peak_power_calculations.py`: Contains the `PeakPowerCalculator` class, which implements all peak power calculations (uniform, optimized, penalized, and eta sweep).
- `truss_visualization.py`: Contains the `TrussPlotter` class for visualizing truss structures and plotting peak power vs penalization parameter.
- `truss_fem_core.py`: Core truss and FEM model definitions (contains `nodes`, `elements`, `truss`).

## How to Use

1. **Install dependencies** (see below).
2. **Run the main script:**
   ```bash
   python run_truss_peak_power_optimization.py
   ```
   This will:
   - Compute and print peak power for uniform, optimized, and penalized trusses.
   - Sweep over multiple penalization parameters (eta) and report the best result.
   - Plot the effect of eta on peak power.
   - Show a 2x2 comparison of truss structures (uniform, optimized, penalized for eta=1, penalized for best eta).

## Code Overview

- All major steps in the main script are clearly commented and sectioned:
  1. Problem and model settings
  2. Geometry and FEM assembly
  3. Load definition
  4. Peak power calculations (uniform, optimized, penalized)
  5. Visualization (plots)
- Utility and plotting classes are documented with docstrings and inline comments.
- The code is modular: you can easily swap out or extend optimization and plotting routines.

## Dependencies
- `picos` (for optimization)
- `mosek` (solver, or another compatible solver)

Install dependencies with:
```bash
pip install numpy matplotlib picos
# For mosek, see https://www.mosek.com/products/mosek/ for installation instructions
```

## Extending or Modifying
- To add new optimization strategies, extend `PeakPowerCalculator` in `peak_power_calculations.py`.
- For new visualizations, add methods to `TrussPlotter` in `truss_visualization.py`.
- For new truss geometries, modify the node and element definitions in the main script.

## Reference
- Original reference notebook(which doesnot work as intended): [peak_power.ipynb](https://gitlab.com/ma.shenyuan/truss_exp/-/blob/main/example/peak_power.ipynb)

## Contact
For questions or contributions, please open an issue or contact me. 
