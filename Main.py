# ------------------------------------------------------------
# Main script for truss peak power optimization and comparison
# ------------------------------------------------------------

import picos as pc
import numpy as np
from truss_fem_core import nodes, elements, truss
from truss_visualization import TrussPlotter
from peak_power_calculations import PeakPowerCalculator
from math import isclose


if __name__ == "__main__":
    # -----------------------------
    # 1. Problem and Model Settings
    # -----------------------------
    E = 2500.0  # Young's modulus
    rho = 1.0   # Density
    m = 1.0     # Total mass
    omega = 12.5  # Frequency

    # -----------------------------------
    # 2. Geometry and FEM Model Assembly
    # -----------------------------------
    # Node coordinates
    x = np.array([0,3] + [0,1,2,3]*2 + [0,3])
    y = np.array([0]*2 + [-1]*4 + [-2]*4 + [-3]*2)
    nodes_obj = nodes(x, y)
    nodes_obj.set_kinematic([0,1,2,3])  # Fix boundary nodes
    # Element connectivity
    n1 = [0,0,1,1,2,2,2,3,3,4,4,4,4,5,6,6,7,7,8,8,9]
    n2 = [2,3,4,5,3,6,7,4,7,5,7,8,9,9,7,10,8,10,9,11,11]
    elements_obj = elements(n1, n2)
    t = truss(nodes_obj, elements_obj)
    t.set_density(rho)
    t.set_module_young(E)
    t.assemble()
    # Degrees of freedom and reduced matrices
    free = np.where(nodes_obj.dof)[0]
    ne = t.elements.Ne
    Ke_red = [Ke[np.ix_(free, free)] for Ke in t.Ke]
    Me_red = [Me[np.ix_(free, free)] for Me in t.Me]

    # -------------------
    # 3. Load Definition
    # -------------------
    fR = np.zeros(2*nodes_obj.Nn)
    fI = np.zeros(2*nodes_obj.Nn)
    fR[[20,22]] = 0.25
    fI[[21,23]] = 0.25
    fR = fR[free]
    fI = fI[free]
    F = np.column_stack([fR, fI])

    # --------------------------------------
    # 4. Peak Power Calculations & Analysis
    # --------------------------------------
    peak_calc = PeakPowerCalculator()

    # Uniform truss (all areas equal)
    print("--- Uniform truss peak power ---")
    peak_power_uniform, uniform_areas = peak_calc.compute_peak_power_uniform(t, free, F, omega, m, rho)
    print(f"Peak power at uniform truss: {peak_power_uniform:.6e}")
    print("Uniform truss element areas:", uniform_areas.flatten())

    # Optimized truss (no penalization)
    print("\n--- Optimization without penalization ---")
    peak_power_opt, actual_peak_power_opt, total_mass_opt, a_opt = peak_calc.compute_peak_power_optimized(
        ne, free, Ke_red, Me_red, F, t, omega, m, rho)
    if peak_power_opt is not None:
        a_opt = np.array(a_opt).flatten()
        print(f"[Optimized] Lower bound peak power: {peak_power_opt:.6e}")
        print(f"[Optimized] Actual peak power: {actual_peak_power_opt:.6e}")
        print(f"[Optimized] Total mass used: {total_mass_opt:.6e}")
        print("Optimized truss element areas:", list(a_opt))
    else:
        print("[Optimized] No optimal solution found.")

    # Penalized optimization: sweep over eta
    etas = [0.01, 0.1, 1, 10, 100, 1000]
    print("\n--- Optimization with penalization (sweep over eta) ---")
    penal_results, best_idx = peak_calc.sweep_eta(etas, ne, free, Ke_red, Me_red, F, t, omega, m, rho)
    lower_bounds = [r['lower_bound'] for r in penal_results]
    actual_peaks = [r['actual_peak_power'] for r in penal_results]
    best_areas = penal_results[best_idx]['areas'] if best_idx is not None else None
    eta1_idx = etas.index(1) if 1 in etas else None
    eta1_areas = penal_results[eta1_idx]['areas'] if eta1_idx is not None else None
    print(f"Best eta: {etas[best_idx] if best_idx is not None else 'N/A'} with actual peak power: {actual_peaks[best_idx] if best_idx is not None else 'N/A'}")
    print("Best penalized truss element areas:", best_areas)
    if eta1_idx is not None:
        print(f"\n--- Penalized results for eta=1 ---")
        print(f"Lower bound peak power: {penal_results[eta1_idx]['lower_bound']}")
        print(f"Actual peak power: {penal_results[eta1_idx]['actual_peak_power']}")
        print(f"Total mass used: {penal_results[eta1_idx]['total_mass']}")
        print(f"Truss element areas: {eta1_areas}")
    if best_idx is not None:
        print(f"\n--- Penalized results for best eta ({etas[best_idx]}) ---")
        print(f"Lower bound peak power: {penal_results[best_idx]['lower_bound']}")
        print(f"Actual peak power: {penal_results[best_idx]['actual_peak_power']}")
        print(f"Total mass used: {penal_results[best_idx]['total_mass']}")
        print(f"Truss element areas: {best_areas}")

    # --------------------------------------
    # 5. Visualization
    # --------------------------------------
    # Create plotter instance before any plotting
    plotter = TrussPlotter()

    # Plot peak power vs eta first (shows effect of penalization)
    plotter.plot_peak_power_vs_eta(etas, lower_bounds, actual_peaks)

    # Prepare areas for all cases for truss structure comparison
    uniform_plot_areas = uniform_areas
    opt_plot_areas = a_opt if peak_power_opt is not None else None
    eta1_plot_areas = eta1_areas
    best_plot_areas = best_areas

    # Plot all four trusses in a single 2x2 figure, with simple titles
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    # 1. Uniform truss
    t.set_cross_section_area(uniform_plot_areas)
    t.assemble()
    plotter.plot_truss(t, a=uniform_plot_areas, ax=axs[0, 0])
    axs[0, 0].set_title('Uniform Truss')
    # 2. Optimized (No Penalization)
    if opt_plot_areas is not None:
        t.set_cross_section_area(opt_plot_areas)
        t.assemble()
        plotter.plot_truss(t, a=opt_plot_areas, ax=axs[0, 1])
        axs[0, 1].set_title('Optimized (No Penalization)')
    else:
        axs[0, 1].set_title('Optimized (No Penalization)\nNo Solution')
    # 3. Penalized (eta=1)
    if eta1_plot_areas is not None:
        t.set_cross_section_area(eta1_plot_areas)
        t.assemble()
        plotter.plot_truss(t, a=eta1_plot_areas, ax=axs[1, 0])
        axs[1, 0].set_title('Penalized (eta=1)')
    else:
        axs[1, 0].set_title('Penalized (eta=1)\nNo Solution')
    # 4. Penalized (Best eta)
    if best_plot_areas is not None:
        t.set_cross_section_area(best_plot_areas)
        t.assemble()
        plotter.plot_truss(t, a=best_plot_areas, ax=axs[1, 1])
        axs[1, 1].set_title(f'Penalized (Best eta={etas[best_idx]})')
    else:
        axs[1, 1].set_title('Penalized (Best eta)\nNo Solution')
    for ax in axs.flat:
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout()
    plt.show()