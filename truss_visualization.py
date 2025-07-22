import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import isclose

class TrussPlotter:
    def plot_truss(self, trs, a=None, forces=None, default_width=2., ax=None):
        """
        Plot a single truss structure with optional cross-sectional areas and forces.
        Args:
            trs: Truss object
            a: Array of cross-sectional areas (optional)
            forces: Not used here (optional)
            default_width: Base line width for plotting
            ax: Matplotlib axis to plot on (optional)
        Returns:
            The matplotlib axis with the plot
        """
        if a is None:
            warnings.warn("value of cross section area not specified, display with default value")
            a = np.ones(trs.elements.Ne)
            a_max = 1.
        else:
            a_max = np.max(a)
        if ax is None:
            ax = plt.gca()
        active_elements_count = 0
        for e, (n1, n2) in enumerate(zip(trs.elements.node1, trs.elements.node2)):
            if np.isclose(a[e], 0.):
                continue
            x = trs.nodes.x[[n1, n2]]
            y = trs.nodes.y[[n1, n2]]
            relative_width = a[e] / a_max if a_max > 0 else 1.0
            if relative_width >= 1e-3:
                ax.plot(x, y, "b", linewidth=default_width * relative_width, zorder=1)
                active_elements_count += 1
            else:
                continue
        # Draw the nodes
        imposed = ~trs.nodes.dof
        for i in range(trs.nodes.Nn):
            dof = np.array([2 * i, 2 * i + 1])
            if np.all(trs.nodes.dof[dof]):  # if both of the dof are free
                ax.scatter(trs.nodes.x[i], trs.nodes.y[i], marker="o", s=80, color="blue", zorder=2)
            else:
                ax.scatter(trs.nodes.x[i], trs.nodes.y[i], marker="x", s=80, color="red", zorder=2)
                if hasattr(trs.nodes, "u_imp"):
                    ux_id = np.sum(imposed[:2 * i])
                    uy_id = ux_id + 1
                    ux, uy = trs.nodes.u_imp[[ux_id, uy_id]]
                    x, y = trs.nodes.x[i], trs.nodes.y[i]
                    if not (isclose(ux, 0) and isclose(uy, 0)):
                        ax.arrow(x, y, np.sign(ux), np.sign(uy), width=.025, color="red", zorder=3)
        # Draw the forces (optional, not used here)
        return ax

    def plot_truss_comparison(self, t, uniform_areas, opt_areas):
        """
        Plot a side-by-side comparison of uniform and optimized (no penalization) truss structures.
        Args:
            t: Truss object
            uniform_areas: Areas for uniform truss
            opt_areas: Areas for optimized truss (no penalization)
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # 1. Uniform truss
        t.set_cross_section_area(uniform_areas)
        t.assemble()
        plt.sca(axs[0])
        self.plot_truss(t, a=uniform_areas, ax=axs[0])
        axs[0].set_title('Uniform Truss')
        # 2. Optimized (No Penalization)
        if opt_areas is not None:
            t.set_cross_section_area(opt_areas)
            t.assemble()
            plt.sca(axs[1])
            self.plot_truss(t, a=opt_areas, ax=axs[1])
            axs[1].set_title('Optimized (No Penalization)')
        else:
            axs[1].set_title('Optimized (No Penalization)\nNo Solution')
        for ax in axs.flat:
            ax.set_aspect('equal')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_peak_power_vs_eta(self, etas, lower_bounds, actual_peaks):
        """
        Plot the effect of penalization parameter eta on peak power (lower bound and actual).
        Args:
            etas: List of eta values
            lower_bounds: List of lower bound peak powers
            actual_peaks: List of actual peak powers
        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(etas, lower_bounds, 'o-', label='Lower bound peak power')
        plt.plot(etas, actual_peaks, 's-', label='Actual peak power')
        plt.xscale('log')
        plt.xlabel('eta (penalization parameter)')
        plt.ylabel('Peak Power')
        plt.title('Effect of Penalization on Peak Power')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show() 