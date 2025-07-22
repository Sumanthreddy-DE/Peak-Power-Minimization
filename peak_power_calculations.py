import numpy as np
import picos as pc

class PeakPowerCalculator:
    @staticmethod
    def get_peak_power(F, K_omega, omega):
        """
        Calculate the peak power for a given force and dynamic stiffness matrix.
        Args:
            F: Force matrix (2D numpy array)
            K_omega: Dynamic stiffness matrix (numpy array)
            omega: Frequency (float)
        Returns:
            Peak power (float)
        """
        FTKF = F.T @ np.linalg.inv(K_omega) @ F
        C0 = np.array([[1,0],[0,-1]])
        C1 = np.array([[0,1],[1,0]])
        return omega/2 * np.sqrt(np.trace(C0@FTKF)**2 + np.trace(C1@FTKF)**2)

    def compute_peak_power_uniform(self, t, free, F, omega, m, rho):
        """
        Compute peak power for a uniform truss (all areas equal).
        Args:
            t: Truss object
            free: Indices of free degrees of freedom
            F: Force matrix
            omega: Frequency
            m: Total mass
            rho: Density
        Returns:
            Tuple (peak_power, uniform_areas)
        """
        uniform_area = m / (np.sum(t.lengths) * rho)
        a_uniform = np.ones(t.elements.Ne) * uniform_area
        t.set_cross_section_area(a_uniform)
        t.assemble()
        K = np.sum(t.Ke, axis=0)[np.ix_(free, free)]
        M = np.sum(t.Me, axis=0)[np.ix_(free, free)]
        K_omega = K - omega**2 * M
        return self.get_peak_power(F, K_omega, omega), a_uniform

    def compute_peak_power_optimized(self, ne, free, Ke_red, Me_red, F, t, omega, m, rho):
        """
        Compute peak power for the optimized truss (no penalization).
        Args:
            ne: Number of elements
            free: Indices of free degrees of freedom
            Ke_red: List of reduced element stiffness matrices
            Me_red: List of reduced element mass matrices
            F: Force matrix
            t: Truss object
            omega: Frequency
            m: Total mass
            rho: Density
        Returns:
            Tuple (lower_bound_peak_power, actual_peak_power, total_mass, areas)
        """
        a = pc.RealVariable('a', ne, lower=0)
        theta = pc.RealVariable('theta')
        X = pc.SymmetricVariable('X', 2)
        Kp = pc.Constant(np.zeros((len(free), len(free))))
        Mp = pc.Constant(np.zeros((len(free), len(free))))
        for i in range(ne):
            Kp += a[i] * pc.Constant(Ke_red[i])
            Mp += a[i] * pc.Constant(Me_red[i])
        Komega = Kp - omega**2 * Mp
        lengths = pc.Constant('lengths', t.lengths.reshape((ne,1)))
        total_mass = a.T * (lengths * rho)
        LMI1 = pc.block([[X, F.T], [F, Komega]])
        LMI2 = (theta & X[0,0]-X[1,1] & X[0,1]) \
               // (X[0,0]-X[1,1] & 4 & 0) \
               // (X[0,1] & 0 & 1)
        P = pc.Problem()
        P.set_objective('min', theta)
        P.add_constraint(a >= 0)
        P.add_constraint(total_mass <= m)
        P.add_constraint(LMI1 >> 0)
        P.add_constraint(LMI2 >> 0)
        P.solve(solver='mosek')
        if theta.value is not None and a.value is not None:
            peak_power = np.sqrt(theta.value) / 2
            # Build numeric Komega for actual peak power
            Komega_np = np.zeros((len(free), len(free)))
            for i in range(ne):
                Komega_np += a.value[i] * Ke_red[i] - omega**2 * a.value[i] * Me_red[i]
            actual_peak_power = self.get_peak_power(F, Komega_np, omega)
            total_mass_val = float(total_mass.value)
            return peak_power, actual_peak_power, total_mass_val, np.array(a.value).flatten()
        else:
            return None, None, None, None

    def compute_peak_power_penalized(self, ne, free, Ke_red, Me_red, F, t, omega, m, rho, eta):
        """
        Compute peak power for penalized optimization (adds eta*trace(X) to objective).
        Args:
            ne: Number of elements
            free: Indices of free degrees of freedom
            Ke_red: List of reduced element stiffness matrices
            Me_red: List of reduced element mass matrices
            F: Force matrix
            t: Truss object
            omega: Frequency
            m: Total mass
            rho: Density
            eta: Penalization parameter
        Returns:
            Tuple (lower_bound_peak_power, actual_peak_power, total_mass, areas)
        """
        a = pc.RealVariable('a', ne, lower=0)
        theta = pc.RealVariable('theta')
        X = pc.SymmetricVariable('X', 2)
        Kp = pc.Constant(np.zeros((len(free), len(free))))
        Mp = pc.Constant(np.zeros((len(free), len(free))))
        for i in range(ne):
            Kp += a[i] * pc.Constant(Ke_red[i])
            Mp += a[i] * pc.Constant(Me_red[i])
        Komega = Kp - omega**2 * Mp
        lengths = pc.Constant('lengths', t.lengths.reshape((ne,1)))
        total_mass = a.T * (lengths * rho)
        LMI1 = pc.block([[X, F.T], [F, Komega]])
        LMI2 = (theta & X[0,0]-X[1,1] & X[0,1]) \
               // (X[0,0]-X[1,1] & 4 & 0) \
               // (X[0,1] & 0 & 1)
        P = pc.Problem()
        P.set_objective('min', theta + eta * pc.trace(X))
        P.add_constraint(a >= 0)
        P.add_constraint(total_mass <= m)
        P.add_constraint(LMI1 >> 0)
        P.add_constraint(LMI2 >> 0)
        P.solve(solver='mosek')
        if theta.value is not None and a.value is not None:
            peak_power = np.sqrt(theta.value) / 2
            Komega_np = np.zeros((len(free), len(free)))
            for i in range(ne):
                Komega_np += a.value[i] * Ke_red[i] - omega**2 * a.value[i] * Me_red[i]
            actual_peak_power = self.get_peak_power(F, Komega_np, omega)
            total_mass_val = float(total_mass.value)
            return peak_power, actual_peak_power, total_mass_val, np.array(a.value).flatten()
        else:
            return None, None, None, None

    def sweep_eta(self, etas, ne, free, Ke_red, Me_red, F, t, omega, m, rho):
        """
        Sweep over multiple eta values for penalized optimization and collect results.
        Args:
            etas: List of eta values to test
            ne: Number of elements
            free: Indices of free degrees of freedom
            Ke_red: List of reduced element stiffness matrices
            Me_red: List of reduced element mass matrices
            F: Force matrix
            t: Truss object
            omega: Frequency
            m: Total mass
            rho: Density
        Returns:
            (results, best_idx):
                results: List of dicts with keys 'eta', 'lower_bound', 'actual_peak_power', 'total_mass', 'areas'
                best_idx: Index of result with lowest actual peak power
        """
        results = []
        best_idx = None
        best_peak = float('inf')
        for idx, eta in enumerate(etas):
            peak_power, actual_peak_power, total_mass, a_pen = self.compute_peak_power_penalized(
                ne, free, Ke_red, Me_red, F, t, omega, m, rho, eta)
            results.append({
                'eta': eta,
                'lower_bound': peak_power,
                'actual_peak_power': actual_peak_power,
                'total_mass': total_mass,
                'areas': a_pen
            })
            if actual_peak_power is not None and actual_peak_power < best_peak:
                best_peak = actual_peak_power
                best_idx = idx
        return results, best_idx 