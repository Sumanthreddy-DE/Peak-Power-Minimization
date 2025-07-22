import numpy as np

_STIFF_SHAPE = np.array([[1, -1], [-1, 1]])
_MASS_SHAPE = {1: np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]]) / 6,
               2: np.eye(4) / 2}

class nodes:
    def __init__(self, x, y, dim=2):
        self.x = x
        self.y = y
        self.Nn = len(self.x)
        self.dim = dim
        self.dof = np.ones(self.dim * self.Nn, dtype="?")

    def set_kinematic(self, ids, u_imp=None):
        self.dof[ids] = False
        self.ndof = np.sum(self.dof)
        if not (u_imp is None):
            self.u_imp = np.array(u_imp)

    def set_boundary(self, ids):
        self.is_boundary = np.zeros(self.dim * self.Nn, dtype='?')
        self.is_boundary[ids] = 1

class elements:
    def __init__(self, node1, node2, node3=None):
        self.Ne = len(node1)
        if node3 is None:
            self.node1 = np.array(node1, dtype=int)
            self.node2 = np.array(node2, dtype=int)
        else:
            self.connectivity = np.array([[n1, n2, n3] for (n1, n2, n3) in zip(node1, node2, node3)])

class truss:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.geometry()
        self.a = np.ones(self.elements.Ne)

    def set_cross_section_area(self, a):
        if isinstance(a, float) or isinstance(a, int):
            self.a = np.ones(self.elements.Ne) * a
        elif isinstance(a, np.ndarray):
            if a.size != self.elements.Ne:
                raise IndexError("Length of area vector does not match number of elements.")
            else:
                self.a = a
        else:
            raise TypeError("Area must be a float, int, or numpy array.")

    def set_module_young(self, E):
        if type(E) is float:
            self.E = np.ones(self.elements.Ne) * E
        elif type(E) is np.ndarray:
            if E.size != self.elements.Ne:
                raise IndexError
            else:
                self.E = E

    def set_density(self, rho):
        if isinstance(rho, float) or isinstance(rho, int):
            self.rho = np.ones(self.elements.Ne) * rho
        elif isinstance(rho, np.ndarray):
            if rho.size != self.elements.Ne:
                raise IndexError("Density array size does not match number of elements.")
            else:
                self.rho = rho
        else:
            raise TypeError("rho must be a float, int, or numpy array.")

    def geometry(self):
        delta_x = self.nodes.x[self.elements.node2] - self.nodes.x[self.elements.node1]
        delta_y = self.nodes.y[self.elements.node2] - self.nodes.y[self.elements.node1]
        lengths = np.sqrt(delta_x ** 2 + delta_y ** 2)
        c = delta_x / lengths
        s = delta_y / lengths
        self.lengths = lengths
        self.cs = np.vstack((c, s))

    def assemble(self, mass_shape=1):
        MS = _MASS_SHAPE[mass_shape]
        ndof = self.nodes.ndof
        nn = self.nodes.Nn
        dim = self.nodes.dim
        ne = self.elements.Ne
        self.Ke = np.zeros((self.elements.Ne, dim * nn, dim * nn))
        self.Me = np.zeros((self.elements.Ne, dim * nn, dim * nn))
        for e, (n1, n2) in enumerate(zip(self.elements.node1, self.elements.node2)):
            E = self.E[e]
            rho = self.rho[e]
            l = self.lengths[e]
            CS = np.kron(_STIFF_SHAPE, np.outer(self.cs[:, e], self.cs[:, e]))
            dof_local = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])
            row, col = np.ix_(dof_local, dof_local)
            self.Ke[e, row, col] = E / l * CS
            self.Me[e, row, col] = rho * l * MS 