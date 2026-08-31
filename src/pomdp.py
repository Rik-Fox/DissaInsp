"""Sparse T, Z, R model skeleton for the disassembly/triage POMDP.

Joint state s = (x, y): x is the physical configuration index, y is the
hidden condition index. Joint index is y-major, x-minor:
    s_idx = y_idx * n_x + x_idx
so that a fixed y forms a contiguous block of n_x states (used below to
build block-diagonal / kron structure without ever storing zero entries).

Action categories (mirrors env.py's action_type/type fields):
    "Disassy"          -> A        (disassembly/fixing)
    "Insp" / "Verify"  -> A_ver    (verification inspections)
    "Insp" / "Inspect" -> A_cond   (condition inspections)
    "Triage"           -> A_R      (terminal triage actions)
"""
import numpy as np
import scipy.sparse as sp

# A_R: terminal triage actions, one per condition y.
TRIAGE_ACTIONS = ("Reuse", "Refurbished", "Recycle")

# Pi(a_term, y): illustrative payoffs, NOT tuned to any real economics yet.
# 0 marks an invalid (condition, action) combination - looked up directly,
# no validity check needed.
TRIAGE_PAYOFF = {
    "Reuse":       {"Pristine": 10.0, "Serviceable": 0.0, "Degraded": 0.0},
    "Refurbished": {"Pristine": 6.0,  "Serviceable": 5.0, "Degraded": 0.0},
    "Recycle":     {"Pristine": 2.0,  "Serviceable": 2.0, "Degraded": 1.0},
}


class JointStateSpace:
    def __init__(self, n_x, n_y):
        self.n_x = n_x
        self.n_y = n_y
        self.n_s = n_x * n_y

    def index(self, x_idx, y_idx):
        return y_idx * self.n_x + x_idx


class TransitionModel:
    """Builds T(x', y' | x, y, a) as a sparse (n_s, n_s) matrix per action."""

    def __init__(self, space: JointStateSpace):
        self.space = space

    def identity(self):
        """a in A_ver or A_cond: x' = x, y' = y."""
        return sp.identity(self.space.n_s, format="csr")

    def disassembly(self, x_succ, p_success):
        """a in A (Disassy): x -> x_succ[x] w.p. p_success[x], else x -> x.
        x_succ, p_success: arrays of shape (n_x,). Repeated identically in
        every y-block since y is static.
        """
        rows, cols, data = [], [], []
        for y in range(self.space.n_y):
            for x in range(self.space.n_x):
                s = self.space.index(x, y)
                p = p_success[x]
                if p > 0:
                    rows.append(s)
                    cols.append(self.space.index(x_succ[x], y))
                    data.append(p)
                if p < 1:
                    rows.append(s)
                    cols.append(s)
                    data.append(1 - p)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.space.n_s, self.space.n_s))

    def terminal(self, x_term):
        """a in A_R: (x, y) -> (x_term, y) deterministically."""
        rows = np.arange(self.space.n_s)
        cols = np.concatenate(
            [np.full(self.space.n_x, self.space.index(x_term, y)) for y in range(self.space.n_y)]
        )
        data = np.ones(self.space.n_s)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.space.n_s, self.space.n_s))


class ObservationModel:
    """Builds Z(o | x', y', a) as a sparse (n_s, n_o) matrix per action."""

    def __init__(self, space: JointStateSpace):
        self.space = space

    def null(self, n_o, o_null=0):
        """a in A or A_R: always observe o_null with probability 1."""
        rows = np.arange(self.space.n_s)
        cols = np.full(self.space.n_s, o_null)
        data = np.ones(self.space.n_s)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.space.n_s, n_o))

    def verification(self, z_x):
        """a in A_ver: depends only on x', uniform over y'.
        z_x: sparse (n_x, n_o) row-stochastic matrix of P(o | x').
        """
        ones_y = sp.csr_matrix(np.ones((self.space.n_y, 1)))
        return sp.kron(ones_y, z_x, format="csr")

    def condition(self, z_y):
        """a in A_cond: depends only on y', uniform over x'.
        z_y: sparse (n_y, n_o) row-stochastic matrix of P(o | y').
        """
        ones_x = sp.csr_matrix(np.ones((self.space.n_x, 1)))
        return sp.kron(z_y, ones_x, format="csr")


class RewardModel:
    """R(x, y, a): flat scalar cost for non-terminal actions, full payoff
    table for terminal actions."""

    def __init__(self, space: JointStateSpace, triage_payoff=None):
        self.space = space
        self.triage_payoff = triage_payoff if triage_payoff is not None else TRIAGE_PAYOFF

    def flat_cost(self, cost):
        """a in A, A_ver, or A_cond: constant -cost regardless of (x, y)."""
        return -float(cost)

    def triage(self, action, condition_order):
        """a in A_R (Triage): Pi(x, y, a_term) via self.triage_payoff, constant
        across x. condition_order lists condition names in y-index order,
        e.g. ["Pristine", "Serviceable", "Degraded"]."""
        row = [self.triage_payoff[action][y] for y in condition_order]
        payoff_matrix = np.tile(row, (self.space.n_x, 1))
        # a in A_R: payoff_matrix has shape (n_x, n_y); flattened to the (n_s,) joint-state order (y-major, x-minor).
        return payoff_matrix.reshape(-1, order="F")
