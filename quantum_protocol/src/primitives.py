"""
primitives.py — core quantum state representations.

Two classes cover everything the framework needs:
  - StateVector   : pure states, fast, memory-efficient
  - DensityMatrix : mixed/noisy states, required for noise channels

Both classes are backend-agnostic — they use `xp` (numpy or cupy)
so protocol code never needs to know which backend is active.

Batching convention
-------------------
Every method operates on a *batch* of states simultaneously.
Shape for n_qubits=2, batch_size=1000:
  StateVector:   (1000, 4)        — 1000 independent 2-qubit states
  DensityMatrix: (1000, 4, 4)     — 1000 independent 2-qubit density matrices

This is the key to GPU throughput: one CuPy kernel over 1M states
instead of a Python loop over 1M sequential states.
"""

from __future__ import annotations
from typing import Optional
import numpy as np  # used only for type hints and CPU-side validation


# ---------------------------------------------------------------------------
# Gates — standard unitary matrices, built lazily on first use
# ---------------------------------------------------------------------------

def _make_gates(xp) -> dict:
    """Return a dict of common gate matrices as xp arrays (float64 complex)."""
    c = xp.array

    I  = c([[1, 0], [0, 1]],           dtype=xp.complex128)
    X  = c([[0, 1], [1, 0]],           dtype=xp.complex128)
    Y  = c([[0, -1j], [1j, 0]],        dtype=xp.complex128)
    Z  = c([[1, 0], [0, -1]],          dtype=xp.complex128)
    H  = c([[1, 1], [1, -1]],          dtype=xp.complex128) / xp.sqrt(xp.array(2.0))
    S  = c([[1, 0], [0, 1j]],          dtype=xp.complex128)
    T  = c([[1, 0], [0, xp.exp(xp.array(1j * xp.pi / 4))]],
                                        dtype=xp.complex128)
    CNOT = c([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                                        dtype=xp.complex128)
    SWAP = c([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
                                        dtype=xp.complex128)
    CZ   = c([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],
                                        dtype=xp.complex128)

    return dict(I=I, X=X, Y=Y, Z=Z, H=H, S=S, T=T, CNOT=CNOT, SWAP=SWAP, CZ=CZ)


def Rz(theta: float, xp) -> "xp.ndarray":
    """Rotation around Z axis by angle theta."""
    return xp.array(
        [[np.exp(-1j * theta / 2), 0],
         [0,                        np.exp(1j * theta / 2)]],
        dtype=xp.complex128,
    )


def Rx(theta: float, xp) -> "xp.ndarray":
    """Rotation around X axis by angle theta."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return xp.array(
        [[c,    -1j * s],
         [-1j * s, c   ]],
        dtype=xp.complex128,
    )


# ---------------------------------------------------------------------------
# StateVector
# ---------------------------------------------------------------------------

class StateVector:
    """
    Batched pure quantum state |ψ⟩.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Hilbert space dimension = 2**n_qubits.
    batch_size : int
        Number of independent states to simulate in parallel.
    xp : module
        Backend (numpy or cupy). Import from quantumsim.backend.
    init : str
        'zero'  → all states initialised to |0...0⟩  (default)
        'rand'  → random normalised pure states
        'plus'  → all states initialised to |+...+⟩ = H⊗n |0...0⟩

    Attributes
    ----------
    data : xp.ndarray, shape (batch_size, dim)
        Complex amplitudes. data[i] is the i-th state vector.
    """

    def __init__(
        self,
        n_qubits: int,
        batch_size: int = 1,
        xp=None,
        init: str = "zero",
    ):
        if xp is None:
            import numpy as _np
            xp = _np
        self.xp = xp
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits # number of possible states is 2^number of qubits
        self.batch_size = batch_size
        self._gates = None  # lazy init

        if init == "zero": # initialize with 0s, set first column of each to 1, |0> state 
            self.data = xp.zeros((batch_size, self.dim), dtype=xp.complex128)
            self.data[:, 0] = 1.0

        elif init == "plus": # initialize with 1s, normalize with sqrt(dim), |+> state
            self.data = xp.ones((batch_size, self.dim), dtype=xp.complex128)
            self.data /= xp.sqrt(xp.array(float(self.dim)))

        elif init == "rand": # initialize randomly
            real = xp.random.randn(batch_size, self.dim)
            imag = xp.random.randn(batch_size, self.dim)
            raw  = real + 1j * imag
            norms = xp.linalg.norm(raw, axis=1, keepdims=True)
            self.data = raw / norms

        else:
            raise ValueError(f"Unknown init mode '{init}'. Use 'zero', 'plus', or 'rand'.")

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    @property
    def gates(self) -> dict:
        if self._gates is None:
            self._gates = _make_gates(self.xp)
        return self._gates

    def apply_gate(self, U: "xp.ndarray") -> "StateVector":
        """
        Apply a unitary gate U to the full state space.

        U must be (dim, dim) or (batch_size, dim, dim) for per-state gates.
        Returns self for method chaining.

        Einsum 'bi,ji->bj':
          b = batch, i = input amplitude index, j = output index
        This is a batched matrix-vector multiply: one gate applied to all states.
        """
        xp = self.xp
        if U.ndim == 2:
            # Same gate for every state in the batch — most common case
            self.data = xp.einsum("bi,ji->bj", self.data, U.conj())
        elif U.ndim == 3:
            # Per-state gates — used e.g. for per-trial random rotations
            self.data = xp.einsum("bi,bji->bj", self.data, U.conj())
        else:
            raise ValueError(f"Gate must be 2D or 3D, got {U.ndim}D")
        return self

    def apply_single_qubit_gate(self, U: "xp.ndarray", qubit: int) -> "StateVector":
        """
        Apply a single-qubit gate U to qubit `qubit`, leaving others unchanged.
        Constructs the full n-qubit operator via tensor product.
        """
        xp = self.xp
        op = xp.array([[1.0+0j]], dtype=xp.complex128).reshape(1, 1)
        for q in range(self.n_qubits):
            op = xp.kron(op, U if q == qubit else self.gates["I"])
        return self.apply_gate(op)

    # Convenience wrappers — these are what protocol code calls
    def h(self, qubit: int = 0) -> "StateVector":
        return self.apply_single_qubit_gate(self.gates["H"], qubit)

    def x(self, qubit: int = 0) -> "StateVector":
        return self.apply_single_qubit_gate(self.gates["X"], qubit)

    def z(self, qubit: int = 0) -> "StateVector":
        return self.apply_single_qubit_gate(self.gates["Z"], qubit)

    def rz(self, theta: float, qubit: int = 0) -> "StateVector":
        return self.apply_single_qubit_gate(Rz(theta, self.xp), qubit)

    def cnot(self) -> "StateVector":
        """CNOT with qubit 0 as control, qubit 1 as target. Requires n_qubits >= 2."""
        return self.apply_gate(self.gates["CNOT"])

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_z(self, qubit: int = 0) -> "xp.ndarray":
        """
        Simulate Z-basis measurement on `qubit` for all batch states.

        Returns an integer array of shape (batch_size,) with values 0 or 1.
        Does NOT collapse the state — for Monte Carlo simulation you usually
        want the outcome statistics, not the post-measurement state.

        Implementation:
          1. Reshape amplitudes into (batch, qubit_dims...) tensor.
          2. Sum |amplitude|^2 over all axes except the target qubit.
          3. That gives P(|1⟩) for each batch state.
          4. Sample Bernoulli(P(|1⟩)).
        """
        xp = self.xp
        shape = (self.batch_size,) + (2,) * self.n_qubits
        tensor = self.data.reshape(shape)

        # Axes to sum over: all qubit axes except `qubit`
        # qubit axis in tensor is at position qubit+1 (position 0 is batch)
        sum_axes = tuple(i + 1 for i in range(self.n_qubits) if i != qubit)

        prob_1 = xp.sum(xp.abs(tensor[..., 1:2, :] if False else tensor) ** 2,
                        axis=sum_axes)
        # prob_1 shape: (batch_size, 2) — index 0 = P(|0⟩), index 1 = P(|1⟩)
        # Select the |1⟩ probability:
        prob_1_flat = xp.moveaxis(
            self.data.reshape(shape), qubit + 1, 1
        ).reshape(self.batch_size, 2, -1)
        p1 = xp.sum(xp.abs(prob_1_flat[:, 1, :]) ** 2, axis=1)  # (batch,)

        outcomes = (xp.random.rand(self.batch_size) < p1).astype(xp.int32)
        return outcomes

    # ------------------------------------------------------------------
    # Conversion and inspection
    # ------------------------------------------------------------------

    def to_density_matrix(self) -> "DensityMatrix":
        """
        Convert |ψ⟩ → ρ = |ψ⟩⟨ψ|.
        Shape: (batch_size, dim) → (batch_size, dim, dim).
        Einsum 'bi,bj->bij': outer product over the batch.
        """
        xp = self.xp
        rho = xp.einsum("bi,bj->bij", self.data, self.data.conj())
        dm = DensityMatrix(self.n_qubits, self.batch_size, xp)
        dm.data = rho
        return dm

    def probabilities(self) -> "xp.ndarray":
        """Return |amplitude|^2 for each basis state. Shape: (batch_size, dim)."""
        return (self.xp.abs(self.data) ** 2).real

    def fidelity(self, other: "StateVector") -> "xp.ndarray":
        """
        Fidelity |⟨ψ|φ⟩|^2 between self and other, per batch element.
        Returns shape (batch_size,), values in [0, 1].
        """
        xp = self.xp
        overlap = xp.einsum("bi,bi->b", self.data.conj(), other.data)
        return xp.abs(overlap) ** 2

    def norm(self) -> "xp.ndarray":
        """L2 norm of each state. Should be 1.0 for valid states."""
        return self.xp.linalg.norm(self.data, axis=1)

    def __repr__(self) -> str:
        backend = type(self.xp).__name__
        return (
            f"StateVector(n_qubits={self.n_qubits}, "
            f"batch_size={self.batch_size}, backend={backend})"
        )


# ---------------------------------------------------------------------------
# DensityMatrix
# ---------------------------------------------------------------------------

class DensityMatrix:
    """
    Batched density matrix ρ — represents mixed (noisy) quantum states.

    Use this when you need noise channels (depolarizing, amplitude damping, etc.).
    More memory-intensive than StateVector: O(dim^2) vs O(dim) per state.

    Parameters
    ----------
    n_qubits : int
    batch_size : int
    xp : module
        Backend (numpy or cupy).
    init : str
        'zero'  → ρ = |0⟩⟨0| for each batch state (pure ground state)
        'mixed' → ρ = I/dim (maximally mixed state)
        'plus'  → ρ = |+⟩⟨+|

    Attributes
    ----------
    data : xp.ndarray, shape (batch_size, dim, dim)
        Density matrices. data[i] is the i-th density matrix.
    """

    def __init__(
        self,
        n_qubits: int,
        batch_size: int = 1,
        xp=None,
        init: str = "zero",
    ):
        if xp is None:
            import numpy as _np
            xp = _np
        self.xp = xp
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.batch_size = batch_size
        self._gates = None

        if init == "zero":
            self.data = xp.zeros((batch_size, self.dim, self.dim), dtype=xp.complex128)
            self.data[:, 0, 0] = 1.0

        elif init == "mixed":
            eye = xp.eye(self.dim, dtype=xp.complex128)
            self.data = xp.broadcast_to(
                eye[None] / self.dim, (batch_size, self.dim, self.dim)
            ).copy()

        elif init == "plus":
            sv = StateVector(n_qubits, batch_size, xp, init="plus")
            self.data = sv.to_density_matrix().data

        else:
            raise ValueError(f"Unknown init mode '{init}'. Use 'zero', 'mixed', or 'plus'.")

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    @property
    def gates(self) -> dict:
        if self._gates is None:
            self._gates = _make_gates(self.xp)
        return self._gates

    def apply_unitary(self, U: "xp.ndarray") -> "DensityMatrix":
        """
        Apply unitary U: ρ → U ρ U†.

        Einsum 'bij,kj->bik' then 'bik,jk->bij':
          Equivalent to batched U @ rho @ U.conj().T
          but written as two einsums to keep memory usage predictable.
        """
        xp = self.xp
        # U @ rho
        rho_new = xp.einsum("ij,bjk->bik", U, self.data)
        # (U @ rho) @ U†
        self.data = xp.einsum("bij,kj->bik", rho_new, U.conj())
        return self

    def apply_kraus(self, kraus_ops: list) -> "DensityMatrix":
        """
        Apply a quantum channel defined by Kraus operators {K_i}.

        ρ → Σ_i K_i ρ K_i†

        This is the standard way to apply noise channels.
        kraus_ops: list of (dim, dim) arrays. Must satisfy Σ K†K = I.

        The loop over Kraus operators is unavoidable, but the inner
        einsum is vectorised over the full batch simultaneously.
        """
        xp = self.xp
        new_data = xp.zeros_like(self.data)
        for K in kraus_ops:
            # K @ rho
            KR = xp.einsum("ij,bjk->bik", K, self.data)
            # K @ rho @ K†
            new_data += xp.einsum("bij,kj->bik", KR, K.conj())
        self.data = new_data
        return self

    # ------------------------------------------------------------------
    # Standard noise channels — each returns self for chaining
    # ------------------------------------------------------------------

    def depolarize(self, p: float) -> "DensityMatrix":
        """
        Single-qubit depolarizing channel with error probability p.
        ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

        At p=0: identity. At p=3/4: maximally mixed state.
        This is the most common noise model for QKD simulations.

        Only valid for n_qubits == 1.
        Use depolarize_qubit(p, qubit) for multi-qubit states.
        """
        if self.n_qubits != 1:
            raise ValueError(
                "depolarize() is for single-qubit states. "
                "Use depolarize_qubit(p, qubit) for multi-qubit."
            )
        xp = self.xp
        g = self.gates
        K0 = xp.sqrt(xp.array(1 - p, dtype=xp.float64) + 0j) * g["I"]
        K1 = xp.sqrt(xp.array(p / 3, dtype=xp.float64) + 0j) * g["X"]
        K2 = xp.sqrt(xp.array(p / 3, dtype=xp.float64) + 0j) * g["Y"]
        K3 = xp.sqrt(xp.array(p / 3, dtype=xp.float64) + 0j) * g["Z"]
        return self.apply_kraus([K0, K1, K2, K3])

    def amplitude_damp(self, gamma: float) -> "DensityMatrix":
        """
        Amplitude damping channel — models energy loss (T1 decay).
        gamma: decay probability (0 = no decay, 1 = full collapse to |0⟩).

        Kraus operators:
          K0 = [[1, 0], [0, sqrt(1-gamma)]]   (no decay)
          K1 = [[0, sqrt(gamma)], [0, 0]]      (decay |1⟩ → |0⟩)
        """
        if self.n_qubits != 1:
            raise ValueError("amplitude_damp() only valid for single-qubit states.")
        xp = self.xp
        K0 = xp.array([[1, 0], [0, xp.sqrt(xp.array(1 - gamma))]], dtype=xp.complex128)
        K1 = xp.array([[0, xp.sqrt(xp.array(gamma))], [0, 0]], dtype=xp.complex128)
        return self.apply_kraus([K0, K1])

    def phase_damp(self, gamma: float) -> "DensityMatrix":
        """
        Phase damping channel — models dephasing (T2 decay, no energy loss).
        gamma: dephasing probability.

        Kraus operators:
          K0 = [[1, 0], [0, sqrt(1-gamma)]]
          K1 = [[0, 0], [0, sqrt(gamma)]]
        """
        if self.n_qubits != 1:
            raise ValueError("phase_damp() only valid for single-qubit states.")
        xp = self.xp
        K0 = xp.array([[1, 0], [0, xp.sqrt(xp.array(1 - gamma))]], dtype=xp.complex128)
        K1 = xp.array([[0, 0], [0, xp.sqrt(xp.array(gamma))]], dtype=xp.complex128)
        return self.apply_kraus([K0, K1])

    def erasure(self, p: float) -> "DensityMatrix":
        """
        Erasure channel: with probability p, replace ρ with |e⟩⟨e|
        (an orthogonal 'erased' state). Approximated here as mixing
        with the maximally mixed state, which is the standard treatment
        in discrete-variable QKD analysis.
        """
        xp = self.xp
        eye = xp.eye(self.dim, dtype=xp.complex128)
        mixed = eye / self.dim
        self.data = (1 - p) * self.data + p * mixed[None]
        return self

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_z(self, qubit: int = 0) -> "xp.ndarray":
        """
        Z-basis measurement outcome probabilities and sampled results.

        P(outcome=1) = Tr(|1⟩⟨1| ρ) = ρ_{1,1} for single qubit.
        For multi-qubit: partial trace over all qubits except `qubit`.

        Returns outcomes array of shape (batch_size,), values 0 or 1.
        """
        xp = self.xp
        p1 = self._prob_1_for_qubit(qubit)
        outcomes = (xp.random.rand(self.batch_size) < p1).astype(xp.int32)
        return outcomes

    def _prob_1_for_qubit(self, qubit: int) -> "xp.ndarray":
        """
        Compute P(|1⟩) for a given qubit via partial trace.
        Returns shape (batch_size,).
        """
        xp = self.xp
        # Reshape density matrix into tensor form
        shape = (self.batch_size,) + (2,) * self.n_qubits + (2,) * self.n_qubits
        tensor = self.data.reshape(shape)

        # Partial trace: trace out all qubits except `qubit`
        # The tensor has axes: [batch, q0, q1, ..., q0', q1', ...]
        # For the diagonal (trace), bra and ket indices of non-target qubits
        # must be equal — we sum over those.

        # This is easier to think about for the single-qubit case:
        # p1 = rho[batch, 1, 1] for n_qubits=1
        if self.n_qubits == 1:
            return self.data[:, 1, 1].real

        # General case: sum diagonal elements where target qubit = 1
        # Rebuild as (batch, dim, dim) and compute per-qubit marginals
        # by grouping basis states where qubit `qubit` is in state |1⟩
        indices_1 = [i for i in range(self.dim) if (i >> (self.n_qubits - 1 - qubit)) & 1]
        p1 = xp.sum(self.data[:, indices_1, :][:, :, indices_1].real, axis=(1, 2))
        return p1

    # ------------------------------------------------------------------
    # Properties and validation
    # ------------------------------------------------------------------

    def trace(self) -> "xp.ndarray":
        """Tr(ρ) for each batch element. Should be 1.0 for valid states."""
        return self.xp.trace(self.data, axis1=1, axis2=2).real

    def purity(self) -> "xp.ndarray":
        """
        Tr(ρ²) for each batch element.
        = 1.0 for pure states, = 1/dim for maximally mixed states.
        """
        xp = self.xp
        rho2 = xp.einsum("bij,bjk->bik", self.data, self.data)
        return xp.trace(rho2, axis1=1, axis2=2).real

    def fidelity(self, other: "DensityMatrix") -> "xp.ndarray":
        """
        Uhlmann fidelity F(ρ, σ) = (Tr√(√ρ σ √ρ))² per batch element.
        Computed via eigendecomposition — exact but O(dim^3) per state.
        Falls back to NumPy for the eigendecomposition even on GPU
        (cupy.linalg.eigh is available but less stable for small matrices).
        """
        xp = self.xp
        # For pure state fidelity: F = Tr(ρσ) (faster path)
        # General case requires matrix square root — use numpy for stability
        try:
            rho_np = xp.asnumpy(self.data)
            sig_np = xp.asnumpy(other.data)
        except AttributeError:
            rho_np = self.data
            sig_np = other.data

        fids = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            eigvals, eigvecs = np.linalg.eigh(rho_np[i])
            eigvals = np.maximum(eigvals, 0)
            sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
            M = sqrt_rho @ sig_np[i] @ sqrt_rho
            eigvals_m = np.linalg.eigvalsh(M)
            fids[i] = (np.sum(np.sqrt(np.maximum(eigvals_m, 0)))) ** 2

        return xp.array(fids)

    def to_numpy(self) -> np.ndarray:
        """Move data to CPU numpy array (no-op if already on CPU)."""
        try:
            return self.xp.asnumpy(self.data)
        except AttributeError:
            return self.data

    def __repr__(self) -> str:
        backend = type(self.xp).__name__
        return (
            f"DensityMatrix(n_qubits={self.n_qubits}, "
            f"batch_size={self.batch_size}, backend={backend})"
        )
