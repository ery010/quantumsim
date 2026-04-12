"""
tests/test_primitives.py

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from primitives import StateVector, DensityMatrix

xp = np  # CPU backend for all tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(a, b, tol=1e-10, label=""):
    """Check two arrays are element-wise close within tolerance."""
    a = np.array(a)
    b = np.array(b)
    max_err = np.max(np.abs(a - b))
    assert max_err < tol, f"{label} — max error {max_err:.2e} exceeds tol {tol:.2e}"


# ---------------------------------------------------------------------------
# StateVector — initialization
# ---------------------------------------------------------------------------

class TestStateVectorInit:

    @pytest.mark.parametrize("n_qubits, batch_size, expected_shape", [
        (1, 10,   (10, 2)),
        (3, 20,   (20, 8)),
        (5, 30,   (30, 32)),
        (10, 100, (100, 1024)),
    ])
    def test_zero_state_shape(self, n_qubits, batch_size, expected_shape):
        """data shape should be (batch_size, dim)."""
        sv = StateVector(n_qubits=n_qubits, batch_size=batch_size, xp=xp)
        assert sv.data.shape == expected_shape

    @pytest.mark.parametrize("n_qubits, batch_size", [
        (1, 10),
        (3, 20),
        (5, 30),
        (10, 100),
    ])
    def test_zero_state_amplitudes(self, n_qubits, batch_size):
        """All amplitude should be in first basis state."""
        sv = StateVector(n_qubits=n_qubits, batch_size=batch_size, xp=xp)
        assert_close(sv.data[:, 0], np.ones(batch_size, dtype=complex))
        assert_close(sv.data[:, 1:], np.zeros((batch_size, 2**n_qubits - 1), dtype=complex))
    
    @pytest.mark.parametrize("n_qubits, batch_size", [
        (1, 10),
        (3, 20),
        (5, 30),
        (10, 100),
    ])
    def test_zero_state_dtype(self, n_qubits, batch_size):
        """dtype should be complex."""
        sv = StateVector(n_qubits=n_qubits, batch_size=batch_size, xp=xp)
        assert sv.data.dtype == np.complex128

    @pytest.mark.parametrize("n_qubits, batch_size", [
        (1, 10),
        (3, 20),
        (5, 30),
        (10, 100),
    ])
    def test_zero_state_normalization(self, n_qubits, batch_size):
        """Sum of |amplitude|^2 per row should equal 1."""
        sv = StateVector(n_qubits=n_qubits, batch_size=batch_size, xp=xp)
        assert_close(np.sum(np.abs(sv.data)**2, axis=1), np.ones(batch_size))

    def test_plus_state_normalization(self):
        pass

    def test_rand_state_normalization(self):
        pass

    def test_dim_derived_correctly(self):
        """dim should equal 2 ** n_qubits."""
        pass


# ---------------------------------------------------------------------------
# StateVector — gates
# ---------------------------------------------------------------------------

class TestStateVectorGates:

    def test_x_gate_flips_zero_to_one(self):
        """X|0> = |1>"""
        pass

    def test_x_gate_flips_one_to_zero(self):
        """X|1> = |0>"""
        pass

    def test_x_twice_is_identity(self):
        """X·X = I"""
        pass

    def test_h_gate_zero_to_plus(self):
        """H|0> = |+> = [1/sqrt(2), 1/sqrt(2)]"""
        pass

    def test_h_gate_one_to_minus(self):
        """H|1> = |-> = [1/sqrt(2), -1/sqrt(2)]"""
        pass

    def test_h_twice_is_identity(self):
        """H·H = I"""
        pass

    def test_z_gate_leaves_zero_unchanged(self):
        """Z|0> = |0>"""
        pass

    def test_norm_preserved_after_gate_sequence(self):
        """Applying multiple gates must preserve normalization."""
        pass

    def test_cnot_creates_bell_state(self):
        """H on qubit 0 then CNOT should produce (|00> + |11>)/sqrt(2)."""
        pass


# ---------------------------------------------------------------------------
# StateVector — measurement
# ---------------------------------------------------------------------------

class TestStateVectorMeasurement:

    def test_zero_state_always_measures_zero(self):
        pass

    def test_one_state_always_measures_one(self):
        pass

    def test_plus_state_measures_half_and_half(self):
        """P(1) for |+> should be ~0.5 over large batch."""
        pass

    def test_outcomes_are_binary(self):
        """All measurement outcomes should be 0 or 1."""
        pass


# ---------------------------------------------------------------------------
# StateVector — other methods
# ---------------------------------------------------------------------------

class TestStateVectorMethods:

    def test_self_fidelity_is_one(self):
        """F(|psi>, |psi>) = 1."""
        pass

    def test_orthogonal_fidelity_is_zero(self):
        """F(|0>, |1>) = 0."""
        pass

    def test_probabilities_sum_to_one(self):
        """Probabilities across all basis states should sum to 1."""
        pass

    def test_to_density_matrix_purity(self):
        """Pure state converted to DM should have purity = 1."""
        pass


# ---------------------------------------------------------------------------
# DensityMatrix — initialization
# ---------------------------------------------------------------------------

class TestDensityMatrixInit:

    def test_zero_state_shape(self):
        """data shape should be (batch_size, dim, dim)."""
        pass

    def test_zero_state_trace(self):
        """Tr(rho) should equal 1."""
        pass

    def test_mixed_state_purity(self):
        """Maximally mixed state should have purity 1/dim."""
        pass


# ---------------------------------------------------------------------------
# DensityMatrix — noise channels
# ---------------------------------------------------------------------------

class TestDensityMatrixNoise:

    def test_depolarize_zero_is_identity(self):
        """depolarize(p=0) should not change the state."""
        pass

    def test_depolarize_preserves_trace(self):
        pass

    def test_depolarize_full_gives_mixed(self):
        """depolarize(p=0.75) should give maximally mixed state."""
        pass

    def test_amplitude_damp_zero_is_identity(self):
        pass

    def test_amplitude_damp_full_collapses_to_zero(self):
        """amplitude_damp(gamma=1) should collapse any state to |0><0|."""
        pass

    def test_erasure_preserves_trace(self):
        pass

    def test_state_remains_hermitian_after_noise(self):
        """rho should equal rho† after any channel."""
        pass


# ---------------------------------------------------------------------------
# DensityMatrix — measurement
# ---------------------------------------------------------------------------

class TestDensityMatrixMeasurement:

    def test_zero_state_always_measures_zero(self):
        pass

    def test_plus_state_measures_half_and_half(self):
        pass

    def test_outcomes_are_binary(self):
        pass


# ---------------------------------------------------------------------------
# Cross-class consistency
# ---------------------------------------------------------------------------

class TestCrossClass:

    def test_sv_and_dm_agree_on_measurement_statistics(self):
        """SV and DM starting from same state should give same P(1)."""
        pass

    def test_sv_to_dm_preserves_purity(self):
        """Converting SV to DM should give purity = 1."""
        pass

    def test_purity_matches_after_noise(self):
        """SV->DM->depolarize and DM->depolarize should give same purity."""
        pass