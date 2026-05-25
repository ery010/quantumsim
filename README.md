# quantum-gpu-projects

A collection of GPU-accelerated quantum simulation projects
built on CuPy and NumPy. Each project explores a different
area of quantum computing and networking, unified by a
shared high-performance simulation framework.

---

## Projects

### 1. Quantum Clock Network Simulator
*Directory: `clock-network/`*

GPU-accelerated simulation of the Komar et al. quantum
clock network protocol — the same protocol studied in the
published IEEE research above.

Simulates GHZ-state preparation across K geographically
distributed clock nodes, phase accumulation during
interrogation, and parity-based phase estimation to
characterize the Heisenberg-limited precision advantage
(1/N scaling) over the standard quantum limit (1/√N).

State vector size scales as 2^(K×n) complex amplitudes.
GPU acceleration via CuPy; benchmarked against NumPy
baseline across network sizes.

**Status:**
- [ ] GHZ state preparation across K nodes
- [ ] Phase rotation under realistic detuning model
- [ ] Parity measurement and phase estimation
- [ ] Allan deviation comparison: quantum/quantum vs classical
- [ ] GPU vs CPU benchmark suite (N = K×n qubits)

---

### 2. Quantum Protocol Simulator
*Directory: `protocol-sim/`*

A backend-agnostic framework for simulating quantum
cryptographic and computing protocols at scale. The core
idea: simulate thousands of independent quantum trials
in parallel as a single GPU kernel rather than a Python loop.

A `StateVector` with `batch_size=1_000_000` runs as one
GPU operation, making large-scale Monte Carlo simulation
of quantum protocols practical.

Backend switching is handled by a single environment variable:

```python