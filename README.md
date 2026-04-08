# quantumsim

A GPU-accelerated quantum protocol simulator built on CuPy and NumPy. Simulates thousands of independent quantum trials in parallel using a backend-agnostic design — the same protocol code runs on CPU or GPU by switching one environment variable.

Built as a portfolio project exploring high-performance computing applied to quantum networking and computing protocols.

---

## Status

- [ ] `StateVector` and `DensityMatrix` primitives (batched, backend-agnostic)
- [ ] Noise channels: depolarizing, amplitude damping, phase damping, erasure
- [ ] 28 unit tests validated against analytic results
- [ ] BB84 QKD protocol
- [ ] E91 entanglement-based QKD
- [ ] Grover's search algorithm
- [ ] GPU vs CPU benchmark suite

---

## Design

The core idea is that every operation runs over a *batch* of quantum states simultaneously rather than simulating trials sequentially. A `StateVector` with `batch_size=1_000_000` runs as a single GPU kernel, not a Python loop.

Backend switching is handled by a single import alias:

```python
# quantumsim/backend.py
xp = cupy  # or numpy — set QUANTUMSIM_BACKEND=gpu to switch
```

All protocol and primitive code uses `xp` instead of `np` or `cp` directly, so nothing above the backend layer needs to change when switching hardware.

---

## Quickstart

```bash
git clone https://github.com/yourusername/quantumsim.git
cd quantumsim
pip install -r requirements.txt

# run tests
python -m pytest tests/ -v

# GPU backend (requires CuPy and CUDA)
QUANTUMSIM_BACKEND=gpu python -m pytest tests/ -v
```

---

## Requirements

- Python 3.10+
- NumPy
- CuPy (optional, for GPU backend — see [CuPy installation](https://docs.cupy.dev/en/stable/install.html))
- pytest

---

## Roadmap

Once the core protocols are implemented, the goal is a benchmark suite comparing CPU vs GPU throughput across trial counts (10k → 10M) and noise levels, with reproducible results exportable to JSON.

---

## Background

This project came out of prior research in quantum network simulation during a graduate program. The focus here is less on the physics and more on what happens when you need to simulate protocols at scale — where the bottleneck shifts from algorithmic complexity to memory bandwidth and kernel utilization.
