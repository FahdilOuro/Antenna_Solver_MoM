# GPU acceleration

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- Installing CuPy for your CUDA version, and how the solver detects it
  (`solver/utils/compute_backend.py`).
- The `use_gpu` argument: `None` (automatic, GPU only above `DEFAULT_GPU_EDGE_THRESHOLD = 100`
  edges), `True`, `False`.
- The memory check before a GPU assembly, and why a CUDA failure during a run raises an explicit
  error instead of silently falling back to the CPU.
- Expected speed-ups, measured on the development machine.
