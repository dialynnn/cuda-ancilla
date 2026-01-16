# cuda-ancilla

Standalone CUDA, CUTLASS, WMMA, and PTX experiments used as supporting infrastructure for GPU programming and GPU-based quantum-adjacent research. 

The name is borrowed from **ancilla bits** in quantum computing: temporary helpers, scratch space, and side computations that exist to make the main operation possible. This repository plays the same role: not a framework, not a product, but the place where ideas get tested, broken, and understood.

---

## What lives here

This repository contains **standalone CUDA programs** and utilities such as:

- WMMA / Tensor Core experiments
- CUTLASS baselines
- PTX and inline-ASM exploration
- Precision experiments (FP16 / FP32 / FP64 / any precision format NVIDIA makes)
- Performance measurement scaffolding
- “Does this even work?” kernels

Most files are:
- Single-purpose
- Minimally abstracted
- Meant to be read, modified, and discarded

---

## Requirements

- NVIDIA GPU with at least sm_70 compute capability
- `nvcc` compiler
- CUTLASS (optional, per-file)

---

## Disclaimer

This repository is experimental by design, and despite being MIT-licensed, each code may or may not contain its own license.