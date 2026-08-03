# Data Structure Optimization for Data Locality in HPC

A Python prototype demonstrating the **data locality optimization** technique
identified in Azad, Iqbal, Hassan, and Roy's (2023) empirical study of
high-performance computing (HPC) performance bugs, *[An Empirical Study of
High Performance Computing (HPC) Performance Bugs](https://foyzulhassan.github.io/files/MSR23_HPC.pdf)*
(MSR '23).

This repository accompanies **Final Project Part 1: Optimization Technique
and Implementation Project Report** for *Optimization in High-Performance
Computing*.

## Overview

Azad et al. (2023) found that inefficient algorithms and data structures were
the single largest root cause of real-world HPC performance bugs (39.3% of
186 confirmed bugs), and that data locality optimization — replacing
pointer-chasing containers (e.g., `std::list`, `std::forward_list`) with
contiguous, cache-friendly containers (e.g., `std::vector`) — was the single
most common fix strategy (21% of fixes), citing examples such as TileDB
commit `d51b082` and CGAL commit `8855eb5`.

This project reproduces that mechanism in Python with two controlled
benchmark experiments:

1. **Linked list vs. contiguous array traversal** — sequential summation over
   a singly linked list (pointer-chasing) compared against a Python list and
   a NumPy array (contiguous memory).
2. **Array-of-Structures (AoS) vs. Structure-of-Arrays (SoA)** — a minimal
   particle-integration kernel compared under two memory layouts that are
   functionally identical but differ only in how fields are laid out in
   memory.

## Repository Structure

```
.
├── src/
│   └── locality_benchmark.py        # Main benchmark script (both experiments)
├── results/
│   ├── experiment1_results.csv      # Raw timing data, Experiment 1
│   ├── experiment2_results.csv      # Raw timing data, Experiment 2
│   ├── experiment1_traversal_time.png
│   ├── experiment1_speedup.png
│   ├── experiment2_integration_time.png
│   └── experiment2_speedup.png
└── README.md
```

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib`

Install with:

```bash
pip install numpy matplotlib
```

## Usage

Run the full benchmark suite from the repository root:

```bash
python3 src/locality_benchmark.py
```

This will:

1. Run Experiment 1 (linked list vs. array traversal) across problem sizes
   N = 1,000 to 1,000,000, using `timeit` with 5 repeats per configuration.
2. Run Experiment 2 (AoS vs. SoA particle integration) across the same
   problem sizes, 20 integration steps per timed run.
3. Write raw timing data to `results/*.csv`.
4. Write comparison plots to `results/*.png`.


## Results Summary

**Experiment 1 — Sequential Sum (best-of-5 timing, ms)**

| N | Linked List | Python List | NumPy Array | Speedup (List/Array) | Speedup (List/NumPy) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.035 | 0.026 | 0.004 | 1.38x | 9.32x |
| 10,000 | 0.393 | 0.256 | 0.010 | 1.54x | 40.97x |
| 100,000 | 6.486 | 3.074 | 0.047 | 2.11x | 137.71x |
| 500,000 | 80.958 | 14.999 | 0.105 | 5.40x | 773.23x |
| 1,000,000 | 173.774 | 38.658 | 0.207 | 4.50x | 840.70x |

**Experiment 2 — 20-Step Particle Integration (best-of-5 timing, ms)**

| N | AoS | SoA | Speedup (AoS/SoA) |
|---:|---:|---:|---:|
| 1,000 | 19.569 | 0.134 | 146.25x |
| 10,000 | 234.982 | 0.407 | 577.35x |
| 100,000 | 2,314.059 | 3.021 | 766.02x |
| 500,000 | 11,846.781 | 94.442 | 125.44x |
| 1,000,000 | 23,580.032 | 189.414 | 124.49x |


## References

- Azad, M. A. K., Iqbal, N., Hassan, F., & Roy, P. (2023). An empirical study
  of high performance computing (HPC) performance bugs. In *Proceedings of
  the 20th IEEE/ACM International Conference on Mining Software Repositories
  (MSR '23)* (pp. 1–12). IEEE.
  [https://foyzulhassan.github.io/files/MSR23_HPC.pdf](https://foyzulhassan.github.io/files/MSR23_HPC.pdf)
- Chilimbi, T. M., Hill, M. D., & Larus, J. R. (2000). Making pointer-based
  data structures cache conscious. *Computer, 33*(12), 67–74.
  https://doi.org/10.1109/2.889095
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array
  programming with NumPy. *Nature, 585*(7825), 357–362.
  https://doi.org/10.1038/s41586-020-2649-2
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful
  visual performance model for multicore architectures. *Communications of
  the ACM, 52*(4), 65–76. https://doi.org/10.1145/1498765.1498785
