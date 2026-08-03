"""
locality_benchmark.py
----------------------------------------------------------------------
Final Project Part 1 -- Prototype implementation

Demonstrates the "Data Structure Optimization for Data Locality"
technique identified in:

    Azad, M. A. K., Iqbal, N., Hassan, F., & Roy, P. (2023). An empirical
    study of high performance computing (HPC) performance bugs. In
    Proceedings of the 20th IEEE/ACM International Conference on Mining
    Software Repositories (MSR '23).

The paper reports that 9 of 186 real-world HPC performance commits were
caused by choosing an inefficient (pointer-chasing / linked) data
structure over a contiguous one, e.g.:

    * TileDB commit d51b082 : std::forward_list -> std::vector
    * CGAL   commit 8855eb5 : std::list        -> std::vector
    * ArrayFire commit ee30e27: std::vector    -> std::array

Two experiments are provided:

  EXPERIMENT 1 -- Pointer-chasing (linked list) vs. contiguous (array)
                  traversal of N elements, replicating the
                  forward_list -> vector / list -> vector fix pattern.

  EXPERIMENT 2 -- Array-of-Structures (AoS) vs Structure-of-Arrays (SoA)
                  layout for a simple N-body / particle-update kernel,
                  replicating the "reorder memory reference for data
                  reuse" / "data structure optimization" fix pattern
                  (c.f. GROMACS float3 packing commit 85c36b9).

Each experiment is run for a range of problem sizes, timed with
`timeit`, and the results are written to CSV and plotted to PNG so they
can be embedded in the project report.
"""

import gc
import os
import sys
import time
import timeit
import statistics
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# EXPERIMENT 1: Linked list (pointer-chasing) vs contiguous array
# ======================================================================

class Node:
    """Minimal singly linked-list node -- the Python analogue of a
    pointer-based C/C++ node allocated individually on the heap."""
    __slots__ = ("value", "next")

    def __init__(self, value, nxt=None):
        self.value = value
        self.next = nxt


def build_linked_list(n, shuffle_allocations=True):
    """Builds a singly linked list of n nodes.

    shuffle_allocations=True intentionally allocates nodes in a
    randomized order and then links them together, which mimics the
    fragmented heap layout that real long-running HPC / STL
    forward_list / list containers accumulate after repeated
    insert/erase cycles. This is the realistic worst case that the
    empirical study's fixes were addressing -- a freshly built,
    perfectly-ordered linked list would understate the effect.
    """
    import random
    indices = list(range(n))
    if shuffle_allocations:
        random.seed(42)
        random.shuffle(indices)

    nodes = {}
    for i in indices:
        nodes[i] = Node(i)

    head = nodes[0]
    cur = head
    for i in range(1, n):
        cur.next = nodes[i]
        cur = nodes[i]
    return head


def build_array(n):
    """Contiguous, cache-friendly container (Python list of ints backed
    by an array of pointers is *not* fully contiguous the way a C++
    std::vector<int> is; to faithfully model the paper's vector/array
    fix we additionally test a NumPy array, which -is- a truly
    contiguous block of raw memory)."""
    return list(range(n))


def build_numpy_array(n):
    return np.arange(n, dtype=np.int64)


def sum_linked_list(head):
    total = 0
    node = head
    while node is not None:
        total += node.value
        node = node.next
    return total


def sum_array(arr):
    total = 0
    for v in arr:
        total += v
    return total


def sum_numpy(arr):
    return int(arr.sum())


def run_experiment_1(sizes, repeats=5):
    rows = []
    for n in sizes:
        head = build_linked_list(n)
        arr = build_array(n)
        np_arr = build_numpy_array(n)

        gc.collect()
        t_list = timeit.repeat(lambda: sum_linked_list(head), number=1, repeat=repeats)
        gc.collect()
        t_arr = timeit.repeat(lambda: sum_array(arr), number=1, repeat=repeats)
        gc.collect()
        t_np = timeit.repeat(lambda: sum_numpy(np_arr), number=1, repeat=repeats)

        row = {
            "n": n,
            "linked_list_s": min(t_list),
            "python_array_s": min(t_arr),
            "numpy_array_s": min(t_np),
            "speedup_array_over_list": min(t_list) / min(t_arr),
            "speedup_numpy_over_list": min(t_list) / min(t_np),
        }
        rows.append(row)
        print(f"[Exp1] n={n:>9,}  linked_list={row['linked_list_s']*1e3:9.3f} ms  "
              f"python_array={row['python_array_s']*1e3:9.3f} ms  "
              f"numpy={row['numpy_array_s']*1e3:9.3f} ms  "
              f"speedup(list/array)={row['speedup_array_over_list']:.2f}x  "
              f"speedup(list/numpy)={row['speedup_numpy_over_list']:.2f}x")
    return rows


# ======================================================================
# EXPERIMENT 2: Array-of-Structures (AoS) vs Structure-of-Arrays (SoA)
# ======================================================================

class Particle:
    """One particle stored as an individual Python object -- the AoS
    analogue: position/velocity/mass fields for a single particle are
    co-located, but successive particles are scattered across the heap
    (pointer array of objects), just like a naive
    std::vector<Particle> of a struct with poor packing referenced in
    the empirical study's micro-architecture-specific optimizations
    (Section IV-A of Azad et al., 2023)."""
    __slots__ = ("x", "y", "z", "vx", "vy", "vz", "mass")

    def __init__(self, x, y, z, vx, vy, vz, mass):
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.mass = mass


def build_aos(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3))
    vel = rng.random((n, 3))
    mass = rng.random(n) + 0.5
    return [Particle(*pos[i], *vel[i], mass[i]) for i in range(n)]


def build_soa(n, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.random(n), "y": rng.random(n), "z": rng.random(n),
        "vx": rng.random(n), "vy": rng.random(n), "vz": rng.random(n),
        "mass": rng.random(n) + 0.5,
    }


def integrate_aos(particles, dt):
    """Explicit Euler position update -- classic HPC kernel."""
    for p in particles:
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.z += p.vz * dt
    return particles


def integrate_soa(soa, dt):
    soa["x"] += soa["vx"] * dt
    soa["y"] += soa["vy"] * dt
    soa["z"] += soa["vz"] * dt
    return soa


def run_experiment_2(sizes, repeats=5, steps=20):
    rows = []
    for n in sizes:
        particles = build_aos(n)
        soa = build_soa(n)

        gc.collect()
        t_aos = timeit.repeat(lambda: [integrate_aos(particles, 0.01) for _ in range(steps)],
                               number=1, repeat=repeats)
        gc.collect()
        t_soa = timeit.repeat(lambda: [integrate_soa(soa, 0.01) for _ in range(steps)],
                               number=1, repeat=repeats)

        row = {
            "n": n,
            "aos_s": min(t_aos),
            "soa_s": min(t_soa),
            "speedup_soa_over_aos": min(t_aos) / min(t_soa),
        }
        rows.append(row)
        print(f"[Exp2] n={n:>9,}  AoS={row['aos_s']*1e3:9.3f} ms  "
              f"SoA={row['soa_s']*1e3:9.3f} ms  "
              f"speedup(AoS/SoA)={row['speedup_soa_over_aos']:.2f}x")
    return rows


# ======================================================================
# Output helpers
# ======================================================================

def write_csv(rows, filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def plot_experiment_1(rows):
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ns, [r["linked_list_s"] * 1e3 for r in rows], marker="o", label="Linked list (pointer-chasing)")
    plt.plot(ns, [r["python_array_s"] * 1e3 for r in rows], marker="s", label="Python list (array-like)")
    plt.plot(ns, [r["numpy_array_s"] * 1e3 for r in rows], marker="^", label="NumPy array (contiguous)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of elements (N)")
    plt.ylabel("Traversal time (ms, best of repeats)")
    plt.title("Experiment 1: Sequential Sum -- Linked List vs. Contiguous Array")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment1_traversal_time.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")

    plt.figure(figsize=(7, 5))
    plt.plot(ns, [r["speedup_array_over_list"] for r in rows], marker="o", label="Python array vs linked list")
    plt.plot(ns, [r["speedup_numpy_over_list"] for r in rows], marker="^", label="NumPy array vs linked list")
    plt.xscale("log")
    plt.xlabel("Number of elements (N)")
    plt.ylabel("Speedup (x)")
    plt.title("Experiment 1: Speedup From Contiguous Data Structures")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment1_speedup.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def plot_experiment_2(rows):
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ns, [r["aos_s"] * 1e3 for r in rows], marker="o", label="Array-of-Structures (AoS)")
    plt.plot(ns, [r["soa_s"] * 1e3 for r in rows], marker="s", label="Structure-of-Arrays (SoA)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of particles (N)")
    plt.ylabel("20-step integration time (ms, best of repeats)")
    plt.title("Experiment 2: Particle Position Update -- AoS vs. SoA")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment2_integration_time.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")

    plt.figure(figsize=(7, 5))
    plt.bar([str(n) for n in ns], [r["speedup_soa_over_aos"] for r in rows], color="teal")
    plt.xlabel("Number of particles (N)")
    plt.ylabel("Speedup (x), SoA over AoS")
    plt.title("Experiment 2: Speedup From Structure-of-Arrays Layout")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "experiment2_speedup.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def main():
    print("=" * 78)
    print("EXPERIMENT 1: Linked List vs. Contiguous Array Traversal")
    print("=" * 78)
    sizes_1 = [1_000, 10_000, 100_000, 500_000, 1_000_000]
    rows_1 = run_experiment_1(sizes_1, repeats=5)
    write_csv(rows_1, "experiment1_results.csv")
    plot_experiment_1(rows_1)

    print()
    print("=" * 78)
    print("EXPERIMENT 2: Array-of-Structures vs. Structure-of-Arrays")
    print("=" * 78)
    sizes_2 = [1_000, 10_000, 100_000, 500_000, 1_000_000]
    rows_2 = run_experiment_2(sizes_2, repeats=5, steps=20)
    write_csv(rows_2, "experiment2_results.csv")
    plot_experiment_2(rows_2)

    print()
    print("Done. Results and plots written to:", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
