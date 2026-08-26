"""
Convergence study of the OUTER quadrature order in the two singular branches of
the EFIE assembly, measured through the reciprocity residual err_T.

Why err_T is the right observable
---------------------------------
The exact Galerkin MoM matrix is complex-symmetric, Z = Z^T, whatever the
geometry. The assembler nevertheless produces two different numbers for Z_mn and
Z_nm when it computes both halves (`full_fill=True`), because two of its four
branches are NOT symmetric in the pair's two triangles:

    SELF          Gauss-Dunavant on the TEST triangle, Duffy on the SOURCE one
    EDGE / VERTEX Gauss-Dunavant on both, but the 1/R half of the scalar
                  potential is integrated in CLOSED FORM over the SOURCE only,
                  leaving the test-side integration purely numerical
    FAR           the same rule on both triangles -- symmetric by construction

So err_T = max|Z - Z^T| / max|Z| is a direct, geometry-free readout of the outer
quadrature's own error on the singular branches. It needs no reference solution,
which is what makes it usable on any mesh.

What this module settles
------------------------
"Structural" is not "correct". If err_T is truncation error it must DECREASE as
the outer rule is refined; if it stagnates, the residual is a bug and no amount
of quadrature will remove it. Deciding between those two requires sweeping the
outer order -- which was impossible until recently for the EDGE/VERTEX branch:
its order is min(quad_order + 5, 11), and 11 was the largest rule in
gauss_dunavant's table, so the default already sat on the ceiling and raising
quad_order moved nothing. Degrees 16/20/24 were added there to lift it.

Run it::

    python -m backend.utils.quadrature.outer_order_study
    python -m backend.utils.quadrature.outer_order_study --mesh <path to _mesh2.mat>

Reading the output
------------------
Per-category tables come from `only_categories=`, which zeroes every sub-pair
outside the requested categories. Because the four branches contribute
additively to each Z_mn, their asymmetries add too, and the script checks that
identity explicitly (`decomposition residual`) rather than assuming it.
Every err_T in every table is divided by the SAME max|Z| -- that of the full
production matrix at the reference order -- so columns are comparable down a
table and across tables.

err_T is reported next to err_Z = max|Z(order) - Z(top order)| / max|Z|, the
matrix's own convergence towards the most refined rule swept. Both are needed,
because err_T has a blind spot: it is a DIFFERENCE of two errors, and when the
outer rule is invariant under permutation of a triangle's vertices (the
symmetric Dunavant orbits, degrees <= 6) and the mesh puts a locally symmetric
pair of triangles across a shared edge, the two errors are equal and cancel.
err_T then reads roundoff while the underlying quadrature error is not small at
all -- so err_T is a LOWER BOUND on the error, not a measure of it. err_Z has no
such cancellation and is the column that decides truncation versus bug.
"""

import argparse
import time

import numpy as np

from backend.rwg.rwg1 import Points, Triangles
from backend.utils.impmet import impedance_matrix_z, impedance_matrix_z_sweep_gpu
from backend.utils.quadrature.gauss_dunavant import dunavant_rule
from backend.utils.quadrature.singularity_classifier import SELF, EDGE, VERTEX, FAR

EPSILON = 8.854e-12
MU = 1.257e-6
C = 1.0 / np.sqrt(EPSILON * MU)

#: Outer orders swept, in increasing accuracy. Point counts 6, 7, 12, 25, 49,
#: 81, 121, 169. 6 is the SELF default, 11 the EDGE/VERTEX default.
OUTER_ORDERS = (4, 5, 6, 7, 11, 16, 20, 24)

CATEGORY_NAMES = {int(SELF): "SELF", int(EDGE): "EDGE", int(VERTEX): "VERTEX", int(FAR): "FAR"}


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------

def right_angle_junction_mesh(n_cells=2, a=0.04):
    """
    Two strips meeting at 90 degrees along a shared edge, refinable in `n_cells`.

    `n_cells=2` reproduces the mesh of test/test_impedance_non_coplanar.py
    exactly (8 triangles, 8 RWG edges). The bend is what makes the study
    meaningful: the closed-form 1/R integral's height term vanishes identically
    for a coplanar observation point, so a flat mesh cannot exercise the term
    that dominates the EDGE/VERTEX error.

    Row A lies on the seam (y=z=0); row B extends one strip out in +y, row C
    extends the other up in +z. Every seam edge is shared by exactly one
    triangle from each strip -- a clean bend, not a T-junction.
    """
    nx = n_cells
    xs = np.arange(nx + 1) * a
    zero = np.zeros(nx + 1)
    row_a = np.vstack([xs, zero, zero])
    row_b = np.vstack([xs, zero + a, zero])
    row_c = np.vstack([xs, zero, zero + a])
    points = np.hstack([row_a, row_b, row_c])

    off_a, off_b, off_c = 0, nx + 1, 2 * (nx + 1)
    faces = []
    for off in (off_b, off_c):
        for i in range(nx):
            faces.append([off_a + i, off_a + i + 1, off + i])
            faces.append([off_a + i + 1, off + i + 1, off + i])
    triangles = np.vstack([np.array(faces).T, np.zeros((1, len(faces)), dtype=int)]).astype(int)

    p = Points(points)
    t = Triangles(triangles)
    t.calculate_triangles_area_and_center(p)
    e = t.get_edges()
    e.compute_edges_length(p)
    return p, t, e


def load_mesh2(path):
    """Load a real mesh from a `_mesh2.mat` file produced by the rwg2 stage."""
    from backend.rwg.rwg2 import DataManager_rwg2
    points, triangles, edges, _rho = DataManager_rwg2.load_data(path)
    return points, triangles, edges


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def asymmetry(Z, denom):
    """max|Z-Z^T|, split into total / real / imaginary, normalised by `denom`."""
    D = Z - Z.T
    return {
        'total': float(np.abs(D).max()) / denom,
        're': float(np.abs(D.real).max()) / denom,
        'im': float(np.abs(D.imag).max()) / denom,
    }


def fill(mesh, frequency, **kwargs):
    """Assemble Z with both halves computed independently; return (Z, seconds)."""
    points, triangles, edges = mesh
    omega = 2.0 * np.pi * frequency
    t0 = time.perf_counter()
    Z = impedance_matrix_z(edges, triangles, points, 1j * omega / C, omega, MU, EPSILON,
                           full_fill=True, **kwargs)
    return Z, time.perf_counter() - t0


def sweep(mesh, frequency, denom, orders=OUTER_ORDERS, categories=None,
          duffy_order=7, quad_order=6, label=""):
    """
    Assemble Z once per outer order and record err_T, err_Z and the fill time.

    `categories` restricts which sub-pair branches contribute (None = all four).
    Both `near_order` (EDGE/VERTEX) and `self_outer_order` (SELF) are set to the
    swept order; the FAR rule stays at `quad_order`, since FAR is symmetric to
    machine precision whatever its order and only inflates the cost.

    err_Z is measured against the LAST order swept, so `orders` must be given in
    increasing order of accuracy. It is the branch's own convergence and, unlike
    err_T, cannot be flattered by an error that cancels between the two halves.
    """
    rows, mats = [], []
    for order in orders:
        Z, secs = fill(mesh, frequency, quad_order=quad_order, duffy_order=duffy_order,
                       near_order=order, self_outer_order=order,
                       only_categories=categories)
        mats.append(Z)
        rows.append({
            'order': order,
            'n_points': len(dunavant_rule(order)[1]),
            'seconds': secs,
            **asymmetry(Z, denom),
        })
        print(f"  [{label or 'all'}] order {order:>2} done in {secs:6.2f} s "
              f"-> err_T = {rows[-1]['total']:.3e}", flush=True)

    reference = mats[-1]
    for row, Z in zip(rows, mats):
        row['err_z'] = float(np.abs(Z - reference).max()) / denom
    return rows


def print_table(rows, title, columns=('total', 'err_z')):
    """Render one sweep as a fixed-width table."""
    head = {'total': 'err_T', 're': 'err_T Re(Z)', 'im': 'err_T Im(Z)', 'err_z': 'err_Z'}
    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'ordre':>6} | {'nb pts':>6} | " + " | ".join(f"{head[c]:>12}" for c in columns) \
             + f" | {'temps (s)':>10} | {'ratio err_T':>11}"
    print(header)
    print("-" * len(header))
    prev = None
    for r in rows:
        cells = " | ".join("      (ref)" if c == 'err_z' and r is rows[-1] else f"{r[c]:12.3e}"
                           for c in columns)
        ratio = "-" if prev is None or r['total'] == 0 else f"{prev / r['total']:11.2f}"
        print(f"{r['order']:>6} | {r['n_points']:>6} | {cells} | {r['seconds']:10.2f} | {ratio:>11}")
        prev = r['total']


def self_branch_grid(mesh, frequency, denom, outer_orders, duffy_orders):
    """
    Cross-sweep the SELF branch over (self_outer_order, duffy_order).

    A one-dimensional sweep of the outer order cannot diagnose this branch. Its
    two rules are coupled: raising the outer order moves outer nodes towards the
    triangle's edges, and Duffy's subdivision from the singular point degenerates
    into slivers there, so a finer outer rule asks a FIXED duffy_order to do
    strictly harder work. Non-monotone convergence in the outer order at fixed
    duffy_order is therefore expected and says nothing on its own; only the grid
    shows whether the branch converges when both are refined together.

    Everything is compared against the most refined corner of the grid.
    """
    print(f"\n--- SELF branch: (self_outer_order x duffy_order) grid ---")
    err_z = np.zeros((len(outer_orders), len(duffy_orders)))
    err_t = np.zeros_like(err_z)
    mats = {}
    for i, o in enumerate(outer_orders):
        for j, d in enumerate(duffy_orders):
            Z, secs = fill(mesh, frequency, self_outer_order=o, duffy_order=d,
                           only_categories={SELF})
            mats[(i, j)] = Z
            err_t[i, j] = asymmetry(Z, denom)['total']
            print(f"  outer {o:>2} x duffy {d:>2}: {secs:6.2f} s", flush=True)

    reference = mats[(len(outer_orders) - 1, len(duffy_orders) - 1)]
    for (i, j), Z in mats.items():
        err_z[i, j] = float(np.abs(Z - reference).max()) / denom

    corner = "outer \\ duffy"
    for name, table in (("err_Z", err_z), ("err_T", err_t)):
        title = f"Table 4 - SELF pairs only, {name} vs (outer order, duffy order)"
        print(f"\n{title}\n{'-' * len(title)}")
        print(f"{corner:>12} | " + " | ".join(f"{d:>10}" for d in duffy_orders))
        print("-" * (14 + 13 * len(duffy_orders)))
        for i, o in enumerate(outer_orders):
            cells = " | ".join("     (ref)" if (i == len(outer_orders) - 1
                                                and j == len(duffy_orders) - 1)
                               else f"{table[i, j]:10.3e}"
                               for j in range(len(duffy_orders)))
            print(f"{o:>12} | {cells}")
    return {'outer_orders': outer_orders, 'duffy_orders': duffy_orders,
            'err_z': err_z, 'err_t': err_t}


def production_cost_and_rank(mesh, frequency, near_orders=(11, 16, 20),
                             self_outer_order=6, duffy_order=7, use_gpu=None):
    """
    What raising `near_order` costs and buys on the path a real solve uses.

    Everything above runs `full_fill=True` on the per-pair CPU assembler, which
    is roughly twice the work of a production fill and skips the batched/GPU
    route entirely -- so its timings say nothing about what a user would pay.
    This measures `impedance_matrix_z_sweep_gpu` instead: mirrored fill, one
    frequency, on whichever backend is live.

    It also reports R = Re(Z)'s numerical rank, because that is the quantity
    CMA's rank-truncated solver actually consumes. The EDGE/VERTEX branch's
    closed-form 1/R term is REAL and carries a purely IMAGINARY prefactor, so
    refining near_order is felt in X = Im(Z) and barely touches R. Printing the
    rank next to the cost keeps that from being assumed either way.
    """
    points, triangles, edges = mesh
    print(f"\n--- Production path (impedance_matrix_z_sweep_gpu, mirrored fill) ---")
    print(f"{'near_order':>10} | {'nb pts':>6} | {'temps (s)':>10} | {'rank(R)':>8} | "
          f"{'d|Re Z|':>10} | {'d|Im Z|':>10}")
    print("-" * 70)

    baseline = None
    for order in near_orders:
        t0 = time.perf_counter()
        Z = impedance_matrix_z_sweep_gpu(edges, triangles, points, [frequency], MU, EPSILON,
                                         near_order=order, self_outer_order=self_outer_order,
                                         duffy_order=duffy_order, use_gpu=use_gpu)[0]
        secs = time.perf_counter() - t0

        R = (Z.real + Z.real.T) * 0.5
        rank = int(np.linalg.matrix_rank(R))
        if baseline is None:
            baseline, d_re, d_im = Z, 0.0, 0.0
        else:
            scale = float(np.abs(baseline).max())
            d_re = float(np.abs(Z.real - baseline.real).max()) / scale
            d_im = float(np.abs(Z.imag - baseline.imag).max()) / scale
        print(f"{order:>10} | {len(dunavant_rule(order)[1]):>6} | {secs:10.2f} | {rank:>8} | "
              f"{d_re:10.3e} | {d_im:10.3e}")


def self_duffy_at_production_outer_order(mesh, frequency, denom, self_outer_order=6,
                                         duffy_orders=(7, 12, 20)):
    """
    Is the SELF branch converged in `duffy_order` AT the production outer order?

    The full (outer x duffy) grid is unaffordable on a real mesh -- its expensive
    corner alone runs for about an hour -- and it answers a question production
    never asks, since production pins the SELF outer rule at quad_order. Holding
    the outer order fixed and refining only Duffy isolates the inner rule's error
    at the setting that actually ships, for a few minutes of compute.
    """
    print(f"\n--- SELF branch at self_outer_order={self_outer_order}: refining duffy_order ---")
    print(f"{'duffy_order':>11} | {'temps (s)':>10} | {'err_T':>11} | {'err_Z vs finest':>15}")
    print("-" * 56)
    mats = []
    for d in duffy_orders:
        Z, secs = fill(mesh, frequency, self_outer_order=self_outer_order, duffy_order=d,
                       only_categories={SELF})
        mats.append((d, secs, Z, asymmetry(Z, denom)['total']))
    finest = mats[-1][2]
    for d, secs, Z, err_t in mats:
        err_z = float(np.abs(Z - finest).max()) / denom
        shown = "        (ref)" if Z is finest else f"{err_z:15.3e}"
        print(f"{d:>11} | {secs:10.2f} | {err_t:11.3e} | {shown}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(mesh, frequency, orders=OUTER_ORDERS, duffy_orders=(7,), name=""):
    points, triangles, edges = mesh
    print(f"\n{'=' * 78}\nMesh: {name}   E = {edges.total_number_of_edges} RWG edges, "
          f"{triangles.total_of_triangles} triangles,  f = {frequency / 1e6:.1f} MHz\n{'=' * 78}")

    # Reference full fill at the production defaults: fixes the normalisation
    # and gives the number the rest of the study has to explain.
    Z_ref, secs_ref = fill(mesh, frequency)
    denom = float(np.abs(Z_ref).max())
    ref = asymmetry(Z_ref, denom)
    print(f"\nProduction default (quad_order=6 -> near_order=11, self_outer_order=6, duffy_order=7)")
    print(f"  max|Z| = {denom:.4e}   fill = {secs_ref:.2f} s")
    print(f"  err_T = {ref['total']:.3e}   (Re {ref['re']:.3e}, Im {ref['im']:.3e})")

    # Control + additivity: FAR must be symmetric to roundoff, and the four
    # branches must reconstruct the full matrix. If either fails, every
    # per-category number below is meaningless.
    Z_far, _ = fill(mesh, frequency, only_categories={FAR})
    Z_self, _ = fill(mesh, frequency, only_categories={SELF})
    Z_near, _ = fill(mesh, frequency, only_categories={EDGE, VERTEX})
    print(f"  FAR-only err_T (control, must be roundoff) = "
          f"{asymmetry(Z_far, denom)['total']:.3e}")
    print(f"  decomposition residual max|Z-(FAR+SELF+EDGE/VERTEX)|/max|Z| = "
          f"{float(np.abs(Z_ref - (Z_far + Z_self + Z_near)).max()) / denom:.3e}")

    results = {'reference': ref, 'denom': denom}

    print(f"\n--- Sweeping the outer order (near_order = self_outer_order = order) ---")
    results['all'] = sweep(mesh, frequency, denom, orders, None, label="all")
    print_table(results['all'],
                "Table 1 - full Z, all categories", ('total', 'err_z'))
    print_table(results['all'],
                "Table 2 - full Z, split Re / Im", ('re', 'im'))

    for key, cats, lbl in (('self', {SELF}, 'SELF'),
                           ('edge', {EDGE}, 'EDGE'),
                           ('vertex', {VERTEX}, 'VERTEX')):
        print(f"\n--- {lbl} pairs only ---")
        results[key] = sweep(mesh, frequency, denom, orders, cats, label=lbl)
        print_table(results[key], f"Table 3{key[0]} - {lbl} pairs only",
                    ('total', 're', 'im', 'err_z'))

    if len(duffy_orders) > 1:
        results['self_grid'] = self_branch_grid(mesh, frequency, denom, orders, duffy_orders)

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--mesh', default=None,
                    help="path to a _mesh2.mat file; default = the built-in 90-degree junction")
    ap.add_argument('--cells', type=int, default=2,
                    help="refinement of the built-in junction mesh (default 2, the test's mesh)")
    ap.add_argument('--frequency', type=float, default=440e6, help="Hz (default 440e6)")
    ap.add_argument('--orders', type=int, nargs='*', default=list(OUTER_ORDERS))
    ap.add_argument('--duffy-orders', type=int, nargs='*', default=[7],
                    help="pass more than one to add the SELF/duffy_order cross-sweep")
    args = ap.parse_args()

    if args.mesh:
        mesh, name = load_mesh2(args.mesh), args.mesh
    else:
        mesh, name = right_angle_junction_mesh(args.cells), f"90-degree junction, {args.cells} cells"

    run(mesh, args.frequency, tuple(args.orders), tuple(args.duffy_orders), name)


if __name__ == '__main__':
    main()
