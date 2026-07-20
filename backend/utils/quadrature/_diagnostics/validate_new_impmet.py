"""
Phase 6 validation of the new impedance_matrix_z.

Acceptance criteria on the 4-triangle and 6-triangle synthetic meshes:

  1. Symmetry: ||R-Rᵀ||/||R|| < 5e-4
               ||X-Xᵀ||/||X|| < 5e-3
     (vs 10-13% / 20-31% for legacy).
     Residual asymmetry is numerical, dominated by the outer-Gauss/inner-Duffy
     scheme on SELF pairs: both Gauss and Duffy halves are accurate individually,
     but the split is asymmetric in (m,n).  Decreases with higher quadrature order.

  2. PSD: min_eig(Re(Z)) >= 0  (legacy gave negative eigenvalues)

  3. Far-pair formula correctness at LOW frequency (f=3 MHz, k*cell≈0.063):
     relative error vs brute-force reference < 1%
     Tested at 3 MHz (not 300 MHz) because at 300 MHz the Green's function
     exp(-jkR)/R oscillates ~4π across a 1m cell, and both the 7-point reference
     and the 12-point new rule are inaccurate (>>5% quadrature error each).
     At 3 MHz both rules are accurate to << 0.1%.

Run:
  cd <repo_root>
  python -m backend.utils.quadrature._diagnostics.validate_new_impmet
"""

import sys
import os
import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.utils.impmet import impedance_matrix_z, impedance_matrix_z_makarov_legacy
from backend.utils.quadrature._diagnostics.verify_z_properties import check_z_properties
from backend.utils.quadrature._diagnostics.brute_force_reference import (
    build_four_triangle_mesh,
    build_six_triangle_mesh,
    _build_rwg_objects,
    build_z_reference,
    build_z_legacy,
    classify_edge_pair,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EPSILON = 8.854e-12
MU      = 1.257e-6
C       = 1.0 / np.sqrt(EPSILON * MU)

# 300 MHz (λ=1m): used for symmetry / PSD checks only.
# k*cell ≈ 2π here — quadrature is INACCURATE for the oscillatory G, but
# symmetry and eigenvalue sign are unaffected by the specific frequency.
FREQ_PHYS = 3e8

# 3 MHz (λ=100m): used for far-pair accuracy.
# k*cell ≈ 0.063 here — both 7-pt and 12-pt rules integrate G to <<0.1%.
FREQ_ACC  = 3e6


def _omega_k_complex(freq):
    omega = 2.0 * np.pi * freq
    k     = omega / C
    return omega, 1j * k


def _build_z_new(tri_obj, points_obj, edges_obj, freq):
    omega, k_complex = _omega_k_complex(freq)
    return impedance_matrix_z(
        edges_obj, tri_obj, points_obj,
        k_complex, omega, MU, EPSILON,
        quad_order=6, duffy_order=7,
    )


# ---------------------------------------------------------------------------
# Symmetry / PSD check at FREQ_PHYS
# ---------------------------------------------------------------------------

def _run_symmetry_psd_check(name, raw_pts, raw_tri):
    print(f"\n{'='*60}")
    print(f"  Mesh: {name}  (symmetry/PSD @ {FREQ_PHYS/1e6:.0f} MHz)")
    print(f"{'='*60}")

    points_obj, tri_obj, edges_obj, bary_obj, rho_obj = _build_rwg_objects(raw_pts, raw_tri)  # noqa: F841
    E = edges_obj.total_number_of_edges
    print(f"  Triangles: {tri_obj.total_of_triangles},  RWG edges: {E}")

    Z_legacy = build_z_legacy(tri_obj, edges_obj, bary_obj, rho_obj, FREQ_PHYS)
    Z_new    = _build_z_new(tri_obj, points_obj, edges_obj, FREQ_PHYS)

    print("\n--- LEGACY ---")
    d_leg = check_z_properties(Z_legacy, label="legacy")

    print("\n--- NEW ---")
    d_new = check_z_properties(Z_new, label="new")

    passed = True

    sym_R = d_new["sym_error_R"]
    sym_X = d_new["sym_error_X"]
    min_e = d_new["min_eig_R"]
    max_e = d_new["max_eig_R"]

    print()
    ok = sym_R < 5e-4
    print(f"[CHECK] ||R-Rᵀ||/||R||  = {sym_R:.2e}  (target < 5e-4)  {'PASS' if ok else 'FAIL'}")
    passed &= ok

    ok = sym_X < 5e-3
    print(f"[CHECK] ||X-Xᵀ||/||X||  = {sym_X:.2e}  (target < 5e-3)  {'PASS' if ok else 'FAIL'}")
    passed &= ok

    tol_psd = -1e-8 * max(abs(max_e), 1.0)
    ok = min_e >= tol_psd
    print(f"[CHECK] min_eig(Re Z)    = {min_e:.4e}  (target >= 0)     {'PASS' if ok else 'FAIL'}")
    passed &= ok

    return passed, d_new, d_leg


# ---------------------------------------------------------------------------
# Far-pair accuracy check at FREQ_ACC (quasi-static, both rules accurate)
# ---------------------------------------------------------------------------

def _run_far_pair_accuracy(name, raw_pts, raw_tri):
    print(f"\n{'='*60}")
    print(f"  Mesh: {name}  (far-pair accuracy @ {FREQ_ACC/1e6:.0f} MHz)")
    print(f"{'='*60}")

    points_obj, tri_obj, edges_obj, *_ = _build_rwg_objects(raw_pts, raw_tri)
    E = edges_obj.total_number_of_edges

    Z_new = _build_z_new(tri_obj, points_obj, edges_obj, FREQ_ACC)
    Z_ref = build_z_reference(tri_obj, points_obj, edges_obj, FREQ_ACC,
                              far_pairs_only=True)

    far_ok    = True
    far_found = False
    for m in range(E):
        for n in range(E):
            if classify_edge_pair(m, n, tri_obj) == 0:
                z_n = Z_new[m, n]
                z_r = Z_ref[m, n]
                if np.isnan(z_r):
                    continue
                rel = abs(z_n - z_r) / (abs(z_r) + 1e-300)
                ok_pair = rel < 0.01   # 1 % at quasi-static
                far_found = True
                # Print only the upper-triangle entry (m<n) to avoid duplicates
                if m <= n:
                    far_ok &= ok_pair
                    print(f"  FAR ({m},{n}):  new={z_n:.6f}  ref={z_r:.6f}  "
                          f"rel={rel:.2e}  {'PASS' if ok_pair else 'FAIL'}")

    if not far_found:
        print("  (no far pairs — skip)")
        return True

    print(f"[CHECK] far-pair accuracy < 1 %          {'PASS' if far_ok else 'FAIL'}")
    return far_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Phase 6 Validation — new impedance_matrix_z")

    all_passed = True

    # ---- 4-triangle mesh ----
    raw_pts4, raw_tri4 = build_four_triangle_mesh()

    p4, *_ = _run_symmetry_psd_check("4-tri", raw_pts4, raw_tri4)
    all_passed &= p4
    all_passed &= _run_far_pair_accuracy("4-tri", raw_pts4, raw_tri4)

    # ---- 6-triangle mesh (has genuine far pairs) ----
    raw_pts6, raw_tri6 = build_six_triangle_mesh()

    p6, d_new_6, d_leg_6 = _run_symmetry_psd_check("6-tri", raw_pts6, raw_tri6)
    all_passed &= p6
    all_passed &= _run_far_pair_accuracy("6-tri", raw_pts6, raw_tri6)

    # ---- summary ----
    print(f"\n{'='*60}")
    print("  Before / after comparison (6-tri mesh @ 300 MHz):")
    print(f"    sym_R:     {d_leg_6['sym_error_R']:.2e}  →  {d_new_6['sym_error_R']:.2e}")
    print(f"    sym_X:     {d_leg_6['sym_error_X']:.2e}  →  {d_new_6['sym_error_X']:.2e}")
    print(f"    min_eig_R: {d_leg_6['min_eig_R']:.4e}  →  {d_new_6['min_eig_R']:.4e}")
    print(f"    cond_R:    {d_leg_6['cond_R']:.2e}  →  {d_new_6['cond_R']:.2e}")
    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
