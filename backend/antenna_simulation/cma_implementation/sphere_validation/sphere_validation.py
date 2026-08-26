"""
External validation of the CMA solver against the analytically known
characteristic modes of a PEC spherical shell.

Why this exists
---------------
Everything measured on this solver so far has been INTERNAL consistency:
Z = Z^T, R positive semi-definite, quadrature converging towards its own refined
self. None of it can detect an error that is the same everywhere -- a wrong
global factor, a flipped sign convention, a missing eta -- because such an error
leaves every internal invariant intact. Only a comparison to an independently
known answer can, and the sphere is the one geometry that supplies one.

Reference
---------
M. Capek, V. Losenicky, L. Jelinek, M. Gustafsson, "Validating the
Characteristic Modes Solvers", IEEE Trans. Antennas Propag., vol. 65, no. 8,
pp. 4134-4145, Aug. 2017.  (preprint: arXiv:1702.07037)

The characteristic numbers of a spherical shell of radius a, eq. (11) and (12):

    lambda_p^TE = - y_p(ka) / j_p(ka)                                     (11)

    lambda_p^TM = - [ (p+1) y_p(ka) - ka y_{p+1}(ka) ]
                  / [ (p+1) j_p(ka) - ka j_{p+1}(ka) ]                    (12)

with j_p, y_p the spherical Bessel functions of the first and second kind of
order p, and the degeneracy of the p-th order, eq. (21):

    N(p) = 2p + 1                                                         (21)

The paper's matrix eigenproblem, eq. (16), is  X I_n = lambda_n R I_n -- the same
convention as backend/cma/eigen_solver.py -- so the eigenvalues are directly
comparable with no sign flip or rescaling.

What "degeneracy" predicts here
-------------------------------
A spherical shell carries TWO families, TM and TE, and each has a multiplet of
2p+1 modes for every order p >= 1. Ordered by |lambda|, the expected sequence of
multiplet SIZES is therefore 3, 3, 5, 5, 7, 7, ... -- one triplet of TM and one
of TE before the first quintuplet -- not 3, 5, 7. Each individual multiplet still
has odd size 2p+1, which is the invariant being tested. TM modes come out
negative (capacitive) and TE positive (inductive) at these electrical sizes.

Run:
    python sphere_validation.py
    python sphere_validation.py --mesh sphere_shell_icosphere.msh   # the control
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.special import spherical_jn, spherical_yn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         *[os.pardir] * 4))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.cma.cma_parameters import CMAParameters                     # noqa: E402
from backend.cma.cma_solver import CharacteristicModeSolver              # noqa: E402
from backend.cma.eigen_solver import GeneralizedEigenSolver              # noqa: E402
from backend.rwg.rwg3 import DataManager_rwg3                            # noqa: E402
from backend.utils.impmet import impedance_matrix_z_sweep_gpu            # noqa: E402
from backend.utils.file_path import setup_save_file_paths                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# The solver's own constants, so that ka is exact by construction rather than
# exact to the four digits of a quoted speed of light.
EPSILON = 8.854e-12
MU = 1.257e-6
C_LIGHT = 1.0 / np.sqrt(EPSILON * MU)

RADIUS = 0.1
KA_VALUES = (0.5, 1.0, 1.5)
N_MODES = 30            # 3+3+5+5+7+7 = 30, i.e. the first six analytic multiplets
# Rank truncation is not a neutral setting here and the report must state it.
# Measured on the geodesic mesh at ka = 1.0 (E = 750), the TM1 error falls
# 6.18% -> 4.69% -> 2.46% -> 2.33% as rank_tol goes 1e-4 -> 1e-8 -> 1e-12 ->
# 1e-14, i.e. the truncation itself biases the eigenvalue: projecting onto R's
# dominant subspace discards part of the true modal current. 1e-12 is where that
# drift has essentially stopped while still sitting above R's own noise floor
# (its most negative eigenvalue is -1.1e-14 of its largest), so it is used as the
# working value and swept explicitly by rank_tol_sweep().
RANK_TOL = 1e-12
MULTIPLET_GROUP_TOL = 0.10   # relative gap that starts a new multiplet
MULTIPLET_CLEAN_TOL = 0.01   # a multiplet is "clean" below 1% internal spread


# ---------------------------------------------------------------------------
# Analytic reference -- Capek et al. (2017), eqs (11), (12), (21)
# ---------------------------------------------------------------------------

def analytic_lambda_te(p, ka):
    """Eq. (11): lambda_p^TE = - y_p(ka) / j_p(ka)."""
    return -spherical_yn(p, ka) / spherical_jn(p, ka)


def analytic_lambda_tm(p, ka):
    """Eq. (12), the TM characteristic number."""
    num = (p + 1) * spherical_yn(p, ka) - ka * spherical_yn(p + 1, ka)
    den = (p + 1) * spherical_jn(p, ka) - ka * spherical_jn(p + 1, ka)
    return -num / den


def analytic_spectrum(ka, p_max=6):
    """
    Every analytic multiplet up to order p_max, ordered by |lambda| ascending --
    the same ordering CMAParameters(sort_by='magnitude') gives the solver, so
    the k-th entry here is what the k-th computed multiplet must reproduce.

    Returns a list of dicts with family, p, lambda and degeneracy 2p+1.
    """
    modes = []
    for p in range(1, p_max + 1):
        modes.append({'family': 'TM', 'p': p, 'lambda': float(analytic_lambda_tm(p, ka)),
                      'degeneracy': 2 * p + 1})
        modes.append({'family': 'TE', 'p': p, 'lambda': float(analytic_lambda_te(p, ka)),
                      'degeneracy': 2 * p + 1})
    modes.sort(key=lambda m: abs(m['lambda']))
    for m in modes:
        m['label'] = f"{m['family']}{m['p']}"
    return modes


def frequency_for_ka(ka, radius=RADIUS):
    """f such that k*radius == ka exactly, with k = 2*pi*f/c and the solver's c."""
    return ka * C_LIGHT / (2.0 * np.pi * radius)


# ---------------------------------------------------------------------------
# Running the solver
# ---------------------------------------------------------------------------

def run_cma(msh_name, ka, n_modes=N_MODES, rank_tol=RANK_TOL, use_gpu=None):
    """
    Solve the CMA at the frequency corresponding to `ka` and return the
    eigenvalues plus the numbers needed to interpret them.

    Uses solver='rank_truncated': the 'regularized' solver populates R's null
    space with eigenvalues set by its own epsilon rather than by the sphere, and
    those fictitious modes would interleave with the real multiplets and make
    any degeneracy count meaningless.
    """
    stem = os.path.splitext(msh_name)[0]
    path = setup_save_file_paths(stem, gmsh_folder_path=os.path.join(HERE, ''))
    params = CMAParameters(n_modes=n_modes, solver='rank_truncated', rank_tol=rank_tol,
                           sort_by='magnitude')

    frequency = frequency_for_ka(ka)
    solver = CharacteristicModeSolver(path, params, frequency=frequency,
                                      use_gpu=use_gpu, write_report=False)
    modal_data = solver.solve()

    # Rank of R under exactly the rule the truncated solver applies, recomputed
    # here so the report can state how many directions actually survived rather
    # than inferring it from how many modes came back.
    _f, _o, _m, _e, _c, _eta, Z = DataManager_rwg3.load_data(path.mat_impedance)
    R = (np.real(Z) + np.real(Z).T) * 0.5
    mu_R = np.linalg.eigvalsh(R)
    rank = int(np.count_nonzero(mu_R > rank_tol * mu_R[-1]))

    return {
        'ka': ka,
        'frequency': frequency,
        'eigenvalues': np.asarray(modal_data.eigenvalues, dtype=float),
        'n_edges': int(R.shape[0]),
        'rank_tol': rank_tol,
        'rank_R': rank,
        'R_eigenvalue_max': float(mu_R[-1]),
        'R_eigenvalue_min': float(mu_R[0]),
    }


# ---------------------------------------------------------------------------
# Multiplet detection
# ---------------------------------------------------------------------------

def detect_multiplets(eigenvalues, group_tol=MULTIPLET_GROUP_TOL):
    """
    Split the |lambda|-ordered spectrum into multiplets.

    A new multiplet starts wherever the relative step between consecutive
    eigenvalues exceeds `group_tol`. That threshold does not need to be delicate:
    on a sphere the within-multiplet spread is sub-percent while the step BETWEEN
    multiplets is tens of percent at worst (the closest pair anywhere in this
    study is TM1 -> TE1 at ka = 1.5, a 69% step), so any cut between the two
    scales gives the same grouping. The number reported per group is the measured
    spread, not the threshold, so the grouping cannot manufacture a clean result.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    if lam.size == 0:
        return []

    groups, current = [], [0]
    for i in range(1, lam.size):
        step = abs(lam[i] - lam[i - 1]) / max(abs(lam[i]), abs(lam[i - 1]))
        if step > group_tol:
            groups.append(current)
            current = []
        current.append(i)
    groups.append(current)

    out = []
    for g in groups:
        vals = lam[g]
        centre = float(np.mean(vals))
        spread = float(vals.max() - vals.min()) / abs(centre) if centre != 0 else np.inf
        out.append({
            'indices': g,
            'values': vals,
            'size': len(g),
            'mean': centre,
            'spread': spread,
            'clean': spread < MULTIPLET_CLEAN_TOL,
        })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_stage1_table(result, multiplets):
    lam = result['eigenvalues']
    print(f"\nka = {result['ka']}  (f = {result['frequency'] / 1e6:.4f} MHz)   "
          f"E = {result['n_edges']} RWG edges, rank_tol = {result['rank_tol']:.0e}, "
          f"rank(R) = {result['rank_R']}, modes returned = {lam.size}")
    print(f"{'idx':>4} | {'lambda':>15} | {'ecart rel. au precedent':>23} | {'multiplet':>20}")
    print("-" * 74)
    member_of = {}
    for gi, g in enumerate(multiplets):
        for pos, idx in enumerate(g['indices']):
            member_of[idx] = (gi, pos, g)
    for i, value in enumerate(lam):
        if i == 0:
            step = "-"
        else:
            step = f"{abs(value - lam[i - 1]) / max(abs(value), abs(lam[i - 1])):.3e}"
        gi, pos, g = member_of[i]
        tag = f"#{gi + 1} size {g['size']}" if pos == 0 else ""
        print(f"{i:>4} | {value:>15.7e} | {step:>23} | {tag:>20}")


def print_multiplet_summary(multiplets, analytic):
    print(f"\n{'multiplet':>10} | {'taille':>6} | {'lambda moyen':>15} | "
          f"{'dispersion interne':>18} | {'propre <1%':>10} | {'attendu':>16}")
    print("-" * 92)
    for gi, g in enumerate(multiplets):
        if gi < len(analytic):
            ref = analytic[gi]
            expect = f"{ref['label']} ({ref['degeneracy']})"
            size_ok = "" if g['size'] == ref['degeneracy'] else "  <-- SIZE MISMATCH"
        else:
            expect, size_ok = "-", ""
        print(f"{'#' + str(gi + 1):>10} | {g['size']:>6} | {g['mean']:>15.7e} | "
              f"{g['spread']:>18.3e} | {'OUI' if g['clean'] else 'NON':>10} | "
              f"{expect:>16}{size_ok}")


def match_multiplets_to_analytic(multiplets, analytic):
    """
    Pair each computed multiplet with an analytic mode by SIGN and SIZE, not by
    position in the |lambda| ordering.

    Pairing by position is what a first draft does, and it breaks as soon as the
    errors are large: at ka = 0.5 the computed TM2 and TE2 are far enough off that
    they swap places in the |lambda| ordering, and a positional match then
    compares a positive multiplet against a negative analytic value and reports a
    194% error for what is really a 5.7% one. Sign (TM negative, TE positive at
    these electrical sizes) and degeneracy 2p+1 are discrete labels that a few
    percent of error cannot move, so they identify the mode; the magnitude is then
    left free to be measured rather than assumed.
    """
    remaining = list(analytic)
    pairs = []
    for g in multiplets:
        candidates = [m for m in remaining
                      if np.sign(m['lambda']) == np.sign(g['mean'])
                      and m['degeneracy'] == g['size']]
        if not candidates:
            pairs.append((g, None))
            continue
        best = min(candidates,
                   key=lambda m: abs(np.log(abs(g['mean']) / abs(m['lambda']))))
        remaining.remove(best)
        pairs.append((g, best))
    return pairs


def print_stage2_table(multiplets, analytic):
    print("")
    print(f"{'mode':>6} | {'taille':>6} | {'lambda calcule':>16} | "
          f"{'lambda analytique':>18} | {'erreur relative':>16} | {'<1%':>5}")
    print("-" * 90)
    rows = []
    for g, ref in match_multiplets_to_analytic(multiplets, analytic):
        if ref is None:
            print(f"{'?':>6} | {g['size']:>6} | {g['mean']:>16.7e} | "
                  f"{'pas d appariement':>18} | {'-':>16} | {'-':>5}")
            continue
        err = abs(g['mean'] - ref['lambda']) / abs(ref['lambda'])
        rows.append({'label': ref['label'], 'computed': g['mean'],
                     'analytic': ref['lambda'], 'rel_error': err,
                     'size': g['size'], 'degeneracy': ref['degeneracy'],
                     'spread': g['spread']})
        print(f"{ref['label']:>6} | {g['size']:>6} | {g['mean']:>16.7e} | "
              f"{ref['lambda']:>18.7e} | {err:>16.3e} | "
              f"{'OK' if err < 0.01 else 'NON':>5}")
    return rows


def rank_tol_sweep(msh_name, ka, rank_tols=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14),
                   n_modes=16, use_gpu=None):
    """
    How much of the Stage 2 error the rank truncation is itself responsible for.

    This belongs in the module rather than in a one-off script because the answer
    is not small -- it is the largest single identified contributor -- and a
    report quoting one rank_tol without this sweep would be presenting an
    artefact of that choice as a property of the solver.
    """
    print("")
    print(f"--- balayage rank_tol, ka = {ka}, maillage = {msh_name} ---")
    print(f"{'rank_tol':>10} | {'rank(R)':>8} | {'modes':>6} | "
          f"{'multiplet #1':>16} {'err':>9} | {'multiplet #2':>16} {'err':>9}")
    print("-" * 84)
    analytic = analytic_spectrum(ka)
    for rt in rank_tols:
        result = run_cma(msh_name, ka, n_modes=n_modes, rank_tol=rt, use_gpu=use_gpu)
        pairs = match_multiplets_to_analytic(
            detect_multiplets(result['eigenvalues']), analytic)
        cells = ""
        for g, ref in pairs[:2]:
            if ref is None:
                cells += f" | {g['mean']:>16.5f} {'?':>9}"
            else:
                cells += (f" | {g['mean']:>16.5f} "
                          f"{abs(g['mean'] - ref['lambda']) / abs(ref['lambda']):>8.2%}")
        print(f"{rt:>10.0e} | {result['rank_R']:>8} | "
              f"{result['eigenvalues'].size:>6}{cells}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(msh_name="sphere_shell.msh", ka_values=KA_VALUES, rank_tol=RANK_TOL,
        n_modes=N_MODES, use_gpu=None):
    meta_path = os.path.join(HERE, os.path.splitext(msh_name)[0] + ".json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.isfile(meta_path) else {}
    mesh_stats = meta.get('mesh', {})

    print("=" * 92)
    print(f"CMA external validation -- PEC spherical shell, a = {RADIUS} m")
    print(f"mesh   : {msh_name}")
    if meta:
        print(f"sha256 : {meta.get('sha256')}")
        print(f"         {mesh_stats.get('n_points')} points, "
              f"{mesh_stats.get('n_triangles')} triangles, "
              f"{mesh_stats.get('n_rwg_edges')} RWG edges")
        print(f"         edge length std/mean = "
              f"{mesh_stats.get('edge_length_std_over_mean', float('nan')):.3%}, "
              f"max/min = {mesh_stats.get('edge_length_max_over_min', float('nan')):.3f}, "
              f"triangle quality min = "
              f"{mesh_stats.get('triangle_quality_min', float('nan')):.3f}")
    print("=" * 92)

    all_results = []
    for ka in ka_values:
        result = run_cma(msh_name, ka, n_modes=n_modes, rank_tol=rank_tol, use_gpu=use_gpu)
        multiplets = detect_multiplets(result['eigenvalues'])
        analytic = analytic_spectrum(ka)

        print_stage1_table(result, multiplets)
        print_multiplet_summary(multiplets, analytic)
        rows = print_stage2_table(multiplets, analytic)

        all_results.append({'result': result, 'multiplets': multiplets,
                            'analytic': analytic, 'comparison': rows})
    return all_results


# ---------------------------------------------------------------------------
# H1 (rank truncation) vs H2 (EFIE low-frequency breakdown)
# ---------------------------------------------------------------------------
#
# Both hypotheses explain the Stage 2 error, and they are told apart by what
# happens when the truncation is REMOVED rather than merely loosened:
#
#   H1  the error is an artefact of projecting onto range(R). Keeping every
#       direction should collapse it, and whatever survives should behave like
#       ordinary discretisation error -- roughly flat in ka.
#   H2  the error is in the operator: as ka falls, the EFIE's loop currents make
#       R = Re(Z) the difference of nearly equal large numbers, so lambda = X/R
#       for those modes is a ratio of two catastrophically cancelled quantities.
#       Removing the truncation then changes little and the error stays strongly
#       ka-dependent, worsening towards small ka.
#
# The experiment needs a mode identifier that works on the UNTRUNCATED spectrum,
# where R's null space contributes hundreds of spurious eigenvalues. Matching a
# cluster to whichever analytic value it happens to sit closest to would be
# circular. The filter used instead is physical and independent of the answer:
# scipy's eigh(X, R + eps*I) returns vectors normalised so that
# J^T (R + eps I) J = 1, hence
#
#       J^T R J = 1 - eps * ||J||^2
#
# A genuinely radiating mode has J^T R J close to 1 and a small norm; a
# null-space vector has J^T R J ~ 0 and ||J||^2 ~ 1/eps. Thresholding on that
# radiated fraction separates the two without ever looking at lambda.


def solve_untruncated(R, X, eps_factor=1e-13, radiating_fraction=0.5):
    """
    Solve X J = lambda (R + eps I) J on the FULL space, keeping only the
    eigenvectors that actually radiate.

    `eps_factor` is relative to lambda_max(R) and only has to make R + eps I
    positive definite for the Cholesky step -- R's most negative eigenvalue sits
    around -1e-14 of its largest, so 1e-13 clears it with a margin while staying
    far below every physically radiating direction.

    Returns (lambda, radiated_fraction, eps, n_kept), sorted by |lambda|.
    """
    Rs = (R + R.T) * 0.5
    Xs = (X + X.T) * 0.5
    N = Rs.shape[0]
    mu_max = float(np.linalg.eigvalsh(Rs)[-1])
    eps = eps_factor * mu_max

    lam, J = eigh(Xs, Rs + eps * np.eye(N))
    radiated = 1.0 - eps * np.einsum('ij,ij->j', J, J)   # = J^T R J, per mode
    keep = radiated > radiating_fraction
    lam, radiated = lam[keep], radiated[keep]
    order = np.argsort(np.abs(lam))
    return lam[order], radiated[order], eps, int(keep.sum())


#: Identification threshold, deliberately looser than MULTIPLET_CLEAN_TOL.
#:
#: The two thresholds answer different questions and conflating them hides
#: results. MULTIPLET_CLEAN_TOL (1%) is the REPORTING criterion from the
#: protocol -- is this multiplet clean enough to count as degenerate. This one
#: only has to be tight enough that no accidental grouping of unrelated
#: eigenvalues can pass, and at 5% it is still an order of magnitude below the
#: gaps BETWEEN multiplets (69% at the closest point anywhere in this study).
#: Using 1% for identification silently dropped the TM1 triplet of the
#: untruncated solve at ka = 0.5 and 1.5, whose spreads are 1.8e-2 and 1.5e-2 --
#: present, correctly sized, correctly valued, and reported as "absent".
MULTIPLET_IDENTIFY_TOL = 5e-2


def identify_by_degeneracy(lam, analytic, spread_tol=MULTIPLET_IDENTIFY_TOL):
    """
    Pick the physical multiplets out of a spectrum and label them.

    Clusters are formed on the |lambda|-sorted spectrum, then only those whose
    size matches an analytic degeneracy AND whose internal spread is below
    `spread_tol` survive. Degeneracy is a symmetry invariant that a few percent
    of error cannot fake, so this identifies a mode without assuming anything
    about its magnitude. The measured spread is returned alongside so the caller
    can still apply the 1% reporting criterion separately.
    """
    multiplets = [g for g in detect_multiplets(lam) if g['spread'] < spread_tol]
    found = {}
    for g, ref in match_multiplets_to_analytic(multiplets, analytic):
        if ref is not None and ref['label'] not in found:
            found[ref['label']] = {
                'computed': g['mean'], 'analytic': ref['lambda'], 'size': g['size'],
                'spread': g['spread'],
                'rel_error': abs(g['mean'] - ref['lambda']) / abs(ref['lambda']),
            }
    return found


def analyse_r_spectrum(R, ka, thresholds=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)):
    """
    R's eigenvalue spectrum and the rank it implies at each tolerance.

    R is the radiation operator, so its numerical rank is not a mesh property but
    a PHYSICAL one: only the spherical harmonics that radiate appreciably at this
    electrical size contribute, and their number grows with ka. That is what the
    rank counts here have to be read against.
    """
    Rs = (R + R.T) * 0.5
    mu = np.linalg.eigvalsh(Rs)[::-1]        # descending
    mu_max = mu[0]
    ranks = {t: int(np.count_nonzero(mu > t * mu_max)) for t in thresholds}
    return {
        'ka': ka, 'mu': mu, 'mu_max': float(mu_max), 'mu_min': float(mu[-1]),
        'mu_min_over_max': float(mu[-1] / mu_max), 'ranks': ranks,
    }


def expected_radiating_count(ka, p_max=8, lambda_cut=1e4):
    """
    How many spherical modes radiate enough to be visible in R at this ka.

    A mode with |lambda| >> 1 stores far more energy than it radiates (its modal
    significance is 1/|1 + j lambda|), so it sits correspondingly lower in R's
    spectrum. Counting the TE and TM orders whose analytic |lambda| stays under
    `lambda_cut`, weighted by their 2p+1 degeneracies, gives the number of
    directions R can be expected to support before dropping into noise.
    """
    total = 0
    for p in range(1, p_max + 1):
        for lam in (analytic_lambda_tm(p, ka), analytic_lambda_te(p, ka)):
            if abs(lam) < lambda_cut:
                total += 2 * p + 1
    return total


def h1_vs_h2(msh_name='sphere_shell_icosphere.msh', ka_values=KA_VALUES,
             rank_tols=(1e-4, 1e-8, 1e-12, 1e-14), labels=('TM1', 'TE1', 'TM2', 'TE2'),
             n_modes=40, use_gpu=None):
    """The discriminating experiment: one mesh, three ka, truncation on and off."""
    from backend.cma.cma_mesh import prepare_cma_mesh

    stem = os.path.splitext(msh_name)[0]
    path = setup_save_file_paths(stem, gmsh_folder_path=os.path.join(HERE, ''))
    points, triangles, edges, _rho = prepare_cma_mesh(path, verbose=False)
    E = edges.total_number_of_edges

    print("=" * 104)
    print("H1 (troncature de rang) vs H2 (degradation basse frequence de l'EFIE)")
    print(f"maillage {msh_name}, E = {E} aretes RWG")
    print("=" * 104)

    out = {}
    for ka in ka_values:
        frequency = frequency_for_ka(ka)
        Z = impedance_matrix_z_sweep_gpu(edges, triangles, points, [frequency],
                                         MU, EPSILON, use_gpu=use_gpu)[0]
        R, X = np.real(Z), np.imag(Z)
        analytic = analytic_spectrum(ka)
        spec = analyse_r_spectrum(R, ka)

        print("")
        print(f"--- ka = {ka}  (f = {frequency / 1e6:.4f} MHz) ---")
        print(f"  spectre de R : mu_max = {spec['mu_max']:.4e}, "
              f"mu_min/mu_max = {spec['mu_min_over_max']:+.2e}")
        print("  rang(R) par seuil : " + "  ".join(
            f"{t:.0e}->{r}" for t, r in spec['ranks'].items()))
        print(f"  modes spheriques attendus a ce ka (|lambda| < 1e4) : "
              f"{expected_radiating_count(ka)}")

        rows = {}
        header = (f"{'methode':>20} | {'directions':>10} | "
                  + " | ".join(f"{lb:>9} {'err':>8}" for lb in labels))
        print("")
        print(header)
        print("-" * len(header))

        mu_all = np.linalg.eigvalsh((R + R.T) * 0.5)
        for rt in rank_tols:
            params = CMAParameters(n_modes=n_modes, solver='rank_truncated',
                                   rank_tol=rt, sort_by='magnitude')
            lam, _J, _o = GeneralizedEigenSolver.solve(R, X, params)
            found = identify_by_degeneracy(lam, analytic)
            rank = int(np.count_nonzero(mu_all > rt * spec['mu_max']))
            rows[f'trunc_{rt:.0e}'] = found
            cells = "".join(
                f" | {found[lb]['computed']:>9.4f} {found[lb]['rel_error']:>7.2%}"
                if lb in found else f" | {'absent':>9} {'-':>8}" for lb in labels)
            print(f"{'tronque ' + format(rt, '.0e'):>20} | {rank:>10}{cells}")

        lam_full, _radiated, eps, n_keep = solve_untruncated(R, X)
        found = identify_by_degeneracy(lam_full, analytic)
        rows['untruncated'] = found
        cells = "".join(
            f" | {found[lb]['computed']:>9.4f} {found[lb]['rel_error']:>7.2%}"
            if lb in found else f" | {'absent':>9} {'-':>8}" for lb in labels)
        print(f"{'NON tronque':>20} | {n_keep:>10}{cells}")
        print(f"  (eps = {eps:.3e} = 1e-13 * mu_max ; {n_keep} vecteurs sur {E} "
              f"rayonnent plus de la moitie de leur norme)")

        out[ka] = {'spectrum': spec, 'rows': rows, 'analytic': analytic}
    return out


def main():
    ap = argparse.ArgumentParser(description="CMA validation on the PEC spherical shell")
    ap.add_argument('--mesh', default='sphere_shell.msh')
    ap.add_argument('--ka', type=float, nargs='*', default=list(KA_VALUES))
    ap.add_argument('--rank-tol', type=float, default=RANK_TOL)
    ap.add_argument('--n-modes', type=int, default=N_MODES)
    ap.add_argument('--cpu', action='store_true', help="force the CPU assembly")
    ap.add_argument('--h1-vs-h2', action='store_true',
                    help="run the truncated/untruncated discriminating experiment")
    args = ap.parse_args()
    if args.h1_vs_h2:
        h1_vs_h2(args.mesh, tuple(args.ka),
                 use_gpu=False if args.cpu else None)
        return
    run(args.mesh, tuple(args.ka), args.rank_tol, args.n_modes,
        use_gpu=False if args.cpu else None)


if __name__ == '__main__':
    main()
