"""
Cross-frequency tracking of characteristic modes.

At every frequency the eigensolver returns an unordered set of modes, and
`GeneralizedEigenSolver._select_and_sign` then labels them by sorting on
`|lambda|`. That label is a property of the sweep point, not of the mode: two
eigenvalue curves that cross exchange their `|lambda|` ranks, so "mode 2"
before the crossing and "mode 2" after it are different physical currents.
Every downstream quantity inherits the problem -- a `lambda_n(f)` curve
develops a kink it does not physically have, a modal-significance peak gets
split between two labels, and the modal current plotted at two frequencies
shows two different patterns under one name.

Tracking replaces the per-frequency label by an identity that follows the
current distribution. Two modes at adjacent frequencies are the same mode when
their currents are the same current, which is measured here by the
`R`-weighted correlation

    C_mn = |J_m^H R J_n| / sqrt( (J_m^H R J_m) (J_n^H R J_n) )

with `R = Re(Z)` the radiation operator. The weighting is not cosmetic. The
plain Euclidean overlap `|J_m^H J_n|` treats all `N` degrees of freedom alike,
including the many that radiate nothing at all -- on a real mesh most of them
(measured on the CubeSat mesh at 440 MHz: numerical rank of `R` is 93 out of
760, see `mode_diagnostics`). Two modes can therefore agree closely in the
directions that radiate and still correlate poorly in the Euclidean metric,
because their non-radiating content differs. `R` is exactly the metric in which
the eigenvectors are orthonormal (`scipy.linalg.eigh(X, R)` returns
`J^T R J = I`), so `C_mn` is the cosine between modes in the inner product the
problem is actually posed in, and `C_mm = 1` holds by construction.

Given `C`, assigning modes one-to-one is a linear assignment problem, and
taking each mode's best match greedily is not the same thing: greedy matching
can send two tracks to the same mode and is order-dependent, whereas maximising
the total correlation is what "these are the same K modes, re-labelled" means.
`scipy.optimize.linear_sum_assignment` (Jonker-Volgenant) solves it exactly.

What this module does NOT claim to fix
--------------------------------------
Correlation-based tracking has a known failure mode, and it is reported rather
than papered over. Where two eigenvalues are quasi-degenerate
(`lambda_m ~ lambda_n`), the two-dimensional eigenspace is very nearly
degenerate and any rotation inside it is an equally valid pair of eigenvectors:
the eigenvectors the solver returns there are set by roundoff, `C` approaches
the same value for both candidate assignments, and no correlation metric can
decide between them. `TrackingEvent`s of kind `degeneracy` and
`avoided_crossing` mark those bands, and the margin between the best and the
runner-up assignment is stored per step, so the confidence is visible in the
data instead of being implied by a smooth-looking curve.

Resolving those bands needs information correlation does not carry -- the local
behaviour of `lambda(f)` and of the operator itself, as in the TECA-style
eigenvalue correction of Chen, Pan and Su (IEEE TAP, 2021).
`ModeTrackingCorrection` is the extension point for that: a correction sees the
whole `TrackingStep` (both operators, both spectra, the correlation matrix and
the assignment) and may return a different assignment and/or corrected
eigenvalues. `TECACorrection` is registered and deliberately unimplemented; see
its docstring for the contract.
"""

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from solver.cma.characteristic_angle import characteristic_angle

# Event kinds, so callers can filter without matching free text.
EVENT_SWAP = 'swap'
EVENT_DEGENERACY = 'degeneracy'
EVENT_AVOIDED_CROSSING = 'avoided_crossing'
EVENT_LOW_CORRELATION = 'low_correlation'


@dataclass
class TrackingParameters:
    """Configuration for cross-frequency mode tracking."""

    # 'R'        weight the correlation by Re(Z) -- the metric the eigenvectors
    #            are orthonormal in, and the one that ignores the non-radiating
    #            directions (see the module docstring).
    # 'identity' plain Euclidean overlap |J_m^H J_n|. Available for comparison
    #            and for callers that have eigenvectors but no Z; not the
    #            recommended default.
    metric: str = 'R'                 # 'R' | 'identity'

    # Which R to weight with, for a pair of adjacent frequencies. R(f) varies
    # smoothly over a sweep step, so the three choices agree closely; 'mean' is
    # the symmetric one, so that correlating i -> i+1 and i+1 -> i give exactly
    # the same numbers.
    weight_at: str = 'mean'           # 'mean' | 'previous' | 'next'

    # Below this accepted correlation, the assignment is reported as unreliable
    # (kind='low_correlation'). 0.7 is not a physical constant: it is roughly
    # where two modes stop looking like the same current, and in practice a
    # sweep with enough points has accepted correlations above 0.99.
    correlation_threshold: float = 0.7

    # Gap between the accepted correlation and the best rejected alternative for
    # the same track. A small margin means the assignment was nearly a coin
    # flip, which is what actually happens inside a degenerate band.
    margin_threshold: float = 0.1

    # Quasi-degeneracy tolerance, as the gap between two characteristic values
    # relative to their own size:
    #
    #     |lambda_m - lambda_n| / max(|lambda_m|, |lambda_n|, 1) < tol
    #
    # The quantity that makes two modes hard to tell apart is how strongly a
    # perturbation mixes them, and first-order perturbation theory puts that
    # mixing at (J_m^T dX J_n) / (lambda_m - lambda_n): it is the gap between
    # the eigenvalues, measured against their own scale, that decides. Hence
    # relative above |lambda| = 1 and absolute below it -- near resonance
    # lambda passes through zero, where a purely relative test would call every
    # pair degenerate.
    #
    # A tempting alternative is to measure the gap in characteristic angle,
    # since delta = 180 - atan(lambda) is bounded and easy to threshold. It is
    # wrong here: atan compresses everything above |lambda| ~ 10 into a few
    # degrees, so lambda = 9 and lambda = 30 -- a factor of three apart in
    # stored energy, and in no way a degenerate pair -- sit 4.4 deg apart and
    # get flagged, while the arctan does nothing useful near resonance either.
    degeneracy_rel_tol: float = 0.05

    # Carry a consistent sign along each track. The per-frequency sign
    # convention in GeneralizedEigenSolver._select_and_sign ("largest component
    # positive") flips whenever the largest component changes edge, which is a
    # relabelling of the same mode but makes a plotted current vector reverse.
    fix_sign: bool = True

    # Optional post-assignment correction stage, see ModeTrackingCorrection.
    correction: str = 'none'          # 'none' | 'teca'


@dataclass
class TrackingEvent:
    """One thing worth knowing about a single sweep step."""

    kind: str                # EVENT_* above
    freq_index: int          # index of the LATER frequency of the step
    frequency: float         # Hz, that later frequency
    modes: tuple             # track indices (0-based) involved
    detail: str              # human-readable measurement behind the flag

    def describe(self):
        modes = ", ".join(f"{m + 1}" for m in self.modes)
        return f"{self.frequency / 1e6:.3f} MHz | {self.kind} | mode(s) {modes} | {self.detail}"

    def as_dict(self):
        return {'kind': self.kind, 'freq_index': int(self.freq_index),
                'frequency': float(self.frequency),
                'modes': [int(m) for m in self.modes], 'detail': self.detail}

    @staticmethod
    def from_dict(d):
        return TrackingEvent(kind=str(d['kind']), freq_index=int(d['freq_index']),
                             frequency=float(d['frequency']), modes=tuple(d['modes']),
                             detail=str(d['detail']))


@dataclass
class ModeTrackingResult:
    """
    The permutation that turns per-frequency labels into mode identities.

    `order[i, k]` is the raw index, in the solver's own `|lambda|`-sorted
    ordering at frequency `i`, of the mode belonging to track `k`. Applying it
    is what `reorder` / `reorder_eigenvectors` do; nothing here modifies the raw
    sweep arrays, so the untracked view remains available and reproducible.
    """

    frequencies: np.ndarray     # (n_freq,)      Hz
    order: np.ndarray           # (n_freq, K)    int, raw index per track
    sign: np.ndarray            # (n_freq, K)    +/-1, continuity of J along a track
    correlation: np.ndarray     # (n_freq-1, K)  accepted C per track, per step
    margin: np.ndarray          # (n_freq-1, K)  accepted C minus best rejected C
    events: list = field(default_factory=list)
    metric: str = 'R'
    correction: str = 'none'
    # Filled only by a correction stage that revises lambda (TECA-style);
    # None means "the raw eigenvalues, reordered, are the answer".
    corrected_eigenvalues: np.ndarray = None

    @property
    def n_modes(self):
        return self.order.shape[1]

    @property
    def n_freq(self):
        return self.order.shape[0]

    # -- applying the permutation ------------------------------------

    def reorder(self, values):
        """
        Re-index a per-frequency, per-mode array `(n_freq, K)` into track order.

        Works for eigenvalues, and equally for anything derived from them
        pointwise (modal significance, characteristic angle), since those are
        computed mode by mode.
        """
        values = np.asarray(values)
        if values.shape[:2] != self.order.shape:
            raise ValueError(
                f"reorder: expected an array shaped {self.order.shape} "
                f"(n_freq, n_modes), got {values.shape}.")
        return np.take_along_axis(values, self.order, axis=1)

    def reorder_eigenvectors(self, eigenvectors):
        """Re-index `(n_freq, N, K)` modal currents, applying the sign continuity."""
        eigenvectors = np.asarray(eigenvectors)
        out = np.empty_like(eigenvectors)
        for i in range(self.n_freq):
            out[i] = eigenvectors[i][:, self.order[i]] * self.sign[i][None, :]
        return out

    def applied(self):
        """
        The same result with the permutation marked as already consumed.

        `order` becomes the identity and `sign` all ones, while the events and
        the per-step confidence are kept. This is what a container attaches to
        arrays it has ALREADY reordered, so that reordering a second time is a
        no-op instead of a second, wrong permutation.
        """
        return ModeTrackingResult(
            frequencies=self.frequencies,
            order=np.tile(np.arange(self.n_modes), (self.n_freq, 1)),
            sign=np.ones((self.n_freq, self.n_modes)),
            correlation=self.correlation,
            margin=self.margin,
            events=list(self.events),
            metric=self.metric,
            correction=self.correction,
            corrected_eigenvalues=(
                None if self.corrected_eigenvalues is None
                else self.reorder(self.corrected_eigenvalues)),
        )

    @property
    def is_applied(self):
        """True when `order` is the identity at every frequency (see `applied`)."""
        return bool(np.array_equal(
            self.order, np.tile(np.arange(self.n_modes), (self.n_freq, 1))))

    # -- reporting ----------------------------------------------------

    def events_of(self, kind):
        return [e for e in self.events if e.kind == kind]

    def summary(self):
        """Short, printable account of what tracking found."""
        lines = [
            f"Mode tracking: {self.n_modes} modes over {self.n_freq} frequencies "
            f"(metric='{self.metric}', correction='{self.correction}')",
        ]
        if self.correlation.size:
            lines.append(
                f"  accepted correlation: min {self.correlation.min():.4f}, "
                f"median {np.median(self.correlation):.4f}")
            lines.append(f"  assignment margin:    min {self.margin.min():.4f}")
        for kind in (EVENT_SWAP, EVENT_DEGENERACY, EVENT_AVOIDED_CROSSING,
                     EVENT_LOW_CORRELATION):
            n = len(self.events_of(kind))
            if n:
                lines.append(f"  {kind:<18} {n}")
        if not self.events:
            lines.append("  no swaps, degeneracies or low-confidence steps flagged")
        return "\n".join(lines)

    # -- (de)serialisation, for DataManager_cma_sweep -----------------

    def events_json(self):
        return json.dumps([e.as_dict() for e in self.events])

    @staticmethod
    def events_from_json(text):
        return [TrackingEvent.from_dict(d) for d in json.loads(text)]


# ----------------------------------------------------------------------
# The correlation metric
# ----------------------------------------------------------------------

def correlation_matrix(J_a, J_b, R=None, return_overlap=False):
    """
    R-weighted correlation between two sets of modal currents.

        C_mn = |J_m^H R J_n| / sqrt( (J_m^H R J_m) (J_n^H R J_n) )

    Parameters
    ----------
    J_a : (N, Ka) modal currents at the first frequency, one mode per column
    J_b : (N, Kb) modal currents at the second frequency
    R   : (N, N) symmetric positive-semidefinite weight, normally Re(Z).
          `None` gives the plain Euclidean overlap (`R = I`).
    return_overlap : also return the SIGNED overlap `M = J_a^H R J_b`, which the
          tracker needs to keep the sign of `J` continuous along a track.

    Returns
    -------
    C : (Ka, Kb) float in [0, 1], 1 for identical currents. Invariant to the
        scaling and the sign of every column.
    M : (Ka, Kb), only if `return_overlap` -- signed (complex if the inputs are).

    Notes
    -----
    A column whose weighted norm is zero radiates nothing, so it has no
    direction in this metric and cannot be correlated with anything; its row or
    column of `C` is set to 0 rather than to NaN, which lets the assignment
    still run and shows up afterwards as a low accepted correlation.
    """
    A = np.asarray(J_a)
    B = np.asarray(J_b)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"correlation_matrix: {A.shape[0]} rows vs {B.shape[0]}; "
                         "both sets must be expressed on the same RWG basis.")

    if R is None:
        RA, RB = A, B
    else:
        R = np.asarray(R)
        if R.shape != (A.shape[0], A.shape[0]):
            raise ValueError(f"correlation_matrix: R is {R.shape}, expected "
                             f"({A.shape[0]}, {A.shape[0]}).")
        R = (R + R.T) * 0.5           # the metric must be symmetric to be one
        RA = R @ A
        RB = R @ B

    M = np.conj(A).T @ RB                                  # (Ka, Kb)
    norm_a = np.real(np.einsum('ik,ik->k', np.conj(A), RA))
    norm_b = np.real(np.einsum('ik,ik->k', np.conj(B), RB))

    # R is only positive SEMI-definite, so a non-radiating column can land at a
    # tiny negative norm through roundoff. Screen those out rather than take the
    # square root of a negative number.
    ok_a = norm_a > 0
    ok_b = norm_b > 0
    denom = np.sqrt(np.outer(np.where(ok_a, norm_a, 1.0),
                             np.where(ok_b, norm_b, 1.0)))
    C = np.abs(M) / denom
    C[~ok_a, :] = 0.0
    C[:, ~ok_b] = 0.0
    # |cos| can exceed 1 by ~1e-16; a caller comparing against a threshold
    # should not have to know that.
    np.clip(C, 0.0, 1.0, out=C)

    return (C, M) if return_overlap else C


# ----------------------------------------------------------------------
# Extension point: post-assignment corrections (TECA and friends)
# ----------------------------------------------------------------------

@dataclass
class TrackingStep:
    """
    Everything known about one sweep step, handed to a correction stage.

    A correction may read all of it and return a revised `assignment` (and
    optionally `corrected_eigenvalues_next`). It must not mutate the arrays it
    is given -- they are views into the sweep.
    """

    freq_index: int              # index of the later frequency
    f_prev: float
    f_next: float
    # Modal currents. `J_prev` is already in TRACK order (track k is column k);
    # `J_next` is in the solver's raw order at the later frequency.
    J_prev: np.ndarray           # (N, K)
    J_next: np.ndarray           # (N, K)
    lambda_prev: np.ndarray      # (K,) track order
    lambda_next: np.ndarray      # (K,) raw order
    R_prev: np.ndarray           # (N, N) or None
    R_next: np.ndarray           # (N, N) or None
    X_prev: np.ndarray = None    # (N, N) or None -- available when the caller has Z
    X_next: np.ndarray = None
    correlation: np.ndarray = None   # (K, K) C[track, raw]
    assignment: np.ndarray = None    # (K,) raw index chosen for each track
    accepted: np.ndarray = None      # (K,) C at the accepted assignment
    margin: np.ndarray = None        # (K,) accepted minus best rejected
    degenerate_pairs: tuple = ()     # ((k, l), ...) track pairs flagged quasi-degenerate
    # A correction may fill this in; None leaves the raw eigenvalues in place.
    corrected_eigenvalues_next: np.ndarray = None


class ModeTrackingCorrection:
    """
    Base class for a correction applied after the Hungarian assignment.

    The contract is deliberately narrow: `refine` receives a `TrackingStep`
    describing one sweep step and returns it, possibly with `assignment` and/or
    `corrected_eigenvalues_next` replaced. It runs once per step, inside the
    tracker's forward loop, so it sees the history only through `J_prev` /
    `lambda_prev`, which are already in track order.

    A correction needing the whole sweep at once (a backward pass, or a fit over
    several steps) should instead be written as a function taking a finished
    `ModeTrackingResult` and returning a revised one; this per-step hook is for
    local corrections.
    """

    name = 'base'

    def refine(self, step: TrackingStep) -> TrackingStep:
        raise NotImplementedError


class NullCorrection(ModeTrackingCorrection):
    """Keep the Hungarian assignment exactly as it is. The default."""

    name = 'none'

    def refine(self, step: TrackingStep) -> TrackingStep:
        return step


class TECACorrection(ModeTrackingCorrection):
    """
    Placeholder for the TECA-style eigenvalue correction of Chen, Pan and Su
    (IEEE Trans. Antennas Propag., 2021).

    NOT IMPLEMENTED, on purpose: the numerical core is the content of the paper,
    and guessing at it would produce corrected eigenvalues that look
    authoritative and are not. What is provided here is the place it plugs into
    and the data it needs, so implementing it is a matter of filling in `refine`
    without touching the tracker.

    What the hook already gives you
    -------------------------------
    - `step.J_prev` (track order) and `step.J_next` (raw order), on the same RWG
      basis, together with `step.lambda_prev` / `step.lambda_next`;
    - `step.R_prev`/`step.R_next` and, when the caller passed them,
      `step.X_prev`/`step.X_next`, so `Z = R + jX` can be rebuilt at both ends
      of the step and a local derivative in frequency formed by finite
      differences;
    - `step.correlation` and `step.assignment`, i.e. the correlation-only answer
      the correction is meant to improve on, plus `step.margin`, which says how
      close the runner-up assignment was;
    - `step.degenerate_pairs`, the tracks the correlation metric cannot separate
      -- exactly the bands such a correction exists to handle.

    What an implementation must return
    ----------------------------------
    The same `TrackingStep` with, at minimum, `assignment` set to a permutation
    of `range(K)` (one raw index per track, no repeats -- the tracker validates
    this). If the scheme also revises the characteristic values, put the revised
    values for the LATER frequency, in RAW index order, into
    `corrected_eigenvalues_next`; they are collected into
    `ModeTrackingResult.corrected_eigenvalues` and left beside the raw
    eigenvalues rather than overwriting them.
    """

    name = 'teca'

    def refine(self, step: TrackingStep) -> TrackingStep:
        raise NotImplementedError(
            "TECA correction (Chen, Pan & Su, IEEE TAP 2021) is not implemented. "
            "solver.cma.mode_tracking.TECACorrection documents the hook it plugs "
            "into and the data it receives; implement refine() there. Use "
            "TrackingParameters(correction='none') for correlation-only tracking."
        )


CORRECTIONS = {
    'none': NullCorrection,
    'teca': TECACorrection,
}


def get_correction(name):
    """Instantiate a registered correction stage by name."""
    try:
        return CORRECTIONS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown tracking correction {name!r}. Registered: "
            f"{sorted(CORRECTIONS)}.") from None


# ----------------------------------------------------------------------
# The tracker
# ----------------------------------------------------------------------

class ModeTracker:
    """
    Follow K characteristic modes across a frequency sweep.

    Streaming by design: `push` one frequency at a time and only two consecutive
    frequencies are ever held, so a sweep can be tracked inside the assembly
    loop without keeping every `R(f)` (an `(n_freq, N, N)` array) alive for the
    tracking's sake.

    >>> tracker = ModeTracker(TrackingParameters())
    >>> for f, lam, J, R in sweep_points:              # doctest: +SKIP
    ...     tracker.push(f, lam, J, R=R)
    >>> result = tracker.finalize()                    # doctest: +SKIP
    >>> lam_tracked = result.reorder(eigenvalues)      # doctest: +SKIP
    """

    def __init__(self, params: TrackingParameters = None):
        self.params = params or TrackingParameters()
        if self.params.metric not in ('R', 'identity'):
            raise ValueError(f"Unknown tracking metric {self.params.metric!r}. "
                             "Use 'R' or 'identity'.")
        if self.params.weight_at not in ('mean', 'previous', 'next'):
            raise ValueError(f"Unknown weight_at {self.params.weight_at!r}. "
                             "Use 'mean', 'previous' or 'next'.")
        self._correction = get_correction(self.params.correction)

        self._freqs = []
        self._order = []
        self._sign = []
        self._corr = []
        self._margin = []
        self._events = []
        self._corrected = []
        self._n_modes = None

        # Carried between pushes (track order for J/lambda, raw for R/X).
        self._J_prev = None
        self._lam_prev = None
        self._R_prev = None
        self._X_prev = None

    # -- public -------------------------------------------------------

    def push(self, frequency, eigenvalues, eigenvectors, R=None, X=None):
        """
        Add one frequency point.

        Parameters
        ----------
        frequency    : float, Hz -- must increase monotonically
        eigenvalues  : (K,)    lambda_n in the solver's raw order
        eigenvectors : (N, K)  J_n in the same raw order
        R            : (N, N)  Re(Z) at this frequency. Required for
                       metric='R'; ignored for metric='identity'.
        X            : (N, N)  Im(Z), optional -- only carried through to a
                       correction stage, never used by the correlation itself.
        """
        lam = np.asarray(eigenvalues, dtype=float).ravel()
        J = np.asarray(eigenvectors)
        if J.ndim == 1:
            J = J[:, None]
        if J.shape[1] != lam.shape[0]:
            raise ValueError(f"push: {lam.shape[0]} eigenvalues but {J.shape[1]} "
                             "eigenvectors.")
        if self._n_modes is None:
            self._n_modes = J.shape[1]
        elif J.shape[1] != self._n_modes:
            raise ValueError(
                f"push: {J.shape[1]} modes at {frequency / 1e6:.3f} MHz but "
                f"{self._n_modes} at the previous frequencies. Tracking needs the same "
                "number of modes at every point -- with cma_solver='rank_truncated', "
                "lower n_modes to the rank at the lowest frequency.")
        if self.params.metric == 'R' and R is None:
            raise ValueError("push: metric='R' needs R = Re(Z) at this frequency. "
                             "Pass R=..., or use TrackingParameters(metric='identity').")
        if self._freqs and frequency <= self._freqs[-1]:
            raise ValueError(
                f"push: frequencies must increase ({frequency:.6g} Hz after "
                f"{self._freqs[-1]:.6g} Hz). Tracking compares ADJACENT points, so the "
                "order they arrive in is the order they are correlated in.")

        K = self._n_modes
        if not self._freqs:
            self._freqs.append(float(frequency))
            self._order.append(np.arange(K))
            self._sign.append(np.ones(K))
        else:
            self._step(frequency, lam, J, R, X)

        self._J_prev = J[:, self._order[-1]] * self._sign[-1][None, :]
        self._lam_prev = lam[self._order[-1]]
        self._R_prev = R
        self._X_prev = X

    def finalize(self) -> ModeTrackingResult:
        """Close the sweep and return the permutation plus the diagnostics."""
        if not self._freqs:
            raise RuntimeError("finalize: no frequency was pushed.")
        n_freq, K = len(self._freqs), self._n_modes
        corrected = None
        if any(c is not None for c in self._corrected):
            corrected = np.full((n_freq, K), np.nan)
            for i, c in enumerate(self._corrected):
                if c is not None:
                    corrected[i] = c
        return ModeTrackingResult(
            frequencies=np.asarray(self._freqs, dtype=float),
            order=np.asarray(self._order, dtype=int).reshape(n_freq, K),
            sign=np.asarray(self._sign, dtype=float).reshape(n_freq, K),
            correlation=np.asarray(self._corr, dtype=float).reshape(n_freq - 1, K),
            margin=np.asarray(self._margin, dtype=float).reshape(n_freq - 1, K),
            events=list(self._events),
            metric=self.params.metric,
            correction=self.params.correction,
            corrected_eigenvalues=corrected,
        )

    # -- internals ----------------------------------------------------

    def _weight(self, R_next):
        """The R the correlation of this step is measured in."""
        if self.params.metric == 'identity':
            return None
        if self.params.weight_at == 'previous':
            return self._R_prev
        if self.params.weight_at == 'next':
            return R_next
        return 0.5 * (np.asarray(self._R_prev) + np.asarray(R_next))

    def _step(self, frequency, lam_next, J_next, R_next, X_next):
        K = self._n_modes
        i_next = len(self._freqs)          # index this point will occupy

        C, M = correlation_matrix(self._J_prev, J_next, self._weight(R_next),
                                  return_overlap=True)

        # Maximise total correlation. Greedy per-track matching would be cheaper
        # and is not the same problem: it can send two tracks to the same mode,
        # and its answer depends on which track is considered first.
        row, col = linear_sum_assignment(-C)
        assignment = np.empty(K, dtype=int)
        assignment[row] = col

        degenerate = self._degenerate_pairs(self._lam_prev, lam_next[assignment])
        accepted = C[np.arange(K), assignment]
        margin = self._margins(C, assignment, accepted)

        step = TrackingStep(
            freq_index=i_next, f_prev=self._freqs[-1], f_next=float(frequency),
            J_prev=self._J_prev, J_next=J_next,
            lambda_prev=self._lam_prev, lambda_next=lam_next,
            R_prev=self._R_prev, R_next=R_next, X_prev=self._X_prev, X_next=X_next,
            correlation=C, assignment=assignment, accepted=accepted, margin=margin,
            degenerate_pairs=degenerate,
        )
        step = self._correction.refine(step)

        # A correction may have revised the assignment, so everything derived
        # from it is recomputed rather than carried over from above.
        assignment = self._validated(step.assignment, K)
        accepted = C[np.arange(K), assignment]
        margin = self._margins(C, assignment, accepted)
        degenerate = self._degenerate_pairs(self._lam_prev, lam_next[assignment])

        prev_order = self._order[-1]
        self._freqs.append(float(frequency))
        self._order.append(assignment)
        self._corr.append(accepted)
        self._margin.append(margin)
        self._corrected.append(step.corrected_eigenvalues_next)

        # Sign continuity along each track: M[k, assignment[k]] is the SIGNED
        # overlap between track k's current at the previous frequency (already
        # sign-corrected) and the mode it was matched to here.
        if self.params.fix_sign:
            overlap = np.real(M[np.arange(K), assignment])
            self._sign.append(np.where(overlap < 0, -1.0, 1.0))
        else:
            self._sign.append(np.ones(K))

        self._flag(i_next, float(frequency), prev_order, assignment,
                   accepted, margin, degenerate, lam_next)

    @staticmethod
    def _validated(assignment, K):
        a = np.asarray(assignment, dtype=int).ravel()
        if a.shape != (K,) or np.unique(a).size != K or a.min() < 0 or a.max() >= K:
            raise ValueError(
                f"A tracking correction returned {a!r}, which is not a permutation of "
                f"range({K}). Every track must be assigned exactly one mode.")
        return a

    @staticmethod
    def _margins(C, assignment, accepted):
        """Accepted correlation minus the best correlation this track gave up."""
        K = C.shape[0]
        masked = C.copy()
        masked[np.arange(K), assignment] = -np.inf
        best_rejected = masked.max(axis=1)
        best_rejected[~np.isfinite(best_rejected)] = 0.0
        return accepted - best_rejected

    def _degenerate_pairs(self, lam_a, lam_b):
        """
        Track pairs whose characteristic values are too close to separate.

        Measured as a gap relative to the eigenvalues' own scale (see
        TrackingParameters.degeneracy_rel_tol) and at BOTH ends of the step: a
        pair that is degenerate at either end has an ill-defined eigenvector
        basis somewhere inside the step.
        """
        tol = self.params.degeneracy_rel_tol
        pairs = []
        for lam in (lam_a, lam_b):
            gap = np.abs(lam[:, None] - lam[None, :])
            scale = np.maximum(np.maximum(np.abs(lam[:, None]), np.abs(lam[None, :])), 1.0)
            close = gap < tol * scale
            for k, l in zip(*np.triu_indices(len(lam), k=1)):
                if close[k, l] and (int(k), int(l)) not in pairs:
                    pairs.append((int(k), int(l)))
        return tuple(pairs)

    def _flag(self, i_next, frequency, prev_order, assignment, accepted, margin,
              degenerate, lam_next):
        p = self.params
        add = self._events.append

        # Mode swapping. The solver relabels every frequency by |lambda|, so two
        # tracks swapped their labels exactly when the raw indices they hold
        # invert between two steps -- which is what a genuine eigenvalue
        # crossing looks like from the label's point of view.
        swapped = set()
        for k, l in zip(*np.triu_indices(len(assignment), k=1)):
            before = np.sign(prev_order[k] - prev_order[l])
            after = np.sign(assignment[k] - assignment[l])
            if before != 0 and after != 0 and before != after:
                swapped.add((int(k), int(l)))
                add(TrackingEvent(
                    EVENT_SWAP, i_next, frequency, (int(k), int(l)),
                    f"solver labels {prev_order[k] + 1}/{prev_order[l] + 1} -> "
                    f"{assignment[k] + 1}/{assignment[l] + 1}; tracked through with "
                    f"C = {accepted[k]:.4f}/{accepted[l]:.4f}"))

        for k, l in degenerate:
            lam_k = lam_next[assignment[k]]
            lam_l = lam_next[assignment[l]]
            delta = characteristic_angle(np.array([lam_k, lam_l]))
            rel = abs(lam_k - lam_l) / max(abs(lam_k), abs(lam_l), 1.0)
            add(TrackingEvent(
                EVENT_DEGENERACY, i_next, frequency, (k, l),
                f"lambda = {lam_k:+.6g} vs {lam_l:+.6g} (delta {delta[0]:.2f} vs "
                f"{delta[1]:.2f} deg); relative gap {rel:.3e} < "
                f"{p.degeneracy_rel_tol:g}. The eigenvectors inside a degenerate pair "
                "are set by roundoff, so this assignment is not decidable from "
                "correlation alone"))

            # Crossing avoidance: the pair comes close, the labels do NOT
            # invert, and the correlation or the margin collapses -- the two
            # modes exchanged character continuously instead of crossing.
            weak = (min(accepted[k], accepted[l]) < p.correlation_threshold
                    or min(margin[k], margin[l]) < p.margin_threshold)
            if (k, l) not in swapped and weak:
                add(TrackingEvent(
                    EVENT_AVOIDED_CROSSING, i_next, frequency, (k, l),
                    f"quasi-degenerate without a label crossing, C = "
                    f"{accepted[k]:.4f}/{accepted[l]:.4f}, margin = "
                    f"{margin[k]:.4f}/{margin[l]:.4f}; the eigenvalue curves veer apart "
                    "while the currents exchange character, so mode identity across "
                    "this step is ambiguous"))

        for k in np.flatnonzero(accepted < p.correlation_threshold):
            add(TrackingEvent(
                EVENT_LOW_CORRELATION, i_next, frequency, (int(k),),
                f"best C = {accepted[k]:.4f} < {p.correlation_threshold:g}; the mode "
                "changed more between these two frequencies than a mode should, "
                "usually a sweep step too coarse for this band"))


# ----------------------------------------------------------------------
# Post-processing entry point
# ----------------------------------------------------------------------

def track_sweep(frequencies, eigenvalues, eigenvectors, R_list=None, X_list=None,
                params: TrackingParameters = None) -> ModeTrackingResult:
    """
    Track a sweep that has already been computed.

    Parameters
    ----------
    frequencies  : (n_freq,)            Hz, increasing
    eigenvalues  : (n_freq, K)          raw per-frequency order
    eigenvectors : (n_freq, N, K)       same order
    R_list       : (n_freq, N, N) or a sequence of (N, N) -- Re(Z) per frequency.
                   Required for the default metric='R'.
    X_list       : optional Im(Z) per frequency, carried to a correction stage.
    params       : TrackingParameters

    Returns
    -------
    ModeTrackingResult
    """
    params = params or TrackingParameters()
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors)
    if eigenvalues.shape[0] != frequencies.size or eigenvectors.shape[0] != frequencies.size:
        raise ValueError("track_sweep: frequencies, eigenvalues and eigenvectors must "
                         "agree on the number of frequency points "
                         f"({frequencies.size}, {eigenvalues.shape[0]}, "
                         f"{eigenvectors.shape[0]}).")
    if params.metric == 'R' and R_list is None:
        raise ValueError(
            "track_sweep: metric='R' needs Re(Z) at every frequency, which a saved CMA "
            "sweep .mat does not contain. Either pass R_list=..., or track inside "
            "cma_frequency_sweep() where Z(f) is still in memory, or fall back to "
            "TrackingParameters(metric='identity').")

    tracker = ModeTracker(params)
    for i, f in enumerate(frequencies):
        tracker.push(f, eigenvalues[i], eigenvectors[i],
                     R=None if R_list is None else np.asarray(R_list[i]),
                     X=None if X_list is None else np.asarray(X_list[i]))
    return tracker.finalize()
