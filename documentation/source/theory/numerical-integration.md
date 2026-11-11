# Numerical integration and singularities

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

The impedance matrix is a sum of double surface integrals whose kernel $e^{-jkR}/R$ is singular when
the two triangles touch. How the solver integrates each case, and how accurate it is.

**This page will cover**

- Classification of triangle pairs into SELF, EDGE, VERTEX and FAR
  (`solver/utils/quadrature/singularity_classifier.py`).
- Gauss–Dunavant rules on triangles, orders available and the defaults (`gauss_dunavant.py`;
  `quad_order`, `near_order`, `NEAR_ORDER_DEFAULT = 16`).
- The Duffy transform for SELF pairs (`duffy.py`) and the weight normalisation between Duffy and
  Dunavant (`impmet._duffy_weight_scale`).
- Closed-form integration of the $1/R$ part for EDGE / VERTEX pairs (`analytic_singular.py`).
- Measured accuracy per category and the reciprocity residual as a diagnostic, with the tests that
  pin them (`solver/utils/quadrature/tests/`).
