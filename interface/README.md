# interface/ — front end

This folder will hold the **front end** (user interface) of Antenna Solver MoM.

It is empty for now. The electromagnetic solver itself (MoM / RWG pipeline, characteristic modes,
optimisation tools) lives in [`../solver/`](../solver/); the front end calls the solver and does not
reimplement any of its physics.

Repository layout:

```
Antenna_Solver_MoM/
├── solver/      ← MoM solver (Python package, imported as `solver.*`)
├── interface/   ← front end (this folder)
├── test/        ← test suite
└── data/        ← meshes and intermediate results
```
