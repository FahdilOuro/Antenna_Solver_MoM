# `solver.rwg` — RWG pipeline

The six stages of the RWG pipeline: mesh topology (`rwg1`), ρ vectors (`rwg2`), impedance matrix
(`rwg3`), linear solve and port results (`rwg4`), surface current density (`rwg5`, `rwg6`).
Theory: {doc}`../theory/rwg-basis-functions`.

## `rwg1` — points, triangles, edges

```{eval-rst}
.. automodule:: solver.rwg.rwg1
```

## `rwg2` — ρ vectors

```{eval-rst}
.. automodule:: solver.rwg.rwg2
```

## `rwg3` — impedance matrix

```{eval-rst}
.. automodule:: solver.rwg.rwg3
```

## `rwg4` — solving Z I = V

```{eval-rst}
.. automodule:: solver.rwg.rwg4
```

## `rwg5` — surface current density

```{eval-rst}
.. automodule:: solver.rwg.rwg5
```

## `rwg6` — surface current distribution

```{eval-rst}
.. automodule:: solver.rwg.rwg6
```
