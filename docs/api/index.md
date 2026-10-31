# API Reference

Each workspace package is imported by its own name. Everything listed in a package's `__all__` is importable from its top level:

```python
import sesmax_core
```

The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

`sesmax_core` receives the core machinery from somax during phase C1 of the workspace and core design; each public module gets its own page here as it lands.

## Modules

| Page | Contents |
|---|---|
| [Types and models](core.md) | `State`, `Params`, `PhysConsts`, `Diagnostics`; `positive`, `interval`, `frozen`, `as_parameter`, `trainable_mask`; `Model`, `TermModel` |
| [Terms](terms.md) | `Term`, `TermFn`, `Sum`, `Scaled`, `Compose`, `explicit`, `implicit`, `partition`, `build_diffrax_terms`, `OperatorTerm` |
| [Scales](scales.md) | `Scales`, `StateAffine`, `ScaledModel` |
| [Layered fluids](gfd.md) | `StratificationProfile`, `ModalTransform`, `HelmholtzCache` and its periodic, Dirichlet, Neumann and multimodal versions |
| [Domains](domain.md) | `Domain`, `TimeDomain` |
| [Solvers](solvers.md) | `imex_solver`, `imex_stepsize_controller` |
| [Guards](guards.md) | `guard_finite`, `guard_positive`, `guard_ceiling` |

## Package overview

::: sesmax_core
    options:
      members: false
      show_root_heading: false
      show_source: false
