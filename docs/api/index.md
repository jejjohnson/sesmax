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
| [Forcing](forcing.md) | `ForcingProtocol` and the constant, seasonal and interpolated forcings; `SpatialBasis`, `TemporalBasis`, `BasisForcing`, `ForcingTerm`, the vector versions; the `spatial_from_*` constructors, `tile_in_time`, `GaussianWindowsInTime`, `matern_spectral_density` |
| [Scales](scales.md) | `Scales`, `StateAffine`, `ScaledModel` |
| [Layered fluids](gfd.md) | `StratificationProfile`, `ModalTransform`, `HelmholtzCache` and its periodic, Dirichlet, Neumann and multimodal versions |
| [Domains](domain.md) | `Domain`, `TimeDomain` |
| [Solvers](solvers.md) | `imex_solver`, `imex_stepsize_controller` |
| [Training rollouts](train.md) | `rollout`, `rollout_loss`, `state_mse`, `ModelStep` |
| [Monitors](monitor.md) | `Monitor`, `BaseMonitor`, `MonitorVerdict`, `ChunkInfo`, the six built-in monitors, `default_monitors` |
| [Operators](operators.md) | `ModelOp`, `OperatorTerm` |
| [Data assimilation](da.md) (`[da]`) | `FilteraxDynamics`, `VardaxForwardModel`, `SubsampleObs`, `state_to_vector`, `make_ensemble` |
| [Flows](flows.md) (`[flows]`) | `StateAffineBijection`, `to_flowjax`, `learned_prior` |
| [I/O](io.md) (`[xarray]`) | `state_to_dataset`, `dataset_to_state`, `snapshots_to_dataset`, `append_to_dataset`, `save_dataset`, `load_dataset`, `scales_attrs`, `transform_attrs`, `apply_scale_metadata` |
| [Checkpointing](checkpoint.md) (`[checkpoint]`) | `SimulationCheckpointer` |
| [Guards](guards.md) | `guard_finite`, `guard_positive`, `guard_ceiling` |
| [Test problems](testing.md) | `Lorenz63`, `Lorenz96`, `Lorenz96t`; `LinearConvection1D`, `NonlinearConvection1D`, `Diffusion1D`, `Burgers1D`; and their state, parameter and diagnostic classes |

## Package overview

::: sesmax_core
    options:
      members: false
      show_root_heading: false
      show_source: false
