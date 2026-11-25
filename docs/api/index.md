# API Reference

Each workspace package is imported by its own name. Everything listed in a package's `__all__` is importable from its top level:

```python
import sesmax_core
import sesmax_experiment
import sesmax_ocean
```

The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Page | Contents |
|---|---|
| [Types and models](core.md) | `State`, `Params`, `PhysConsts`, `Diagnostics`; `positive`, `interval`, `frozen`, `as_parameter`, `trainable_mask`; `Model`, `TermModel` |
| [Terms](terms.md) | `Term`, `TermFn`, `Sum`, `Scaled`, `Compose`, `explicit`, `implicit`, `partition`, `build_diffrax_terms`, `OperatorTerm` |
| [Forcing](forcing.md) | `ForcingProtocol` and the constant, seasonal and interpolated forcings; `AbstractForcing`, `ForcingSum`, `ForcingProduct`, `TimeVaryingForcing`, the wind, drag and Rayleigh operators and their tendency functions, `linear_drag`, `rayleigh_relaxation`; `SpatialBasis`, `TemporalBasis`, `BasisForcing`, `ForcingTerm`, the vector versions; the `spatial_from_*` constructors, `tile_in_time`, `GaussianWindowsInTime`, `HatInTime`, `matern_spectral_density` |
| [Scales](scales.md) | `Scales`, `StateAffine`, `ScaledModel` |
| [Layered fluids](gfd.md) | `StratificationProfile`, `ModalTransform`, `streamfunction_from_vorticity`, `pressure_from_divergence`, `pv_inversion`; `resolve_burger`, `burger_to_g_prime`, `resolve_burger_spherical` |
| [Vertical modes](vertical.md) | `build_coupling_matrix`, `decompose_vertical_modes`, `layer_to_mode`, `mode_to_layer`, `multilayer`, `barotropic_filter` |
| [Domains](domain.md) | `Domain`, `TimeDomain` |
| [Solvers](solvers.md) | `imex_solver`, `imex_stepsize_controller`, `stable_step` |
| [Training rollouts](train.md) | `rollout`, `rollout_loss`, `state_mse`, `ModelStep` |
| [Operators](operators.md) | `ModelOp`, `OperatorTerm` |
| [Data assimilation](da.md) (`[da]`) | `FilteraxDynamics`, `VardaxForwardModel`, `WindowOperator`, `SubsampleObs`, `state_to_vector`, `make_ensemble` |
| [Flows](flows.md) (`[flows]`) | `StateAffineBijection`, `to_flowjax`, `learned_prior` |
| [I/O](io.md) (`[xarray]`) | `state_to_dataset`, `dataset_to_state`, `snapshots_to_dataset`, `append_to_dataset`, `save_dataset`, `load_dataset`, `scales_attrs`, `transform_attrs`, `apply_scale_metadata` |
| [Guards](guards.md) | `guard_finite`, `guard_positive`, `guard_ceiling` |
| [Fields](fields.md) | `each`, `map_fields`, `placed`, `values`, `is_field` |
| [Validation](validation.md) | `require_positive`, `require_non_negative`, `reject_derived_kwargs` |
| [Components](component.md) | `Component`, `BuiltRun`, `Builder`, `Check` |
| [Test problems](testing.md) | `Lorenz63`, `Lorenz96`, `Lorenz96t`; `LinearConvection1D`, `NonlinearConvection1D`, `Diffusion1D`, `Burgers1D` and their 2D versions; `Burgers2DTermModel`, `Burgers2DOp`; `IncompressibleNS2D`; `PoissonSolver2D`, `LaplaceSolver2D`, `HelmholtzSolver2D`; `LinearShallowWater1D`, `NonlinearShallowWater1D`, `LinearShallowWater2D`, `LinearSWM2DTermModel` and its terms; `gravity_wave_1d`, `inertial_oscillation_1d`, `geostrophic_adjustment_2d`; and their state, parameter and diagnostic classes |

### sesmax-experiment

| Page | Contents |
|---|---|
| [Runs in chunks](experiment_run.md) | `run_chunks`, `Timeline`, `ChunkedRun`, `ChunkFrame`, `RunStopped`; the schedules `every_chunks`, `every_sim_seconds`, `every_wall_seconds`, `at_snapshots`; the stages `AdvanceChunk`, `RunMonitors` |
| [Monitors](experiment_monitor.md) | `Monitor`, `BaseMonitor`, `MonitorVerdict`, `ChunkInfo`, the six built-in monitors, `default_monitors` |
| [Checkpointing](experiment_checkpoint.md) (`[checkpoint]`) | `SimulationCheckpointer` |
| [Run specs](experiment_sim.md) (`[sim]`) | `RunSpec` and its blocks, `load_yaml`, `dump_yaml`; `simulate`, `spinup`, `restart`, `SimulationResult`, `IntegrationDivergedError`; `build_run`, `timeline_of`; `Checkpointing`, `CheckpointStore`; `MetricWriter`, `JsonLinesWriter`; `check_cfl`, `check_bounded_metric`; `model_op`; the command `sesmax-experiment` |
| [Components](experiment_components.md) | `components`, `find_model` |

### sesmax-ocean

| Page | Contents |
|---|---|
| [QG models](ocean_qg.md) | `BarotropicQG`, `BaroclinicQG`, with their state, parameter, constant and diagnostic classes; `ReparameterizedQG` and its diagnostics |
| [Shallow-water models](ocean_swm.md) | `NonlinearShallowWater2D`, `MultilayerShallowWater2D`, with their state, parameter, constant and diagnostic classes |
| [Models on the sphere](ocean_spherical.md) | `SphericalSWM`, `SphericalQG`, with their state, parameter, constant and diagnostic classes |
| [Resolution checks](ocean_resolution.md) | `check_munk_width`, `check_stommel_width`, `check_deformation_radius`, `check_equatorial_deformation_radius`, `AssertionFailedError` |
| [Metrics](ocean_eval.md) | `rms_divergence`, `total_enstrophy`, `kinetic_energy`, `geostrophic_imbalance`, `qg_balance_residual`, `compute_eval_metrics` |
| [Forcing presets](ocean_forcing.md) | `ssh_geostrophic`, `sss_coastal`, `sst_frontal`, `geostrophic_currents` |

## Package overview

::: sesmax_core
    options:
      members: false
      show_root_heading: false
      show_source: false
