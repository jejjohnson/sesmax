# sesmax

> Simple Earth-system models in JAX: cheap, differentiable forward models to use as priors in short-window data assimilation.

sesmax is a workspace of packages: a shared core, one package per Earth-system component, and later a coupler on top. Its models are the forward models inside an assimilation window of hours to weeks, so they are built to be cheap enough for ensembles, differentiable through a whole window, honest about their error, and learnable.

:::{note} Status
The workspace is being scaffolded. `sesmax-core` is empty until it receives the model-agnostic machinery from [somax](https://github.com/jejjohnson/somax), and the ocean component follows. The plan, its phases and its gates are in the [workspace and core design](design/workspace_and_core.md).
:::

## Packages

| Distribution | Import | Holds | Status |
| --- | --- | --- | --- |
| `sesmax-core` | `sesmax_core` | state and parameter types, term algebra, steppers, forcing and bases, DA adapters, monitors, I/O | scaffolded (C0) |
| `sesmax-ocean` | `sesmax_ocean` | shallow-water, QG and spherical ocean models, ocean diagnostics, the double-gyre scenarios | planned (C2) |
| `sesmax` | — | code-free umbrella that installs the packages | scaffolded (C0) |

## Installation

From source, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
make install
```
