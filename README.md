# sesmax

[![Tests](https://github.com/jejjohnson/sesmax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/sesmax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/sesmax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/sesmax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/sesmax/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple Earth-system models in JAX.

Author: J. Emmanuel Johnson

sesmax is a workspace of packages whose models are cheap, differentiable forward models to use as priors in short-window data assimilation: a window of hours to weeks, inside the predictability time of the scales the model resolves. A model here is judged as a prior: cheap enough to run ensembles and take gradients through a whole window, good where the observations look, honest about its error, and learnable. It grows out of [somax](https://github.com/jejjohnson/somax), a simple ocean model in JAX.

The workspace is being scaffolded. The plan, its phases and its gates are in the [workspace and core design](docs/design/workspace_and_core.md).

## Packages

| Tier | Distribution | Import | Holds | Status |
| --- | --- | --- | --- | --- |
| — | `sesmax` | `sesmax` | code-free umbrella that installs the packages | scaffolded |
| 1 | `sesmax-ocean` | `sesmax_ocean` | shallow-water, QG and spherical ocean models, ocean diagnostics, scenarios | planned (C2) |
| 0 | `sesmax-core` | `sesmax_core` | state and parameter types, term algebra, steppers, forcing and bases, DA adapters, monitors, I/O | empty until C1 |

A package imports only packages in a lower tier.

## Layout

```text
sesmax/
├── pyproject.toml          # the workspace root (never published)
├── packages/
│   ├── sesmax-core/        # tier 0: src/sesmax_core, tests/
│   └── sesmax/             # the umbrella: dependencies only
├── tests/                  # workspace tests: tiers, docs build, capability index
├── docs/                   # one site: MyST prose + MkDocs API reference
└── scripts/                # build_docs.py, capabilities.py
```

## Development

With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
make install    # every package and dependency group, plus pre-commit hooks
make test       # every test and doctest
make lint       # ruff on the whole repo
make typecheck  # ty on packages/*/src and scripts
make docs       # build the documentation site into public/
```

`make help` lists every target. The rules for contributors and coding agents are in [AGENTS.md](AGENTS.md), and the review checklist is in [CODE_REVIEW.md](CODE_REVIEW.md).

## License

MIT; see [LICENSE](LICENSE).
