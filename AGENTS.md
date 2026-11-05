# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

sesmax ("simple Earth-system models in JAX") is a uv workspace of packages whose models are cheap, differentiable forward models to use as priors in short-window data assimilation (a window of hours to weeks). The plan, the phases and what moves here from [somax](https://github.com/jejjohnson/somax) are in [`docs/design/workspace_and_core.md`](docs/design/workspace_and_core.md); read it before changing the layout.

People build on the core, so **the core is a library of primitives**: new code composes what is here, and anything genuinely new lands where the next person will find and reuse it.

Each distribution lives in `packages/<dist>/` with its own `pyproject.toml`, `src/<import>/`, tests, README and CHANGELOG. The workspace root (`sesmax-workspace`) is never published. Imports only point down the tiers:

| Tier | Distribution | Import | Holds | Status |
|---|---|---|---|---|
| — | `sesmax` | `sesmax` | code-free umbrella that installs the packages | scaffolded |
| 2 | `sesmax-couple` | `sesmax_couple` | flux exchange, regridding, coupled runs | later |
| 1 | `sesmax-ocean` | `sesmax_ocean` | the ocean models, ocean diagnostics, scenarios | phase C2 |
| 0 | `sesmax-core` | `sesmax_core` | state and parameter types, term algebra, steppers, forcing and bases, scales, DA adapters, monitors, I/O, test problems | moved from somax; first release 0.1.0 |

The public API of each package is its top-level `__all__`, plus its public submodules.

### Boundaries

- **Tiers.** A package imports only packages in a lower tier; `sesmax_core` imports no component and nothing from `somax` (`tests/test_workspace.py` reads the sources and fails otherwise). Two components never import each other: what they share goes into the core, and what couples them goes into the coupler.
- **Dependencies.** `sesmax-core`'s base install: JAX, jaxtyping, Equinox, diffrax, lineax, optimistix, einops, paramax, finitevolX, spectraldiffx, pipekit (with pipekit-cycle and pipekit-jax) and geonnax; git dependencies are direct references in `packages/sesmax-core/pyproject.toml`, kept in step with the overrides in the root `pyproject.toml`. An optional dependency goes behind an extra (`[da]` for filterax and vardax, `[flows]`, `[xarray]`, `[checkpoint]`) and an import that raises an `ImportError` naming the extra.
- **Nobody imports sesmax yet.** When other projects build on it, list them here and keep their imports working: a rename or removal of a public name then goes through a deprecation (the old name keeps working with a `DeprecationWarning` that names the replacement for at least one release).

## What it is built on

Each foundation brings a rule. Breaking one runs fine on your example and fails in someone else's pipeline or in CI.

| Foundation | Used for | The rule it brings |
|---|---|---|
| **uv workspace** | one lock file and one environment for every package | `make install` (`uv sync --all-packages --all-groups`); a new package is a directory under `packages/`, a `[tool.uv.sources]` entry and a `dev` group entry in the root `pyproject.toml`, and a release-please entry. |
| **JAX and Equinox** | every model, term and step | A step is a pure function of state, parameters and forcing that `jit`, `grad` and `vmap` go through; models and states are `eqx.Module`s; shapes and flags are static fields, physical parameters are leaves. |
| **The stack** (finitevolX, spectraldiffx, diffrax, lineax, gaussx, vardax, filterax) | stencils, spectral solves, time stepping, linear algebra, bases, assimilation | Compose them; never re-implement a stencil, a solver or a basis here. |
| **pytest + `--doctest-modules`** | the suites and every docstring `Examples:` block | An example's expected output is what the code prints; a stale example is a failing test. |

## Reuse before you write

Before writing a helper, a stencil, a solver, a basis or a model building block,
find out whether it exists:

1. **Search the capability index.**
   [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every
   public name once, at its public home, grouped by the module that defines
   it, with a one-line summary. It is generated (`make capabilities`) and
   checked by `tests/test_capabilities.py`, so it is current.
2. **Search the stack** in the table below: the upstream libraries already
   own most numerical building blocks.
3. **If it is missing, add it to the module that owns the concept**, at the
   lowest layer every caller can reach, not inline in the one caller you
   have today; a helper two components need goes into `sesmax-core`.
4. **One object, one name.** A name means one thing across the package and
   its submodules (`tests/test_capabilities.py`); a deliberate exception is
   listed with its reason in `ALLOWED_SHARED_NAMES` in
   `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| A finite-difference stencil, a C-grid interpolation, a vorticity, divergence or kinetic-energy formula | finitevolX |
| A spectral derivative, a Helmholtz or Poisson solve | spectraldiffx |
| A time-stepping loop or a Runge–Kutta scheme | diffrax through `Model.integrate` and `Model.step`; `rollout` for a trajectory; `imex_solver` for a stiff term |
| `jnp.linalg.inv`, `solve` or a Cholesky on a covariance | lineax, gaussx |
| An RBF, Fourier, wavelet or EOF basis | the `spatial_from_*` constructors (`geonnax.basis` today, vardax's bases after #25) |
| Flattening a state pytree into a vector | `state_to_vector`, `make_ensemble` (`sesmax_core.da`), or `jax.flatten_util.ravel_pytree` where the DA stack is not installed |
| Summing physics contributions by hand in a `vector_field` | `Term`, its `+` and `*` algebra, `TermModel` |
| Lifting a forcing field onto a tendency component | `ForcingTerm`, `add_to`, `add_vector_to` |
| A softplus or sigmoid on a parameter, or `stop_gradient` to freeze one | `positive`, `interval`, `frozen`, `trainable_mask` |
| Non-dimensionalising or standardising a state | `Scales`, `StateAffine`, `ScaledModel` |
| A vertical-mode transform | `ModalTransform` |
| A blow-up check inside a tendency, or between chunks | `guard_finite`, `guard_positive`, `guard_ceiling`; `sesmax_core.monitor` |
| A Lorenz system, a 1D or 2D PDE or a linear shallow-water toy for a test or example | `sesmax_core.testing` |

## The contracts

Everything composes because it keeps these contracts; the tests in "What enforces them" check most of them.

### Models

- A model subclasses `Model` and implements `vector_field(t, state, args=None)`, which returns a tendency pytree with the state's structure, and `apply_boundary_conditions(state)`, which returns a state. `diagnose(state)` is optional and returns a `Diagnostics`.
- `Model` wraps those three methods so they run on the unwrapped model, so a `positive` or `interval` parameter behaves like a plain array inside them. Anything else reads a constrained parameter through the `diagnostics` method or `paramax.unwrap`, because arithmetic on a wrapper is a type error.
- `integrate(state0, t0, t1, dt, **kw)` returns a diffrax `Solution`, and `step(state, dt, *, t0=0.0, **kw)` returns a bare state with the same structure as `state`. The DA adapters and `rollout` call `step`, so a model changes how it integrates by overriding `build_terms` (SDE or IMEX), never `step`.
- `as_forward_model(dt=, dt0=, **settings)` returns a `pipekit_jax.DiffraxForwardModel`, which holds the solver, step controller, adjoint and step budget and satisfies `pipekit_cycle.ForwardModel`. It holds one vector field, so a model whose `build_terms` gives anything but one `ODETerm` (IMEX, SDE, or a `ScaledModel` around either) raises `ValueError`. pipekit is imported inside the method, so `import sesmax_core` stays pipekit-free.
- Boundary conditions usually live in `apply_boundary_conditions`, which fills the ghosts once per state. A term may fill its own instead, because advection and diffusion need different Dirichlet ghost values: the first reads the exterior value, the second a reflection about the face. The model's method is then a pass-through.
- A model whose right-hand side is an assembled term tree subclasses `TermModel`, which owns `vector_field` and `build_terms`; it overrides `apply_boundary_conditions` when it has boundaries.
- States subclass `State`, parameters `Params`, fixed constants `PhysConsts` and diagnostics `Diagnostics`. Physical parameters are leaves; shapes and flags are `eqx.field(static=True)`. A state sets `scale_kinds` and `mask_locations` where a field's name does not determine its meaning (`StateAffine` reads them).

### Terms

- A term subclasses `Term` and implements `__call__(t, state, args=None)`, which returns a tendency with the state's structure, zero in the fields it does not touch.
- Terms compose with `+`, `-`, `*` and `@` (`Sum`, `Scaled`, `Compose`). A coefficient is a JAX leaf, so `jax.grad` flows through it.
- A term is explicit unless wrapped with `implicit`. `partition` flattens a `Sum` by kind, and `build_diffrax_terms` returns one `ODETerm` when all summands share a kind, or a `MultiTerm` of (explicit, implicit) for an IMEX solver.
- Use `TermFn` to wrap a plain callable and `OperatorTerm` to lift a pipekit operator. `OperatorTerm` is a lazy top-level export, so `import sesmax_core` stays pipekit-free.

### Forcing

- A forcing subclasses `ForcingProtocol` and implements `__call__(t, grid)`, which returns a field. `ForcingTerm(forcing, place=add_to("q"), grid=grid)` lifts it into the term algebra; a forcing never edits a tendency itself.
- `BasisForcing` is `SpatialBasis.Phi @ (coeffs * TemporalBasis.weights(t))`. `coeffs` is the only control: `control_filter` selects it for an optimiser, and the dictionary, the temporal gate and the prior `std` stay fixed.
- Each `SpatialBasis` carries its per-mode prior standard deviation, from a spectral density for a spectral basis and prescribed for a frame. `whiten` and `regularization` read it.

### Data-assimilation adapters

- filterax and vardax work on flat vectors and sesmax on pytrees. An adapter stores a `template` state and calls `ravel_pytree` per call rather than holding an `unravel` closure, so it stays a clean pytree under `jit` and `filter_vmap`. It drives the model only through `step`.
- A `transform` given to `state_to_vector` or `make_ensemble` must also be given to the adapter, or the model receives standardised values as if they were physical ones.
- `VardaxForwardModel` passes the same `t0` on every substep, so it is exact only for autonomous models.
- `WindowOperator` is the route for a time-varying forcing: it maps a control through a whole model run to the stacked observations, for `vardax.ThreeDVar`, and the model integrates the window with its own clock. It inverts for a forcing or a parameter, not an initial state.
- Adapters import their library, so `sesmax_core.da` sits behind `[da]`, stays out of the top-level `__init__` and raises an `ImportError` naming the extra.

### Monitors

- A monitor subclasses `BaseMonitor`, sets `name` and overrides what it needs of `on_run_start(model, state0, *, spec)`, `on_chunk_end(model, state, info)` and `on_run_end(model, final_state, *, ok)`.
- Monitors run host-side between jitted chunks, so they may hold Python state and never run inside a trace. A halt that must happen at the offending step is a `guard_finite`, `guard_positive` or `guard_ceiling` in the tendency instead.
- `on_chunk_end` returns a `MonitorVerdict`; `terminate=True` needs a `reason`.
- Nothing in the core calls monitors yet: the run loop that does arrives with `sesmax_core.sim` (#28).

### Migrated code

- Code copied from somax keeps its behaviour. A copy changes only its imports and the renames listed in the design doc (`SomaxModel` → `Model`, `SomaxModelOp` → `ModelOp`, `SomaxDynamics` → `FilteraxDynamics`, `SomaxForwardModel` → `VardaxForwardModel`), and names the somax commit it came from in the commit message.
- The somax tests that cover a moved module move with it and pass with only the same import and rename changes.

### Moving code from somax

Code moves from somax at commit `9cb19b3` (phase C1 moved the core, and phase C2 moves the ocean). Every moved file is checked against its source with the import rewriter:

```bash
diff <(git -C ../somax show 9cb19b3:somax/_src/core/types.py | sed -f scripts/somax_imports.sed) \
     packages/sesmax-core/src/sesmax_core/_src/types.py
```

The diff may show only the changes allowed above (imports, the four renames, docstrings that say "somax", behaviour-neutral lint or type fixes). The somax tests for the module are copied to the same relative path under `packages/sesmax-core/tests/`. The commit message names the source: `Moved from somax 9cb19b3: somax/_src/core/types.py`.

### The public API

- Export a public name from its module's `__all__` and from the package's `__init__.py` (import and `__all__`); both stay sorted (ruff's `RUF022` and `test_all_is_sorted_and_unique` agree on the order).
- A new public module declares a sorted `__all__`, joins `SUBMODULES` in its package's `tests/test_public_api.py`, and gets its own page in `docs/api/` (a `::: sesmax_core.<module>` block), a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- Run `make capabilities` after any public API change.
- Link prose to the API by the **top-level** name (``[`Name`](xref:api#sesmax_core.Name)``), see "Documentation".

## What enforces them

Most rules here are tests, so CI tells you when one breaks. Read a test's docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `packages/*/tests/test_public_api.py` | `__all__` sorted, unique and importable; `SUBMODULES` is exactly the set of public modules; every submodule declares a sorted `__all__`; the `py.typed` marker ships; `__version__` is dotted |
| `tests/test_workspace.py` | The tiers (the core imports no component and nothing from somax); the umbrella holds no code; each `__version__` matches its `pyproject.toml` |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name is bound to two different objects |
| `packages/sesmax-core/tests/test_lazy_exports.py` | `OperatorTerm` is exported lazily, so `import sesmax_core` does not import pipekit |
| `packages/sesmax-core/tests/da/test_missing_extra.py` | Importing `sesmax_core.da` or `sesmax_core.flows` without its extra raises an `ImportError` naming the extra |
| `packages/sesmax-core/tests/core/test_model.py`, `test_paramax_integration.py` | The model contract; a model with wrapped parameters behaves like one with plain arrays, and gradients land on the stored values |
| `packages/sesmax-core/tests/core/test_terms.py` | The term algebra, IMEX partitioning and the diffrax terms it builds |
| `packages/sesmax-core/tests/core/test_forcing.py`, `core/test_basis.py` | Forcing fields, the reduced-order bases, and that a one-column `BasisForcing` reproduces a hand-written wind forcing |
| `packages/sesmax-core/tests/test_monitor.py` | The monitor protocol and the built-in monitors, without an integration |
| `packages/sesmax-core/tests/da/test_filterax.py`, `da/test_vardax.py` | A twin experiment through each DA adapter improves on the background (skipped without `[da]`) |
| `packages/sesmax-core/tests/testing/` | The test problems against analytic or published solutions, for example Ghia's cavity and Poiseuille flow in `test_navier_stokes.py` |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, anchor repair, link checking, the inventory port and the absolute nav URL in `docs/myst.yml`, the files copied to the site root (`ROOT_FILES`) |
| The `bare-install` job in `ci.yml` | Each package installs and imports alone, with no workspace around it |
| `--doctest-modules` on each package's sources (`addopts`) | Every docstring `Examples:` block runs and prints what it says |
| `filterwarnings = ["error", …]` | A warning in a test is a failure (deprecation warnings excepted) |
| `[tool.coverage.report] fail_under = 30` | Coverage of the packages in every coverage-enabled run (CI, `make test-cov`); deliberately low to start, raised as the moved code gains tests |
| `strict_markers = true` | Only the registered tier markers (`slow`, `integration`) exist |
| ruff (`select = E, F, I, UP, B, SIM, RUF`), ty | Lint (the whole repo, notebook code cells included), format, types on `packages/*/src` and `scripts` |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in `.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any agent can read and follow them), and two read-only review checklists in `.claude/agents/`:

| Job | Recipe |
|---|---|
| Add or update a guide page (MyST prose) | `add-guide-page` |
| Add or update an example notebook | `add-notebook` |
| Add a physics term or a term model | `add-term` |
| Add a forcing, a spatial or temporal basis | `add-forcing-basis` |
| Add a simulation monitor | `add-monitor` |
| Adapt the models to a new DA library | `add-da-adapter` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `sesmax-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `correctness-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `make gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-packages --all-groups + pre-commit hooks
make test                 # every test + doctests, no coverage (fast)
make test-cov             # the same with coverage and its gate (what CI runs)
make lint                 # ruff check .   (entire repo)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check packages/*/src scripts (what CI runs)
make capabilities         # regenerate docs/api/capabilities.md
make docs                 # build both docs halves into public/ and check every link
make docs-check           # strict validation of both halves, no themed render
make docs-api             # MkDocs API reference only (mkdocs build --strict)
make docs-serve           # build, then serve public/ at :8000
make precommit            # every pre-commit hook on every file
make build                # a wheel + sdist per package into dist/
```

`addopts` turns coverage on, and a partial run may not reach the coverage gate,
so pass `--no-cov` when running part of the suite:

```bash
uv run pytest --no-cov tests/test_workspace.py -v
```

### Test tiers

The markers are registered in `pyproject.toml` (`strict_markers = true`):

- **Unmarked (fast):** unit tests and doctests, each well under a second.
- **`@pytest.mark.slow`:** a test over ~1 s (a large sweep, a long fit). For
  a parametrised test, mark only the expensive cases
  (`pytest.param(..., marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows, optional
  dependencies, external tools.

Every test whose name starts with `test_grad` is tagged `slow` automatically
(`packages/sesmax-core/tests/conftest.py`, as in somax). CI ("Tests",
`ci.yml`) runs the fast tier with coverage on Python 3.12 and 3.13
(`uv run pytest -v -m "not slow"`); `full-tests.yml` runs every tier on
demand and weekly. Select the fast tier locally with
`uv run pytest -m "not slow and not integration"`.

### Tests that assert on random draws

Models, ensembles and DA tests draw random numbers:

- **Incidental randomness** (any draw would do): seed it
  (`random.Random(0)`, `numpy.random.default_rng(0)`, a fixed key) so a
  failure reproduces.
- **Sampling behaviour under test** (a Monte Carlo estimate, a statistical
  property): bound the statistic by its own sampling distribution (a number
  of standard errors, a test's p-value floor), not by a tolerance tuned on
  one draw, and say in a comment where the bound comes from.

### Before every commit

All of these must pass, from the repo root:

1. `make test`: zero failures (doctests included).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes
   `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`.
   Never lint a subdirectory: `ruff check packages/sesmax-core/` misses errors CI
   catches.
3. `uv run --group lint ruff format --check .`
4. `make typecheck` (`ty check packages/*/src scripts`, as CI runs it).
5. After changing a public API: `make capabilities`, and the `docs/api`
   page.
6. After changing docs, docstrings, `mkdocs.yml` or `docs/myst.yml`:
   `make docs` (or at least `make docs-api` when only docstrings changed).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; if something is
   unclear, stop and ask; push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change;
   remove only what your change made unused, and mention (don't delete)
   unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a
   reproduced bug, a closed form to match) and loop until it passes; for
   multi-step work, state the plan with a check per step.

Also: Python 3.12+, `from __future__ import annotations` in every module,
type hints on every public function, `X | None` over `Optional`, specific
exceptions with `raise … from …`, Google-style docstrings, pure functions
with side effects isolated and explicit. When you suggest a change, propose
the code; when you fix a bug or add a feature, add the test that checks it.

## Git, commits and pull requests

- **Never** push to or merge into `main` unless explicitly told to ("push
  to main", "merge to main"). Work on a feature branch and commit locally;
  never run `git push` unless asked. "Merge the branch" means push the
  feature branch, not merge into `main`. Confirm before any action that
  affects a shared branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(core): add …`); CI validates PR titles
  (`conventional-commits.yml`). Types: `feat`, `fix`, `docs`, `style`,
  `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Breaking
  changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please (`release-please-config.json`), one
  component per package with tags such as `sesmax-core-v0.1.0`: it bumps
  each package's version and writes its `packages/<dist>/CHANGELOG.md` from
  the `feat`, `fix`, `perf` and `revert` commits; don't bump versions by
  hand. `sesmax-core` carries `"release-as": "0.1.0"` for its first release;
  remove it once that release is cut. The umbrella `sesmax` joins
  release-please with `sesmax-ocean` in phase C2.
- **Never replace or remove an existing PR title or description.** An agent
  called for a small follow-up tends to write a fresh description scoped to
  its own work and silently discard the rest: read the existing description
  first and only append checklist items or update their status. Change the
  title only to fix a Conventional Commits violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the
  label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

The docs are built by **two tools** and deployed as one site
(`docs/README.md` has the rationale); `docs.yml` builds them on every PR and
push to `main`, and `pages.yml` deploys `public/` to GitHub Pages from
`main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, example notebooks | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/` (toc in `docs/myst.yml`) | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` (`docs_dir` in `mkdocs.yml`) | `/reference/` |

`make docs` runs `scripts/build_docs.py`: it builds the API with
`mkdocs build --strict`, serves `site/` on port 8910 so mystmd can read its
`objects.inv` (mystmd loads inventories only over http), runs
`myst build --html --strict`, rewrites the localhost API URLs to
`/reference/`, repairs anchors lowercased by mystmd's `$`-expansion bug, and
fails if any internal link in the assembled `public/` site does not resolve.
mystmd is a Node CLI (`npm install -g mystmd`), not a uv dependency; if its
theme download is blocked, `docs/README.md` says how to build against a local
clone of the theme.

- **Prose is MyST Markdown**, not MkDocs-Material syntax: `:::{note}`,
  `:::{tab-set}`, `:::{dropdown}` directives, not `!!!` / `===` / `???`.
- **Link to the API** with the `xref:` protocol and the **top-level**
  exported name: ``[`Name`](xref:api#sesmax_core.Name)``, not
  `sesmax_core.module.Name` (mystmd lowercases `$`-abbreviated anchors;
  see `docs/README.md`). A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename, with `_` turned into `-`
  (`guide/quickstart.md` → `/quickstart/`,
  `notebooks/smoothing_deep_dive.ipynb` → `/smoothing-deep-dive/`), so keep
  basenames unique across `docs/`, `docs/guide/` and `docs/notebooks/`;
  frontmatter `slug:` is ignored.
- **Site-nav URLs stay absolute** (`site.nav` in `docs/myst.yml`):
  the theme prefixes root-relative ones with `BASE_URL` a second time
  (`tests/test_build_docs.py` checks it).
- **API pages** (`docs/api/*.md`) are MkDocs pages: prose plus `:::` blocks,
  `!!!` admonitions, listed in the `nav` of `mkdocs.yml`; docstrings are
  Google style (`docstring_style: google`), maths through MathJax
  (`pymdownx.arithmatex`).
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb`
  files, authored as jupytext percent `.py` and deleted after conversion;
  mystmd renders the stored outputs without re-executing, so each must be
  listed in the `docs/myst.yml` toc. Figures render inline with
  `plt.show()`: no `savefig`, no committed PNGs. `notebooks/` at the root
  holds scratch notebooks, outside the docs. Full standards:
  `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues. Draft a whole wave of issues from
`docs/templates/wave-backlog.md` (see `docs/contributing.md`).
