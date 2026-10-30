# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the module map,
"reuse before you write", the contracts, the tests that enforce them,
commands, the pre-commit checklist, the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- A uv workspace: one package per directory under `packages/` (`sesmax-core` in tier 0, components such as `sesmax-ocean` in tier 1, a coupler in tier 2, and the code-free umbrella `sesmax`). A package imports only packages below its tier (`tests/test_workspace.py`). Search [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a helper.
- The plan, the phases and what moves from somax are in [`docs/design/workspace_and_core.md`](../docs/design/workspace_and_core.md); code copied from somax keeps its behaviour and changes only imports and the listed renames.
- Model steps are pure functions of state, parameters and forcing that `jit`, `grad` and `vmap` go through; optional dependencies sit behind an extra and raise an `ImportError` naming it.

The rules for every project:

- Every docstring `Examples:` block runs (`--doctest-modules`): the expected
  output must be what the code prints.
- Before committing, from the repo root: `make test`,
  `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`,
  `make typecheck`; `make capabilities` after a public API change. Lint the
  whole repo, never a subdirectory.
- When you suggest a change, propose the code; when you fix a bug, add the
  test that reproduces it. Don't nitpick what ruff and the formatter handle.
- Partial test runs need `--no-cov` (the coverage gate fails a subset).
- Never push or merge to `main` unless told to; never replace an existing PR
  description, only append to it.
- Step-by-step recipes (adopt the template, add a guide page or notebook;
  pre-PR check; review) are plain Markdown
  in `.claude/skills/<name>/SKILL.md`, and the two review checklists in
  `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
