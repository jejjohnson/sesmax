---
name: sesmax-review
description: Review a change or pull request in this repository against its own rules — CODE_REVIEW.md, the contracts in AGENTS.md, the workspace tiers, reuse of what the packages and the stack already provide, and numerical correctness (jit / grad / vmap safety, precision, gradients, physics, doctest accuracy). Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which modules, tests and docs it touches.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: a re-written
   stencil, solver, stepper, basis or pytree helper is the main way a
   package drifts from itself and from the stack.
3. **Correctness** — run the `correctness-reviewer` subagent in parallel
   with step 2: jit / grad / vmap safety, leaves and static fields,
   precision, gradients, physics and units, doctest accuracy, test
   tolerances.
4. **Contracts** — the workspace tiers (a package imports only below
   itself; `tests/test_workspace.py`), optional dependencies behind their
   extra with an `ImportError` that names it, and the rules under "The
   contracts" in `AGENTS.md`.
5. **Checklist** — the rest of `CODE_REVIEW.md` (public API, API page and
   `docs/api/index.md` row, capability index, MyST prose, tests and tiers),
   skipping what ruff, ty or the tests already enforce.
6. **Verify claims** — for anything you flag as a bug, run it
   (`uv run python -c "..."`) and report what it printed.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
