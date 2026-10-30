# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. Read "Reuse before you write", "The contracts" and "What enforces
them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the
report format. Most defects worth finding are a contract broken in a way the
tests cannot see (a step that fails under `jit` or `grad` on one path, a
gradient that is NaN where the value is fine, a package importing above its
tier, a docstring that promises what the code does not do) or a
re-implementation of something the workspace or the stack already has, not
style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order, `__all__` ordering, doctest output, the coverage gate); review what
they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against
  [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private
  stack table in "Reuse before you write" in `AGENTS.md`. A re-implemented
  stencil, solver, stepper, basis or pytree helper is a **High** finding,
  with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind
  an extra and an in-function import that names the extra. `uv.lock` is
  updated with any dependency change.
- Imports keep pointing down the tiers (the package table in `AGENTS.md`,
  checked by `tests/test_workspace.py`); two components never import each
  other.
- A rename or removal of a public name keeps the old name working with a
  `DeprecationWarning` for at least one release.

### 2. The contracts

- **Migrated code** from somax changes only its imports and the renames in
  `docs/design/workspace_and_core.md`, and the commit message names the
  somax commit it came from; a behaviour change hidden in a move is a
  **High** finding.
- **Model steps** are pure functions of state, parameters and forcing that
  `jit`, `grad` and `vmap` go through: no Python control flow on traced
  values, no in-place updates, no host round-trips (`float(x)`, `.item()`).
- **Optional dependencies** are imported inside the function or module that
  needs them and raise an `ImportError` naming the extra; the package's
  top-level import never pulls one in.

### 3. Public API and documentation

- New names: in the module's `__all__` and the top-level `__all__`; a new
  module in `SUBMODULES` (`tests/test_public_api.py`), with its
  `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in
  `docs/api/index.md`; `docs/api/capabilities.md` regenerated
  (`make capabilities`).
- Docstrings: Google style; the formula in `$…$`; `Args:`, `Returns:`,
  `Raises:` matching the code; an executable `Examples:` block whose output
  is what the code prints (print rounded values, not 17-digit floats).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the
  top-level name (`xref:api#sesmax_core.<Name>`); a new page or notebook in
  the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number
  quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against a closed form, a hand computation or a
  published value (a conserved quantity, a linear wave's dispersion
  relation, a manufactured solution); a gradient is checked against finite
  differences where it matters.
- New code is covered (the coverage gate is global and starts low, so look
  at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the
  statistic's own sampling distribution, with the bound's source in a
  comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for
  end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public
  function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `dataclasses` with `slots=True` for data containers (`frozen=True` when
  immutable); `Enum` / `StrEnum` for fixed sets of options; `pathlib`;
  f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over
  deep nesting.
- A dependency, when one is justified, follows the preferences in
  `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not
  joined into places they could escape.

---

## Package-Specific Checks

These are the defects JAX model code is prone to. Every example here runs (with JAX installed).

### Branch with `jnp.where`, not Python `if`, on traced values

```python
import jax
import jax.numpy as jnp


# ❌ Python control flow on a traced array: fails under jit with a
#    TracerBoolConversionError
def limiter_bad(x):
    if x > 0.0:
        return x
    return 0.0


# ✅ Both branches traced, the choice made elementwise
def limiter(x):
    return jnp.where(x > 0.0, x, 0.0)


assert float(jax.jit(limiter)(jnp.asarray(-1.0))) == 0.0
```

### A gradient that is NaN where the value is fine

```python
import jax
import jax.numpy as jnp


# ❌ |u| = sqrt(u²) has value 0 at u = 0, but its gradient is 0/0 = NaN,
#    and the NaN poisons every gradient through a 4D-Var window
def speed_bad(u):
    return jnp.sqrt(u**2)


# ✅ The double-where: keep the sqrt's argument away from zero on both paths
def speed(u):
    safe = jnp.where(u == 0.0, 1.0, u**2)
    return jnp.where(u == 0.0, 0.0, jnp.sqrt(safe))


assert jnp.isnan(jax.grad(speed_bad)(0.0))
assert float(jax.grad(speed)(0.0)) == 0.0
```

### Functional updates, never in-place ones

```python
import jax.numpy as jnp


h = jnp.zeros(4)
# ❌ h[0] = 1.0 raises a TypeError: JAX arrays are immutable
# ✅ A new array, which jit turns into an in-place update when it can
h_new = h.at[0].set(1.0)
assert float(h[0]) == 0.0
assert float(h_new[0]) == 1.0
```

### Parameters are leaves, shapes are static

```python
import equinox as eqx
import jax
import jax.numpy as jnp


# ❌ A differentiable parameter marked static: its gradient silently vanishes
class DragBad(eqx.Module):
    r: float = eqx.field(static=True)

    def __call__(self, u):
        return -self.r * u


# ✅ The parameter is a leaf, so filter_grad sees it
class Drag(eqx.Module):
    r: jax.Array

    def __call__(self, u):
        return -self.r * u


def loss(model, u):
    return jnp.sum(model(u) ** 2)


u = jnp.ones(3)
grads = eqx.filter_grad(loss)(Drag(jnp.asarray(0.5)), u)
assert float(grads.r) == 3.0  # d/dr Σ (r u)² = 2 r Σ u² = 2 · 0.5 · 3
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
