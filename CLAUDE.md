# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/api/capabilities.md` (every public name in
  the package) before writing a helper; the "Reuse before you write" table in
  `AGENTS.md` maps the usual hand-rolled code to what already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - finishing the template adoption in phase C1: `adopt-template`;
  - building: `add-guide-page`, `add-notebook` (phase C1 adds the core's
    extension points: `add-term`, `add-forcing-basis`, `add-monitor`,
    `add-da-adapter`);
  - shipping: `pre-pr-check`, `sesmax-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `sesmax-review`; run them on any diff that adds code, before
  committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/api/capabilities.md` or a library of the stack?
  - `correctness-reviewer`: numerics — jit / grad / vmap safety, leaves
    and static fields, precision, NaN-safe gradients, physics and units,
    doctest accuracy, test tolerances.
- **Partial test runs** need `--no-cov` (`addopts` turns coverage on and its
  gate can fail a subset).
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
