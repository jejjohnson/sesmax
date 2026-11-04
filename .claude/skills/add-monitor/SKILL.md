---
name: add-monitor
description: Add a simulation monitor to sesmax — a BaseMonitor subclass with its lifecycle hooks and MonitorVerdict, whether it joins default_monitors, the public exports, the API page and the tests. Use when asked to add run-time observability, a chunk-boundary check or a termination rule.
---

# Add a monitor

Read "Monitors" under "The contracts" in `AGENTS.md` first. A monitor runs host-side between jitted chunks. If the check must stop the run at the offending step, it is a guard inside the tendency (`guard_finite`, `guard_positive`, `guard_ceiling`), not a monitor.

Nothing in the core calls monitors yet: the run loop that does arrives with `sesmax_core.sim` (#28). Until then a monitor is exercised directly in its tests.

## 1. Where it goes

`packages/sesmax-core/src/sesmax_core/_src/monitor/builtins.py` for one every model can use. Anything ocean-specific goes in `sesmax-ocean`, once it exists (phase C2).

## 2. Write it

Subclass `BaseMonitor`, set `name` and override only the hooks you need.

```python
import numpy as np

from sesmax_core.monitor import BaseMonitor, ChunkInfo, MonitorVerdict


class MaxValueMonitor(BaseMonitor):
    """Stop the run when a state field exceeds a ceiling."""

    name = "max_value"

    def __init__(self, ceiling: float) -> None:
        self.ceiling = ceiling

    def on_chunk_end(self, model, state, info: ChunkInfo) -> MonitorVerdict:
        peak = float(np.abs(np.asarray(state.x)).max())
        if peak > self.ceiling:
            return MonitorVerdict(
                terminate=True, reason=f"max |x| = {peak:.3g} > {self.ceiling:.3g}"
            )
        return MonitorVerdict(metrics={"max_abs_x": peak})
```

- `terminate=True` always carries a `reason`.
- `metrics` hold scalars, and `messages` hold lines for the run log.
- `ChunkInfo.stats` may be empty, because not every solver exposes its stats.
- A monitor reads a constrained parameter through `paramax.unwrap(model)`.

## 3. Export and document it

Add the name to `__all__` in `_src/monitor/__init__.py` and in `sesmax_core/monitor.py`. `docs/api/monitor.md` renders it from there; if it is a built-in, update the row in `docs/api/index.md`, and add it to `default_monitors()` only if every run should have it. Then run `make capabilities`.

## 4. Test it

Tests join `packages/sesmax-core/tests/test_monitor.py`, using its lightweight fakes rather than an integration, so they stay in the fast tier. Check the quiet case, the triggering case with its `reason`, and each hook you overrode.

```bash
uv run pytest --no-cov packages/sesmax-core/tests/test_monitor.py -v
```

Finish with `pre-pr-check`.
