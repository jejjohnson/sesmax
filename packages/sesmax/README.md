<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/logo-dark.svg">
  <img alt="sesmax" src="../../docs/assets/logo-light.svg" width="300">
</picture>

**Simple Earth-system models in JAX, in one install.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📖 [Documentation](https://jejjohnson.github.io/sesmax/) ·
📚 [API reference](https://jejjohnson.github.io/sesmax/reference/) ·
🧭 [Design](https://jejjohnson.github.io/sesmax/workspace-and-core/)

</div>

---

This package holds no code. It installs the packages of the [sesmax workspace](https://github.com/jejjohnson/sesmax), which are imported by their own names.

| Package | Import | Holds | Status |
|---|---|---|---|
| [`sesmax-core`](../sesmax-core/README.md) | `sesmax_core` | states, terms, forcing, rescaling, rollouts and assimilation adapters | released |
| [`sesmax-experiment`](../sesmax-experiment/README.md) | `sesmax_experiment` | runs in chunks, monitors and checkpoints | moving from somax |
| [`sesmax-ocean`](../sesmax-ocean/README.md) | `sesmax_ocean` | shallow-water, QG and spherical ocean models | moving from somax |

## 📥 Install

The package is not on PyPI, and a plain `pip install` from git cannot see the packages in the workspace beside it. So install it from a clone, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jejjohnson/sesmax.git
cd sesmax
uv sync --package sesmax
```

To use the core alone in another project, install [`sesmax-core`](../sesmax-core/README.md) directly.

## 📄 License

MIT. See [LICENSE](LICENSE).
