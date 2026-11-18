# Fix import regressions on `main` and reconcile with `msite_to_assembly`

Status: steps 0-6 done 2026-11-18 (option (a)); step 7 (downstream bumps) pending, after push. Drafted 2026-11-12.

## Outcome (2026-11-18)

| Step | Commit | Notes |
|---|---|---|
| lint on `msite_to_assembly` | `d53d21a` | pending ruff fixes, 22 errors → 0 |
| 1-3 on `fix/import-regressions` (from `main`) | `97bcf69` | tests: 13/14 import tests failed on `main` before the fix |
| 5 merge into `msite_to_assembly` | `c7aba45` | conflicts in `waterflow.py`, `fitting.py`, `examples/flow_params.py` |
| 6 CI | `bed072d` | `.github/workflows/tests.yml`, extras `dev` + `fitting` |

Changes from the plan:
- **#2:** `PumpSpeedFit`/`FlowPressureFit` are imported from `fitting` under `TYPE_CHECKING`, with quoted
  annotations, the form the lint commit already used on `msite_to_assembly`, rather than a direct import.
  This avoids a conflict at the merge.
- **Warning-scope test:** it records the `filterwarnings`/`simplefilter` calls made from `python_magnetcooling`
  modules. Diffing `warnings.filters` does not work: importing numpy alone adds bare `ignore` entries, and
  `warnings` moves an existing identical filter to the front instead of adding it.
- **Merge:** `fitting.py` keeps `WaterFlow` under `TYPE_CHECKING`, since both runtime uses import it locally.
  `main` also brings `cooling.py` `uguess` 0 → 1.0 (clamped at 1e-3) and `thermohydraulics.py` using
  `dataclasses.replace` instead of mutating its inputs.

Result on `msite_to_assembly`: `ruff check .` clean; pytest 129 passed, 1 skipped, 1 failed. The failure is
`test_fit_hydraulic_system_piecewise`: `fit_hydraulic_system` formats `'N/A'` with `:.4f` when the piecewise pump
fit has no `fit_result`. It only runs once `pwlf` is installed, which is why it was skipped before. It is a
separate bug, fixed with the fitting changes.

## Symptoms

On `main` (`adfa1c4`), `import python_magnetcooling` fails, so every entry point and the test suite fail too:

```
NameError: name 'warnings' is not defined            # waterflow.py:23
ImportError: cannot import name 'FlowPressureFit'
             from 'python_magnetcooling.waterflow'   # waterflow_factory.py:12
```

`pytest` on `main` stops at collection with 4 errors. On `origin/msite_to_assembly` (`53f2ea0`) the same suite
gives 120 passed and 4 skipped. The physics is unaffected: `heatexchanger_primary.estimate_temperature_elevation`
and `mixing_temp` give bit-identical results on both branches (checked on the M9 Tout analysis in magnet-scipy).

## Root causes

| # | File | Introduced by | What happened |
|---|------|---------------|---------------|
| 1 | `waterflow.py` | merge `b2c4932` (main → `claude/update-code-review-PESmk`), released by PR #7 (`adfa1c4`) | `c3dfec9` replaced `from warnings import simplefilter` + `simplefilter("ignore")` with `import warnings` + a pint-scoped `warnings.filterwarnings(...)`. The merge kept the new call but the old import. |
| 2 | `waterflow_factory.py` | `a693b90` "fix ruff linting" | `PumpSpeedFit` / `FlowPressureFit` were used only in annotations of `from_fits()`. To satisfy ruff F821 they were imported from `waterflow`, but they are defined in `fitting.py` (l. 111, 192). |
| 3 | `waterflow.py` on `msite_to_assembly` | branch predates `c3dfec9` | Still calls `simplefilter("ignore")` at import time. This silences **every** warning in the importing process, including `DeprecationWarning` from downstream packages (e.g. magnet-scipy's camelCase deprecation shims). |
| 4 | repo | n/a | No CI. `tests/test_import.py` already catches #1 and #2, but nothing runs it on push or PR. |

## Downstream state (submodule pins, checked 2026-11-12)

| Repo | Pinned commit | Importable | Global ignore (#3) |
|---|---|---|---|
| python_magnetrun | `53f2ea0` (msite_to_assembly) | yes | yes |
| magnetdb-duckdb-demo | `53f2ea0` (msite_to_assembly) | yes | yes |
| 2026-m1-hifimagnet | `53f2ea0` (msite_to_assembly) | yes | yes |
| python_magnetapi | `a94c4f8` (main, 2026-03-03) | yes | yes |
| python_magnetworkflows | `a1194d3` (main, 2026-02-25) | yes | yes |

No downstream repo is pinned to the broken `main`. All of them carry #3.

## Plan

### 0. Branch
`fix/import-regressions` from `main`.

### 1. Tests first → verify: they fail on `main`
Extend `tests/test_import.py`:
- import, in a fresh interpreter (`subprocess`, `sys.executable -c ...`), each of `python_magnetcooling`, `waterflow`,
  `waterflow_factory`, `fitting`, `heatexchanger_primary`, `thermohydraulics`. A fresh interpreter avoids a
  cached module from a previous test hiding the failure.
- **warning-scope test**: in a fresh interpreter, snapshot `warnings.filters`, run `import python_magnetcooling`, and
  diff the lists. Every added entry must have a module pattern matching `pint`. No entry may be a bare
  `("ignore", None, Warning, None, 0)`. This test fails on `msite_to_assembly` (#3) and passes after step 5.

### 2. Fix #1: `waterflow.py` → verify: tests from step 1 pass the import part
Replace `from warnings import simplefilter` with `import warnings`. Keep the pint-only filter.

### 3. Fix #2: `waterflow_factory.py` → verify: `from_fits` tests in `test_fitting.py` pass
```python
from python_magnetcooling.fitting import FlowPressureFit, PumpSpeedFit
from python_magnetcooling.waterflow import WaterFlow
```
There is no import cycle: `fitting` imports `waterflow` at module level and imports `waterflow_factory` only lazily.
If a cycle appears later, switch to `if TYPE_CHECKING:` + `from __future__ import annotations`.

### 4. Lint + full suite → verify: `ruff check .` clean, `pytest` green
`pytest` needs `pytest-cov` because `addopts` has `--cov`. Use the `dev` extra, or `-o addopts=""` locally.

### 5. Reconcile with `msite_to_assembly` → verify: the suite passes on the merged result
`git merge-tree` shows a single conflict, in `python_magnetcooling/waterflow.py`.
Resolve it as follows:
- keep `import warnings` + the pint-scoped `filterwarnings` (main);
- keep the hysteresis import position (either side, equivalent);
- drop `simplefilter("ignore")`.

All other files auto-merge, including the `msite` → `assembly` rename in `fitting.py`, `feelpp.py` and the examples.

**Decision needed:** which branch becomes the integration target?
- (a) Merge the fixed `main` into `msite_to_assembly`, then PR `msite_to_assembly` → `main`. Recommended: three
  downstream repos already track `msite_to_assembly`, and `main` gains the rename.
- (b) Keep `msite_to_assembly` separate and cherry-pick only the `waterflow.py` warning fix onto it.

### 6. CI → verify: a deliberately broken import fails the workflow
Add `.github/workflows/tests.yml`: Python 3.11 and 3.12, `pip install -e ".[dev]"`, `ruff check .`, `pytest`.
Both regressions in this plan would have been caught by the existing `test_import.py`.

### 7. Downstream bumps → verify per repo: `python -c "import python_magnetcooling"` + that repo's tests
After the merge, bump the submodule in python_magnetrun, magnetdb-duckdb-demo and 2026-m1-hifimagnet
(currently `53f2ea0`), then in python_magnetapi and python_magnetworkflows (older `main`).
**Risk:** removing the global ignore makes warnings from other libraries visible again. A downstream
`pytest -W error` or `filterwarnings = error` setting could start failing. Check each repo's pytest config before bumping.

## Out of scope
- API changes from the `msite` → `assembly` rename (already on the branch).
- Physics or numerical changes. `heatexchanger_primary` differs between branches only by a commented-out `heatBalance`.
