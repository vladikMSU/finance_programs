# Repository Guidelines

## Project Structure & Module Organization
This workspace is split by workflow: exploratory notebooks in `EDA/`, reinforcement-learning experiments in `MPT/`, reusable utilities in `MPT2/`, and statistical summaries in `STATISTICS/`. Pure Python modules such as `EDA/process_indices.py`, `MPT2/helpers.py`, and `STATISTICS/analysis.py` should hold logic that notebooks call into so charts remain reproducible. Data inputs live under `data/` (`daily/`, `monthly/`) while temporary or vendor-specific pulls stay in `data downloads/` or `artifacts/`.

## Build, Test & Development Commands
Create an isolated toolchain before editing: `python -m venv .venv && source .venv/bin/activate`, then install `numpy pandas matplotlib seaborn statsmodels scipy python-dateutil`. Launch notebooks with `jupyter lab EDA/explore_ru_industries.ipynb MPT2/00_BASE.ipynb` to preserve kernels and relative paths, and run scripts headlessly via `python STATISTICS/analysis.py` or `python EDA/process_indices.py` when validating pipelines.

## Coding Style & Naming Conventions
Code targets modern Python (3.10+) with PEP 8 spacing, 4-space indentation, and type hints mirroring `STATISTICS/analysis.py`. Prefer snake_case for functions/variables, PascalCase only for dataclasses such as `AssetConfig`, and keep modules short with focused helpers. Route plotting styles through shared helpers like `MPT2.helpers.set_style` so notebooks and scripts render consistently, and reserve inline comments for domain-specific math (e.g., `calendar_days_per_year`).

## Testing Guidelines
Unit-test deterministic helpers (parsers, resamplers, catalog builders) with `pytest`; place new suites under `tests/` (create it if needed) and mirror the module path, e.g., `tests/test_helpers.py` covering `build_assets_catalog`. Use fixtures to mock CSV fragments instead of touching the large files in `data/`. For notebooks, execute `jupyter nbconvert --execute MPT2/01_DATA_PROC.ipynb` before opening a pull request to ensure paths resolve and charts refresh, and treat failing numeric assertions as merge blockers.

## Commit & Pull Request Guidelines
Follow the existing short, imperative style seen in `Improve data processing notebook presentation`: a single-line summary (<60 chars) plus optional body when context is non-obvious. Scope commits per feature or fix (one dataset import, one helper tweak) to simplify reviews. PRs should explain the objective, note which data folders changed, attach key plots if visuals update, and list the verification commands that were re-run; also flag any large files that must stay untracked so reviewers can double-check storage decisions.

## Data & Security Notes
Many CSVs contain licensed or proprietary series; do not commit fresh vendor exports—store them under `data downloads/` locally and document them separately. Scrub notebooks of embedded credentials, clear outputs that show raw account data, and only share sanitized samples under `data/sample/` with comments describing the anonymization.
