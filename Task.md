# Task 1

## Scope
- Build a minimal dummy baseline for `fastgs_core2` using root-level build entry.
- Keep `fastgs_core/` as code-only directory.
- Use conda env `fastgs_core`.
- Add `scripts/` for utility python or shell scripts.
- Add `python_package/` for pip package build/install artifacts (for example egg-info).
- Add pip package config files for `fastgs_core` (based on scratch reference).

## Completed
- [x] Remove previous docs-based workflow files (`docs/`).
- [x] Create code directory structure under `fastgs_core/`:
  - [x] `fastgs_core/include/`
  - [x] `fastgs_core/metal/`
  - [x] `fastgs_core/binding/`
  - [x] `fastgs_core/test/`
- [x] Create dummy source files:
  - [x] `fastgs_core/include/dummy.h`
  - [x] `fastgs_core/dummy.cpp`
  - [x] `fastgs_core/metal/dummy.metal`
  - [x] `fastgs_core/binding/binding.cpp`
  - [x] `fastgs_core/test/test.cpp`
- [x] Move build entry to repository root:
  - [x] `CMakeLists.txt` at `fastgs_core2/`
  - [x] `Makefile` at `fastgs_core2/`
- [x] Ensure build outputs are generated in root build folders:
  - [x] `build/`
  - [x] `build_xcode/` (configured by target)
- [x] Remove old subdirectory build entry to avoid confusion:
  - [x] Delete `fastgs_core/CMakeLists.txt`
  - [x] Clean old `fastgs_core/build*` outputs
- [x] Create `scripts/` directory for python/sh helpers.
- [x] Create and update `scripts/dummy.py` to call package function `fastgs_core.dummy_add`.
- [x] Create `python_package/` directory for package artifacts.
- [x] Create pip package files:
  - [x] `setup.py`
  - [x] `pyproject.toml`
  - [x] `python_package/fastgs_core/__init__.py`
- [x] Update `Makefile` with package build/install targets:
  - [x] `pip-install`
  - [x] `pip-develop`
  - [x] `pip-wheel`

## Validation Completed
- [x] `make cmake-configure`
- [x] `make pyext-build`
- [x] `make test-run`
- [x] Python smoke test:
  - [x] `import _fastgs_core`
  - [x] `_fastgs_core.dummy_add(1, 2) == 3`

## Notes
- This baseline is intentionally dummy-only for build/link/runtime validation.
- Next step is to replace dummy API with the first FastGS forward primitive skeleton.
