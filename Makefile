CONDA_ENV ?= fastgs_core
BUILD_DIR ?= build
XCODE_BUILD_DIR ?= build_xcode
CONFIG ?= Release
CONDA_BASE := $(shell conda info --base 2>/dev/null)

.PHONY: help cmake-configure pyext-build test-build test-run xcode-configure xcode-build pip-install pip-develop pip-wheel clean

help:
	@printf "Targets:\n"
	@printf "  make cmake-configure   Configure Ninja build for Python extension.\n"
	@printf "  make pyext-build       Build _fastgs_core extension.\n"
	@printf "  make test-build        Build C++ dummy test target.\n"
	@printf "  make test-run          Run C++ dummy test target.\n"
	@printf "  make xcode-configure   Generate Xcode project at repo root.\n"
	@printf "  make xcode-build       Build _fastgs_core with Xcode generator.\n"
	@printf "  make pip-install       pip install . --no-build-isolation\n"
	@printf "  make pip-develop       pip install -e . --no-build-isolation\n"
	@printf "  make pip-wheel         Build wheel/sdist via python -m build.\n"
	@printf "  make clean             Remove root build folders and dist artifacts.\n"

cmake-configure:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && cmake -S . -B $(BUILD_DIR) -G Ninja -DPython_EXECUTABLE="$$(which python)" -DFASTGS_BUILD_PYTHON=ON -DFASTGS_BUILD_TEST=ON -DFASTGS_BUILD_METAL=ON'

pyext-build: cmake-configure
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && cmake --build $(BUILD_DIR) --config $(CONFIG) --target _fastgs_core'

test-build: cmake-configure
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && cmake --build $(BUILD_DIR) --config $(CONFIG) --target fastgs_core_dummy_test'

test-run: test-build
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && $(BUILD_DIR)/fastgs_core_dummy_test'

xcode-configure:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && cmake -S . -B $(XCODE_BUILD_DIR) -G Xcode -DCMAKE_CXX_COMPILER="$$(xcrun --find clang++)" -DPython_EXECUTABLE="$$(which python)" -DFASTGS_BUILD_PYTHON=ON -DFASTGS_BUILD_TEST=ON -DFASTGS_BUILD_METAL=ON'

xcode-build: xcode-configure
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && cmake --build $(XCODE_BUILD_DIR) --config $(CONFIG) --target _fastgs_core'

pip-install:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && pip install . --no-build-isolation'

pip-develop:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && pip install -e . --no-build-isolation'

pip-wheel:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python -m build'

clean:
	rm -rf $(BUILD_DIR) $(XCODE_BUILD_DIR) dist *.egg-info python_package/*.egg-info
