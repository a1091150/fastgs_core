CONDA_ENV ?= fastgs_core
BUILD_DIR ?= build
XCODE_BUILD_DIR ?= build_xcode
CONFIG ?= Release
CONDA_BASE := $(shell conda info --base 2>/dev/null)
CLASS ?=
FORCE ?= 0

.PHONY: help env-check gen-primitive cmake-configure pyext-build test-build test-run xcode-configure xcode-build pip-install pip-develop pip-wheel train-scanner-fixed train-scanner-fastgs train-scanner-fastgs-smoke train-scanner-fastgs-bbox clean

help:
	@printf "Targets:\n"
	@printf "  make env-check        Print python/cmake paths and mlx/nanobind versions.\n"
	@printf "  make gen-primitive CLASS=Foo [FORCE=1]  Generate primitive .h/.cpp/.metal files.\n"
	@printf "  make cmake-configure   Configure Ninja build for Python extension.\n"
	@printf "  make pyext-build       Build _fastgs_core extension.\n"
	@printf "  make test-build        Build C++ dummy test target.\n"
	@printf "  make test-run          Run C++ dummy test target.\n"
	@printf "  make xcode-configure   Generate Xcode project at repo root.\n"
	@printf "  make xcode-build       Build _fastgs_core with Xcode generator.\n"
	@printf "  make pip-install       pip install . --no-build-isolation\n"
	@printf "  make pip-develop       pip install -e . --no-build-isolation\n"
	@printf "  make pip-wheel         Build wheel/sdist via python -m build.\n"
	@printf "  make train-scanner-fixed Run scripts/train_scanner_fixed.py with the active conda python.\n"
	@printf "  make train-scanner-fastgs Run scripts/train_scanner_fastgs.py with FastGS-style densify/prune.\n"
	@printf "  make train-scanner-fastgs-smoke Short smoke run for train_scanner_fastgs.py.\n"
	@printf "  make train-scanner-fastgs-bbox FastGS training with bbox extra-point seeding.\n"
	@printf "  make clean             Remove root build folders and dist artifacts.\n"

env-check:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && \
	echo "CONDA_ENV=$(CONDA_ENV)" && \
	echo "python=$$(which python)" && \
	echo "cmake=$$(which cmake)" && \
	python -c "import importlib.metadata as md, sys; print(\"python_version=\"+sys.version.split()[0]); print(\"mlx=\"+md.version(\"mlx\")); print(\"nanobind=\"+md.version(\"nanobind\"))"'

gen-primitive:
	@if [ -z "$(CLASS)" ]; then \
		echo "Usage: make gen-primitive CLASS=FastGSPreprocess [FORCE=1]"; \
		exit 1; \
	fi
	@if [ "$(FORCE)" = "1" ]; then \
		python3 scripts/mlx_cxx_primitive_generate.py "$(CLASS)" --force; \
	else \
		python3 scripts/mlx_cxx_primitive_generate.py "$(CLASS)"; \
	fi

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

train-scanner-fixed:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fixed.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14'

test-scanner:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/test_gaussian_render.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14 --eval-index 0 --render-all'

train-scanner-fixed-bbox:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fixed.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14 --extra-points-ratio 0.5 --extra-points-mode bbox'

train-scanner-fastgs:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14' --final-prune-min-opacity 0.03 --final-prune-score-thresh 0.95 --final-prune-min-gaussians 128

train-scanner-fastgs-smoke:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14 --steps 200 --save-every 100 --log-every 10 --max-frames 24 --densify-from-step 50 --densification-interval 50 --densify-until-step 200 --final-prune-start 100000 --final-prune-end 100000'

train-scanner-fastgs-bbox:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14 --extra-points-ratio 0.5 --extra-points-mode bbox'

train-scanner-fastgs-densify:
	python scripts/train_scanner_fastgs.py \
		--data /path/to/your_scanner_dataset \
		--steps 6000 \
		--log-every 20 \
		--save-every 200 \
		--opacity-reset-interval 3000 \
		--opacity-reset-value 0.82 \
		--opacity-cap-after-densify 0.82 \
		--densify-from-step 500 \
		--densify-until-step 6000 \
		--densification-interval 500 \
		--importance-score-threshold 2.0 \
		--grad-thresh 1e-4 \
		--grad-abs-thresh 6e-4 \
		--split-factor 2 \
		--min-opacity 0.005 \
		--max-screen-size 20.0 \
		--max-world-scale-factor 0.1 \
		--data /Users/yangdunfu/Downloads/2026_03_01_16_36_14

train-scanner-fastgs-densify2:
	python scripts/train_scanner_fastgs.py \
		--data /path/to/your_scanner_dataset \
		--steps 6000 \
		--log-every 20 \
		--save-every 500 \
		--densify-from-step 500 \
		--densify-until-step 6000 \
		--densification-interval 500 \
		--importance-score-threshold 1.0 \
		--grad-thresh 5e-5 \
		--grad-abs-thresh 3e-4 \
		--max-screen-size 0 \
		--opacity-reset-value 0.82 \
		--opacity-cap-after-densify 0.82 \
		--data /Users/yangdunfu/Downloads/2026_03_01_16_36_14

train-scanner-fastgs-densify3:
	python scripts/train_scanner_fastgs.py \
		--data /path/to/your_scanner_dataset \
		--steps 6000 \
		--log-every 20 \
		--save-every 500 \
		--densify-from-step 500 \
		--densify-until-step 6000 \
		--densification-interval 500 \
		--importance-score-threshold 0.5 \
		--grad-thresh 5e-6 \
		--grad-abs-thresh 5e-5 \
		--dense 0.02 \
		--max-screen-size 0 \
		--opacity-reset-value 0.82 \
		--opacity-cap-after-densify 0.82 \
		--data /Users/yangdunfu/Downloads/2026_03_01_16_36_14



clean:
	rm -rf $(BUILD_DIR) $(XCODE_BUILD_DIR) dist *.egg-info python_package/*.egg-info
