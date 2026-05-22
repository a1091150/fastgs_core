CONDA_ENV ?= fastgs_core
BUILD_DIR ?= build
XCODE_BUILD_DIR ?= build_xcode
CONFIG ?= Release
CONDA_BASE := $(shell conda info --base 2>/dev/null)
CLASS ?=
FORCE ?= 0
SWIFT_RECORDED_DATASET ?= /Users/yangdunfu/Downloads/2026_05_04_16_51_29
SWIFT_RECORDED_REF_DIR ?= /private/tmp/fastgs_recorded_reference
SWIFT_RECORDED_LARGE_REF_DIR ?= /private/tmp/fastgs_recorded_reference_16384
SWIFT_XCODE_DERIVED_DATA ?= /private/tmp/fastgs_swift_xcode_derived
SWIFT_PREPROCESS_BACKWARD_REF ?= /private/tmp/fastgs_preprocess_backward_ref.json
SWIFT_PREPROCESS_BACKWARD_DERIVED_DATA ?= /private/tmp/fastgs_swift_xcode_derived_preprocess_backward_parity

.PHONY: help env-check gen-primitive cmake-configure pyext-build test-build test-run xcode-configure xcode-build pip-install pip-develop pip-wheel swift-recorded-reference swift-recorded-xcode-test swift-recorded-compare test-swift-recorded-forward swift-preprocess-backward-reference swift-preprocess-backward-parity-xcode test-swift-preprocess-backward-parity swift-recorded-training-smoke-xcode test-swift-recorded-training-smoke swift-recorded-training-loop-xcode test-swift-recorded-training-loop train-scanner-fixed train-scanner-fastgs train-scanner-fastgs2 train-scanner-fastgs2-base train-scanner-fastgs2-smoke train-scanner-fastgs-no-prune train-scanner-fastgs-smoke train-scanner-fastgs-bbox clean

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
	@printf "  make test-swift-recorded-forward  Regenerate recorded Swift refs, run Xcode tests, compare stage summaries.\n"
	@printf "  make test-swift-preprocess-backward-parity  Generate preprocess backward refs and run slow Xcode parity tests.\n"
	@printf "  make test-swift-recorded-training-smoke  Generate recorded Swift refs and run one-step training smoke.\n"
	@printf "  make test-swift-recorded-training-loop  Generate recorded Swift refs and run a 3-step training loop smoke.\n"
	@printf "  make train-scanner-fixed Run scripts/train_scanner_fixed.py with the active conda python.\n"
	@printf "  make train-scanner-fastgs Run scripts/train_scanner_fastgs.py with FastGS-style densify/prune.\n"
	@printf "  make train-scanner-fastgs2 Run self-contained scanner FastGS2 training.\n"
	@printf "  make train-scanner-fastgs2-base Run scanner FastGS2 with FastGS train_base-style 30k params.\n"
	@printf "  make train-scanner-fastgs2-smoke Short smoke run for train_scanner_fastgs2.py.\n"
	@printf "  make train-scanner-fastgs-no-prune Temporary FastGS run that never removes Gaussians.\n"
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

swift-recorded-reference:
	conda run -n $(CONDA_ENV) python swift/FastGSSwiftTools/generate_recorded_reference.py --dataset-dir $(SWIFT_RECORDED_DATASET) --max-points 4096 --out-dir $(SWIFT_RECORDED_REF_DIR)
	conda run -n $(CONDA_ENV) python swift/FastGSSwiftTools/generate_recorded_reference.py --dataset-dir $(SWIFT_RECORDED_DATASET) --max-points 16384 --out-dir $(SWIFT_RECORDED_LARGE_REF_DIR)

swift-recorded-xcode-test:
	cd swift/FastGSSwiftApps && xcodebuild test -project FastGSSwift.xcodeproj -scheme FastGSSwiftMac -destination 'platform=macOS' -derivedDataPath $(SWIFT_XCODE_DERIVED_DATA)

swift-recorded-compare:
	python3 swift/FastGSSwiftTools/compare_recorded_stage_summary.py --manifest $(SWIFT_RECORDED_REF_DIR)/recorded_manifest.json --swift-summary $(SWIFT_RECORDED_REF_DIR)/recorded_swift_stage_summary.json
	python3 swift/FastGSSwiftTools/compare_recorded_stage_summary.py --manifest $(SWIFT_RECORDED_LARGE_REF_DIR)/recorded_manifest.json --swift-summary $(SWIFT_RECORDED_LARGE_REF_DIR)/recorded_swift_stage_summary.json

test-swift-recorded-forward: swift-recorded-reference swift-recorded-xcode-test swift-recorded-compare

swift-preprocess-backward-reference:
	conda run -n $(CONDA_ENV) python swift/FastGSSwiftTools/fastgs_preprocess_backward_ref.py --out $(SWIFT_PREPROCESS_BACKWARD_REF)

swift-preprocess-backward-parity-xcode:
	cd swift/FastGSSwiftApps && FASTGS_RUN_SLOW_PARITY=1 xcodebuild test -quiet -project FastGSSwift.xcodeproj -scheme FastGSSwiftMac -destination 'platform=macOS' -derivedDataPath $(SWIFT_PREPROCESS_BACKWARD_DERIVED_DATA) -test-timeouts-enabled NO -only-testing:FastGSSwiftXcodeTests/FastGSSmokeXcodeTests/testPreprocessBackwardMatchesReferenceSummaryUnderXcode -only-testing:FastGSSwiftXcodeTests/FastGSSmokeXcodeTests/testPreprocessBackwardSHDegree3MatchesReferenceSummaryUnderXcode

test-swift-preprocess-backward-parity: swift-preprocess-backward-reference swift-preprocess-backward-parity-xcode

swift-recorded-training-smoke-xcode:
	cd swift/FastGSSwiftApps && xcodebuild test -quiet -project FastGSSwift.xcodeproj -scheme FastGSSwiftMac -destination 'platform=macOS' -derivedDataPath $(SWIFT_XCODE_DERIVED_DATA) -test-timeouts-enabled NO -only-testing:FastGSSwiftXcodeTests/FastGSSmokeXcodeTests/testRecordedSmallTrainingStepUpdatesParametersUnderXcode

test-swift-recorded-training-smoke: swift-recorded-reference swift-recorded-training-smoke-xcode

swift-recorded-training-loop-xcode:
	cd swift/FastGSSwiftApps && xcodebuild test -quiet -project FastGSSwift.xcodeproj -scheme FastGSSwiftMac -destination 'platform=macOS' -derivedDataPath $(SWIFT_XCODE_DERIVED_DATA) -test-timeouts-enabled NO -only-testing:FastGSSwiftXcodeTests/FastGSSmokeXcodeTests/testRecordedSmallTrainingLoopReducesSyntheticLossUnderXcode

test-swift-recorded-training-loop: swift-recorded-reference swift-recorded-training-loop-xcode

train-scanner-fixed:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fixed.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14'

test-scanner:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/test_gaussian_render.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14 --eval-index 0 --render-all'

train-scanner-fixed-bbox:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fixed.py --data /Users/yangdunfu/Downloads/2026_03_01_16_36_14 --extra-points-ratio 0.5 --extra-points-mode bbox'

train-scanner-fastgs:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs.py --data /Users/yangdunfu/Downloads/2026_05_04_16_51_29' --final-prune-min-opacity 0.03 --final-prune-score-thresh 0.95 --final-prune-min-gaussians 128

train-scanner-fastgs2:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs2.py --data /Users/yangdunfu/Downloads/2026_05_04_16_51_29 --final-prune-min-opacity 0.03 --final-prune-score-thresh 0.95 --final-prune-min-gaussians 128'

train-scanner-fastgs2-base:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs2.py --data /Users/yangdunfu/Downloads/2026_05_04_16_51_29 --steps 30000 --save-every 30000 --log-every 10 --lr-means 0.00016 --lr-colors 0.0025 --lr-opacity 0.025 --lr-scales 0.005 --lr-rotations 0.001 --densify-from-step 500 --densify-until-step 15000 --densification-interval 500 --opacity-reset-interval 3000 --grad-thresh 0.0002 --grad-abs-thresh 0.0012 --dense 0.001 --loss-thresh 0.1 --final-prune-min-opacity 0.1 --final-prune-start 15000 --final-prune-end 30000 --final-prune-interval 3000 --final-prune-score-thresh 0.9 --final-prune-min-gaussians 64'

train-scanner-fastgs2-smoke:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs2.py --data /Users/yangdunfu/Downloads/2026_05_04_16_51_29 --steps 200 --save-every 100 --log-every 10 --densify-from-step 50 --densification-interval 50 --densify-until-step 200 --final-prune-start 100000 --final-prune-end 100000'

train-scanner-fastgs-no-prune:
	/bin/zsh -lc 'source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV) && python scripts/train_scanner_fastgs.py --data /Users/yangdunfu/Downloads/2026_05_04_16_51_29 --final-prune-min-opacity 0.03 --final-prune-score-thresh 0.95 --final-prune-min-gaussians 128 --no-prune-gaussians --reset-optimizer'

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
