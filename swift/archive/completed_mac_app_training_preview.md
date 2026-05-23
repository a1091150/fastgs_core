# Completed Mac App Training And Preview Tasks

- [x] Keep the first real runner Mac-App oriented rather than adding a
  command-line executable. A CLI runner can remain a later convenience idea.
- [x] Start with fixed-point training only: no densify, prune, opacity reset,
  or optimizer-state migration until the rendered previews and debug logs look
  trustworthy.
- [x] Represent runner configuration as Swift structs so the macOS app can own
  and mutate training parameters cleanly.
  - `FastGSRecordedTrainingRunConfig` owns total steps, cache limit, learning
    rates, and the current recorded reference set.
  - `FastGSRecordedTrainingReferenceSet` scans per-camera
    `recorded_manifest.json` files and falls back to the root manifest when
    per-camera references have not been generated yet.
- [x] Add a first macOS app training button that runs one 200-step fixed-point
  training pass and displays the completed target/render result in the app.
- [x] Generate and consume a full-point 512x512 recorded reference from
  `/private/tmp/fastgs_recorded_reference_full_512`.
- [x] Show current training progress as `Step current / total` in the macOS app
  toolbar.
- [x] Add previous/next camera buttons in the macOS app toolbar.
  - [x] Switch by sorted scanner frame-pair offset instead of treating the UI
    camera index as a raw file frame index.
  - [x] Load the selected frame target preview immediately when switching, and
    clear the stale render.
- [x] Add `make swift-recorded-full-512-camera-references` to generate
  per-camera 512x512 full-point manifests.
- [x] Clean up the macOS app so it only exposes training-related controls.
  - [x] Remove static fixture forward, mock camera frame, and early reload/test
    forward buttons from the primary app UI.
- [x] Add macOS app training settings controls.
  - dataset directory picker, defaulting to
    `/Users/yangdunfu/Downloads/2026_05_04_16_51_29`
  - output directory picker, defaulting to
    `/private/tmp/fastgs_swift_mac_training`
  - editable training width and height
  - editable `maxFrames` loader setting
  - editable training step count
  - settings live in a SwiftUI sheet opened from the toolbar.
- [x] Add an explicit `Load` button and keep navigation/training disabled until
  Load succeeds.
- [x] Run one initial forward render during Load so the right-hand Swift Render
  pane shows the current untrained Gaussian render before pressing Train.
- [x] Make camera switching run a Swift render for the newly selected scanner
  frame.
- [x] Preserve trained Gaussian parameters after the Mac App training run.
- [x] Add a single-entry MLX runtime gate in the macOS app.
- [x] Add `FastGSRenderPreviewScheduler` for training-time preview render
  requests.
- [x] Add the first simple multi-view training loop. The training runner can
  accept multiple recorded scenes and cycles through them with
  `(step - 1) % sceneCount`.
- [x] Change the Mac App default `maxFrames` to `9999`.
- [x] Allow camera switching during training as a queued preview request.
- [x] Add a manual performance report test for camera switching:
  `FASTGS_RUN_PERF_REPORT=1 swift test --filter FastGSScannerDatasetLoaderTests/testFixedScannerCameraSwitchPerformanceReport`.
- [x] Add first GPU presentation fast path:
  `FastGSOutColorTextureRenderer` calls `MLXArray.asMTLBuffer(noCopy: true)`
  for contiguous `[3, width * height]` float32 `outColor`, then dispatches a
  small Metal compute kernel to write RGBA8 into an app-owned `MTLTexture`.
- [x] Wire macOS Load/camera-switch preview to prefer the GPU texture path.
