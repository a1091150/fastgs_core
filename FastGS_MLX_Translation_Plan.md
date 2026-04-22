# FastGS To MLX Translation Plan

## Goal
Translate the behavior of FastGS training from the PyTorch reference into an MLX-based scanner trainer.

The intent is not only:

- add densification
- add pruning

but rather:

- preserve FastGS training loop semantics
- preserve Gaussian state transitions
- preserve optimizer cadence and topology-mutation behavior
- preserve the multi-view scoring logic that drives densify / prune

Primary references:

- [submodules/FastGS/train.py](/Users/yangdunfu/Documents/cxxPractice/fastgs_core2/submodules/FastGS/train.py)
- [submodules/FastGS/scene/gaussian_model.py](/Users/yangdunfu/Documents/cxxPractice/fastgs_core2/submodules/FastGS/scene/gaussian_model.py)
- [submodules/FastGS/train_base.sh](/Users/yangdunfu/Documents/cxxPractice/fastgs_core2/submodules/FastGS/train_base.sh)
- [submodules/fastgs_core_old/training/optimizer_policy.py](/Users/yangdunfu/Documents/cxxPractice/fastgs_core2/submodules/fastgs_core_old/training/optimizer_policy.py)

## What FastGS Actually Does
FastGS training is structured as:

1. Update learning rate.
2. Increase SH degree every 1000 iterations.
3. Sample one random train camera.
4. Render current Gaussians.
5. Compute photometric loss.
6. Backpropagate.
7. In a no-grad section:
   - update densification statistics from `viewspace_points.grad`
   - periodically compute multi-view Gaussian scores
   - densify and prune using those scores
   - reset opacity on schedule
   - after 15k iterations, periodically run final aggressive prune
   - apply optimizer step on a cadence schedule

Important point:
FastGS is not “gradient densification only”. It is a two-stage decision:

- gradient-based candidate selection
- multi-view-consistency-based filtering and pruning

## Canonical FastGS State
The reference `GaussianModel` keeps more than trainable parameters.

Trainable tensors:

- `_xyz`
- `_features_dc`
- `_features_rest`
- `_scaling`
- `_rotation`
- `_opacity`

Persistent training state:

- `active_sh_degree`
- `max_radii2D`
- `xyz_gradient_accum`
- `xyz_gradient_accum_abs`
- `denom`
- `tmp_radii`
- optimizer state
- SH optimizer state
- `spatial_lr_scale`
- `percent_dense`

This means a faithful MLX translation should not treat densification as an external post-process. It must be part of model state management.

## FastGS Densification Logic
Reference path:

- `train.py` updates `max_radii2D` and `add_densification_stats(...)`
- `gaussian_model.py::densify_and_prune_fastgs(...)`

The algorithm is:

1. Accumulate `viewspace_points.grad[:, :2]` norm into `xyz_gradient_accum`.
2. Accumulate `viewspace_points.grad[:, 2:]` norm into `xyz_gradient_accum_abs`.
3. Divide both by `denom` to get average statistics.
4. Form candidate masks:
   - clone candidates: small Gaussians with large standard gradient
   - split candidates: large Gaussians with large absolute gradient
5. Use `importance_score > 5` as the multi-view consistency gate.
6. Clone qualifying small Gaussians.
7. Split qualifying large Gaussians into `N=2`.
8. Prune Gaussians with:
   - low opacity
   - too-large screen-space size
   - too-large world-space scale
9. Apply a budgeted pruning sample biased by pruning score.
10. Cap opacity after densify/prune.

Important translation detail:
the split/clone decision is not purely geometric. It depends on both:

- gradient statistics
- multi-view importance score

## FastGS Final Pruning Logic
Reference path:

- `train.py` runs this every 3000 iterations after 15000 and before 30000
- `gaussian_model.py::final_prune_fastgs(...)`

Behavior:

- prune if `opacity < 0.1`
- prune if `pruning_score > 0.9`

This is intentionally aggressive.

Important note:
the original code has no minimum-Gaussian safeguard. So if the score distribution changes in the MLX port, it is completely possible to prune everything.

## FastGS Multi-View Score Logic
Reference path:

- `utils/fast_utils.py`

The score pass does this:

1. Sample 10 cameras.
2. Render each view.
3. Compute photometric loss and a normalized L1 error map.
4. Threshold the normalized map with `loss_thresh`.
5. Re-render with a per-pixel metric map enabled.
6. Collect per-Gaussian `accum_metric_counts`.
7. Accumulate:
   - view counts for densification importance
   - weighted photometric score for pruning

Outputs:

- `importance_score`
- `pruning_score`

Translation implication:
this score computation is a first-class part of FastGS, not an optional heuristic.

## Optimizer Behavior In FastGS
Reference path:

- `gaussian_model.py::training_setup`
- `gaussian_model.py::optimizer_step`

FastGS uses two optimizers:

- main optimizer for `xyz`, `f_dc`, `opacity`, `scaling`, `rotation`
- SH optimizer for `f_rest`

Cadence:

- `<= 15000`: main every step, SH every 16 steps
- `<= 20000`: both every 32 steps
- `> 20000`: both every 64 steps

Also important:
when tensors are replaced, appended, or pruned, FastGS mutates optimizer state to match the new topology.

This is done through:

- `replace_tensor_to_optimizer`
- `_prune_optimizer`
- `cat_tensors_to_optimizer`

This is the hardest part of the MLX translation.

## What The Old MLX Optimizer Policy Already Solves
Reference:

- [submodules/fastgs_core_old/training/optimizer_policy.py](/Users/yangdunfu/Documents/cxxPractice/fastgs_core2/submodules/fastgs_core_old/training/optimizer_policy.py)

Useful pieces already implemented:

- split main optimizer and SH optimizer
- xyz exponential LR scheduler
- FastGS-style optimizer cadence
- state resize helpers for prune / append
- state replacement helper for tensor reset
- optimizer moment reset utilities

This old file is not the full training translation, but it is a strong base for the optimizer/topology part.

Best reuse candidates:

- `get_expon_lr_func`
- `FastGSOptimizerPolicy`
- `prune_states`
- `append_states`
- `replace_state`
- cadence logic in `apply_gradients`

## What Must Change In The Current MLX Port
To become a true “FastGS train.py translated to MLX”, the current code should be reworked in these areas.

### 1. Training Loop Structure
The MLX trainer should be reorganized so its control flow mirrors `train.py`.

Desired loop:

1. `update_learning_rate(iteration)`
2. `oneupSHdegree()` every 1000
3. sample random camera from a recyclable camera pool
4. render once for main training pass
5. compute FastGS loss
6. backpropagate
7. inside host-side update block:
   - update radii/stat buffers
   - densify/prune on interval
   - reset opacity on interval
   - final prune on interval
   - apply optimizer cadence

### 2. State Ownership
The scanner trainer should own a single coherent model/state object, not:

- MLX model in one place
- NumPy densification buffers in another
- optimizer state rebuilt ad hoc

Suggested MLX state object:

- trainable MLX tensors
- non-trainable NumPy or MLX bookkeeping buffers
- optimizer policy
- SH degree
- topology mutation helpers

### 3. Topology Mutation API
Implement MLX analogues for:

- `reset_opacity`
- `add_densification_stats`
- `densify_and_clone_fastgs`
- `densify_and_split_fastgs`
- `densify_and_prune_fastgs`
- `final_prune_fastgs`
- `prune_points`
- `densification_postfix`

These should be named and scoped similarly to the reference code so line-by-line comparison stays easy.

### 4. Optimizer State Preservation
Avoid rebuilding MLX optimizers blindly after densify/prune.

Instead:

- prune optimizer moments when points are removed
- append zero moments when points are cloned/split
- replace moments when opacity logits are reset

The old `optimizer_policy.py` provides the right direction here.

### 5. Photometric Score Fidelity
The MLX port should align score computation with FastGS more closely.

That means:

- preserve normalized L1 error map thresholding
- preserve the rerender with `metric_map`
- preserve weighted accumulation of `metric_count`
- keep the same score normalization semantics

If SSIM is unavailable or too expensive in MLX for the score pass, document that deviation clearly.

### 6. Final Prune Policy
There are two reasonable modes:

- faithful mode: exactly match FastGS, even if pruning can go to zero
- safe mode: add a configurable minimum Gaussian floor for debugging and migration

During migration, safe mode is practical.
For paper-faithful experiments, faithful mode should exist.

## Recommended Implementation Order
### Phase 1. Structural Translation
Build a clean MLX trainer skeleton that mirrors `train.py`.

Deliverables:

- MLX FastGS trainer class or stateful module
- matching iteration schedule
- matching camera sampling
- matching SH progression

### Phase 2. Optimizer Translation
Integrate the old optimizer policy ideas.

Deliverables:

- main optimizer + SH optimizer
- xyz LR scheduler
- cadence stepping
- state prune/append/replace support

### Phase 3. Densification Translation
Implement the exact FastGS mutation pipeline.

Deliverables:

- viewspace grad stat accumulation
- clone path
- split path
- densification postfix
- pruning budget logic

### Phase 4. Multi-View Score Translation
Match `compute_gaussian_score_fastgs(...)`.

Deliverables:

- camera sampling pass
- metric map rerender
- `importance_score`
- `pruning_score`

### Phase 5. Final Prune And Debuggability
Add transparent logging around prune behavior.

Deliverables:

- opacity-hit count
- score-hit count
- remove budget
- before/after point counts
- optional safe floor

## Concrete Design Suggestion For This Repo
### Proposed file responsibilities
Keep:

- [scripts/train_scanner_fixed.py](/Users/yangdunfu/Documents/cxxPractice/fastgs_core2/scripts/train_scanner_fixed.py)
as the source of:

- scanner dataset parsing
- camera creation
- image IO
- render helper
- export helpers

Refactor or rewrite:

- [scripts/train_scanner_fastgs.py](/Users/yangdunfu/Documents/cxxPractice/fastgs_core2/scripts/train_scanner_fastgs.py)
into a more faithful FastGS translation.

Suggested internal modules inside that script or future helper files:

- `FastGSScannerState`
- `FastGSOptimizerPolicy`
- `compute_gaussian_score_fastgs_mlx`
- `FastGSDensifier`
- `FastGSTrainer`

## Known Translation Risks
### Score distribution mismatch
Even if the algorithm matches, score thresholds may shift because:

- MLX numerics differ
- the dataset is scanner-based, not original benchmark scenes
- loss implementation may not be identical

### Topology mutation semantics
This is the area most likely to drift from PyTorch if optimizer state is not preserved carefully.

### Small-point-count instability
FastGS thresholds were tuned for larger populations than some scanner runs currently produce.
That makes final pruning much more fragile.

## Bottom Line
The correct interpretation of “port all FastGS strategy” is:

- translate `train.py` behavior
- translate `gaussian_model.py` mutation/state behavior
- translate optimizer cadence and state mutation behavior

not merely:

- borrow a few pruning thresholds

The old MLX optimizer policy is valuable because it already solves much of the optimizer-side translation. The missing work is to connect it to a faithful MLX implementation of the FastGS model-state lifecycle.

