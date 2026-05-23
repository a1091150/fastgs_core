# Extra VR Work Notes

## Question

For a VR stereo camera setup, can visible Gaussian compaction reuse work between
the left and right eye when the two camera views are very close?

## Short Answer

Yes, but it is not a single shared render. The left and right eye still need
separate projection, tile binning, sorting, and rasterization because their
view/projection matrices and final images differ. The useful shared work is the
visibility/compaction step: find a conservative Gaussian subset that is visible
to either eye, then render each eye from that compact subset.

## Useful Existing References

- gsplat has a close conceptual match through packed multi-camera projection.
  Its projection utilities can output flattened visible Gaussian data such as
  camera ids, Gaussian ids, and CSR-style pointers when `packed=True`.
- gsplat tile intersection also supports tighter bounds when conics and
  opacities are provided. This enables AccuTile/SNUGBOX-style conservative
  ellipse intersection instead of a looser axis-aligned bounding box.
- gsplat also exposes segmented sort options for tile intersections, which is
  relevant because sorting is one of the larger costs in the current Swift
  renderer.
- VRSplat is a VR-specific 3DGS research direction. It focuses on VR issues such
  as popping, stereo-disrupting floaters, high field of view, HMD resolution,
  foveated rendering, and frame rate. It is more of a full VR renderer direction
  than a small compaction patch.

Reference links:

- https://docs.gsplat.studio/main/apis/utils.html
- https://arxiv.org/abs/2505.10144
- https://arxiv.org/abs/2308.04079

## Proposed Stereo Compaction Direction

First implementation should avoid changing training/backward. Treat this as a
preview/rendering optimization.

Preferred first shape:

```text
all gaussian parameters
  -> project/cull for left and right camera as a small camera batch
  -> build packed visible ids with camera ids
  -> per-eye tile intersection, sort, and rasterize
```

More aggressive shape after the first version works:

```text
left visible ids + right visible ids
  -> union visible ids
  -> compact gaussian parameters once
  -> left preprocess/binning/rasterize
  -> right preprocess/binning/rasterize
```

## Why Union Visibility Helps VR

VR left/right cameras usually have a small baseline, so their visible Gaussian
sets are highly overlapping. A conservative union set avoids running later
pipeline stages over completely irrelevant Gaussians while preventing one eye
from losing Gaussians only visible from the other eye.

Expected savings:

- fewer Gaussians passed into later preview stages
- fewer tile intersections
- less sort input
- less rasterization work
- better preview FPS for large scenes

## Important Caveats

- Projection is not fully shared. Left and right eye projected `xy`, depth,
  conics, tile coverage, and sorted tile ranges are still eye-specific.
- Training should not use this shortcut at first. Backward, optimizer state,
  densify, prune, and other Gaussian-count-changing logic make subset
  compaction risky during optimization.
- A first Swift/Metal version should be conservative. Missing visible Gaussians
  in VR is more noticeable than drawing a few extra Gaussians.
- If the scene is already mostly visible to both eyes, union compaction may not
  save much. The bigger wins may come from gsplat-style conic/opacity-aware tile
  intersection, segmented sort, and foveated rendering.

## Suggested Future Tasks

- Add a tiny stereo recorded-camera fixture or synthetic two-camera fixture.
- Prototype `FastGSVisibleGaussianCompaction` for preview-only use.
- Add `FastGSTrainableParameters.take(indices:)` or an equivalent compact
  parameter view/copy helper.
- Benchmark single-eye baseline, two independent eye renders, and shared-union
  compact stereo render.
- Investigate gsplat-style conic/opacity-aware tile intersection before or
  alongside full stereo compaction.
- Keep VRSplat as a later research direction for foveated rendering and
  stereo-specific artifact reduction.
