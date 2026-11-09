# Texture tiling: Python port of MATLAB `findTile`

Python port of SightTex's MATLAB `findTile` (2022), used by `fabsim package`.
Generated with an LLM and reviewed by a human; not yet compared numerically with MATLAB.

## Signature and semantics

`find_tile(images, window_size, shift, interp_dist, area=None, guidance_images=None)`
The package uses `find_tile_detailed(..., joint_search=False)`, which also returns the seam paths and the
resolved parameters.

* `images`: one array or a list, same height and width, any channel count. They are stacked along the
  channel axis, so one seam path and one `area` serve every image (a joint search). MATLAB expects
  floats in [0, 1], `uint8` and `uint16` inputs are divided by 255 and 65535 and returned in the input
  dtype (rounded and clipped). No gamma or colour-space change.
* Computation uses `float32`, in-place crop blending and batches of at most 48 planes. A build
  with 72 + 72 states and 1000-pixel atlases peaked at about 2 GB per orientation.
* `window_size`: scalar or `[vertical, horizontal]`. `interp_dist`: scalar or `[vertical, horizontal]`
  (default 0). A MATLAB quirk is kept: `findTile` takes the height of the top and bottom windows from
  `window_size(2)` and `idv = interp_dist(1)`.
* `shift`: empty, a scalar, or `[left right top down]`. Two values (`[20 20]`) are invalid.
* `area`: empty (whole image), a scalar (centred square), `[h w]` (centred) or `[top left h w]` with
  **0-based** `top` and `left`.
* `guidance_images`: image **arrays** (not indices) of the same size. The search
  (`automaticCoordinatesForMaxTile`, `findPaths`) uses them instead of `images`, the blend still runs
  on `images`. In `fabsim package` they are given as state indices (loading states first, then release)
  and mapped to arrays.
* Output `area`: `[top, left, height, width]`, **1-based**, MATLAB convention. The returned tile is one
  row and one column smaller than this area. Tile pixel (0, 0) is atlas pixel `(left - 1, top - 1)`.

**Differences from MATLAB**

* If a window is larger than the selected tile (the top/bottom or left/right windows overlap), MATLAB's
  index ranges become empty and it silently builds a meaningless tile. The port raises `ValueError`
  ("window_size too large for the selected tile ...").
* `joint_search=True` (`--tiling-joint-search`) is an addition. The greedy MATLAB search fixes the
  bottom edge from the top and bottom windows alone and then moves the right edge, so the left/right
  seam cannot influence the vertical period. The joint variant scores every (bottom, right) pair in the
  range with the mean top/bottom cost plus the mean left/right cost and takes the best pair. Slower,
  off by default.
* `cut_tile` applies seams found earlier to any other layer (see below). It is the part of `createTile`
  after `findPaths`.

## In the pipeline

Find one area and seam pair across all loading/release colour atlases, then apply the same cuts
to albedo and normals. `apply_tiling(layer, seams, normals=True)` renormalises blended normals.
The [albedo scale](photometric_stereo.md#albedo-scale) includes blended values.

`Seams.save` writes `tiling_seams.npz` for recutting layers later; the viewer does not read it.
See the [archive format](formats.md#tiling_seamsnpz).

## Limits

Blending cannot fix a vertical phase offset between ribs. Use the tiling scripts in
`experiments/` to compare candidate areas.
