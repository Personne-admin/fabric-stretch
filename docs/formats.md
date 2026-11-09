# File formats

Run files and package fields. See [Pipeline](pipeline.md) for the build procedure.

Contents: [project and run files](#project-and-run-files) | [per-run `ps/`](#per-run-ps-products) |
[material package v3](#material-package-v3) | [build report](#build_reportjson) | [tiling seams](#tiling_seamsnpz) |
[v2 packages](#version-2-packages-and-how-the-viewer-treats-them) | [what the viewer requires](#what-the-viewer-requires)

Conventions: JSON files are UTF-8. Numbers in pixels are image pixels (x right, y down) unless a
field says otherwise. `NNNN` is a `frame_id` padded to four digits.

## Project and run files

```
<project>/<orientation>_<loading|release>/
  images/ frames.json force.json session.json tracks.json  ps/ (derived)
```

Run names must be exactly `<orientation>_loading` or `<orientation>_release`, with a
`frames.json` inside. The orientation may be any string, such as `warp`, `weft` or `bias`.
Suffixed names such as `warp_loading_2` are ignored and reported by validation and packaging.
Packaging needs both branches for an orientation.

Image filenames come from frame entries. Tracking uses image pixels (y down); normals use the
[photometric frame](photometric_stereo.md#coordinate-conventions).

### `frames.json`

```json
{
  "sample_id": "fabric_002", "test_id": "warp_stretch_20261020_215722",
  "direction": "warp", "branch": "loading", "motion_mode": "loading",
  "clamp_width_m": 0.202,
  "commanded_motor_steps_per_increment": 8000,
  "requested_displacement_um_per_increment": 1224.49,
  "screw_pitch_um_per_output_rev": 1500.0, "gear_ratio": 49.0,
  "motor_full_steps_per_rev": 200, "microsteps": 1,
  "motor_steps_per_output_rev": 9800, "linear_um_per_motor_step": 0.153061,
  "frames": [
    {"frame_id": 0, "step_index": 0, "displacement_mm": 0.0, "force_json_index": 0,
     "settle_ms": 50, "is_relaxation_hold": false,
     "image_diffuse": "img_0000_diffuse.png",
     "image_ps_1": "img_0000_ps_1.png", "image_ps_2": "img_0000_ps_2.png", "image_ps_3": "img_0000_ps_3.png"}
  ]
}
```

| Field | Used by the pipeline |
| --- | --- |
| `frames` (required) | list in chronological order |
| `frames[].frame_id` | identity of the frame; falls back to the list position if absent. Names the state files (`frame_NNNN.png`) and joins tracks and `ps/` entries |
| `frames[].image_diffuse` | file in `images/`; a frame without it is not usable |
| `frames[].image_ps_1..3` | the three single-LED captures, in LED-azimuth order; `has_ps` is true when all three are listed. The pipeline does not use the optional `image_ps_b_*` polarisation set |
| `frames[].force_json_index` | the `index` of the matching `force.json` reading |
| `frames[].displacement_mm` | copied to the package state as `travel_mm` |
| `clamp_width_m` | clamp width in metres; falls back to `session.json` |
| other fields | motor mechanics and bookkeeping; not read by the packager. Older files may also have `force_csv_row` (alias of `force_json_index`) |

### `force.json`

```json
{"forces": [
  {"index": 0, "label": "0", "timestamp": "...", "raw": "TENSION force_n=-4.300000 raw_counts=-284 zero_counts=-628 calibration_n_per_count=0.01250000",
   "value": -4.3, "force_n": -4.3, "raw_counts": -284, "zero_counts": -628,
   "calibration_n_per_count": 0.0125, "status": "ok"}
]}
```

`force_n = (zero_counts - raw_counts) * calibration_n_per_count`. The controller decodes
`raw_counts` as a signed 16-bit value from the first 16 bits of a 24-bit frame. A reading is **valid** when
`status == "ok"` and `force_n` is a finite number. It is addressed by `index` (not list position);
retried or aborted readings leave gaps. If a valid index occurs twice, the Python pipeline
(`dataset.load_force`) keeps the last one and the Rust crates (`fabric-data`) the first; real captures
do not repeat an index.
`status` is `"timeout"` for a reading that never arrived and `"invalid"` for an unusable one.
`zero_counts` and `calibration_n_per_count` are used for the tare when the two runs of an orientation
have different zeros.

### `session.json`

Acquisition settings written by the rig (`schema_version` 2; older files have none). The pipeline reads:

| Field | Use |
| --- | --- |
| `led_config` | snapshot of the LED configuration (below). Decides the photometric light directions |
| `clamp_width_m` | fallback for `frames.json` |
| everything else | informational: sample and test id, `direction`, `branch`, step size and mechanics, `cam_exposure_ms`, `led_power`, `settle_ms`, `max_force_n`, `color_calibration` stamp, `capture_plan`, `status`, `abort_reason`, `frames_written` |

`led_config` (example from a capture, not the current packaged defaults):

```json
{"pol_a_bits": [1, 3, 5], "pol_b_bits": [2, 4, 6], "diffuse_mask": 1,
 "light_elevation_deg": 45.0, "light_azimuths_deg": [91.91, 331.45, 208.24]}
```

`light_azimuths_deg[i]` belongs to `image_ps_{i+1}` (set A). Angles use the
[photometric coordinate conventions](photometric_stereo.md#coordinate-conventions). Either the
long names (`light_azimuths_deg`,
`light_elevation_deg`) or `azimuths_deg` / `elevation_deg` are accepted, plus an optional `gains`
(three positive numbers). Anything without three azimuths and an elevation is an error. The same JSON
is accepted by `--led-config`. See [Light configuration](photometric_stereo.md#light-configuration)
for selection precedence and [Lights](hardware.md#lights) for calibration.

### `tracks.json`

Format `fabsim-tracks` version 1, written by `src/fabsim/cowtracker_to_fabsim.py`:

```json
{"format": "fabsim-tracks", "version": 1,
 "provenance": {"tool": "cowtracker_to_fabsim", "model": "CoWTracker (facebookresearch/cowtracker)",
                "repo_commit": "1454f200...", "role": "diffuse", "source_dimensions": [192, 154],
                "reference_frame": 0, "coordinate_convention": "pixel_center",
                "reference_error_limit_px": 2.0, "timestamp": "..."},
 "frames": [{"frame_id": 0, "points": [{"id": "grid_0", "x": 4.3, "y": 4.3125, "confidence": 0.999998}]}]}
```

What `fabsim package` requires: ids `grid_0 ... grid_N-1` per frame without gaps, a rectangular grid
(points of the first row share a `y` within 1e-4), `provenance.reference_frame` (default 0) naming the
loading reference frame. Points with `confidence < 0.5` or outside the image count as not tracked.
Coordinates are image pixels at the resolution of the diffuse images.

## Per-run `ps/` products

Written by `fabsim ps`; never overwrites `ps_output/`.

```
<run>/ps/img_NNNN_albedo.png   8-bit RGB: albedo = value / 255 * albedo_scale (linear)
<run>/ps/img_NNNN_normal.png   16-bit RGB: n = 2 * v / 65535 - 1; x image right, y image up, z toward camera
<run>/ps/ps_manifest.json
```

Both images have the size of the source images. The albedo is *not* multiplied by the exposure gain
(the package's is). Encodings: `v = round(clip(a / scale, 0, 1) * 255)` and
`v = round((n * 0.5 + 0.5) * 65535)` (rounding, never flooring).

`ps_manifest.json` (trimmed, from `examples/demo/warp_loading/ps/`):

```json
{
  "format": "fabsim-ps-frames", "version": 1,
  "light": {"azimuths_deg": [91.91, 331.45, 208.24], "elevation_deg": 45.0,
            "source": "session.json:led_config"},
  "gains": [1.0217, 1.0, 0.9783],
  "gains_mode": "per-frame",
  "albedo_scale": 0.5872,
  "albedo_encoding": "linear RGB 8-bit; albedo = value/255*albedo_scale",
  "normal_encoding": "16-bit RGB; n = 2*v/65535-1; x=image right, y=image up, z=toward camera",
  "quality": {"reference_frame_id": 0, "albedo_diffuse_correlation": 0.916,
              "mean_normal_tilt_deg": 0.53, "exposure_gain": 1.9464, "warnings": []},
  "frames": [{"frame_id": 0, "albedo": "img_0000_albedo.png", "normal": "img_0000_normal.png",
              "gains": [1.02174, 1.0, 0.97826]}],
  "skipped_frames": [{"frame_id": 3, "reason": "cannot read image ..."}]
}
```

| Field | Meaning |
| --- | --- |
| `light.source` | `session.json:led_config`, `file:<name>` or `packaged:led_config.json` |
| `gains`, `frames[].gains` | LED radiances relative to their mean. Top level: the reference frame. Per frame: that frame's own estimate (`gains_mode: per-frame`) or the configured gains (`fixed`) |
| `albedo_scale` | one scale shared by both runs; see [Albedo scale](photometric_stereo.md#albedo-scale) |
| `quality` | measured on the reference frame. `exposure_gain` is informational here |
| `skipped_frames` | present only if frames could not be read; not in `frames` |

## Material package v3

One folder per orientation:

```
<out>/<orientation>/package.json
                    loading/frame_NNNN.png  albedo_NNNN.png  normal_NNNN.png
                    release/frame_NNNN.png  albedo_NNNN.png  normal_NNNN.png
                    tiling_seams.npz                           (tiled package only)
```

* `frame_*.png`: 8-bit RGB, sRGB-encoded colour. Interpolate in linear light.
* `albedo_*.png`: 16-bit RGB, linear: `albedo = v / 65535 * ps.albedo_scale`.
* `normal_*.png`: 16-bit RGB: `n = 2 * v / 65535 - 1`, axes `x` = atlas column (right), `y` = atlas row
  UP (the negative row direction), `z` toward the camera.
* Size: the atlas (`atlas.dimensions_px`) in an untiled package, one tile period
  (`tiling.tile_size_px`) in a tiled one. All layers of all states have the same size.
* Layers exist for every state or for none. With `--no-ps`, or when the photometric solve is not
  possible, there are no `albedo`/`normal` keys and no `ps` block.

### `package.json`

Full example: [demo package](../examples/demo/material_package_v3/warp/package.json).

### Top level

| Key | Type | Meaning |
| --- | --- | --- |
| `format` | string | always `fabsim-material-rgb` |
| `version` | int | `3` |
| `material` | string | project folder name |
| `orientation` | string | `warp`, `weft`, ... |
| `atlas` | object | registration and size, below |
| `lref_mm`, `lref_status` | null, string | reference length of the sample; not measured, so null |
| `stretch_definition` | string | how `stretch_ratio` is defined (text) |
| `release_residual_stretch` | number | stretch at the last (relaxed) release state, measured against the loading reference. Above 1 when the cloth stays stretched. Can come out slightly below 1 (measurement noise; the build warns) |
| `color_space` | string | `sRGB PNG; decode to linear RGB before interpolation` |
| `force` | object | tare and quality, below |
| `ps` | object | present only with photometric layers |
| `branches` | object | `loading` and `release`, each a list of states |
| `tiling` | object | present only in a tiled package |

### `atlas`

| Key | Meaning |
| --- | --- |
| `grid_cells` | `[col, row, cols, rows]`: the common rectangle of tracker cells, in cell units of the loading reference grid |
| `reference_bounds_px` | `[x0, y0, x1, y1]` of that rectangle in the loading reference image |
| `dimensions_px` | `[width, height]` of the atlas |
| `origin` | `Loading reference grid top-left` (the viewer requires exactly this) |
| `loading_reference_frame`, `release_reference_frame` | frame ids of the relaxed reference states |
| `tracker_grid` | `[cols, rows]` of the tracked point grid |
| `active_grid_axis` | `vertical` (pull along atlas rows) or `horizontal` (along columns); the axis that stretches more |
| `loading_to_release_homography` | 3x3 matrix from the registration of the two reference photographs |
| `release_reference_error_p95_px` | registration error of the release grid (the build fails above 2 px) |

### `force`

| Key | Meaning |
| --- | --- |
| `unit` | `N` |
| `tare` | text describing the method |
| `tare_states` | number of loading states the offset was fitted to; `1` = the loading reference reading itself |
| `offset_n` | subtracted from every reading: `force_n = reading - offset_n` |
| `zero_counts` | `{loading, release}` zero counts of the two reference readings; release readings with another zero are shifted by `(loading zero - own zero) * calibration_n_per_count` before the offset is subtracted |
| `clamp_width_m` | clamp width of the loading run (null if unknown); converts N of the strip to N/m |
| `quality.<branch>` | `states`, `span_n` (range of a quadratic fit of force against stretch), `noise_sigma_n` (scatter about that fit), `snr` (span / noise), `spearman` (rank correlation of force and stretch), `usable` |

`usable` is `false` when `snr < 3` or `spearman < 0.5`, and `null` with fewer than 8 states (then only
`states` and `usable` are given). The viewer does not read `quality`; it computes its own fit check.

With `tare_states > 1` the loading reference state shows the fit residual, not exactly 0 N.
The fitting procedure is described in [Package build](pipeline.md#what-it-does-per-orientation).
This quality flag is separate from the viewer's [ForceFit check](press_view.md#force-and-indentation).

### `ps`

| Key | Meaning |
| --- | --- |
| `albedo_scale` | `albedo = v / 65535 * albedo_scale`. One scale for both branches; calculation in [Albedo scale](photometric_stereo.md#albedo-scale) |
| `albedo_scale_basis`, `exposure_basis`, `normal_correction`, `albedo_encoding`, `normal_encoding`, `frame` | text descriptions |
| `exposure_gain` | already applied to the stored albedo; do not apply it again when decoding. See [Exposure gain](photometric_stereo.md#exposure-gain-package-only) for calculation and capture reproduction |
| `light` | `azimuths_deg`, `elevation_deg`, `source` of the LED geometry used |
| `gains` | LED gains of the loading reference frame |
| `gains_mode` | `per-state` (estimated for every state; see `branches.<branch>[].ps_gains`) or `fixed` (configured, or `--ps-gains reference`) |
| `quality` | `albedo_diffuse_correlation`, `normal_tilt_deg` `{mean, max}` over the states, `warnings` |

### `branches.<loading|release>[]` (a state)

| Key | Meaning |
| --- | --- |
| `frame_id` | frame id in its run |
| `sequence_index` | position within the branch in capture order |
| `stretch_ratio` | stretch (1 = loading reference). On the release branch the first state is the most stretched one and the last is `release_residual_stretch` |
| `force_n` | tared tension in N (strip test, whole sheet) |
| `force_raw_n` | the reading as stored in `force.json` |
| `travel_mm` | machine travel of the run so far, from `displacement_mm`; restarts at 0 for the release run. v2 name: `machine_displacement_mm` |
| `texture` | RGB state image, relative to the package folder |
| `source_image` | diffuse file the state came from |
| `albedo`, `normal` | photometric layers (only with `ps`) |
| `ps_gains` | gains used for this state |
| `normal_tilt_deg` | tilt of the state's mean normal before the plane correction |

The lists are in capture order; the viewer sorts them by `stretch_ratio`.

### Dropped and skipped states

States that cannot be built are not in `branches`. The reasons are in [build_report.json](#build_reportjson)
(`skipped_states`: not tracked, no diffuse image, no valid force; `dropped_states`: PS images missing
or truncated, each `{branch, frame_id, reason}`). `package.json` itself carries no list of them. A
dropped state is removed from the **whole** package (colour and photometric layers), because layers
are all-or-nothing.

### `tiling`

| Key | Meaning |
| --- | --- |
| `enabled` | `true` |
| `window_size`, `shift`, `interp_dist` | the `findTile` inputs used |
| `guidance_images_used`, `guidance_state_indices` | whether a subset of states drove the search |
| `area_input_0based` | the `--tiling-area` input, or null |
| `joint_search` | `--tiling-joint-search` |
| `states_in_joint_search` | number of states the tile was found on |
| `source_atlas_size_px` | atlas size `[w, h]` |
| `tile_size_px`, `period_px` | size of one tile period (equal) |
| `area_matlab_1based` | MATLAB-convention output area `[top, left, height, width]`; the tile is one row and column smaller |
| `origin_px` | atlas pixel `[x, y]` (0-based) that is tile pixel (0, 0) |
| `origin_uv`, `period_uv` | origin and period in atlas units (`origin_px / atlas`, `tile_size / atlas`) |
| `quality` | wrap jumps and neighbour differences of the tiles (informational) |
| `search_seconds`, `total_seconds` | timings |

The viewer uses these fields when [mapping the atlas onto the sheet](press_view.md#mapping-the-atlas-onto-the-sheet).

### `tiling_seams.npz`

Compressed NumPy archive next to the tiled `package.json`: `area` (4 ints, 1-based `y1 y2 x1 x2`),
`window`, `interp` (resolved window and interpolation distances), `horizontal` and `vertical` (seam
paths, n x 2, 1-based `[row, col]` inside the working crop), `atlas_size` (`[w, h]` of the atlas the
seams were found on). `fabsim.tiling.Seams.load` reads it and `fabsim.tiling.apply_tiling` cuts any
layer registered to the same atlas with these seams. The viewer does not read it.

## `build_report.json`

Both output roots receive this report. Abbreviated example (comments and ellipses are omitted
from the actual JSON):

```jsonc
{
  "project": "demo", "format": "fabsim-material-rgb", "version": 3,
  "ignored_run_folders": [{"name": "warp_loading_2", "reason": "name is not ..."}],
  "orientations": {
    "warp": {
      "status": "built",
      "atlas": {...},
      "states": {"loading": 8, "release": 8},
      "skipped_states": [],
      "stretch_range": {"loading": [1.0, 1.292], "release": [1.004, 1.295]},
      "release_residual_stretch": 1.00397,
      "force": {"offset_n": -4.3, "tare_states": 1, "zero_counts": {...},
                "zero_counts_differ": false, "range_n": {...}, "quality": {...}},
      "tiling": {...},            // or "tiling_skipped": "<reason>"
      "ps": {"albedo_scale": 1.0667, "exposure_gain": 1.946, "gains": [...], "gains_mode": "per-state",
             "quality": {...}, "clipped_fraction_max": {"untiled": 0.0008, "tiled": 0.0009}},
                                   // or "ps_skipped": "<reason>"
      "dropped_states": [...],    // only when states were left out
      "clamp_width_warning": "loading 0.2 m, release 0.201 m",   // only when they differ
      "warnings": ["..."],
      "seconds": 0.6
    },
    "weft": {"status": "skipped", "reason": "no release run (both loading and release are needed)"}
  }
}
```

The console also prints the report's warnings:

| Warning | Meaning |
| --- | --- |
| `<branch> force is not usable: span ... against noise ...` | the force of this branch is about as large as its noise; do not drive a view or a fit with it |
| `release residual stretch ... is below 1` | the cloth ends shorter than its loading reference (measurement noise) |
| `albedo and diffuse photograph correlate poorly` | the PS capture and the diffuse shot disagree; relighting will look wrong |
| `<branch> frame N left out: photometric-stereo image ...` | see [Damaged images](pipeline.md#damaged-images) |

A `clamp_width_warning` means the two runs recorded different clamp widths. The package uses the
loading run's width for the N-to-N/m conversion hint.

## Version 2 packages and how the viewer treats them

The old `material_package/` and `material_package_untiled/` folders contain v2 packages:

| | v2 | v3 |
| --- | --- | --- |
| `force_n` | raw load-cell reading | tared; raw kept in `force_raw_n` |
| `force` block | at most `clamp_width_m` | tare, zero counts, quality |
| `release_residual_stretch` | absent | present; release stretch is relative to the loading reference |
| `ps`, `albedo`, `normal` | absent | present when photometric stereo was possible |
| travel | `machine_displacement_mm` | `travel_mm` |

Version 1 (separate loading and release atlases) and the old `material_package_tiled/` folder are not
read: the viewer reports `unsupported material package version 1 ... rebuild with fabsim package`.

The viewer reads a v2 package like this:

* The viewer subtracts the first loading state's force from both branches and keeps the raw
  reading in `force_raw_n`. This assumes a shared zero; v2 has no zero-count or quality metadata.
* Relight is disabled. Rebuild with `fabsim package` to add photometric layers.
* `machine_displacement_mm` is read as `travel_mm`.
* The stretch is used as written. Check `stretch_definition` in the file for how the release branch
  was referenced.

## What the viewer requires

Use [`fabsim validate-package`](pipeline.md#validate-a-package) to check packages without
opening the viewer. It also decodes images and checks their bit depths; the viewer checks headers.

`MaterialPackage::load` accepts a `package.json` when:

* `format == "fabsim-material-rgb"` and `version` is 2 or 3 (anything else gives a specific message);
* `atlas.dimensions_px` has no zero, `atlas.grid_cells` has non-zero `cols`/`rows`, and
  `atlas.origin == "Loading reference grid top-left"`;
* `branches.loading` and `branches.release` exist; every state has a finite, positive
  `stretch_ratio`, a finite `force_n` and a `texture`; each branch has a stretch range (not all
  states at the same value);
* every path in the manifest (`texture`, `albedo`, `normal`) is relative and stays inside the package
  folder (no absolute path, no `..`); every texture has the expected size (the atlas, or the tile
  size for a tiled package), checked from the PNG headers;
* if `tiling.enabled`: `tile_size_px` > 0, `source_atlas_size_px` equals the atlas size,
  `period_px` equals `tile_size_px`, `period_uv` is finite and positive, and
  `period_uv == period_px / atlas size` within 1e-4.

Optional and used when present: `orientation`, `atlas.active_grid_axis` (anything but `horizontal`
means vertical), `force.clamp_width_m`, `release_residual_stretch`, `ps.albedo_scale` (must be finite
and positive; every state needs `albedo` and `normal`, and the layers need the texture size, or the
package loads with Relight disabled and a reason). Unknown keys are ignored, so a newer writer does not
break this viewer.

The viewer ignores `ps.exposure_gain` (already applied to the stored albedo), `ps.gains`, `ps_gains`,
`force.quality`, `tiling_seams.npz` and `build_report.json`.
