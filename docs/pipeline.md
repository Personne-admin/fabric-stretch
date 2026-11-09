# Pipeline: from a captured project to a material package

From capture to a package the viewer can open:

```
rig (fabsim rig)        runs: images, frames.json, force.json, session.json
  -> tracking           server/fabsim-track     -> <run>/tracks.json
  -> fabsim validate    sanity report
  -> fabsim ps          <run>/ps/               albedo + normal per frame (optional step)
  -> fabsim package     material_package_v3/    tiled package
                        material_package_v3_untiled/
  -> Fabric Stretch Viewer (Press View)
```

Install with `pip install -e .` in a virtual environment. The commands below take a project folder.

## 1. The project folder

Use `<orientation>_loading` and `<orientation>_release` folders inside the project; both branches
are needed. Rename suffixed captures such as `warp_loading_2` before processing them.
See [Run files](formats.md#project-and-run-files) for the layout.

## 2. Tracking

Each run needs a rectangular point grid in `tracks.json`. Use [the tracker](self_hosting_tracker.md#first-run)
or another tool that writes the [same format](formats.md#tracksjson).

## 3. `fabsim validate`

```sh
fabsim validate <project>
```

Per run it reads `frames.json` and `force.json` and prints `ok`, `warn` or `ERROR`:

| Level | Condition |
| --- | --- |
| error | `frames.json` or `force.json` unreadable; no frames; a diffuse (or listed PS) image file missing; a frame without a valid force reading (by `index`, `status == "ok"`); `tracks.json` is not valid JSON |
| warning | frames without PS images; no `tracks.json` yet; no `session.json`; a folder ignored because of its name |

Exit code 1 if any run has an error, or if the project has no run at all; 0 otherwise. Validate checks
that files exist. It does not decode images, so a truncated or damaged PNG passes; see
[Damaged images](#damaged-images).

## 4. `fabsim ps`: per-frame albedo and normals

```sh
fabsim ps <project> [--runs RUN [RUN ...]] [--led-config FILE] [--jobs N] [--force]
```

Writes [albedo, normals and a manifest](formats.md#per-run-ps-products) to `<run>/ps/`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--runs` | all | only these run folders (names such as `warp_loading`) |
| `--led-config FILE` | see [Light configuration](photometric_stereo.md#light-configuration) | geometry override when the run has no snapshot |
| `--jobs N` | `min(4, CPU count)` | worker processes per run |
| `--force` | off | regenerate runs that already have `ps/ps_manifest.json`; without it such runs print `skipped: ps/ exists (use --force)` |

Unreadable frames are skipped and listed in the manifest. The maps keep the raw solve, without
[package corrections](photometric_stereo.md#exposure-gain-package-only). The rig GUI's
`ps_output/` is left alone.

This step is optional: packaging solves its own photometric layers. Gain and scale calculations
are described in [Photometric stereo](photometric_stereo.md#per-led-gains).

## 5. `fabsim package`

```sh
fabsim package <project> [options]
```

Builds tiled and untiled packages for each orientation.

| Option | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | `<project>/material_package_v3` | tiled package folder |
| `--untiled-out DIR` | `<project>/material_package_v3_untiled` | untiled package folder |
| `--no-tiling` | off | write only the untiled package (`--out` is then unused) |
| `--no-ps` | off | no albedo/normal layers (the package then cannot be relit) |
| `--ps-gains per-state\|reference` | `per-state` | estimate LED gains for every state, or once on the loading reference frame |
| `--tare-states N` | 10 | fit the force tare to the first N loading states; `1` = use the loading reference reading alone |
| `--orientations NAME ...` | all found | build only these |
| `--led-config FILE` | see [Light configuration](photometric_stereo.md#light-configuration) | geometry override when the loading run has no snapshot |
| `--diagnostics DIR` | none | write tiling previews to `DIR/<orientation>/` (not part of a package) |
| `--force` | off | overwrite non-empty output folders |
| `--tiling-window-size N [N]` | `100 200` | `[vertical horizontal]` or one value |
| `--tiling-shift N [N N N]` | `20` | one value or `left right top down` |
| `--tiling-interp-dist N [N]` | `2` | one value or `vertical horizontal` |
| `--tiling-guidance STATE ...` | all states | state indices (loading first, then release) that guide the seam search |
| `--tiling-area TOP LEFT H W` | automatic | 0-based tile area in the atlas |
| `--tiling-joint-search` | off | score the bottom and right edges together |

An existing non-empty output folder is not overwritten without `--force`; the command stops with
exit code 2 before doing any work. Exit code 0 means at least one orientation was built, 1 means none.
Memory usage and algorithm details are in [Texture tiling](texture_tiling.md#signature-and-semantics).

### What it does, per orientation

1. **Load.** Keep frames with tracks, a diffuse image and valid force. The loading reference is
   the tracker reference; the release reference is the last usable release frame.
2. **Register.** Match the reference photos with SIFT (at least 30 matches covering a quarter of
   the image), then fit a homography. Use the largest rectangle of tracker cells valid across
   both runs. Sample loading through its grid and release through its grid and the homography.
   Reject orientations that fail the [registration check](formats.md#atlas).
3. **Stretch.** Take the median tracked-edge ratio along the pull axis, relative to the loading
   reference for both branches. Choose the axis that stretches more at the last loading frame.
4. **Tare.** Fit a line to the first `--tare-states` loading readings with stretch ≤ 1.03;
   its value at stretch 1 is `offset_n`. With fewer than three readings, or `--tare-states 1`,
   use the loading reference reading. Shift release readings to the loading zero before
   subtracting the offset. Keep originals as `force_raw_n`; record [quality](formats.md#force).
5. **Solve PS**, unless `--no-ps`: register albedo and normals, apply
   [corrections](photometric_stereo.md#exposure-gain-package-only), and choose a shared scale.
6. **Tile**, unless `--no-tiling`: [find seams](texture_tiling.md#in-the-pipeline) on the colour
   states and reuse them for other layers. A window that does not fit leaves only untiled output.
7. **Write** `package.json` for each variant and `build_report.json` in each output root.

### Output

```
material_package_v3/                 tiled
  build_report.json
  warp/
    package.json
    tiling_seams.npz
    loading/  frame_NNNN.png  albedo_NNNN.png  normal_NNNN.png
    release/  frame_NNNN.png  albedo_NNNN.png  normal_NNNN.png
material_package_v3_untiled/         same layout, atlas size, no tiling_seams.npz
  build_report.json
  warp/ ...
```

Failed orientations leave no directory; partial builds are removed. If tiling fails, only the
untiled variant appears in the viewer.

### Reading the build report

Check each orientation's `status`, `reason` and `warnings` in
[`build_report.json`](formats.md#build_reportjson). Warnings also appear on the console.

### Validate a package

```sh
fabsim validate-package <project>
```

Also accepts a package set, an orientation folder or `package.json`. Checks manifest fields,
paths (including symlink escapes), PNG decoding, sizes, bit depths and complete PS layers.
Exit status is 1 for invalid or missing packages, 0 otherwise. v2 packages give a warning.

### Damaged images

Validation checks file existence, not image decoding. Packaging handles damaged PS images as follows:

| Problem | Result |
| --- | --- |
| Missing or truncated PNG | drop the state from all layers; record it in `dropped_states` |
| Damaged reference-frame PS image | omit photometric layers (`ps_skipped`) |
| Complete PNG with decode errors, such as a bad CRC | fail the orientation; `fabsim ps` instead skips that frame |
| Frames without PS image entries | build without photometric layers |

Replace the damaged image and rebuild, or use `--no-ps`.

### LED configuration

Packaging uses the loading snapshot when present; see
[configuration precedence](photometric_stereo.md#light-configuration).

## 6. Other tools

- `fabsim colorcal` fits diffuse and point-light models from `(N*30, 3)` RGB patch arrays:

  ```sh
  fabsim colorcal --diffuse-patches d.npy --point-light-patches p.npy \
      --diffuse-exposure-ms 15 --point-light-exposure-ms 15 --output bundle.json
  ```

  The rig applies the bundle when saving; packaging does not read it.
- `fabsim stereo`: Tk PS GUI, previews and batches to `<run>/ps_output/`.
- `fabsim ps-verify RUN`: [checks](photometric_stereo.md#quality-metrics) and diagnostics in
  `<run>/ps_verify/`. Pass several runs to compare them.
- `fabsim rig`: [acquisition app](hardware.md#the-acquisition-app).

## 7. Try it on the demo

Work on a copy, since packaging replaces derived outputs when `--force` is used:

```sh
WORK=$(mktemp -d)
cp -r examples/demo "$WORK/demo"
fabsim validate "$WORK/demo"
```

Follow the [demo recipe](../examples/demo/README.md#regenerate), substituting `"$WORK/demo"`.
Open that folder in the viewer or render a [snapshot](press_view.md#headless-snapshots-press_snapshot).

## 8. Rebuilds and selected orientations

`--orientations` limits which orientations are built. Others are left untouched, but
`build_report.json` is rewritten to describe only the requested ones. Requesting an orientation
with no runs gives it a skipped entry; if nothing is built, the command exits with 1.

`--out` and `--untiled-out` accept any path; viewer discovery uses the
[standard folders](press_view.md#package-set-discovery).
