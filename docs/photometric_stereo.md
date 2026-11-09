# Photometric stereo

The Python solver is `src/fabsim/photometric.py`, shared by the pipeline and rig tools.
The viewer shades the maps with `crates/fabric-viewer/src/lighting.rs`.

## Rig geometry

The camera looks at the fabric strip from above. At every state the rig takes, one after the other:

* `diffuse`: the diffuse light, used for colour and tracking,
* `ps_1`, `ps_2`, `ps_3`: three single LEDs of set A, one image each.

The model uses an azimuth for each LED and a shared elevation, not LED positions.
The rig records calibrated angles in `session.json`; see [Light configuration](#light-configuration).
The optional second polarisation set is not used by the solver.

## Coordinate conventions

Solver and viewer conventions (tracking pixels still use y down):

| Item | Convention |
| --- | --- |
| Camera / data frame | x = image right, y = image **up**, z = toward the camera |
| Azimuth | counter-clockwise from image right, looking at the image (90 degrees = image up) |
| Elevation | angle above the image plane (0 = grazing, 90 = straight on) |
| Light direction | `L = (cos e cos a, cos e sin a, sin e)` |
| Image rows | run downward, so a row index is `-y` |
| Atlas of a package | same axes as the camera image: x = atlas column, y = atlas row UP, z toward the camera |
| Pixel values | camera bytes are raw and close to linear (no gamma); bytes / 255 are linear intensities. Never sRGB-decode them |
| Quantisation | see [image encodings](formats.md#per-run-ps-products) |

## The solver

### Model

A Lambertian surface lit by LED `i` gives

```
I_i = gain_i * albedo * max(n . L_i, 0)
```

for the grey intensity `I_i` of one pixel (`n` is the unit normal). Ignoring the clamp (shadows),
three lights give three linear equations in `g = albedo * n`:

```
(I_1/gain_1, I_2/gain_2, I_3/gain_3)^T = L g          L = rows L_i
g = pinv(L) (I / gain)        albedo = |g|        n = g / |g|
```

Pixels without signal (`|g| < 1e-8`) get the normal `(0, 0, 1)`. `solve` takes an `(H, W, 3)` float
stack with one channel per LED and returns the grey albedo and the normals. Shadowed pixels (a light
below the horizon of a steep slope) break the linear model locally; there is no shadow handling.

### Input

Each PS image is read as 8-bit BGR and reduced to its **Rec. 601 luma** (the same weights as
`cv2.cvtColor(..., COLOR_BGR2GRAY)`: 0.299 R + 0.587 G + 0.114 B), divided by 255, so the three
channels of the PS stack are the luma of `image_ps_1..3`. The diffuse image is read as linear RGB.

### Light configuration

Precedence (`load_light_config`): the `led_config` snapshot in the run's `session.json`, then the
file given with `--led-config`, then the packaged default `src/fabsim/rig/defaults/led_config.json`.
The choice is recorded as `light.source` in the manifests. Packaging uses the loading run's
snapshot. A selected configuration without three azimuths and an elevation is an error.
The JSON keys are defined under [session.json](formats.md#sessionjson); use the rig's
[light calibration dialogs](hardware.md#lights) to measure your own geometry.

### Per-LED gains

Unequal LED brightness biases the normals. `estimate_gains` estimates relative radiance:

1. the **support** is the cloth: pixels whose mean over the three images exceeds 5 % of the median
   of that mean image;
2. per LED the median intensity over the support;
3. gains = medians divided by their mean (so they average 1).

This assumes the cloth faces the camera on average. Configured `gains` override estimation.
Otherwise:

* `fabsim ps`: per frame (`gains_mode: per-frame`), recorded in the manifest.
* `fabsim package`: for every state (`--ps-gains per-state`, the default; recorded as
  `branches.<branch>[].ps_gains`) or once on the loading reference frame (`--ps-gains reference`, recorded as
  `gains_mode: fixed`).

### Colour albedo

The solved albedo is grey. The colour is taken from the diffuse photograph by its chromaticity:

```
albedo_rgb = albedo_grey * diffuse_rgb / luma(diffuse_rgb)
```

with the diffuse image as linear RGB. The albedo therefore has the neutral chromaticity of the diffuse
shot, not the colour cast of the LED spectrum that the PS photos carry.

### Encoding

The [per-run products](formats.md#per-run-ps-products) and
[package layers](formats.md#material-package-v3) use different albedo bit depths.
Those references define the encoding, decoding and rounding rules.

### Albedo scale

* `fabsim ps`: the 99.9th percentile of the brightest channel over the support, measured on the
  loading reference frame and shared by both runs of the orientation. The reference is the
  tracker's reference frame if it has PS images, otherwise the first frame with them.
* `fabsim package`: measured over **all states of both branches**, on the atlas and on the tile if
  tiled: the smallest value that at most 0.1 % of the brightest-channel pixels of any state exceed,
  plus 0.1 % headroom. One scale for the package keeps the albedo comparable between states, and the
  build report lists the clipped fraction actually reached (about 0.001 on real data).

### Exposure gain (package only)

A single LED is dimmer than the all-LED diffuse light, so the albedo from the solver is darker than
the diffuse photograph (a factor of about 2 on the demo). The package multiplies the albedo by

```
exposure_gain = mean(luma(diffuse)) / mean(albedo * max(n_z, 0))     over the support, loading reference frame
```

so that a relight straight on (`elevation 90`, `intensity 1`, `ambient 0`) is as bright as the diffuse
photograph. The factor is stored as `ps.exposure_gain`; the stored albedo already includes it and the
viewer does not apply it again. To reproduce the photograph of LED `i` from a package use
`intensity = gains[i] / exposure_gain`. The `ps/` products are not scaled by it (they use
`intensity = gains[i]`).

### Normal plane removal (package only)

The parallel-light assumption leaves a broad tilt in the normals when the LEDs are close.
On real captures this reached roughly ±11 degrees across a flat sheet.

For every state the package fits a plane (a constant and a slope in each image direction) to the x and
y components of the normal over the whole atlas, subtracts it and renormalises. The plane is fitted on
a subsample of at most about 200 000 pixels. `normal_tilt_deg` of a state records the tilt of its mean
normal before the correction (the angle between the mean normal and the camera axis).

This assumes the sheet is flat on average: real large-scale slopes are removed too, while
finer detail remains. Correction happens before tiling. `fabsim ps` keeps the raw solve.

### Quality metrics

| Metric | Where | Meaning |
| --- | --- | --- |
| `albedo_diffuse_correlation` | `ps/` manifest, package `ps.quality` | Pearson correlation of the grey albedo and the luma of the diffuse photograph over the support. Good captures reach 0.86 to 0.95. Below 0.6 a warning says relighting will look wrong (dim or mis-lit PS capture, opposing illumination gradients) |
| `mean_normal_tilt_deg` | `ps/` manifest | angle of the mean normal from the camera axis on the reference frame |
| `normal_tilt_deg` | package, per state, and `ps.quality.normal_tilt_deg {mean, max}` | the same before the plane correction |
| `exposure_gain` | both manifests | see above |
| `clipped_fraction_max` | `build_report.json` | largest fraction of albedo pixels at full scale, untiled and tiled |
| `gains` | both | spread of LED radiances. A gain far from 1 (e.g. 0.79) flags a weak LED |

`fabsim ps-verify <run>` goes further for one run: exposure of the captures, conditioning and spread
of the light directions, hemisphere and unit length of the normals, the re-render error and the
held-out prediction of the diffuse image (the discriminating test), clipping and mask coverage, and
optionally integrability. Several run folders give a table that ranks LED power and exposure
settings.

## The relight model

```
displayed = clamp( albedo * (ambient + intensity * max(n . l, 0)) , 0, 1 )
```

* Python: `fabsim.photometric.shade(albedo, normals, direction, intensity, ambient, gain)`.
* Rust: `lighting::shade_pixel` in `crates/fabric-viewer/src/lighting.rs`. The Press View GPU shader
  and CPU renderer use the same shading. The displayed byte is `round(clamp(...) * 255)`.

Shading is linear and display-raw: `albedo = decoded value * albedo_scale`, no sRGB decode of
the albedo and no sRGB encode of the result. The light is a neutral white light given by azimuth
(counter-clockwise from image right), elevation, `intensity` and `ambient`. The
[Press View controls](press_view.md#controls) define the viewer defaults. `shade` accepts an
optional per-LED `gain` (default 1) for comparisons with captures. The viewer's `Light` controls
the preview and has no per-LED gain setting.

`l` is the unit vector towards the light in the data frame (`light_direction`). In the Image Viewer
the pad frame is the image frame. In the Press View the pad is in screen space and converted to the
data frame; see [press_view.md](press_view.md).

## Verification facts

Compare relights with captures in luma. PS photographs carry the LED colour cast; the albedo
takes its colour from the diffuse shot, so a neutral relight will not match each RGB channel.

Test bounds (mean absolute error on the 8-bit scale):

| Test | What is compared | Bound |
| --- | --- | --- |
| `test_photometric.py::test_sphere_round_trip` | synthetic: solve then render vs input | albedo and normals within 1e-4 on fully lit pixels; re-render mean error < 1/255 |
| `test_photometric.py::test_real_frame_rerender_matches_capture` (real data, `FABSIM_DATA_DIR`) | captured `fabric_002` frame re-rendered, gains estimated | < 2/255 |
| `test_ps_frames.py::test_demo_products_rerender_the_captured_photos` | demo `ps/` products under each LED with its gain, luma | < 1/255, every frame and LED |
| `images/relight.rs::relight_of_the_demo_project_matches_its_captured_photos` | Image Viewer relight of the demo `ps/` products under each LED, luma | < 3/255, every frame and LED |
| `images/relight.rs::eight_bit_normals_cost_less_than_half_a_level` | Image Viewer's 8-bit normals vs exact 16-bit | < 0.5/255 |
| `press_sim.rs` relight at a capture LED | Press View with the **package** layers, light pad rotated to the screen, vs demo photo `ps_i` | 3x3-box-filtered luma MAE < 3/255 (measured 1.14, 0.81, 0.83) and raw MAE < 4/255 (measured 3.49, 3.17, 2.95). Without the 90 degree rotation of the pad the error is 12 to 15/255, which checks that the rotation is applied |
| `test_material_package.py::test_head_on_relight_is_as_bright_as_the_diffuse_photo` | package layers, head-on light vs diffuse photo | mean within 3 %; LED `i` at `gains[i]/exposure_gain` matches the captured mean within 3 % |
| `test_photometric.py::test_gains_estimation_and_solve` | gain recovery and the tilt it removes | gains within 1e-5; tilt below 0.1 degrees |

Atlas resampling softens fine texture, so Press View has a larger residual than the Image Viewer.

For capture comparisons, use `intensity = gains[i] / exposure_gain` and ambient 0.

## Limits

* Lambertian, no specular term, no shadows, no inter-reflection; a glossy or very deep structure
  violates the model.
* Near-field effects are only corrected by a global plane in the normals. The albedo keeps whatever
  illumination fall-off the LEDs have.
* A neutral light cannot reproduce the colour cast of the photos (see above).
* The geometry depends on the calibrated azimuths. A wrong configuration tilts the normals; compare
  `fabsim ps-verify` between configurations. Older sessions without a snapshot use the selected
  fallback geometry, which may differ from the calibration used during capture.
