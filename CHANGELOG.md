# Changelog

## Unreleased

- `fabsim validate-package` checks manifests and image files without the viewer.
- Viewer package loading runs in the background, with progress and cancellation.
- Fixed the Python sphere fixture and lint/format checks.

## [0.1.0] - 2026-11-08

First release of the consolidated repository. This release contains `fabsim` 0.3.0 and `fabric-viewer` 0.1.0.

### Added

- Material package format v3 (`fabsim-material-rgb`, version 3).
- Packaged rig UI, replacing the single-file `fullui.py`.
- `fabsim` command line: `package`, `ps`, `colorcal`, `validate`, `rig`, `stereo`, `ps-verify`.
- `fabsim ps`: full-precision albedo and normal maps per run in `<run>/ps/`.
- `fabsim.photometric`: one photometric-stereo implementation shared by the pipeline and the rig
  GUIs. LED geometry comes from the session snapshot, a given file or the packaged default.
- Press View in the viewer: package-set discovery, Tiled and Untiled textures, Relight with a
  light pad, Oblique and Top views, force or indentation control.
- `press_snapshot` example: renders Press View headless to a PNG, on the CPU or with `--gpu`.
- Self-hosted tracking: `server/fabsim-track` (`setup`, `status`, `dry-run`, `run`, `fetch`,
  `dataset`, `validate`) with ssh, local and Docker modes, and `server/bootstrap.sh`, which
  installs CoWTracker at a pinned commit with a checksum-verified checkpoint.
- Demo project `examples/demo` with its v3 packages, used by the README and the tests.
- Apache-2.0 licence and NOTICE, CI for Python, Rust and shell scripts.

### Changed

- Rig app moved to `fabsim.rig`: per-user config, typed settings, simulated hardware,
  session writer and headless PS processing using `fabsim.photometric`.
- Shared `lighting` module for both views, with linear display-raw shading. Package albedo
  includes `exposure_gain` so a head-on relight matches diffuse brightness.
- Viewer reads v3 packages and legacy v2 packages.
- Package build uses float32 and preallocated buffers in tiling to cut peak memory.
- The tracker adapter `src/fabsim/cowtracker_to_fabsim.py` stays self-contained (the server
  copies only that file) and records provenance in `tracks.json`.

### Removed

- The 3D View tab. Saved layouts that contain it still load.
- Package format v1 and `material_package_tiled/` support in the viewer.
- RGB-mode photometric stereo in the rig GUI (it assumed fixed 0/120/240 geometry).
- Old tracker scripts (replaced by `server/fabsim-track`), `ps_material_package.py` (merged into
  the package build), `rough_tile_gui.py`.

### Known issues

For current open work, see [ROADMAP.md](ROADMAP.md). Controller defects are maintained in
[Known firmware issues](firmware/README.md#known-issues).

[0.1.0]: https://github.com/personne-admin/fabric-stretch/releases/tag/v0.1.0
