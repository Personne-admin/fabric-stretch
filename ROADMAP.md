# Roadmap

## Viewer

- Reuse draw buffers and bind groups.
- Share CPU render workers and caches between tabs.
- Antialias silhouettes on CPU and GPU.
- Add captured-LED presets and per-LED colour to the light pad.
- Show typed motor mechanics in the inspector.
- Test the windowed egui-wgpu integration on a real display; only headless software GPU tests
  were run during refactoring.

## Pipeline

- Check [fallback LED geometry](docs/photometric_stereo.md#light-configuration) against older captures.
- Replace [normal-plane correction](docs/photometric_stereo.md#normal-plane-removal-package-only)
  with a near-field light model.
- Improve tare estimates for forces near the noise floor.
- Compare the tiling port against MATLAB and handle vertical phase offsets.

## Rig and firmware

- Fix the [firmware issues](firmware/README.md#known-issues).
- Confirm the driver's [microstepping setting](docs/hardware.md#motor-and-clamp).
- Test the uEye camera and Tk app on hardware; so far they have only been tested with fakes.
- Check the PS GUI's handling of old sessions stamped `applied: true` with reason
  "images saved uncorrected". It applies the point-light model from a bundle beside the run.

## Tracker and builds

- Run tracker setup, smoke tests and Docker on a real GPU host. Verify the checkpoint hash
  by downloading it; the current hash came from Hugging Face metadata. Compare new tracks
  with archived results.
- Support batch systems such as Apptainer on HPC.
- Build the viewer for Windows.
- Test the rig build under Wine and Windows. CI does not cover hardware, Windows or GPU servers.
