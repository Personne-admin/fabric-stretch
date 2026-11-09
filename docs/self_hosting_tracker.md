# Self-hosting the point tracker

Run CoWTracker on your GPU machine and bring back `<run>/tracks.json` for packaging.
Read [NOTICE](../NOTICE) before setup; CoWTracker has a non-commercial licence.

Tracking uploads one Python file, `src/fabsim/cowtracker_to_fabsim.py`, to the GPU host.
The shell frontend runs locally. Results record model, checkpoint and runtime provenance. Other trackers can be
used if they write the [same format](formats.md#tracksjson).

## What you need

* An NVIDIA GPU with a driver compatible with the torch/CUDA build selected in
  [`tracker.env.example`](../server/tracker.env.example). There is no CPU or ROCm path.
  `TRACKER_MIN_DRIVER` is the minimum driver major version checked by setup.
* GPU memory requirements have not been measured. Sequences of up to 200 frames at the
  adapter's default size have been tested. For longer sequences use `--windowed`; for memory
  errors, also try smaller `--inference-width/--inference-height`.
* About 15 GB of free disk under `TRACKER_ROOT` (torch about 6 GB, checkpoint about 4 GB).
* On the GPU machine: `git`, `curl`, `sha256sum`, and Python meeting `TRACKER_PYTHON_VERSION`, with `venv`
  (docker mode: Docker with the NVIDIA container toolkit instead).
* On your machine: Linux or WSL, bash >= 4.4, GNU coreutils, `ssh`, `rsync`, `python3`.
  macOS needs a newer bash and GNU tools. `dry-run` also needs numpy and Pillow, for example the
  repository venv: `TRACKER_LOCAL_PYTHON=<repo>/.venv/bin/python`.

## The three modes

Select `TRACKER_MODE` in the config, or pass `--mode`:

| Mode | GPU | Inputs and results |
| --- | --- | --- |
| `ssh` | remote host | rsync upload, then fetch results |
| `local` | this machine | read inputs in place |
| `docker` | this machine | mount inputs and checkpoint read-only |

SSH uses `BatchMode`; load a key in your agent or set `TRACKER_SSH_KEY`. Use `TRACKER_SSH_JUMP`
for a gateway and `TRACKER_SSH_OPTS` for extra options.

## Configuration

Copy the annotated configuration and fill in the required paths:

```bash
cp server/tracker.env.example server/tracker.env
$EDITOR server/tracker.env
```

The [example file](../server/tracker.env.example) documents keys, defaults and version pins.
It is sourced by bash (`KEY=value`, no spaces around `=`). Keep pins when reproducing a run.

The lookup order is `--env FILE`, `$FABSIM_TRACKER_ENV`, `server/tracker.env`, then
`~/.config/fabsim/tracker.env`. Variables already set in the environment win over the file.
Set `TRACKER_MODE`, `TRACKER_ROOT` and `FABSIM_DATA_ROOT` in every mode. Also set
`TRACKER_SSH_HOST` in ssh mode.

`FABSIM_ADAPTER` overrides the adapter path.

`HF_TOKEN` is needed only for gated or private weights. If you put it in the config file, use
`chmod 600`. It is passed to the target and to curl on stdin, never as a command-line argument.

## First run

From the repository root, after configuring `tracker.env`:

```bash
server/fabsim-track dry-run my_dataset/warp_loading   # check inputs locally; no GPU needed
server/fabsim-track setup                            # install on the chosen target
server/fabsim-track status
server/fabsim-track run my_dataset/warp_loading
server/fabsim-track fetch my_dataset/warp_loading --run-id <id printed by run>
server/fabsim-track validate my_dataset/warp_loading
```

For a dataset, use `server/fabsim-track dataset my_dataset`. In local/Docker modes, `fetch`
only places the already-local result in the run folder.

## Commands

| Command | What it does |
| --- | --- |
| `setup [--force] [--skip-smoke] [--checkpoint-only]` | Uploads the adapter and runs `bootstrap.sh` on the target (docker mode: builds the image first). Safe to repeat. |
| `status [--deep]` | Checks GPU, driver, torch/CUDA/cuDNN, the CoWTracker revision and submodules, the checkpoint hash, the adapter version, disk space and the latest runs. Exit status 1 on any `FAIL`. |
| `dry-run <run> [-- args]` | The adapter's own `--dry-run` on this machine: frames, resolution, point grid, reference frame. |
| `run <run> [--role R] [--grid-cols N --grid-rows N] [--gpu I] [--run-id ID] [--force] [-- args]` | Uploads and tracks one run. |
| `fetch <run> --run-id ID [--no-place] [--refetch] [--force]` | Downloads the results and places `tracks.json` in the run directory. |
| `validate <run> [--run-id ID \| --tracks FILE] [--any-points]` | Checks a result against its run, prints a JSON report. |
| `dataset <dataset> [run ...] [--run-id ID] [-- args]` | `run` + `fetch` + `validate` for every `*_loading` / `*_release` run. |

`<run>` is relative to `FABSIM_DATA_ROOT` or absolute. Arguments after `--` go to the adapter;
see `python src/fabsim/cowtracker_to_fabsim.py --help`. For example, use `-- --windowed` for
long sequences or `-- --reference-frame 3`. Custom `--points FILE` is read on the target and
replaces the grid.

Global options, before the command: `--env FILE` (config file) and `--mode MODE`.

### Where things end up

| | ssh mode | local / docker mode |
| --- | --- | --- |
| uploaded inputs | `$TRACKER_ROOT/measurements/<dataset>/<run>/` | not copied |
| results | `$TRACKER_ROOT/outputs/<dataset>/<run>/<run id>/` | `<run>/cowtracker-results/<run id>/` |
| downloaded copy | `<run>/cowtracker-results/<run id>/` | (same directory) |
| placed for the viewer | `<run>/tracks.json` | `<run>/tracks.json` |

A results directory holds `tracks.json`, `tracks.csv`, `diagnostics/` (still images
of the tracks over the first, 25 %, 50 %, 75 % and last frame) and `run.log`.
`$TRACKER_ROOT` additionally contains `env/`, `code/cowtracker/`,
`code/adapter/cowtracker_to_fabsim.py` (+ `.version` stamp), `checkpoints/`,
`env.lock` (pip freeze) and `setup.json` (versions at setup time).

### What counts as a valid result

Validation requires one frame per tracked image, the same points in every frame, finite
coordinates, confidence in [0, 1], and reference reconstruction error within the configured
limit. `TRACKER_REFERENCE_ERROR_LIMIT` is passed to the adapter and recorded in the tracks;
`validate` checks that recorded limit. Override one run with `-- --reference-error-limit PX`.

`dataset` continues after failures and exits 1 if any failed. It writes
`<dataset>/cowtracker-logs/<run id>-summary.json`; repeating that run ID skips valid results.

`fetch` needs `--force` to replace `<run>/tracks.json`, backing it up as `tracks.prev-<time>.json`.
See [tracks.json](formats.md#tracksjson) for packaging requirements.

## What `setup` does (`bootstrap.sh`)

Setup sends `bootstrap.sh` to the target. Re-running it reuses the installation; `--force`
moves old environments and checkpoints aside.

1. Check tools, Python, GPU driver and disk space (warn below 12 GB).
2. Create `$TRACKER_ROOT/env`, install configured torch, torchvision, xformers and other packages;
   check that pip did not replace torch.
3. Clone the pinned CoWTracker revision and its submodules. Refuse dirty revision changes
   unless `--force` is given.
4. Download the checkpoint resumably, verify SHA256 and write a `.sha256` file. A bad hash
   leaves a `.mismatch` file and fails setup.
5. Test GPU convolution and imports. If cuDNN fails, `TRACKER_CUDNN_PREFIX` allows a separate install.
6. Track three synthetic frames on a 4 × 4 grid; require reference error below 2 px.
   `--skip-smoke` skips this check.
7. Write `env.lock` (pip freeze) and `setup.json` (versions and hardware).

Docker setup builds the image first, then runs only the checkpoint step.

## Docker mode

Set `TRACKER_MODE=docker`, then use the same setup/run commands. Setup builds
[`server/Dockerfile`](../server/Dockerfile), passing version settings from the config as build
arguments. `BASE_IMAGE` selects the CUDA base image.

The image contains the environment and code, but no weights. The working-tree adapter is mounted
at run time, so editing it needs no rebuild.

| Host | In the container | Mode |
| --- | --- | --- |
| the run directory | `/data/run` | read-only |
| `<run>/cowtracker-results/<run id>` | `/data/out` | read-write |
| `$TRACKER_ROOT/checkpoints` | `/ckpt` | read-only |
| your working tree's adapter | `/opt/fabsim/cowtracker_to_fabsim.py` | read-only |

The container uses `--gpus "device=<gpu>"` and your user/group ID.
To check GPU access: `docker run --rm --gpus all <base image> nvidia-smi`.

## Troubleshooting

| Problem | Check or fix |
| --- | --- |
| Cannot reach SSH host | Try `ssh <host> true`. Check VPN, jump host and loaded key. For repeated 2FA, open a `ControlMaster` session with `ControlPersist` first. |
| Driver too old | Match `TORCH_INDEX_URL` to the driver and adjust `TRACKER_MIN_DRIVER`. |
| cuDNN/libcublas error | Conflicting CUDA libraries. Set a per-host `TRACKER_CUDNN_PREFIX` and rerun setup, or install cuDNN separately and set `TRACKER_LD_LIBRARY_PATH`. |
| CUDA unavailable | Check `nvidia-smi`, `TRACKER_GPU`, and the NVIDIA container toolkit in Docker mode. |
| Checkpoint hash mismatch | Check `CHECKPOINT_REVISION` and `CHECKPOINT_URL`. For different weights, set their SHA256 explicitly. |
| Cannot import CoWTracker | Rerun setup to initialise submodules and install matching xformers. |
| Over 200 frames / out of memory | Use `-- --windowed` or reduce `--inference-width` and `--inference-height`. |
| Results directory exists | Pick a new run ID, or use `--force` to replace partial results (the log is appended). `dataset` resumes IDs with an existing summary. |
| Expected 320 points | For custom points, use `--any-points` or `--expect-points N`. Arguments in `TRACKER_EXTRA_ARGS` are detected automatically. |
| rsync missing | Install it on both machines. |

## Tests

`bash server/tests/e2e_local.sh` tests local/SSH flows, failures and resume using fakes, without
a GPU or network. Adapter tests are in `tests/python/test_cowtracker_to_fabsim.py`.
