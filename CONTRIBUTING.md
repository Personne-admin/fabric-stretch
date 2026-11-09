# Contributing

Open an issue before starting a large change.

## Setup

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'       # use '.[rig,dev]' for the acquisition app
```

Rust uses the toolchain in `rust-toolchain.toml`; the minimum version is in
[`Cargo.toml`](Cargo.toml). On Linux, install the development packages for X11 or Wayland,
xkbcommon, GTK 3 and Vulkan or OpenGL. See the Ubuntu [CI package list](.github/workflows/ci.yml).
The rig GUIs also need Tk.

## Checks

```sh
scripts/check.sh              # or: scripts/check.sh python | rust | shell
bash server/tests/e2e_local.sh
```

The shell tests need no GPU. For rig tests, see
[Developing without hardware](docs/hardware.md#developing-without-hardware).

## Documentation

Keep each rule in one reference and link to it elsewhere. File layouts and encodings belong
in `docs/formats.md`, processing in the pipeline and algorithm guides, protocol details in
`docs/serial_protocol.md`, and firmware bugs in `firmware/README.md`.
Tracker defaults and pins belong in `server/tracker.env.example`. The roadmap should list
unfinished work, not repeat the component guides.

## Images and demo

Use the [snapshot recipes](docs/press_view.md#headless-snapshots-press_snapshot) for README
images and the [demo recipe](examples/demo/README.md#regenerate) to rebuild the demo.
Keep images under about 300 KB each and the demo under about 15 MB.

Contributions are licensed under Apache 2.0.
