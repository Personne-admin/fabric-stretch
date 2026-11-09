# Releases

Download from [GitHub Releases](https://github.com/Personne-admin/fabric-stretch/releases/latest).

| System | Installer | Portable |
| --- | --- | --- |
| Windows x64 | `.msi` | `.zip` |
| Linux x64 | `.deb` | `.tar.gz` |

The bundles contain `fabric-viewer`, `fabsim`, `fabsim-rig`, `press_snapshot` and `examples/demo`.
Python and Tk are included. The camera needs the IDS uEye driver installed separately.
Copy the demo to a writable folder before rebuilding it. Third-party notices are in `licenses/`.

On Windows, run the MSI or extract the ZIP. The installer adds Start menu shortcuts and adds
its folder to PATH for new terminals. On Debian/Ubuntu, install with
`sudo apt install ./fabric-stretch_*_amd64.deb`. Portable Linux builds need the desktop libraries
listed in `packaging/README.txt`; they target Ubuntu 22.04 or newer (glibc 2.35+).
A desktop session is needed for the viewer and rig app.

To check a download on Linux, run `sha256sum -c SHA256SUMS` beside the downloaded assets.
On Windows, compare `Get-FileHash <file> -Algorithm SHA256` with its entry in `SHA256SUMS`.

## Making a release

1. Set the same version in `src/fabsim/__init__.py` and the workspace section of `Cargo.toml`.
   Cargo crates and Python package metadata inherit these values. Update `Cargo.lock` and
   add a dated entry in `CHANGELOG.md`.
2. Push to `master` and wait for CI. It runs lint and tests, builds on Windows and Linux,
   then installs and tests the MSI and DEB. The build artifacts are available on that run.
3. Tag that commit and push the tag:

   ```sh
   python scripts/release.py version --tag v0.4.0
   git tag -a v0.4.0 -m "Fabric Stretch 0.4.0"
   git push origin v0.4.0
   ```

The Release workflow checks the tag against both versions and runs CI again. After every
check passes, it uploads the bundles, Python wheel/source archive and `SHA256SUMS` to a draft,
then publishes it. Release notes come from the matching changelog entry. A failed run leaves
no public release; an existing draft can be retried. Published releases are never overwritten.

## Local builds

Use a native Windows or Linux host with Rust stable, Python 3.12 and the
[development system packages](../CONTRIBUTING.md). Windows installers also need
WiX 5.0.2 (`dotnet tool install --global wix --version 5.0.2`).

```sh
python -m pip install -e '.[rig]' -r scripts/release-requirements.txt
python scripts/release.py build --stage build/stage/fabric-stretch --target x86_64-unknown-linux-gnu
python scripts/release.py smoke --stage build/stage/fabric-stretch --gui
python scripts/release.py pack --stage build/stage/fabric-stretch --out dist/release --target linux-x86_64
```

For Windows, use `x86_64-pc-windows-msvc` for the build target and `windows-x86_64` for packaging.
The smoke test opens and closes the rig UI, rebuilds the demo with two workers, validates its
packages and renders a snapshot. It does not exercise a camera or controller.
