from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABOUT_VERSION = "0.9.3"
UPGRADE_CODE = "19A04FD9-C920-4DAF-9F3E-C8E2C3948AC2"
COMMANDS = ("fabric-viewer", "press_snapshot", "fabsim", "fabsim-rig")


def run(*args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def version(root: Path = ROOT, tag: str | None = None) -> str:
    tree = ast.parse((root / "src/fabsim/__init__.py").read_text())
    value = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    )
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
        raise ValueError(f"invalid version: {value}")
    cargo = (root / "Cargo.toml").read_text()
    if not re.search(rf'^version = "{re.escape(value)}"$', cargo, re.M):
        raise ValueError("Rust workspace and Python versions differ")
    if tag and tag != f"v{value}":
        raise ValueError(f"tag {tag!r} does not match v{value}")
    return value


def executable(stage: Path, name: str) -> Path:
    return stage / (name + (".exe" if sys.platform == "win32" else ""))


def freeze(stage: Path):
    stage.mkdir(parents=True, exist_ok=True)
    for name, entry in (("fabsim", "fabsim_entry.py"), ("fabsim-rig", "rig_entry.py")):
        args = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--name",
            name,
            "--paths",
            str(ROOT / "src"),
            "--collect-submodules",
            "fabsim",
            "--collect-data",
            "fabsim",
            "--hidden-import",
            "pyueye.ueye",
            "--exclude-module",
            "torch",
            "--exclude-module",
            "pytest",
            "--distpath",
            str(stage),
            "--workpath",
            str(ROOT / "build/frozen" / name),
            "--specpath",
            str(ROOT / "build/frozen"),
        ]
        if name == "fabsim-rig" and sys.platform == "win32":
            args.append("--windowed")
        run(*args, ROOT / "scripts" / entry, cwd=ROOT)


def python_licenses(stage: Path):
    dest = stage / "licenses/python"
    dest.mkdir(parents=True, exist_ok=True)
    packages = ("numpy", "pillow", "opencv-python-headless", "pyserial", "pyueye", "pyinstaller")
    for name in packages:
        dist = importlib.metadata.distribution(name)
        folder = dest / f"{dist.metadata['Name']}-{dist.version}"
        folder.mkdir()
        (folder / "METADATA.txt").write_text(dist.read_text("METADATA") or "", encoding="utf-8")
        for file in dist.files or []:
            if file.name.lower().startswith(("license", "licence", "copying", "copyright", "notice")):
                source = Path(dist.locate_file(file))
                if source.is_file():
                    target = folder / Path(*[p for p in file.parts if p not in ("..", ".")])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    for source in (Path(sys.base_prefix) / "LICENSE.txt", Path(sysconfig.get_path("stdlib")) / "LICENSE.txt"):
        if source.is_file():
            shutil.copyfile(source, dest / "Python-LICENSE.txt")
            break
    import tkinter

    interp = tkinter.Tcl()
    for variable in ("tcl_library", "tk_library"):
        try:
            library = Path(interp.eval(f"set {variable}"))
        except tkinter.TclError:
            # Loading Tk solely to find its licence needs a display on Linux.
            library = Path(interp.eval("set tcl_library")).parent / "tk8.6"
        source = library / "license.terms"
        if source.is_file():
            shutil.copyfile(source, dest / f"{variable}-LICENSE.txt")


def rust_licenses(stage: Path, target: str):
    triple = "x86_64-pc-windows-msvc" if sys.platform == "win32" else "x86_64-unknown-linux-musl"
    tools = ROOT / "build/release-tools"
    tools.mkdir(parents=True, exist_ok=True)
    filename = f"cargo-about-{ABOUT_VERSION}-{triple}.tar.gz"
    archive = tools / filename
    base = f"https://github.com/EmbarkStudios/cargo-about/releases/download/{ABOUT_VERSION}/{filename}"
    if not archive.exists():
        urllib.request.urlretrieve(base, archive)
    with urllib.request.urlopen(base + ".sha256", timeout=60) as response:
        expected = response.read().decode().split()[0]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("cargo-about checksum mismatch")
    binary = "cargo-about.exe" if sys.platform == "win32" else "cargo-about"
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if Path(m.name).name == binary and m.isfile())
        with tar.extractfile(member) as source, (tools / binary).open("wb") as dest:
            shutil.copyfileobj(source, dest)
    (tools / binary).chmod(0o755)
    dest = stage / "licenses"
    dest.mkdir(exist_ok=True)
    run(
        tools / binary,
        "generate",
        "--locked",
        "--workspace",
        "--config",
        ROOT / "packaging/about.toml",
        "--target",
        target,
        "--output-file",
        dest / "rust.html",
        ROOT / "packaging/licenses.hbs",
        cwd=ROOT,
    )


def build(stage: Path, target: str):
    run(
        "cargo",
        "build",
        "--locked",
        "--release",
        "--workspace",
        "--bins",
        "--example",
        "press_snapshot",
        cwd=ROOT,
    )
    stage.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    cargo_target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release"
    shutil.copy2(cargo_target / f"fabric-viewer{suffix}", executable(stage, "fabric-viewer"))
    shutil.copy2(cargo_target / "examples" / f"press_snapshot{suffix}", executable(stage, "press_snapshot"))
    freeze(stage)
    shutil.copytree(
        ROOT / "examples/demo", stage / "examples/demo", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / name, stage / name)
    shutil.copyfile(ROOT / "packaging/README.txt", stage / "README.txt")
    (stage / "VERSION").write_text(version() + "\n")
    python_licenses(stage)
    rust_licenses(stage, target)


def smoke(stage: Path, gui: bool):
    expected = version()
    for command in ("fabric-viewer", "fabsim"):
        output = run(executable(stage, command), "--version", capture_output=True, text=True, timeout=60)
        if expected not in output.stdout:
            raise ValueError(f"{command} version mismatch: {output.stdout!r}")
    run(executable(stage, "fabsim"), "rig", "--paths", timeout=60)
    if gui:
        run(executable(stage, "fabsim-rig"), "--smoke-test", timeout=60)
    with tempfile.TemporaryDirectory(prefix="fabric stretch ") as temp:
        project = Path(temp) / "demo"
        shutil.copytree(stage / "examples/demo", project)
        run(executable(stage, "fabsim"), "validate", project, timeout=60)
        run(executable(stage, "fabsim"), "ps", project, "--force", "--jobs", "2", timeout=180)
        run(
            executable(stage, "fabsim"),
            "package",
            project,
            "--force",
            "--tiling-window-size",
            "8",
            "16",
            "--tiling-shift",
            "3",
            "--tiling-interp-dist",
            "1",
            timeout=180,
        )
        run(executable(stage, "fabsim"), "validate-package", project, timeout=60)
        image = Path(temp) / "snapshot.png"
        run(
            executable(stage, "press_snapshot"),
            project,
            "--relight",
            "--size",
            "320x240",
            "--out",
            image,
            timeout=60,
        )
        from PIL import Image

        with Image.open(image) as rendered:
            rendered.load()
            if rendered.size != (320, 240) or rendered.getextrema() == ((0, 0),) * 4:
                raise ValueError("snapshot smoke test produced no image")


def wix_source(stage: Path, output: Path):
    ns = "http://wixtoolset.org/schemas/v4/wxs"
    ET.register_namespace("", ns)

    def node(parent, tag, **attrs):
        return ET.SubElement(parent, f"{{{ns}}}{tag}", attrs)

    root = ET.Element(f"{{{ns}}}Wix")
    package = node(
        root,
        "Package",
        Name="Fabric Stretch",
        Manufacturer="fabric-stretch",
        Version=version(),
        UpgradeCode=UPGRADE_CODE,
        Scope="perMachine",
        InstallerVersion="500",
    )
    node(package, "MajorUpgrade", DowngradeErrorMessage="A newer version is already installed.")
    node(package, "MediaTemplate", EmbedCab="yes")
    program_files = node(package, "StandardDirectory", Id="ProgramFiles64Folder")
    install = node(program_files, "Directory", Id="INSTALLFOLDER", Name="Fabric Stretch")
    feature = node(package, "Feature", Id="Main", Title="Fabric Stretch", Level="1")
    directories = {Path("."): install}
    for source in sorted(stage.rglob("*")):
        relative = source.relative_to(stage)
        parent = directories[relative.parent]
        identifier = "p" + hashlib.sha256(relative.as_posix().encode()).hexdigest()[:24]
        if source.is_dir():
            directories[relative] = node(parent, "Directory", Id=identifier, Name=source.name)
        else:
            component = node(parent, "Component", Id=identifier, Guid="*")
            node(component, "File", Id="f" + identifier, Source=str(source.resolve()), KeyPath="yes")
            node(feature, "ComponentRef", Id=identifier)
    menu = node(
        node(package, "StandardDirectory", Id="ProgramMenuFolder"),
        "Directory",
        Id="ProgramMenuDir",
        Name="Fabric Stretch",
    )
    component = node(
        install, "Component", Id="Shortcuts", Guid=str(uuid.uuid5(uuid.UUID(UPGRADE_CODE), "shortcuts"))
    )
    for name, title in (("fabric-viewer", "Fabric Stretch Viewer"), ("fabsim-rig", "Fabric Stretch Rig")):
        node(
            component,
            "Shortcut",
            Id=name.replace("-", "_"),
            Name=title,
            Directory=menu.attrib["Id"],
            Target=f"[INSTALLFOLDER]{name}.exe",
            WorkingDirectory="INSTALLFOLDER",
        )
    node(component, "RemoveFolder", Id="RemoveMenu", Directory="ProgramMenuDir", On="uninstall")
    node(
        component,
        "RegistryValue",
        Root="HKLM",
        Key="Software\\FabricStretch",
        Name="Installed",
        Type="integer",
        Value="1",
        KeyPath="yes",
    )
    node(
        component,
        "Environment",
        Id="Path",
        Name="PATH",
        Value="[INSTALLFOLDER]",
        Action="set",
        Part="last",
        System="yes",
    )
    node(feature, "ComponentRef", Id="Shortcuts")
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def deb(stage: Path, output: Path):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        app = root / "opt/fabric-stretch"
        shutil.copytree(stage, app)
        binaries = root / "usr/bin"
        binaries.mkdir(parents=True)
        for name in COMMANDS:
            wrapper = binaries / name
            wrapper.write_text(f'#!/bin/sh\nexec /opt/fabric-stretch/{name} "$@"\n')
            wrapper.chmod(0o755)
        desktops = root / "usr/share/applications"
        desktops.mkdir(parents=True)
        for name, title in (("fabric-viewer", "Fabric Stretch Viewer"), ("fabsim-rig", "Fabric Stretch Rig")):
            (desktops / f"{name}.desktop").write_text(
                f"[Desktop Entry]\nType=Application\nName={title}\nExec={name}\n"
                "Terminal=false\nCategories=Science;\n"
            )
        control = root / "DEBIAN"
        control.mkdir()
        depends = (
            "libc6 (>= 2.35), libstdc++6, libgtk-3-0 | libgtk-3-0t64, libgl1, libegl1, libvulkan1, "
            "libxkbcommon0, libx11-6, libxcb1, libxcursor1, libxi6, libxrandr2, libxrender1, libxext6, "
            "libxfixes3, libxft2, libxss1, libfontconfig1, libfreetype6, libwayland-client0, libdbus-1-3"
        )
        size = sum(p.stat().st_size for p in app.rglob("*") if p.is_file()) // 1024
        (control / "control").write_text(
            f"Package: fabric-stretch\nVersion: {version()}\nArchitecture: amd64\n"
            "Maintainer: fabric-stretch authors\n"
            f"Depends: {depends}\nInstalled-Size: {size}\nSection: science\nPriority: optional\n"
            "Homepage: https://github.com/Personne-admin/fabric-stretch\n"
            "Description: Fabric appearance measurement and viewer\n"
        )
        run("dpkg-deb", "--root-owner-group", "--build", "-Zxz", root, output)


def pack(stage: Path, output: Path, target: str):
    output.mkdir(parents=True, exist_ok=True)
    basename = f"fabric-stretch-{version()}-{target}"
    if sys.platform == "win32":
        shutil.make_archive(str(output / basename), "zip", stage.parent, stage.name)
        wxs = stage.parent / "installer.wxs"
        wix_source(stage, wxs)
        run("wix", "build", "-arch", "x64", "-o", output / f"{basename}.msi", wxs)
        (output / f"{basename}.wixpdb").unlink(missing_ok=True)
    else:
        with tarfile.open(output / f"{basename}.tar.gz", "w:gz") as tar:
            tar.add(stage, arcname=stage.name)
        deb(stage, output / f"fabric-stretch_{version()}_amd64.deb")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checksums(folder: Path):
    names = sorted(
        p for p in folder.iterdir() if p.name.endswith((".zip", ".tar.gz", ".deb", ".msi", ".whl"))
    )
    if not names:
        raise ValueError("no release assets found")
    lines = [f"{digest(p)}  {p.name}\n" for p in names]
    (folder / "SHA256SUMS").write_text("".join(lines))


def verify_assets(folder: Path):
    current = version()
    required = {
        f"fabric-stretch-{current}-windows-x86_64.zip",
        f"fabric-stretch-{current}-windows-x86_64.msi",
        f"fabric-stretch-{current}-linux-x86_64.tar.gz",
        f"fabric-stretch_{current}_amd64.deb",
        f"fabsim-{current}-py3-none-any.whl",
        f"fabsim-{current}.tar.gz",
    }
    missing = required - {p.name for p in folder.iterdir() if p.is_file() and p.stat().st_size}
    if missing:
        raise ValueError(f"missing release assets: {', '.join(sorted(missing))}")


def notes(output: Path):
    changelog = (ROOT / "CHANGELOG.md").read_text()
    heading = f"## [{version()}]"
    if heading not in changelog:
        raise ValueError(f"CHANGELOG.md needs a {heading} entry")
    body = changelog.split(heading, 1)[1].split("\n## ", 1)[0].split("\n", 1)[1].strip()
    body += (
        "\n\nDownloads: Windows x64 `.msi` or portable `.zip`; Linux x64 `.deb` or `.tar.gz` "
        "(Ubuntu 22.04+/glibc 2.35+). Each includes the viewer, pipeline, rig app, and demo. "
        "Python is bundled. The camera still needs the IDS uEye driver.\n\n"
        "`SHA256SUMS` covers every asset. Install and build notes: "
        "https://github.com/Personne-admin/fabric-stretch/blob/master/docs/releases.md\n"
    )
    output.write_text(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("version", "build", "freeze", "smoke", "pack", "checksums", "verify-assets", "notes"),
    )
    parser.add_argument("--tag")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.command == "version":
        print(version(tag=args.tag))
        return
    version(tag=args.tag)
    stage = args.stage.resolve() if args.stage else None
    output = args.out.resolve() if args.out else None
    if args.command == "build":
        build(stage, args.target)
    elif args.command == "freeze":
        freeze(stage)
    elif args.command == "smoke":
        smoke(stage, args.gui)
    elif args.command == "pack":
        pack(stage, output, args.target)
    elif args.command == "checksums":
        checksums(output)
    elif args.command == "verify-assets":
        verify_assets(output)
    elif args.command == "notes":
        notes(output)


if __name__ == "__main__":
    main()
