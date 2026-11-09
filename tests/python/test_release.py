import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release", ROOT / "scripts/release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_version_matches_tag_and_component_versions(tmp_path):
    package = tmp_path / "src/fabsim"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n')
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text('[workspace.package]\nversion = "1.2.3"\n')
    assert release.version(tmp_path, "v1.2.3") == "1.2.3"
    with pytest.raises(ValueError, match="tag"):
        release.version(tmp_path, "v1.2.4")
    cargo.write_text('[workspace.package]\nversion = "1.2.4"\n')
    with pytest.raises(ValueError, match="versions differ"):
        release.version(tmp_path)


def test_checksums_cover_assets_in_name_order(tmp_path):
    for name in ("z.zip", "a.deb", "source.tar.gz", "package.whl"):
        (tmp_path / name).write_bytes(b"hello\n")
    (tmp_path / "notes.md").write_text("not an asset")
    release.checksums(tmp_path)
    lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert [line.split("  ")[1] for line in lines] == ["a.deb", "package.whl", "source.tar.gz", "z.zip"]
    assert all(
        line.startswith("5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03  ")
        for line in lines
    )
    before = (tmp_path / "SHA256SUMS").read_bytes()
    release.checksums(tmp_path)
    assert (tmp_path / "SHA256SUMS").read_bytes() == before


def test_installer_includes_nested_files_and_stable_upgrade_code(tmp_path):
    stage = tmp_path / "stage"
    nested = stage / "examples/demo & test"
    nested.mkdir(parents=True)
    for path in (stage / "fabric-viewer.exe", nested / "a & b.json"):
        path.write_text("test")
    output = tmp_path / "installer.wxs"
    release.wix_source(stage, output)
    ns = {"w": "http://wixtoolset.org/schemas/v4/wxs"}
    tree = ET.parse(output)
    assert tree.find("w:Package", ns).attrib["UpgradeCode"] == release.UPGRADE_CODE
    sources = {node.attrib["Source"] for node in tree.findall(".//w:File", ns)}
    assert sources == {str(p.resolve()) for p in stage.rglob("*") if p.is_file()}
    assert len(tree.findall(".//w:Shortcut", ns)) == 2


def test_publish_requires_both_platforms_and_python_distributions(tmp_path):
    with pytest.raises(ValueError, match="missing release assets"):
        release.verify_assets(tmp_path)
    current = release.version()
    for name in (
        f"fabric-stretch-{current}-windows-x86_64.zip",
        f"fabric-stretch-{current}-windows-x86_64.msi",
        f"fabric-stretch-{current}-linux-x86_64.tar.gz",
        f"fabric-stretch_{current}_amd64.deb",
        f"fabsim-{current}-py3-none-any.whl",
        f"fabsim-{current}.tar.gz",
    ):
        (tmp_path / name).write_bytes(b"asset")
    release.verify_assets(tmp_path)
