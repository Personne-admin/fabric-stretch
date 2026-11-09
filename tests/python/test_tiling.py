from __future__ import annotations

import cv2
import numpy as np
import pytest

from fabsim import photometric as ph
from fabsim import tiling


@pytest.fixture(scope="module")
def solved():
    rng = np.random.default_rng(5)
    base = rng.integers(0, 256, (96, 120, 3), dtype=np.uint8)
    textures = [np.clip(base.astype(int) + 10 * i, 0, 255).astype(np.uint8) for i in range(4)]

    cfg = tiling.TilingConfig(window_size=(8, 8), shift=2, interp_dist=1)
    tiles, meta, details = tiling.tile_states(textures, cfg)

    return textures, tiles, meta, tiling.Seams.from_details(details, (120, 96))


def test_apply_reproduces_the_joint_tiles(solved):
    textures, tiles, _, seams = solved
    for texture, tile in zip(textures, tiles, strict=True):
        assert np.array_equal(tiling.apply_tiling(texture, seams), tile)


def test_seams_round_trip(solved, tmp_path):
    textures, tiles, _, seams = solved

    seams.save(tmp_path / "s.npz")
    loaded = tiling.Seams.load(tmp_path / "s.npz")

    assert loaded.area == seams.area and loaded.atlas_size == seams.atlas_size
    assert np.array_equal(tiling.apply_tiling(textures[2], loaded), tiles[2])


def test_dtype_and_gray_layers(solved):
    textures, tiles, _, seams = solved

    gray16 = (textures[0][..., 0].astype(np.uint16)) * 257
    out = tiling.apply_tiling(gray16, seams)

    assert out.dtype == np.uint16 and out.shape == tiles[0].shape[:2]
    assert tiling.apply_tiling(textures[0].astype(np.float32) / 255, seams).dtype == np.float32


def test_normals_are_renormalised(solved):
    _, tiles, _, seams = solved

    rng = np.random.default_rng(1)
    n = rng.normal(size=(96, 120, 3)).astype(np.float32)
    n[..., 2] = np.abs(n[..., 2]) + 2
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    encoded = ph.encode_normals_u16(n)

    cut = ph.decode_normals_u16(tiling.apply_tiling(encoded, seams, normals=True))

    assert cut.shape == tiles[0].shape
    assert np.abs(np.linalg.norm(cut, axis=-1) - 1).max() < 1e-3

    plain = ph.decode_normals_u16(tiling.apply_tiling(encoded, seams))
    assert np.linalg.norm(plain, axis=-1).min() < 0.999  # blending alone shortens the vectors


def test_layer_of_the_wrong_size_is_refused(solved):
    _, _, _, seams = solved
    with pytest.raises(ValueError, match="atlas"):
        tiling.apply_tiling(np.zeros((50, 60, 3), np.uint8), seams)


def test_guidance_indices_must_exist(solved):
    textures = solved[0]
    for guidance in ([0, 4], [-1], [99]):
        cfg = tiling.TilingConfig(window_size=(8, 8), shift=2, interp_dist=1, guidance=guidance)
        with pytest.raises(ValueError, match="out of range"):
            tiling.tile_states(textures, cfg)

    cfg = tiling.TilingConfig(window_size=(8, 8), shift=2, interp_dist=1, guidance=[0, 3])
    assert tiling.tile_states(textures, cfg)[1]["guidance_state_indices"] == [0, 3]


def test_seam_shift_finds_a_rib_offset():
    rng = np.random.default_rng(0)

    profile = cv2.GaussianBlur(rng.random((120, 1)).astype(np.float32), (0, 0), 2.0).ravel()
    profile = (profile - profile.min()) / (profile.max() - profile.min())
    ribs = np.tile(np.rint(255 * profile).astype(np.uint8)[:, None, None], (1, 160, 3))

    assert tiling.seam_shift_px(ribs)[0] == 0
    ribs[:, 80:] = np.roll(ribs[:, 80:], 6, axis=0)
    assert tiling.seam_shift_px(ribs)[0] == 6
