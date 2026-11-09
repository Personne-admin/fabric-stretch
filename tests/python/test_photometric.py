from __future__ import annotations

import json

import numpy as np
import pytest

from fabsim import photometric as ph

CFG = ph.LightConfig(
    azimuths_deg=(91.27, 334.26, 211.16), elevation_deg=45.0, source="test"
)


def _sphere(size=96, albedo=0.6):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (xs - size / 2) / (size * 0.4)
    y = -(ys - size / 2) / (size * 0.4)  # image up = +y
    r2 = x * x + y * y
    inside = r2 < 0.9
    z = np.sqrt(np.clip(1 - r2, 0, None))
    n = np.stack([x, y, z], axis=-1)
    n[~inside] = (0, 0, 1)
    a = np.where(inside, albedo * (0.8 + 0.2 * np.cos(6 * x)), 0.5).astype(np.float32)

    return a, n.astype(np.float32), inside


def test_packaged_default_config_loads():
    cfg = ph.load_light_config()
    assert ph.default_led_config_path().is_file()
    assert len(cfg.azimuths_deg) == 3
    assert cfg.elevation_deg == pytest.approx(45.0)
    assert cfg.gains is None
    assert cfg.source.startswith("packaged")


def test_config_precedence(tmp_path):
    f = tmp_path / "led.json"
    f.write_text(
        json.dumps({"light_azimuths_deg": [0, 120, 240], "light_elevation_deg": 30})
    )
    assert ph.load_light_config(f).azimuths_deg == (0.0, 120.0, 240.0)
    session = {
        "led_config": {
            "light_azimuths_deg": [10, 20, 30],
            "light_elevation_deg": 50,
            "gains": [1, 1, 1],
        }
    }
    cfg = ph.load_light_config(f, session)
    assert cfg.azimuths_deg == (10.0, 20.0, 30.0) and cfg.elevation_deg == 50.0
    assert cfg.source == "session.json:led_config"
    assert cfg.gains == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        ph.load_light_config(session={"led_config": {"light_azimuths_deg": [1, 2]}})


def test_light_matrix_is_unit_and_ccw_from_right():
    L = ph.light_matrix(ph.LightConfig((0.0, 90.0, 180.0), 45.0))
    np.testing.assert_allclose(np.linalg.norm(L, axis=1), 1.0)
    np.testing.assert_allclose(
        L[0], [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)], atol=1e-12
    )
    assert L[1, 1] > 0.7 and L[2, 0] < -0.7  # 90 deg = image up, 180 deg = image left


def test_sphere_round_trip():
    albedo, n, inside = _sphere()
    ps = ph.render(albedo, n, CFG)

    # keep pixels lit by all three LEDs so the linear model is exact
    lit = inside & np.all(n @ ph.light_matrix(CFG).T.astype(np.float32) > 0.02, axis=-1)
    a2, n2 = ph.solve(ps, CFG)

    assert lit.sum() > 500

    np.testing.assert_allclose(a2[lit], albedo[lit], atol=1e-4)
    np.testing.assert_allclose(n2[lit], n[lit], atol=1e-4)

    # re-rendering the solution reproduces all pixels, including the shadowed ones, within 1/255
    again = ph.render(a2, n2, CFG)
    assert np.abs(again - ps).mean() < 1 / 255


def test_gains_estimation_and_solve():
    albedo = np.full((64, 64), 0.5, np.float32)
    n = np.zeros((64, 64, 3), np.float32)
    n[..., 2] = 1.0  # flat cloth facing the camera

    gains = np.array([1.0, 0.65, 1.2])
    gains /= gains.mean()

    ps = ph.render(albedo, n, CFG, gains)
    est = ph.estimate_gains(ps)

    np.testing.assert_allclose(est, gains, atol=1e-5)
    _, n_plain = ph.solve(ps, CFG)
    _, n_fixed = ph.solve(ps, CFG, est)

    tilt = lambda m: np.degrees(
        np.arccos(np.clip(m[..., 2].mean(), -1, 1))
    )  # noqa: E731

    assert tilt(n_plain) > 5
    assert tilt(n_fixed) < 0.1

    # cfg.gains is used when no explicit gains are passed
    _, n_cfg = ph.solve(ps, CFG.with_gains(est))
    np.testing.assert_allclose(n_cfg, n_fixed, atol=1e-6)


def test_estimate_gains_respects_support_and_empty():
    ps = np.zeros((8, 8, 3), np.float32)
    ps[:4] = (0.2, 0.4, 0.6)
    ps[4:] = 0.9

    support = np.zeros((8, 8), bool)
    support[:4] = True

    np.testing.assert_allclose(
        ph.estimate_gains(ps, support), np.array([0.2, 0.4, 0.6]) / 0.4, atol=1e-6
    )

    np.testing.assert_array_equal(
        ph.estimate_gains(ps, np.zeros((8, 8), bool)), np.ones(3)
    )


def test_normal_encoding_round_trip():
    rng = np.random.default_rng(0)

    n = rng.normal(size=(50, 50, 3)).astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    v = ph.encode_normals_u16(n)

    assert v.dtype == np.uint16
    assert np.abs(ph.decode_normals_u16(v) - n).max() <= 1.0 / 65535 + 1e-6

    # rounding, not flooring: 0 maps to 32767.5 -> 32768
    assert ph.encode_normals_u16(np.zeros(1))[0] == 32768
    assert ph.encode_normals_u16(np.array([-1.0, 1.0])).tolist() == [0, 65535]


def test_albedo_encoding_round_trip_and_scale():
    rng = np.random.default_rng(1)

    a = rng.uniform(0, 0.8, size=(40, 40, 3)).astype(np.float32)
    a[0, 0] = 5.0  # an outlier must not set the scale

    scale = ph.albedo_scale(a)
    assert 0.7 < scale < 0.81

    v = ph.encode_albedo_u16(a, scale)
    back = ph.decode_albedo_u16(v, scale)
    ok = a <= scale

    assert np.abs(back[ok] - a[ok]).max() <= scale / 65535
    assert (v[a >= scale] == 65535).all()

    u8 = ph.encode_albedo_u8(a, scale)
    assert (
        u8.dtype == np.uint8
        and np.abs(u8[ok] / 255 * scale - a[ok]).max() <= scale / 255 / 2 + 1e-6
    )
    assert ph.albedo_scale(np.zeros((4, 4, 3))) == 1.0


def test_color_albedo_keeps_luma():
    gray = np.full((4, 4), 0.4, np.float32)
    diffuse = np.zeros((4, 4, 3), np.float32)
    diffuse[...] = (0.8, 0.4, 0.2)

    rgb = ph.color_albedo(gray, diffuse)
    np.testing.assert_allclose(rgb @ ph.LUMA, 0.4, atol=1e-6)

    assert rgb[0, 0, 0] > rgb[0, 0, 1] > rgb[0, 0, 2]


def test_solve_rejects_bad_shape():
    with pytest.raises(ValueError):
        ph.solve(np.zeros((4, 4, 2)), CFG)


@pytest.mark.realdata
def test_real_frame_rerender_matches_capture(data_dir):
    run = data_dir / "fabric_002" / "warp_loading"
    frame = json.loads((run / "frames.json").read_text())["frames"][0]
    ps = ph.read_ps_gray([run / "images" / frame[f"image_ps_{i}"] for i in (1, 2, 3)])
    cfg = ph.load_light_config()
    support = ph.support_mask(ps)
    cfg = cfg.with_gains(ph.estimate_gains(ps, support))
    albedo, n = ph.solve(ps, cfg)
    again = ph.render(albedo, n, cfg)
    assert np.abs(again - ps).mean() < 2 / 255


@pytest.mark.realdata
def test_real_mm003_gains_remove_tilt(data_dir):
    run = data_dir / "mm003" / "warp_loading"
    frame = json.loads((run / "frames.json").read_text())["frames"][0]
    ps = ph.read_ps_gray([run / "images" / frame[f"image_ps_{i}"] for i in (1, 2, 3)])
    cfg = ph.load_light_config()
    support = ph.support_mask(ps)

    def tilt_deg(gains):
        _, n = ph.solve(ps, cfg, gains)
        mean = n[support].mean(axis=0)

        return float(np.degrees(np.arccos(mean[2] / np.linalg.norm(mean))))

    gains = ph.estimate_gains(ps, support)
    assert gains.min() < 0.85  # the dim LED is detected
    assert tilt_deg(None) > 6
    assert tilt_deg(gains) < 3


def test_shade_is_linear_clamped_and_gain_is_optional():
    up = np.array([0.0, 0.0, 1.0], np.float32)
    n = np.broadcast_to(up, (2, 2, 3))
    albedo = np.full((2, 2, 3), 0.5, np.float32)
    np.testing.assert_allclose(ph.shade(albedo, n, up), 0.5)

    d = ph.light_direction(0.0, 45.0)
    np.testing.assert_allclose(
        ph.shade(albedo, n, d), 0.5 * np.cos(np.pi / 4), atol=1e-6
    )

    # ambient fills in when the light is behind the surface; gain scales the diffuse term only
    np.testing.assert_allclose(ph.shade(albedo, n, -up, ambient=0.2), 0.1, atol=1e-6)
    np.testing.assert_allclose(ph.shade(albedo, n, up, gain=0.5), 0.25)
    np.testing.assert_allclose(ph.shade(albedo, n, up, intensity=9.0), 1.0)  # clamps


def test_light_direction_frame():
    np.testing.assert_allclose(
        ph.light_direction(90.0, 0.0), [0, 1, 0], atol=1e-12
    )  # image up

    np.testing.assert_allclose(ph.light_direction(0.0, 90.0), [0, 0, 1], atol=1e-12)
    assert ph.light_matrix(CFG).shape == (3, 3)


def test_remove_normal_tilt_removes_the_ramp_and_keeps_detail():
    size = 120
    rng = np.random.default_rng(0)

    bump = np.zeros((size, size), np.float32)
    bump[50:70, 50:70] = 0.2  # a local feature that is much smaller than the image

    ramp = np.tile(np.linspace(-0.2, 0.2, size, dtype=np.float32), (size, 1))
    n = np.dstack(
        [
            ramp + bump,
            0.1 + 0.01 * rng.standard_normal((size, size)),
            np.ones((size, size)),
        ]
    )

    n = (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)
    fixed, tilt = ph.remove_normal_tilt(n)
    assert tilt == pytest.approx(
        np.degrees(np.arctan(0.1)), abs=0.5
    )  # mean tilt comes from the y offset

    assert np.abs(np.linalg.norm(fixed, axis=-1) - 1).max() < 1e-5
    assert ph.mean_normal_tilt_deg(fixed) < 0.7

    columns = fixed[..., 0].mean(axis=0)
    assert columns[:10].mean() - columns[-10:].mean() == pytest.approx(0, abs=0.02)
    assert fixed[60, 60, 0] - fixed[60, 20, 0] > 0.15  # the bump survives

    flat = np.broadcast_to(np.float32([0, 0, 1]), (30, 40, 3))
    same, zero = ph.remove_normal_tilt(flat)
    np.testing.assert_allclose(same, flat, atol=1e-6)
    assert zero == 0


def test_exposure_gain_scales_a_head_on_relight_to_the_diffuse_photo():
    albedo, normals, support = _sphere()
    diffuse = np.dstack([albedo] * 3) * 2.5
    factor = ph.exposure_gain(albedo, normals, diffuse, support)

    head_on = (albedo * np.maximum(normals[..., 2], 0))[support].mean() * factor
    assert head_on == pytest.approx((diffuse @ ph.LUMA)[support].mean(), rel=1e-4)
    assert (
        factor > 2.5
    )  # the sphere's normals tilt away, so the head-on light is dimmer than the albedo

    assert (
        ph.exposure_gain(
            np.zeros((4, 4)),
            np.broadcast_to([0, 0, 1.0], (4, 4, 3)),
            np.ones((4, 4, 3)),
        )
        == 1.0
    )


def test_albedo_correlation_flags_a_negative_and_a_flat_capture():
    albedo, _, support = _sphere()
    diffuse = np.dstack([albedo] * 3)

    assert ph.albedo_correlation(albedo, diffuse, support) == pytest.approx(
        1.0, abs=1e-4
    )

    assert ph.albedo_correlation(albedo, 1 - diffuse, support) == pytest.approx(
        -1.0, abs=1e-4
    )

    assert ph.albedo_correlation(np.ones((8, 8)), np.ones((8, 8, 3))) == 0.0
