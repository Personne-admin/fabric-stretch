import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

ADAPTER_PATH = Path(__file__).resolve().parents[2] / "src" / "fabsim" / "cowtracker_to_fabsim.py"


def _import_adapter():
    try:
        return importlib.import_module("fabsim.cowtracker_to_fabsim")
    except ImportError:
        name = "cowtracker_to_fabsim"
        spec = importlib.util.spec_from_file_location(name, ADAPTER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolves annotations via sys.modules
        spec.loader.exec_module(module)

        return module


ct = _import_adapter()

SRC_W, SRC_H = 1280, 1024
INF_W, INF_H = 560, 336


def make_points(positions):
    return [ct.TrackPoint(id=f"p{i}", x=x, y=y) for i, (x, y) in enumerate(positions)]


class CoordinateTransformTests(unittest.TestCase):
    def assert_round_trip(self, x, y, convention, places=6):
        xi, yi = ct.source_to_inference(x, y, SRC_W, SRC_H, INF_W, INF_H, convention)
        xs, ys = ct.inference_to_source(xi, yi, SRC_W, SRC_H, INF_W, INF_H, convention)
        self.assertAlmostEqual(xs, x, places=places, msg=f"x round trip ({convention})")
        self.assertAlmostEqual(ys, y, places=places, msg=f"y round trip ({convention})")

    def test_pixel_center_top_left(self):
        xi, yi = ct.source_to_inference(0.0, 0.0, SRC_W, SRC_H, INF_W, INF_H)
        # (0 + 0.5) * 560/1280 - 0.5 = 0.21875 - 0.5 = -0.28125
        self.assertAlmostEqual(xi, -0.28125, places=5)
        self.assertAlmostEqual(yi, (0.5) * INF_H / SRC_H - 0.5, places=5)

    def test_pixel_center_center(self):
        xi, yi = ct.source_to_inference(SRC_W / 2 - 0.5, SRC_H / 2 - 0.5, SRC_W, SRC_H, INF_W, INF_H)
        self.assertAlmostEqual(xi, INF_W / 2 - 0.5, places=5)
        self.assertAlmostEqual(yi, INF_H / 2 - 0.5, places=5)

    def test_pixel_center_bottom_right(self):
        xi, yi = ct.source_to_inference(SRC_W - 1.0, SRC_H - 1.0, SRC_W, SRC_H, INF_W, INF_H)
        self.assertAlmostEqual(xi, (SRC_W - 0.5) * INF_W / SRC_W - 0.5, places=5)
        self.assertAlmostEqual(yi, (SRC_H - 0.5) * INF_H / SRC_H - 0.5, places=5)

    def test_pixel_center_round_trips(self):
        for x, y in [
            (0.0, 0.0),
            (SRC_W - 1.0, SRC_H - 1.0),
            (SRC_W / 2 - 0.5, SRC_H / 2 - 0.5),
            (123.456, 789.012),
            (1279.5, 3.25),
            (0.25, 1023.75),
        ]:
            self.assert_round_trip(x, y, ct.PIXEL_CENTER)

    def test_corner_aligned_round_trips(self):
        for x, y in [
            (0.0, 0.0),
            (SRC_W - 1.0, SRC_H - 1.0),
            (SRC_W / 2 - 0.5, SRC_H / 2 - 0.5),
            (123.456, 789.012),
        ]:
            xi, yi = ct.source_to_inference(x, y, SRC_W, SRC_H, INF_W, INF_H, ct.CORNER_ALIGNED)
            self.assertAlmostEqual(xi, x * INF_W / SRC_W, places=6)
            xs, ys = ct.inference_to_source(xi, yi, SRC_W, SRC_H, INF_W, INF_H, ct.CORNER_ALIGNED)
            self.assertAlmostEqual(xs, x, places=6)
            self.assertAlmostEqual(ys, y, places=6)

    def test_conventions_differ_by_half_pixel(self):
        xi_pc, yi_pc = ct.source_to_inference(0.0, 0.0, SRC_W, SRC_H, INF_W, INF_H, ct.PIXEL_CENTER)
        xi_ca, yi_ca = ct.source_to_inference(0.0, 0.0, SRC_W, SRC_H, INF_W, INF_H, ct.CORNER_ALIGNED)
        # pc: (c+0.5)*s - 0.5 ; ca: c*s  ->  pc = ca + 0.5*(s - 1)
        self.assertAlmostEqual(xi_pc, xi_ca + 0.5 * (INF_W / SRC_W - 1.0), places=6)
        self.assertAlmostEqual(yi_pc, yi_ca + 0.5 * (INF_H / SRC_H - 1.0), places=6)

    def test_bad_convention_and_dimensions_raise(self):
        with self.assertRaises(ValueError):
            ct.source_to_inference(1.0, 1.0, SRC_W, SRC_H, INF_W, INF_H, "bogus")
        with self.assertRaises(ValueError):
            ct.source_to_inference(1.0, 1.0, 0, SRC_H, INF_W, INF_H)


class PixelCenterResizeTests(unittest.TestCase):
    def test_identity_size_is_noop(self):
        img = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
        out = ct.pixel_center_resize(img, 20, 30)
        np.testing.assert_array_equal(out, img)

    def test_downscale_of_linear_image_is_exact(self):
        xs, ys = np.meshgrid(np.arange(SRC_W), np.arange(SRC_H), indexing="xy")
        img = (xs.astype(np.float64) * 0.5 + ys.astype(np.float64) * 0.25).astype(np.float32)
        out = ct.pixel_center_resize(img, INF_H, INF_W)
        self.assertEqual(out.shape, (INF_H, INF_W))
        # Expected output value: linear field evaluated at the sample points.
        x_in = (np.arange(INF_W, dtype=np.float64) + 0.5) * (SRC_W / INF_W) - 0.5
        y_in = (np.arange(INF_H, dtype=np.float64) + 0.5) * (SRC_H / INF_H) - 0.5
        expected = x_in[None, :] * 0.5 + y_in[:, None] * 0.25
        np.testing.assert_allclose(out, expected.astype(np.float32), atol=1e-4)

    def test_border_pixels_are_true_edge_clamped(self):
        w, h = 10, 6
        # Value == column index, constant along rows.
        img = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        out = ct.pixel_center_resize(img, h, 2 * w)  # 2x upscale
        self.assertEqual(out.shape, (h, 2 * w))
        # ox=0 samples source x_in = -0.25 -> clamped to 0 -> value 0.0
        # (the old blend produced 0.75 * p0 + 0.25 * p1 = 0.25).
        np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-6)
        # ox=1 samples x_in = 0.25 -> interior bilinear value 0.25.
        np.testing.assert_allclose(out[:, 1], 0.25, atol=1e-6)
        # ox=2*w-1 samples x_in = 9.25 -> clamped to the last column -> 9.0.
        np.testing.assert_allclose(out[:, -1], float(w - 1), atol=1e-6)


class SyntheticDenseFieldTests(unittest.TestCase):
    def _field(self, T):
        gy, gx = np.meshgrid(np.arange(INF_H), np.arange(INF_W), indexing="ij")
        track = np.stack([gx, gy], axis=-1)[None].astype(np.float64)  # [1, Hi, Wi, 2]
        track = np.repeat(track, T, axis=0)
        for t in range(T):
            track[t, :, :, 0] += 3.0 * t
            track[t, :, :, 1] += 5.0 * t

        return track.astype(np.float32)

    def test_sampling_recovers_linear_motion(self):
        T = 5
        field = self._field(T)

        # Queries map inside the model's pixel-center span [0, Wi-1]x[0, Hi-1].
        queries = make_points([(100.0, 80.0), (640.0, 511.5), (1200.0, 950.25), (1270.0, 1020.0)])

        query_src = np.asarray([[p.x, p.y] for p in queries])
        query_inf = np.asarray(
            [ct.source_to_inference(p.x, p.y, SRC_W, SRC_H, INF_W, INF_H) for p in queries]
        )

        sampled_inf = ct.sample_dense_field(field, query_inf[:, 0], query_inf[:, 1])
        self.assertEqual(sampled_inf.shape, (T, len(queries), 2))
        track_src = np.empty_like(sampled_inf, dtype=np.float64)
        for t in range(T):
            for n in range(len(queries)):
                track_src[t, n] = ct.inference_to_source(
                    sampled_inf[t, n, 0],
                    sampled_inf[t, n, 1],
                    SRC_W,
                    SRC_H,
                    INF_W,
                    INF_H,
                )

        np.testing.assert_allclose(track_src[0], query_src, atol=1e-4)

        # Linear motion: each frame moves 3*scale_x px in x, 5*scale_y in y.
        sx, sy = INF_W / SRC_W, INF_H / SRC_H
        for t in range(T):
            np.testing.assert_allclose(track_src[t, :, 0], query_src[:, 0] + 3.0 * t / sx, atol=1e-3)
            np.testing.assert_allclose(track_src[t, :, 1], query_src[:, 1] + 5.0 * t / sy, atol=1e-3)

    def test_reference_error_is_zero_for_identity_reference(self):
        T = 4
        field = self._field(T)
        queries = make_points([(10.0, 20.0), (300.0, 400.0), (900.0, 700.0)])
        query_src = np.asarray([[p.x, p.y] for p in queries])
        query_inf = np.asarray(
            [ct.source_to_inference(p.x, p.y, SRC_W, SRC_H, INF_W, INF_H) for p in queries]
        )

        sampled = ct.sample_dense_field(field, query_inf[:, 0], query_inf[:, 1])
        track_src = np.empty_like(sampled, dtype=np.float64)
        for t in range(T):
            for n in range(len(queries)):
                track_src[t, n] = ct.inference_to_source(
                    sampled[t, n, 0], sampled[t, n, 1], SRC_W, SRC_H, INF_W, INF_H
                )

        err = ct.reference_error_px(track_src, query_src, ref_seq_index=0)
        np.testing.assert_allclose(err, 0.0, atol=1e-3)

    def test_out_of_field_query_clamps_to_edge_with_exact_error(self):
        T = 3
        field = self._field(T)

        # Source x = 0.0 maps to inference x = -0.28125 (outside [0, Wi-1]).
        query = make_points([(0.0, 512.0)])
        query_src = np.asarray([[p.x, p.y] for p in query])
        query_inf = np.asarray([ct.source_to_inference(p.x, p.y, SRC_W, SRC_H, INF_W, INF_H) for p in query])
        self.assertLess(query_inf[0, 0], 0.0)
        sampled = ct.sample_dense_field(field, query_inf[:, 0], query_inf[:, 1])

        np.testing.assert_allclose(sampled[0, 0, 0], 0.0, atol=1e-6)
        track_src = np.empty_like(sampled, dtype=np.float64)
        for t in range(T):
            track_src[t, 0] = ct.inference_to_source(
                sampled[t, 0, 0], sampled[t, 0, 1], SRC_W, SRC_H, INF_W, INF_H
            )

        err = ct.reference_error_px(track_src, query_src, 0)

        # x_s of clamped inference x_i = 0: (0 + 0.5) * SRC_W/INF_W - 0.5.
        expected_xs = 0.5 * SRC_W / INF_W - 0.5
        np.testing.assert_allclose(track_src[0, 0, 0], expected_xs, atol=1e-4)
        self.assertAlmostEqual(err[0], expected_xs, places=5)

        # The preflight warning still flags such points.
        notes = ct.validate_points_in_image(query, SRC_W, SRC_H, INF_W, INF_H, ct.PIXEL_CENTER)
        self.assertTrue(any("inference field" in n for n in notes))

    def test_rotation_remap_anchors_reference(self):
        T = 5
        ref = 3
        rot = ct.compute_rotation(ref, T)
        self.assertEqual(list(rot), [2, 3, 4, 0, 1])
        order = list(np.argsort(rot))
        self.assertEqual(order, [3, 4, 0, 1, 2])

        # Rotated-order dense field anchored at rotated position 0.
        field = self._field(T)
        query_inf = np.asarray([ct.source_to_inference(500.0, 400.0, SRC_W, SRC_H, INF_W, INF_H)])

        # Sample in rotated order, then remap back to original order.
        sampled_rot = ct.sample_dense_field(field, query_inf[:, 0], query_inf[:, 1])
        remapped = np.stack([sampled_rot[int(p)] for p in rot], axis=0)
        # Original frame `ref` was rotated to position 0 -> identity.
        np.testing.assert_allclose(remapped[ref, 0], query_inf[0], atol=1e-6)
        # Original frame 0 was rotated to position 2 -> 2 steps of motion.
        np.testing.assert_allclose(remapped[0, 0, 0], query_inf[0, 0] + 6.0, atol=1e-6)
        # Original frame 4 -> 1 step of motion.
        np.testing.assert_allclose(remapped[4, 0, 0], query_inf[0, 0] + 3.0, atol=1e-6)


class DenseFieldEdgeClampingTests(unittest.TestCase):
    W, H = 10, 8

    def _x_ramp(self, T=1):
        # Nonconstant along x (value == column index), constant in y and t.
        gy, gx = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        field = gx.astype(np.float32)[None, :, :, None]

        return np.repeat(field, T, axis=0)  # [T, H, W, 1]

    def test_low_side_clamps_to_first_column(self):
        field = self._x_ramp()
        # Negative half-pixel coordinates (x = -0.5, x = -3) must sample the
        # first column exactly -- never blend columns 0 and 1 inward.
        out = ct.sample_dense_field(field, np.array([-0.5, -3.0]), np.array([2.5, 2.5]))
        np.testing.assert_allclose(out[0, :, 0], [0.0, 0.0], atol=1e-6)
        # In-range control: plain bilinear at x = 2.5 gives 2.5.
        mid = ct.sample_dense_field(field, np.array([2.5]), np.array([2.5]))
        np.testing.assert_allclose(mid[0, 0, 0], 2.5, atol=1e-6)

    def test_high_side_clamps_to_last_column(self):
        field = self._x_ramp()
        # x = W - 0.5 maps beyond the last pixel center; x = W + 5 far beyond.
        # Both must sample the last column exactly (value W - 1).
        out = ct.sample_dense_field(field, np.array([self.W - 0.5, self.W + 5.0]), np.array([3.0, 3.0]))
        np.testing.assert_allclose(out[0, :, 0], [self.W - 1, self.W - 1], atol=1e-6)

    def test_y_edges_clamp_to_first_and_last_row(self):
        # Nonconstant along y: value == row index.
        gy, gx = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        field = gy.astype(np.float32)[None, :, :, None]
        out = ct.sample_dense_field(field, np.array([1.5, 1.5]), np.array([-0.5, self.H + 2.0]))

        np.testing.assert_allclose(out[0, :, 0], [0.0, self.H - 1], atol=1e-6)

    def test_3d_field_edge_clamp_preserves_shape(self):
        field = np.zeros((2, self.H, self.W), dtype=np.float32)
        out = ct.sample_dense_field(field, np.array([-2.0]), np.array([self.H + 1.0]))
        self.assertEqual(out.shape, (2, 1))
        np.testing.assert_allclose(out, 0.0)


class SequenceBuildingTests(unittest.TestCase):
    def _write_run(self, frames_doc, images=None):
        tmp = tempfile.mkdtemp(prefix="ct_adapt_")
        os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
        with open(os.path.join(tmp, "frames.json"), "w") as fh:
            json.dump(frames_doc, fh)

        for name, data in (images or {}).items():
            with open(os.path.join(tmp, "images", name), "wb") as fh:
                fh.write(data)

        return tmp

    def _png(self, w=4, h=4):
        import struct
        import zlib

        # Minimal valid PNG (RGBA). Contents do not matter for ordering tests.
        raw = b"".join(b"\x00" + bytes([10, 20, 30, 255]) * w for _ in range(h))

        def chunk(typ, payload):
            c = struct.pack(">I", len(payload)) + typ + payload
            c += struct.pack(">I", zlib.crc32(typ + payload) & 0xFFFFFFFF)

            return c

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )

    def test_order_and_mapping_follow_frames_json(self):
        doc = {
            "frames": [
                {"frame_id": 7, "image_diffuse": "a.png"},
                {"frame_id": 3, "image_diffuse": "b.png"},
                {"frame_id": 11, "image_diffuse": "c.png"},
            ]
        }
        png = self._png()
        tmp = self._write_run(doc, {"a.png": png, "b.png": png, "c.png": png})
        frames, skipped = ct.build_sequence(tmp, "diffuse", skip_missing=False)
        self.assertEqual(len(skipped), 0)
        self.assertEqual([f.frame_id for f in frames], [7, 3, 11])
        self.assertEqual([f.sequence_index for f in frames], [0, 1, 2])
        self.assertTrue(all(os.path.isfile(f.image_path) for f in frames))
        import shutil

        shutil.rmtree(tmp)

    def test_missing_role_rejected_by_default(self):
        doc = {
            "frames": [
                {"frame_id": 0, "image_diffuse": "a.png"},
                {"frame_id": 1},  # no image_diffuse
                {"frame_id": 2, "image_diffuse": "c.png"},
            ]
        }
        png = self._png()
        tmp = self._write_run(doc, {"a.png": png, "c.png": png})
        with self.assertRaises(ct.AdapterError) as ctx:
            ct.build_sequence(tmp, "diffuse", skip_missing=False)
        self.assertIn("--skip-missing", str(ctx.exception))
        import shutil

        shutil.rmtree(tmp)

    def test_skip_missing_reports_mapping(self):
        doc = {
            "frames": [
                {"frame_id": 0, "image_diffuse": "a.png"},
                {"frame_id": 1},
                {"frame_id": 2, "image_diffuse": "c.png"},
                {"frame_id": 3, "image_ps_1": "x.png"},  # wrong role
            ]
        }
        png = self._png()
        tmp = self._write_run(doc, {"a.png": png, "c.png": png})
        frames, skipped = ct.build_sequence(tmp, "diffuse", skip_missing=True)
        self.assertEqual([f.frame_id for f in frames], [0, 2])
        # Skipped frames keep their sequence position for the report.
        self.assertEqual(
            [(s["sequence_index"], s["frame_id"]) for s in skipped],
            [(1, 1), (3, 3)],
        )
        import shutil

        shutil.rmtree(tmp)

    def test_missing_image_file_handled(self):
        doc = {"frames": [{"frame_id": 0, "image_diffuse": "gone.png"}]}
        tmp = self._write_run(doc, {})
        with self.assertRaises(ct.AdapterError):
            ct.build_sequence(tmp, "diffuse", skip_missing=False)
        # With --skip-missing every frame is skipped: nothing left to track.
        with self.assertRaises(ct.AdapterError):
            ct.build_sequence(tmp, "diffuse", skip_missing=True)
        import shutil

        shutil.rmtree(tmp)


class PointsLoadingTests(unittest.TestCase):
    def test_load_points_primary_mode(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(
                {
                    "points": [
                        {"id": "a", "x": 1.5, "y": 2.5},
                        {"id": 2, "x": 3.0, "y": 4.0},
                    ]
                },
                fh,
            )
            path = fh.name
        try:
            points = ct.load_points(path)
            self.assertEqual([p.id for p in points], ["a", "2"])
            self.assertEqual((points[0].x, points[0].y), (1.5, 2.5))
        finally:
            os.unlink(path)

    def test_load_points_rejects_duplicates(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(
                {
                    "points": [
                        {"id": "a", "x": 1.0, "y": 1.0},
                        {"id": "a", "x": 2.0, "y": 2.0},
                    ]
                },
                fh,
            )
            path = fh.name
        try:
            with self.assertRaises(ct.AdapterError):
                ct.load_points(path)
        finally:
            os.unlink(path)

    def test_load_points_rejects_non_numeric(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"points": [{"id": "a", "x": "NaN", "y": 2.0}]}, fh)
            path = fh.name
        try:
            with self.assertRaises(ct.AdapterError):
                ct.load_points(path)
        finally:
            os.unlink(path)

    def test_grid_points_deterministic_and_cover_image(self):
        pts = ct.grid_points(4, 3, SRC_W, SRC_H)
        self.assertEqual(len(pts), 12)
        self.assertEqual(pts[0].id, "grid_0")
        # With cols == width the grid reproduces the exact pixel centers.
        exact = ct.grid_points(SRC_W, 1, SRC_W, SRC_H)
        self.assertEqual(int(exact[0].x), 0)
        self.assertEqual(int(exact[-1].x), SRC_W - 1)
        # Determinism.
        again = ct.grid_points(4, 3, SRC_W, SRC_H)
        self.assertEqual([(p.id, p.x, p.y) for p in pts], [(p.id, p.x, p.y) for p in again])

    def test_grid_spacing_mode(self):
        pts = ct.grid_points(0, 0, SRC_W, SRC_H, spacing=(100.0, 200.0))
        xs = sorted({p.x for p in pts})
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(xs[1], 100.0)
        self.assertEqual(xs[-1], 1200.0)  # 1200 + 100 > 1280
        ys = sorted({p.y for p in pts})
        self.assertEqual(ys[-1], 1000.0)  # 1000 + 200 > 1024


class ConfidenceAndSamplingTests(unittest.TestCase):
    def test_confidence_modes(self):
        vis = np.array([[0.9, 0.1], [0.5, 0.0]], dtype=np.float32)
        conf = np.array([[0.8, 0.2], [0.4, 0.3]], dtype=np.float32)
        np.testing.assert_allclose(
            ct.combine_confidence(vis, conf, "vis*conf"),
            np.array([[0.72, 0.02], [0.2, 0.0]]),
        )
        np.testing.assert_allclose(ct.combine_confidence(vis, conf, "vis"), vis)
        np.testing.assert_allclose(ct.combine_confidence(vis, conf, "conf"), conf)
        np.testing.assert_allclose(ct.combine_confidence(vis, conf, "min"), np.minimum(vis, conf))

    def test_low_quality_points_are_retained_not_dropped(self):
        T, N = 3, 4
        rng = np.random.default_rng(42)
        vis = rng.random((T, N)).astype(np.float32)
        conf = rng.random((T, N)).astype(np.float32)
        emitted = ct.combine_confidence(vis, conf, "vis*conf")
        # Every point has an emitted confidence (nothing disappears).
        self.assertEqual(emitted.shape, (T, N))
        self.assertTrue(np.all(np.isfinite(emitted)))
        low = emitted < 0.1
        if np.any(low):
            idx = np.argwhere(low)[0]
            self.assertEqual(emitted[idx[0], idx[1]], vis[idx[0], idx[1]] * conf[idx[0], idx[1]])

    def test_sampling_clamps_out_of_range_without_nan(self):
        field = np.ones((2, 10, 10, 2), dtype=np.float32) * 3.0
        out = ct.sample_dense_field(field, np.array([-50.0, 50.0]), np.array([0.5, 0.5]))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 3.0)


class OutputWriterTests(unittest.TestCase):
    def _plan_and_data(self):
        run_dir = "/nonexistent/run"
        frames = [ct.SequenceFrame(i, i, f"/nonexistent/run/images/img_{i}.png") for i in range(2)]
        plan = ct.RunPlan(
            run_dir=run_dir,
            role="diffuse",
            frames=frames,
            skipped=[],
            source_w=SRC_W,
            source_h=SRC_H,
            inference_w=INF_W,
            inference_h=INF_H,
            scale_x=INF_W / SRC_W,
            scale_y=INF_H / SRC_H,
            points=make_points([(10.0, 20.0), (30.0, 40.0)]),
            points_mode="points-file",
            reference_frame_id=0,
            reference_seq_index=0,
            rotation_active=False,
            convention=ct.PIXEL_CENTER,
            confidence_mode="vis*conf",
            reference_error_limit=ct.DEFAULT_REFERENCE_ERROR_LIMIT,
            output_path="",
            csv_output=None,
            save_sampled=None,
            save_dense=None,
            diagnostics_dir=None,
            diagnostic_video=None,
            device="cuda",
        )
        track_src = np.array(
            [[[10.0, 20.0], [30.0, 40.0]], [[11.0, 21.0], [31.0, 41.0]]],
            dtype=np.float64,
        )
        vis = np.full((2, 2), 0.9, dtype=np.float32)
        raw = np.full((2, 2), 0.8, dtype=np.float32)
        conf = vis * raw

        return plan, track_src, vis, raw, conf

    def test_write_tracks_json_matches_fabsim_schema(self):
        plan, track_src, vis, raw, conf = self._plan_and_data()
        prov = {
            "tool": "cowtracker_to_fabsim",
            "source": plan.run_dir,
            "frame_mapping": [
                {"sequence_index": 0, "frame_id": 0, "rotated_position": 0},
                {"sequence_index": 1, "frame_id": 1, "rotated_position": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "tracks.json")
            ct.write_tracks_json(out, plan, track_src, conf, prov)
            with open(out) as fh:
                doc = json.load(fh)
        self.assertEqual(doc["format"], ct.FABSIM_TRACKS_FORMAT)
        self.assertEqual(doc["version"], ct.FABSIM_TRACKS_VERSION)
        self.assertEqual(doc["provenance"]["source"], plan.run_dir)
        self.assertEqual(len(doc["frames"]), 2)
        f0 = doc["frames"][0]
        self.assertEqual(f0["frame_id"], 0)
        self.assertEqual(len(f0["points"]), 2)
        self.assertEqual(f0["points"][0]["id"], "p0")
        self.assertEqual(f0["points"][0]["x"], 10.0)
        self.assertEqual(f0["points"][0]["confidence"], 0.72)
        # IDs stable across frames.
        self.assertEqual([p["id"] for p in doc["frames"][1]["points"]], ["p0", "p1"])

    def test_write_tracks_csv_column_order(self):
        plan, track_src, vis, raw, conf = self._plan_and_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "tracks.csv")
            ct.write_tracks_csv(out, plan, track_src, conf, vis, raw)
            with open(out) as fh:
                lines = fh.read().splitlines()
        self.assertEqual(
            lines[0],
            "frame_id,track_id,x,y,confidence,visibility,raw_confidence,sequence_index",
        )
        self.assertEqual(len(lines), 1 + 4)
        self.assertIn("0,p0,10,20,0.720000,0.900000,0.800000,0", lines[1])
        self.assertIn("1,p1,31,41,0.720000,0.900000,0.800000,1", lines[-1])

    def test_sampled_npz_round_trip(self):
        plan, track_src, vis, raw, conf = self._plan_and_data()
        prov = {"tool": "cowtracker_to_fabsim", "source": plan.run_dir}
        with tempfile.TemporaryDirectory() as tmp:
            npz = os.path.join(tmp, "sampled.npz")
            ct.write_sampled_npz(npz, plan, track_src, vis, raw, conf, prov)
            data = ct.load_sampled_npz(npz)
        self.assertEqual([p.id for p in data["points"]], ["p0", "p1"])
        np.testing.assert_allclose(data["track_src"], track_src)
        np.testing.assert_allclose(data["visibility"], vis)
        self.assertEqual(data["metadata"]["source"], plan.run_dir)
        self.assertEqual(list(data["frame_id"]), [0, 1])


class ReferenceErrorTests(unittest.TestCase):
    def test_limit_is_enforced(self):
        T, N = 3, 2
        track_src = np.zeros((T, N, 2), dtype=np.float64)
        query = np.zeros((N, 2), dtype=np.float64)
        err = ct.reference_error_px(track_src, query, 0)
        np.testing.assert_allclose(err, 0.0)
        # Move the reference-frame track away from the query by 3 px.
        track_src[0, :, 0] = 3.0
        err = ct.reference_error_px(track_src, query, 0)
        np.testing.assert_allclose(err, [3.0, 3.0])


def write_test_png(path, w=8, h=8):
    from PIL import Image

    Image.new("RGB", (w, h), (40, 80, 120)).save(path)


class ModuleAndCliTests(unittest.TestCase):
    def test_module_import_does_not_touch_torch(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("cowtracker", sys.modules)

    def test_help_works_without_torch(self):
        try:
            ct.main(["--help"])
        except SystemExit as exc:
            self.assertEqual(exc.code, 0)

    def test_dry_run_against_synthetic_run(self):
        doc = {
            "frames": [
                {"frame_id": 0, "image_diffuse": "a.png"},
                {"frame_id": 1, "image_diffuse": "b.png"},
            ]
        }
        tmp = tempfile.mkdtemp(prefix="ct_cli_")
        os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
        with open(os.path.join(tmp, "frames.json"), "w") as fh:
            json.dump(doc, fh)
        write_test_png(os.path.join(tmp, "images", "a.png"))
        write_test_png(os.path.join(tmp, "images", "b.png"))
        points = os.path.join(tmp, "points.json")
        with open(points, "w") as fh:
            json.dump({"points": [{"id": "q", "x": 1.0, "y": 1.0}]}, fh)

        import shutil

        try:
            code = ct.main([tmp, "--points", points, "--dry-run"])
            self.assertEqual(code, 0)
            # Missing role without --skip-missing is a clear failure.
            doc2 = {"frames": [{"frame_id": 0}, {"frame_id": 1, "image_diffuse": "a.png"}]}
            with open(os.path.join(tmp, "frames.json"), "w") as fh:
                json.dump(doc2, fh)
            code = ct.main([tmp, "--points", points, "--dry-run"])
            self.assertEqual(code, 1)
        finally:
            shutil.rmtree(tmp)

    def test_no_points_is_a_clear_error(self):
        doc = {"frames": [{"frame_id": 0, "image_diffuse": "a.png"}]}
        tmp = tempfile.mkdtemp(prefix="ct_cli2_")
        os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
        with open(os.path.join(tmp, "frames.json"), "w") as fh:
            json.dump(doc, fh)
        write_test_png(os.path.join(tmp, "images", "a.png"))
        import shutil

        try:
            captured = io.StringIO()
            old = sys.stderr
            sys.stderr = captured
            try:
                code = ct.main([tmp, "--dry-run"])
            finally:
                sys.stderr = old
            self.assertEqual(code, 1)
            self.assertIn("--points", captured.getvalue())
        finally:
            shutil.rmtree(tmp)

    def test_grid_requires_both_cols_and_rows(self):
        doc = {"frames": [{"frame_id": 0, "image_diffuse": "a.png"}]}
        tmp = tempfile.mkdtemp(prefix="ct_grid_")
        os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
        with open(os.path.join(tmp, "frames.json"), "w") as fh:
            json.dump(doc, fh)
        write_test_png(os.path.join(tmp, "images", "a.png"))
        import shutil

        try:
            for argv in (
                ["--grid-cols", "4"],
                ["--grid-rows", "3"],
                ["--grid-cols", "4", "--grid-rows", "0"],
            ):
                captured = io.StringIO()
                old = sys.stderr
                sys.stderr = captured
                try:
                    code = ct.main([tmp, *argv, "--dry-run"])
                finally:
                    sys.stderr = old
                self.assertEqual(code, 1, f"must reject {argv}")
                self.assertIn("--grid-cols", captured.getvalue())
                self.assertIn("--grid-rows", captured.getvalue())
            # Both dimensions supplied works.
            self.assertEqual(ct.main([tmp, "--grid-cols", "4", "--grid-rows", "3", "--dry-run"]), 0)
            # --grid-spacing alone is a valid alternative.
            self.assertEqual(ct.main([tmp, "--grid-spacing", "2", "2", "--dry-run"]), 0)
        finally:
            shutil.rmtree(tmp)


class FullPipelineTests(unittest.TestCase):
    RUN_W, RUN_H = 32, 24
    INF_W, INF_H = 64, 48
    T = 3

    def _make_run(self):
        tmp = tempfile.mkdtemp(prefix="ct_pipe_")
        os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
        doc = {
            "frames": [
                {"frame_id": 10, "image_diffuse": "a.png"},
                {"frame_id": 11, "image_diffuse": "b.png"},
                {"frame_id": 12, "image_diffuse": "c.png"},
            ]
        }
        with open(os.path.join(tmp, "frames.json"), "w") as fh:
            json.dump(doc, fh)
        for name in ("a.png", "b.png", "c.png"):
            write_test_png(os.path.join(tmp, "images", name), self.RUN_W, self.RUN_H)
        points = os.path.join(tmp, "points.json")
        with open(points, "w") as fh:
            json.dump(
                {
                    "points": [
                        {"id": "q0", "x": 4.0, "y": 4.0},
                        {"id": "q1", "x": 24.0, "y": 18.0},
                    ]
                },
                fh,
            )

        return tmp, points

    def _dense_field(self):
        # Identity at rotated position 0, linear motion afterwards.  Field is
        # already in "model output" space: [T, Hi, Wi, 2] / [T, Hi, Wi].
        gy, gx = np.meshgrid(np.arange(self.INF_H), np.arange(self.INF_W), indexing="ij")
        track = np.stack([gx, gy], axis=-1).astype(np.float32)
        track = np.repeat(track[None], self.T, axis=0).copy()
        for t in range(1, self.T):
            track[t, :, :, 0] += 2.0 * t
            track[t, :, :, 1] += 3.0 * t
        vis = np.full((self.T, self.INF_H, self.INF_W), 0.95, dtype=np.float32)
        conf = np.full((self.T, self.INF_H, self.INF_W), 0.8, dtype=np.float32)

        return {"track": track, "vis": vis, "conf": conf}

    def test_full_export_path_and_sampled_reexport(self):
        import shutil

        tmp, points = self._make_run()
        try:
            args = ct.build_arg_parser().parse_args(
                [
                    tmp,
                    "--points",
                    points,
                    "--inference-width",
                    str(self.INF_W),
                    "--inference-height",
                    str(self.INF_H),
                    "--output",
                    os.path.join(tmp, "tracks.json"),
                    "--csv-output",
                    os.path.join(tmp, "tracks.csv"),
                    "--save-sampled",
                    os.path.join(tmp, "sampled.npz"),
                    "--save-dense",
                    os.path.join(tmp, "dense.npz"),
                ]
            )
            plan = ct.preflight(args)
            dense = self._dense_field()
            provenance: dict[str, Any] = {}
            ct.sample_and_export(plan, args, dense, provenance)

            # tracks.json is the fabsim-tracks schema with provenance.
            with open(os.path.join(tmp, "tracks.json")) as fh:
                doc = json.load(fh)
            self.assertEqual(doc["format"], ct.FABSIM_TRACKS_FORMAT)
            self.assertEqual(len(doc["frames"]), self.T)
            self.assertEqual(doc["frames"][0]["frame_id"], 10)
            f0 = doc["frames"][0]["points"]
            self.assertEqual([p["id"] for p in f0], ["q0", "q1"])
            self.assertEqual(f0[0]["x"], 4.0)  # reference frame: identity
            self.assertEqual(f0[0]["y"], 4.0)
            self.assertEqual(doc["provenance"]["coordinate_convention"], ct.PIXEL_CENTER)
            # Frame ids in output = fabsim ids, not sequence positions.
            self.assertEqual([f["frame_id"] for f in doc["frames"]], [10, 11, 12])

            # Frame 2: query q0 moved 2*2 px (inference) -> 2*2*RUN_W/INF_W px source.
            f2 = doc["frames"][2]["points"]
            sx = self.RUN_W / self.INF_W
            self.assertAlmostEqual(f2[0]["x"], 4.0 + 4.0 * sx, places=4)
            # Confidence = vis*conf retained for every point.
            self.assertEqual(f0[0]["confidence"], 0.76)

            # CSV: core columns first.
            with open(os.path.join(tmp, "tracks.csv")) as fh:
                csv_lines = fh.read().splitlines()
            self.assertTrue(csv_lines[0].startswith("frame_id,track_id,x,y,confidence"))
            self.assertEqual(len(csv_lines), 1 + self.T * 2)

            # Dense NPZ written only when requested.
            dense_data = np.load(os.path.join(tmp, "dense.npz"))
            self.assertEqual(dense_data["track"].shape, (self.T, self.INF_H, self.INF_W, 2))

            # Re-export from the sampled NPZ without any inference.
            code = ct.main(
                [
                    tmp,
                    "--points",
                    points,
                    "--inference-width",
                    str(self.INF_W),
                    "--inference-height",
                    str(self.INF_H),
                    "--sampled-input",
                    os.path.join(tmp, "sampled.npz"),
                    "--output",
                    os.path.join(tmp, "tracks_again.json"),
                ]
            )
            self.assertEqual(code, 0)
            with open(os.path.join(tmp, "tracks_again.json")) as fh:
                doc2 = json.load(fh)
            self.assertEqual(doc2["frames"][0]["points"][0]["x"], 4.0)
        finally:
            shutil.rmtree(tmp)

    def test_reference_error_limit_fails_run(self):
        import shutil

        tmp, points = self._make_run()
        try:
            args = ct.build_arg_parser().parse_args(
                [
                    tmp,
                    "--points",
                    points,
                    "--inference-width",
                    str(self.INF_W),
                    "--inference-height",
                    str(self.INF_H),
                    "--reference-error-limit",
                    "0.01",
                ]
            )
            plan = ct.preflight(args)
            dense = self._dense_field()
            # Break the reference frame (shift all positions by 5 px).
            dense["track"][0, :, :, 0] += 5.0
            with self.assertRaises(ct.AdapterError) as ctx:
                ct.sample_and_export(plan, args, dense, {})
            # Every offending point is reported, not just the first.
            self.assertIn("2 point(s) exceed", str(ctx.exception))
            self.assertIn("q0", str(ctx.exception))
            self.assertIn("q1", str(ctx.exception))
        finally:
            shutil.rmtree(tmp)

    def _write_sampled_npz(self, tmp):
        points = os.path.join(tmp, "points.json")
        with open(points, "w") as fh:
            json.dump(
                {
                    "points": [
                        {"id": "q0", "x": 4.0, "y": 4.0},
                        {"id": "q1", "x": 24.0, "y": 18.0},
                    ]
                },
                fh,
            )
        args = ct.build_arg_parser().parse_args(
            [
                tmp,
                "--points",
                points,
                "--inference-width",
                str(self.INF_W),
                "--inference-height",
                str(self.INF_H),
                "--save-sampled",
                os.path.join(tmp, "sampled.npz"),
            ]
        )
        plan = ct.preflight(args)
        ct.sample_and_export(plan, args, self._dense_field(), {})

        return os.path.join(tmp, "sampled.npz")

    def test_sampled_reexport_is_self_contained_without_points(self):
        import shutil

        tmp, _ = self._make_run()
        try:
            npz = self._write_sampled_npz(tmp)
            out = os.path.join(tmp, "tracks_self.json")
            code = ct.main([tmp, "--sampled-input", npz, "--output", out])
            self.assertEqual(code, 0)
            with open(out) as fh:
                doc = json.load(fh)
            self.assertEqual(len(doc["frames"]), self.T)
            self.assertEqual([p["id"] for p in doc["frames"][0]["points"]], ["q0", "q1"])
            # Reference frame identity preserved from the NPZ.
            self.assertEqual(doc["frames"][0]["points"][0]["x"], 4.0)
            self.assertEqual(doc["frames"][0]["points"][0]["y"], 4.0)
            # Motion preserved from the NPZ (2*2 inference px -> source px).
            sx = self.RUN_W / self.INF_W
            self.assertAlmostEqual(doc["frames"][2]["points"][0]["x"], 4.0 + 4.0 * sx, places=4)
        finally:
            shutil.rmtree(tmp)

    def test_sampled_reexport_rejects_mismatched_points(self):
        import shutil

        tmp, _ = self._make_run()
        try:
            npz = self._write_sampled_npz(tmp)

            # Same ids, different coordinates.
            wrong_coords = os.path.join(tmp, "wrong_coords.json")
            with open(wrong_coords, "w") as fh:
                json.dump(
                    {
                        "points": [
                            {"id": "q0", "x": 100.0, "y": 4.0},
                            {"id": "q1", "x": 24.0, "y": 18.0},
                        ]
                    },
                    fh,
                )
            captured = io.StringIO()
            old = sys.stderr
            sys.stderr = captured
            try:
                code = ct.main(
                    [
                        tmp,
                        "--points",
                        wrong_coords,
                        "--sampled-input",
                        npz,
                        "--output",
                        os.path.join(tmp, "never.json"),
                    ]
                )
            finally:
                sys.stderr = old
            self.assertEqual(code, 1)
            self.assertIn("differ from the sampled NPZ", captured.getvalue())

            # Different ids: refusing to relabel.
            renamed = os.path.join(tmp, "renamed.json")
            with open(renamed, "w") as fh:
                json.dump(
                    {
                        "points": [
                            {"id": "z0", "x": 4.0, "y": 4.0},
                            {"id": "z1", "x": 24.0, "y": 18.0},
                        ]
                    },
                    fh,
                )
            captured = io.StringIO()
            old = sys.stderr
            sys.stderr = captured
            try:
                code = ct.main(
                    [
                        tmp,
                        "--points",
                        renamed,
                        "--sampled-input",
                        npz,
                        "--output",
                        os.path.join(tmp, "never.json"),
                    ]
                )
            finally:
                sys.stderr = old
            self.assertEqual(code, 1)
            self.assertIn("refusing to relabel", captured.getvalue())

            # Matching points file still works alongside --sampled-input.
            good = os.path.join(tmp, "points.json")
            with open(good, "w") as fh:
                json.dump(
                    {
                        "points": [
                            {"id": "q0", "x": 4.0, "y": 4.0},
                            {"id": "q1", "x": 24.0, "y": 18.0},
                        ]
                    },
                    fh,
                )
            code = ct.main(
                [
                    tmp,
                    "--points",
                    good,
                    "--sampled-input",
                    npz,
                    "--output",
                    os.path.join(tmp, "tracks_ok.json"),
                ]
            )
            self.assertEqual(code, 0)
        finally:
            shutil.rmtree(tmp)

    def test_sampled_reexport_rejects_mismatched_run(self):
        import shutil

        tmp, _ = self._make_run()
        try:
            npz = self._write_sampled_npz(tmp)
            # Drop the last frame from frames.json -> mapping mismatch.
            doc = {
                "frames": [
                    {"frame_id": 10, "image_diffuse": "a.png"},
                    {"frame_id": 11, "image_diffuse": "b.png"},
                ]
            }
            with open(os.path.join(tmp, "frames.json"), "w") as fh:
                json.dump(doc, fh)
            captured = io.StringIO()
            old = sys.stderr
            sys.stderr = captured
            try:
                code = ct.main(
                    [
                        tmp,
                        "--sampled-input",
                        npz,
                        "--output",
                        os.path.join(tmp, "never.json"),
                    ]
                )
            finally:
                sys.stderr = old
            self.assertEqual(code, 1)
            self.assertIn("does not match the run/CLI", captured.getvalue())
            self.assertIn("frame_id", captured.getvalue())
        finally:
            shutil.rmtree(tmp)


class ProvenanceTests(unittest.TestCase):
    # Reuse the synthetic run and dense field of the full-pipeline tests.
    T, INF_W, INF_H, RUN_W, RUN_H = (
        FullPipelineTests.T,
        FullPipelineTests.INF_W,
        FullPipelineTests.INF_H,
        FullPipelineTests.RUN_W,
        FullPipelineTests.RUN_H,
    )
    _make_run = FullPipelineTests._make_run
    _dense_field = FullPipelineTests._dense_field

    def _export(self, runtime, env=None):
        import shutil

        tmp, points = self._make_run()
        try:
            args = ct.build_arg_parser().parse_args(
                [
                    tmp,
                    "--points",
                    points,
                    "--inference-width",
                    str(self.INF_W),
                    "--inference-height",
                    str(self.INF_H),
                    "--checkpoint",
                    "/secret/dir/ckpt/cowtracker_model.pth",
                ]
            )
            plan = ct.preflight(args)
            with mock.patch.dict(os.environ, env or {}):
                ct.sample_and_export(plan, args, self._dense_field(), runtime)
            with open(os.path.join(tmp, "tracks.json")) as fh:
                return json.load(fh)["provenance"], tmp
        finally:
            shutil.rmtree(tmp)

    def test_runtime_block_and_env_stamps(self):
        runtime = {"torch": "2.x", "cuda": "13.0", "gpu": "Test GPU", "host": "h"}
        prov, _ = self._export(
            runtime,
            {"FABSIM_ADAPTER_COMMIT": "abc123", "COWTRACKER_CHECKPOINT_SHA256": "ff00"},
        )
        self.assertEqual(prov["runtime"], runtime)
        self.assertEqual(prov["adapter_git_commit"], "abc123")
        self.assertEqual(prov["checkpoint_sha256"], "ff00")
        self.assertEqual(prov["adapter_version"], ct.ADAPTER_VERSION)

    def test_no_absolute_paths_in_provenance(self):
        prov, tmp = self._export({})
        self.assertEqual(prov["checkpoint"], "cowtracker_model.pth")
        self.assertNotIn(tmp, json.dumps(prov))
        self.assertNotIn("runtime", prov)

    def test_run_label_is_never_an_absolute_path(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FABSIM_RUN_KEY", None)
            self.assertEqual(ct.run_label("/data/sets/ds_a/run_1/"), "ds_a/run_1")
            self.assertEqual(ct.run_label("/data/run"), "data/run")
            # A relative directory is resolved first, so "." keeps its real name.
            cwd = os.path.abspath(os.curdir).rstrip(os.sep).split(os.sep)[-2:]
            self.assertEqual(ct.run_label("."), "/".join(cwd))
        with mock.patch.dict(os.environ, {"FABSIM_RUN_KEY": "ds_a/run_1"}):
            self.assertEqual(ct.run_label("/data/run"), "ds_a/run_1")

    def test_collect_runtime_info_with_fake_torch(self):
        torch_mod = mock.Mock()
        torch_mod.__version__ = "9.9.9"
        torch_mod.version.cuda = "13.0"
        torch_mod.backends.cudnn.version.return_value = 91000
        torch_mod.cuda.get_device_name.return_value = "Fake GPU"
        info = ct.collect_runtime_info(torch_mod, "cuda")
        self.assertEqual(info["torch"], "9.9.9")
        self.assertEqual(info["cudnn"], 91000)
        self.assertEqual(info["gpu"], "Fake GPU")
        self.assertEqual(
            set(ct.collect_runtime_info(torch_mod, "cpu")),
            {"python", "host", "torch", "cuda", "device"},
        )

    def test_verbose_prints_traceback_for_unexpected_failure(self):
        import contextlib
        import shutil

        tmp, points = self._make_run()
        try:
            argv = [
                tmp,
                "--points",
                points,
                "--inference-width",
                "28",
                "--inference-height",
                "20",
            ]
            with mock.patch.object(ct, "run_inference", side_effect=RuntimeError("boom")):
                quiet, loud = io.StringIO(), io.StringIO()
                with contextlib.redirect_stderr(quiet):
                    self.assertEqual(ct.main(argv), 1)
                with contextlib.redirect_stderr(loud):
                    self.assertEqual(ct.main([*argv, "--verbose"]), 1)
            self.assertNotIn("Traceback", quiet.getvalue())
            self.assertIn("--verbose", quiet.getvalue())
            self.assertIn("Traceback", loud.getvalue())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
