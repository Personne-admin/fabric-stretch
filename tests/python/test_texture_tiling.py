import unittest

import numpy as np

from fabsim import texture_tiling as tt


class PrimitiveTests(unittest.TestCase):
    def test_distance_sums_squares_over_planes(self):
        A = np.zeros((2, 3, 3))
        B = np.ones((2, 3, 3))
        B[0, 0] = [1, 2, 3]
        D = tt.distance(A, B)

        self.assertEqual(D.shape, (2, 3))
        self.assertEqual(D[0, 0], 14)
        self.assertEqual(D[1, 1], 3)

    def test_distance4_from_mean(self):
        z = np.zeros((1, 1, 1))
        o = np.ones((1, 1, 1))

        self.assertAlmostEqual(tt.distance4(z, z, z, z)[0, 0], 0)
        # values 0,0,0,4 -> mean 1 -> 1+1+1+9
        self.assertAlmostEqual(tt.distance4(z, z, z, 4 * o)[0, 0], 12)

    def test_compute_costs_chebyshev(self):
        C0 = np.full((5, 7), 99.0)
        C0[2, 3] = 0
        C = tt.compute_costs(np.ones((5, 7)), C0)
        for y in range(5):
            for x in range(7):
                self.assertEqual(C[y, x], max(abs(y - 2), abs(x - 3)))

    def test_paint_four_connected_no_diagonal(self):
        I = np.array([[1, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]], float)
        out = tt.paint(I, 7, [[1, 1]])  # 1-based seed (row 1, col 1)

        self.assertEqual(out[0, 0], 7)
        self.assertEqual(out[0, 1], 7)
        self.assertEqual(out[1, 1], 7)
        self.assertEqual(out[2, 0], 1)  # only diagonal-connected -> untouched
        self.assertEqual(out[0, 3], 1)

    def test_paint_rejects_mixed_seeds(self):
        with self.assertRaises(ValueError):
            tt.paint(np.array([[0.0, 1.0]]), 5, [[1, 1], [1, 2]])

    def test_interp_mask_zero_dist(self):
        h, w = 6, 5
        P = np.array([[r, 3] for r in range(1, h + 1)])  # vertical path at col 3
        M = tt.interp_mask_left_vertical(h, w, P, 0)

        self.assertTrue(np.all(M[:, :2] == 1))
        self.assertTrue(np.all(M[:, 2] == 0.5))
        self.assertTrue(np.all(M[:, 3:] == 0))

    def test_interp_mask_ramp(self):
        h, w = 6, 9
        P = np.array([[r, 5] for r in range(1, h + 1)])
        M = tt.interp_mask_left_vertical(h, w, P, 2)  # d=3
        # distance-to-path / d (capped at 1), signed by side, mapped to [0,1]
        expect = [
            1,
            (1 + 3 / 3) / 2,
            (1 + 2 / 3) / 2,
            (1 + 1 / 3) / 2,
            0.5,
            (1 - 1 / 3) / 2,
            (1 - 2 / 3) / 2,
            0,
            0,
        ]

        np.testing.assert_allclose(M[3], expect)

    def test_interp_mask_wraps_rows(self):
        P = np.array([[0, 2], [1, 2], [2, 2], [3, 2]])  # row 0 wraps to row 3 (mod)
        M = tt.interp_mask_left_vertical(3, 4, P, 0)
        self.assertTrue(np.all(M[:, 1] == 0.5))


class AreaAndArgumentTests(unittest.TestCase):
    def p(self, **kw):
        a = dict(shape_hw=(300, 400), window_size=[20, 30], shift=0, interp_dist=2, area=None)
        a.update(kw)

        return tt.resolve_params(**a)

    def test_window_size_order_and_quirk(self):
        p = self.p(window_size=[20, 30])
        self.assertEqual((p["wv"], p["wh"]), (20, 30))

    def test_scalar_and_four_shift(self):
        self.assertEqual(self.p(shift=5)["shift"], [5, 5, 5, 5])
        self.assertEqual(self.p(shift=[1, 2, 3, 4])["shift"], [1, 2, 3, 4])

        with self.assertRaises(ValueError):
            self.p(shift=[20, 20])

        with self.assertRaises(ValueError):
            self.p(shift=[1, 2, 3])

    def test_interp_dist_order_quirk(self):
        p = self.p(interp_dist=[1, 4])
        self.assertEqual((p["idv"], p["idh"]), (1, 4))

    def test_automatic_area(self):
        p = self.p(shift=[1, 2, 3, 4])
        wt, wb, wl, wr = 15, 14, 10, 9
        y1_min, y2_max = 1 + 2 + wt, 300 - 2 - wb
        x1_min, x2_max = 1 + 2 + wl, 400 - 2 - wr

        self.assertEqual((p["y1"], p["x1"]), (y1_min, x1_min))
        self.assertEqual(p["y2"], y2_max - 4)
        self.assertEqual(p["x2"], x2_max - 2)

    def test_explicit_area_is_zero_based_input(self):
        p = self.p(area=[50, 60, 100, 120])
        self.assertEqual((p["y1"], p["x1"], p["y2"], p["x2"]), (51, 61, 150, 180))

    def test_center_area(self):
        p = self.p(area=[100, 120])
        self.assertEqual((p["y1"], p["x1"]), (101, 141))

    def test_invalid_area(self):
        with self.assertRaises(ValueError):
            self.p(area=[0, 0, 100, 100])

        with self.assertRaises(ValueError):
            self.p(area=[1, 2, 3])

        with self.assertRaises(ValueError):
            self.p(window_size=[1000, 1000])

    def test_matlab_round_half_away(self):
        self.assertEqual(tt._round(2.5), 3)
        self.assertEqual(tt._round(-2.5), -3)


class AutomaticCoordinatesTests(unittest.TestCase):
    def test_finds_period_shift_vertical_and_horizontal(self):
        rng = np.random.default_rng(1)
        base = rng.random((60, 70, 1))
        Py, Px = 40, 50
        I = np.tile(base[:Py, :Px], (3, 3, 1))[:120, :150]

        # start with y2/x2 too large by 6/4 -> shifts back to the true period
        y1, x1 = 30, 30
        res = tt.automatic_coordinates_for_max_tile(I, (10, 10), (8, 8), [y1, y1 + Py + 3, x1, x1 + Px + 2])

        self.assertEqual(res[1], y1 + Py)
        self.assertEqual(res[3], x1 + Px)

    def test_ties_choose_lowest_shift(self):
        I = np.zeros((100, 100, 1))
        res = tt.automatic_coordinates_for_max_tile(I, (10, 10), (5, 5), [30, 70, 30, 70])
        self.assertEqual(res[1], 65)
        self.assertEqual(res[3], 65)  # strict '<'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.Py, self.Px = 40, 50
        base = rng.random((self.Py, self.Px, 3))
        self.img = np.tile(base, (4, 4, 1))[:150, :180]

    def run_tile(self, **kw):
        return tt.find_tile(self.img, [10, 10], kw.pop("shift", 0), kw.pop("interp_dist", 0), **kw)

    def test_periodic_image_reproduced_and_tile_size(self):
        tile, area, insp = self.run_tile(area=[30, 30, self.Py + 1, self.Px + 1])

        self.assertIsNone(insp)
        self.assertEqual(area, (31, 31, self.Py + 1, self.Px + 1))
        self.assertEqual(tile.shape, (self.Py, self.Px, 3))  # one smaller than area

        np.testing.assert_allclose(tile, self.img[30 : 30 + self.Py, 30 : 30 + self.Px], atol=1e-12)

    def test_uint8_roundtrip_and_dtype(self):
        u8 = np.round(self.img * 255).astype(np.uint8)
        tile, _, _ = tt.find_tile(u8, [10, 10], 0, 0, area=[30, 30, self.Py + 1, self.Px + 1])

        self.assertEqual(tile.dtype, np.uint8)
        np.testing.assert_array_equal(tile, u8[30 : 30 + self.Py, 30 : 30 + self.Px])

    def test_joint_multi_image_and_guidance(self):
        gray = self.img[:, :, :1]
        tiles, area, _ = tt.find_tile(
            [self.img, gray[:, :, 0]],
            [10, 10],
            4,
            1,
            area=[30, 30, self.Py + 1, self.Px + 1],
        )

        tg, ag, _ = tt.find_tile(
            [self.img, gray[:, :, 0]],
            [10, 10],
            4,
            1,
            area=[30, 30, self.Py + 1, self.Px + 1],
            guidance_images=gray,
        )

        self.assertEqual(tiles[0].shape[:2], tiles[1].shape[:2])
        self.assertEqual(tiles[1].ndim, 2)
        self.assertEqual(area, ag)  # same search result on periodic data
        self.assertEqual(tg[0].shape, tiles[0].shape)

    def test_window_larger_than_tile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            tt.find_tile(self.img, [10, 60], 0, 0, area=[30, 30, 41, 51])

    def test_guidance_size_mismatch(self):
        with self.assertRaises(ValueError):
            tt.find_tile(self.img, [10, 10], 0, 0, guidance_images=np.zeros((5, 5)))

    def test_tile_is_seamless_when_wrapped(self):
        # smooth non-periodic image
        rng = np.random.default_rng(9)

        n = rng.random((150, 180, 1))
        k = np.ones(9) / 9
        for ax in (0, 1):
            n = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 4, mode="edge"), k, "valid"), ax, n)

        y = np.linspace(0, 1, 150)[:, None, None]
        x = np.linspace(0, 1, 180)[None, :, None]
        img = n + 2 * y + 1.5 * x  # strong gradient: crop wraps badly
        area = [30, 30, 61, 71]
        tile, a, _ = tt.find_tile(img, [10, 10], 0, 4, area=area)
        crop = img[30 : 30 + tile.shape[0], 30 : 30 + tile.shape[1]]

        def jump(t):
            return np.abs(t[0] - t[-1]).mean(), np.abs(t[:, 0] - t[:, -1]).mean()

        inner = np.abs(np.diff(tile, axis=0)).mean()
        self.assertTrue(np.isfinite(tile).all())
        self.assertGreater(jump(crop)[0], 5 * inner)
        self.assertLess(jump(tile)[0], 3 * inner + 0.02)
        self.assertLess(jump(tile)[1], 3 * np.abs(np.diff(tile, axis=1)).mean() + 0.02)


class PathSearchTests(unittest.TestCase):
    def test_min_costs_values_and_padding(self):
        E = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        C = tt.min_costs(E, [[1, 1]])

        self.assertEqual(C.shape, (4, 5))
        self.assertTrue(np.isinf(C[0]).all() and np.isinf(C[-1]).all())
        self.assertTrue(np.isinf(C[:, 0]).all() and np.isinf(C[:, -1]).all())

        # start = E(1,1)=1; (1,2)=1+2; (2,1)=1+4; (2,2)=min(1,3,5)+5=6 ; (1,3)=3+3=6; (2,3)=min(3,6,..)+6
        np.testing.assert_array_equal(C[1:-1, 1:-1], [[1, 3, 6], [5, 6, 9]])

    def test_min_costs_matches_full_relaxation(self):
        rng = np.random.default_rng(5)
        E = rng.random((12, 9)) + 0.1
        starts = [[3, 1], [7, 1]]

        C = tt.min_costs(E, starts)[1:-1, 1:-1]
        C0 = np.full(E.shape, np.inf)
        for r, c in starts:
            C0[r - 1, c - 1] = E[r - 1, c - 1]

        np.testing.assert_allclose(C, tt.compute_costs(E, C0))

    def test_multiple_starts_use_cheapest(self):
        E = np.ones((5, 5))
        E[0, :] = 50
        C = tt.min_costs(E, [[1, 1], [5, 1]])[1:-1, 1:-1]

        self.assertEqual(C[2, 0], 3)  # reached from the row-5 start: 1+1+1
        self.assertEqual(C[4, 0], 1)

    def test_exact_straight_path_and_orientation(self):
        E = np.ones((5, 7))
        P, C = tt.optimal_path(E, [[1, 1]], [[5, 1]])
        np.testing.assert_array_equal(P, [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]])

        self.assertEqual(C.shape, (7, 9))

    def test_path_follows_cheap_corridor(self):
        E = np.full((5, 7), 9.0)
        for r, c in [(1, 1), (2, 2), (3, 3), (3, 4), (4, 5), (5, 6), (5, 7)]:
            E[r - 1, c - 1] = 1

        P, _ = tt.optimal_path(E, [[1, 1]], [[5, 7]])
        np.testing.assert_array_equal(P, [[1, 1], [2, 2], [3, 3], [3, 4], [4, 5], [5, 6], [5, 7]])

    def test_end_selection_takes_cheapest_end_first_on_tie(self):
        E = np.ones((3, 3))
        C = tt.min_costs(E, [[2, 1]])
        P = tt.costs_to_path(C, [[2, 1]], [[1, 3], [2, 3], [3, 3]])

        self.assertEqual(P[0].tolist(), [2, 1])
        self.assertEqual(P[-1].tolist(), [1, 3])  # first of equal ends

    def test_tie_break_column_major(self):
        E = np.ones((3, 3))
        C = tt.min_costs(E, [[1, 1], [2, 1], [3, 1]])
        P = tt.costs_to_path(C, [[1, 1], [2, 1], [3, 1]], [[2, 3]])

        self.assertEqual(P.tolist(), [[1, 1], [1, 2], [2, 3]])

    def test_path_ends_on_start_and_end_sets(self):
        rng = np.random.default_rng(2)
        E = rng.random((20, 6)) + 0.05
        P, _ = tt.optimal_path(E, [[3, 2]], [[18, 2]])

        self.assertEqual(P[0].tolist(), [3, 2])
        self.assertEqual(P[-1].tolist(), [18, 2])
        self.assertTrue((np.abs(np.diff(P, axis=0)).max(axis=1) <= 1).all())
        self.assertEqual(len({tuple(p) for p in P.tolist()}), len(P))  # visited cells are never reused

    def test_no_path_guard(self):
        C = np.full((5, 5), np.inf)
        with self.assertRaises(RuntimeError):
            tt.costs_to_path(C, [[5, 1]], [[3, 3]])

    def test_find_paths_endpoints(self):
        rng = np.random.default_rng(4)
        I = rng.random((60, 70, 1))
        wh, wv, idh, idv = 10, 12, 2, 3
        PH, PV = tt.find_paths(I, wh, wv, idh, idv)
        wt, wl = wh // 2, wv // 2
        self.assertEqual(PH[0].tolist(), [idh + 1 + wt, idv + 1 + wl])
        self.assertEqual(PV[0].tolist(), [idh + 1 + wt, idv + 1 + wl])


if __name__ == "__main__":
    unittest.main()
