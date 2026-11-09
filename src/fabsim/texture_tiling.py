from __future__ import annotations

import numpy as np


def _s(a: int, b: int) -> slice:
    return slice(a - 1, b)


def _idx(v) -> np.ndarray:
    return np.asarray(v, dtype=int) - 1


def _round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


_CHUNK = 48


def distance(A, B):
    total = None
    for k in range(0, A.shape[2], _CHUNK):
        D = A[:, :, k : k + _CHUNK] - B[:, :, k : k + _CHUNK]
        part = np.einsum("ijk,ijk->ij", D, D)
        total = part if total is None else total + part

    return total if total is not None else np.zeros(A.shape[:2], dtype=A.dtype)


def distance4(A, B, C, D):
    total = None
    for k in range(0, A.shape[2], _CHUNK):
        X = np.stack([a[:, :, k : k + _CHUNK] for a in (A, B, C, D)], axis=3)
        R = X - np.mean(X, axis=3, keepdims=True)
        part = np.einsum("ijkl,ijkl->ij", R, R)
        total = part if total is None else total + part

    return total if total is not None else np.zeros(A.shape[:2], dtype=A.dtype)


def compute_costs(E, C0):
    h, w = E.shape[:2]
    C = np.full((h + 2, w + 2), np.inf)
    C[1:-1, 1:-1] = C0

    while True:
        D = np.stack(
            [
                C[0:-2, 0:-2] + E,
                C[0:-2, 1:-1] + E,
                C[0:-2, 2:] + E,
                C[1:-1, 0:-2] + E,
                C[1:-1, 2:] + E,
                C[2:, 0:-2] + E,
                C[2:, 1:-1] + E,
                C[2:, 2:] + E,
            ],
            axis=2,
        )

        D2 = D.min(axis=2)
        C0 = C[1:-1, 1:-1].copy()
        A = D2 < C0
        if not A.any():
            break

        C0[A] = D2[A]
        C[1:-1, 1:-1] = C0

    return C0


def paint(I, value, seeds):
    I = I.copy()
    h, w = I.shape[:2]
    seeds = np.asarray(seeds, dtype=int).reshape(-1, 2)
    if seeds.shape[0] == 0:
        return I

    r, c = seeds[:, 0] - 1, seeds[:, 1] - 1
    v = I[r, c]
    if np.any(v != v[0]):
        raise ValueError("all seeds not on the same value")

    B = I == v[0]
    A0 = np.zeros((h, w), bool)
    A0[r, c] = True
    A2 = np.zeros((h + 2, w + 2), bool)
    A2[1:-1, 1:-1] = A0
    while True:
        A2[0:-2, 1:-1] |= A0
        A2[1:-1, 0:-2] |= A0
        A2[1:-1, 2:] |= A0
        A2[2:, 1:-1] |= A0
        A2[:, 0] = False
        A2[:, -1] = False
        A2[0, :] = False
        A2[-1, :] = False
        A2[1:-1, 1:-1] &= B
        if np.array_equal(A2[1:-1, 1:-1], A0):
            break

        A0 = A2[1:-1, 1:-1].copy()

    I[A0] = value

    return I


def interp_mask_left_vertical(h, w, P, interp_dist):
    d = max(int(_round(interp_dist)) + 1, 1)
    A = -np.ones((h, w))
    P = np.asarray(P, dtype=int)
    rows = (P[:, 0] - 1) % h  # == mod(P-1,h)+1, zero-based
    cols = P[:, 1] - 1
    A[rows, cols] = 0
    A = np.tile(A, (3, 1))
    a = np.nonzero(A[:, 0] == -1)[0] + 1
    seeds = np.stack([a, np.ones_like(a)], axis=1)
    A = paint(A, 1, seeds)
    A = A[h : 2 * h, :]
    D = d * np.ones((h, w))
    D[rows, cols] = 0
    if d > 1:
        D = compute_costs(np.ones((3 * h, w)), np.tile(D, (3, 1)))
        D = D[h : 2 * h, :]
        D = D / d
    D = D * A

    return (D + 1) / 2


def automatic_coordinates_for_max_tile(images, window_size, shift, coordinates):
    wv, wh = window_size
    sv, sh = shift
    y1, y2, x1, x2 = coordinates
    wt, wb = wh // 2, (wh - 1) // 2
    wl, wr = wv // 2, (wv - 1) // 2

    ET = images[_s(y1 - wt, y1 + wb), _s(x1, x2 - 1), :]
    best_err, best_y2 = np.inf, y2
    for sy2 in range(y2 - sh, y2 + 1):
        EB = images[_s(sy2 - wt, sy2 + wb), _s(x1, x2 - 1), :]
        err = distance(ET, EB).sum()
        if err < best_err:
            best_err, best_y2 = err, sy2
    y2 = best_y2

    EL = images[_s(y1, y2 - 1), _s(x1 - wl, x1 + wr), :]
    best_err, best_sx2 = np.inf, x2
    for sx2 in range(x2 - sv, x2 + 1):
        ER = images[_s(y1, y2 - 1), _s(sx2 - wl, sx2 + wr), :]
        err = distance(EL, ER).sum()
        if err < best_err:
            best_err, best_sx2 = err, sx2
    x2 = best_sx2

    return [y1, y2, x1, x2]


def joint_coordinates_for_max_tile(images, window_size, shift, coordinates):
    wv, wh = window_size
    sv, sh = shift
    y1, y2, x1, x2 = coordinates
    wt, wb = wh // 2, (wh - 1) // 2
    wl, wr = wv // 2, (wv - 1) // 2

    ET = images[_s(y1 - wt, y1 + wb), _s(x1, x2 - 1), :]
    best_err, best = np.inf, (y2, x2)
    for sy2 in range(y2 - sh, y2 + 1):
        EB = images[_s(sy2 - wt, sy2 + wb), _s(x1, x2 - 1), :]
        top = distance(ET, EB).mean()
        EL = images[_s(y1, sy2 - 1), _s(x1 - wl, x1 + wr), :]
        for sx2 in range(x2 - sv, x2 + 1):
            ER = images[_s(y1, sy2 - 1), _s(sx2 - wl, sx2 + wr), :]
            err = top + distance(EL, ER).mean()
            if err < best_err:
                best_err, best = err, (sy2, sx2)
    y2, x2 = best

    return [y1, y2, x1, x2]


def min_costs(E, P):
    h, w = E.shape[:2]
    P = np.asarray(P, dtype=int).reshape(-1, 2)
    C = np.full((h + 2, w + 2), np.inf)
    C[P[:, 0], P[:, 1]] = E[P[:, 0] - 1, P[:, 1] - 1]  # padded idx = P (1-based +1, then -1)

    F = np.zeros(h + 2, bool)  # padded rows, 0-based
    F[P[:, 0]] = True  # MATLAB F(P+1)
    F[P[:, 0] - 1] = True  # F(P)
    F[P[:, 0] + 1] = True  # F(P+2)
    F[[0, -1]] = False

    D = np.zeros((h, w, 8))
    while True:
        F0 = F[1:-1].copy()
        Ft = np.concatenate([F0, [False, False]])
        Fc = np.concatenate([[False], F0, [False]])
        Fb = np.concatenate([[False, False], F0])

        D[F0, :, 0] = C[Ft, 0:-2]
        D[F0, :, 1] = C[Ft, 1:-1]
        D[F0, :, 2] = C[Ft, 2:]
        D[F0, :, 3] = C[Fc, 0:-2]
        D[F0, :, 4] = C[Fc, 2:]
        D[F0, :, 5] = C[Fb, 0:-2]
        D[F0, :, 6] = C[Fb, 1:-1]
        D[F0, :, 7] = C[Fb, 2:]

        D2 = D[F0, :, :].min(axis=2)
        D2 = D2 + E[F0, :]

        CC = C[F, 1:-1].copy()
        A = D2 < CC
        a = A.any(axis=1)
        if not a.any():
            break
        CC[A] = D2[A]
        C[F, 1:-1] = CC

        F[Fc] = a
        F[Ft] = F[Ft] | a
        F[Fb] = F[Fb] | a
        F[[0, -1]] = False

    return C


def costs_to_path(C, PS, PE):
    C = C.copy()
    PS = np.asarray(PS, dtype=int).reshape(-1, 2)
    PE = np.asarray(PE, dtype=int).reshape(-1, 2)
    vals = C[PE[:, 0], PE[:, 1]]  # padded 0-based index == 1-based coord
    idx = int(np.argmin(vals))  # first minimum
    y, x = int(PE[idx, 0]), int(PE[idx, 1])
    C[PE[:, 0], PE[:, 1]] = np.inf

    path = [(y, x)]
    limit = C.size + 2
    while True:
        if not (1 <= y <= C.shape[0] - 2 and 1 <= x <= C.shape[1] - 2):
            raise RuntimeError("costs_to_path: no path to a start position")  # MATLAB: index error
        c = C[y - 1 : y + 2, x - 1 : x + 2]  # MATLAB C(y+1+(-1:1), x+1+(-1:1))
        k = int(np.argmin(c.ravel(order="F")))  # column-major, first minimum
        y += k % 3 - 1
        x += k // 3 - 1
        path.append((y, x))
        C[y, x] = np.inf  # MATLAB C(y+1, x+1) = Inf
        if np.any((PS[:, 0] == y) & (PS[:, 1] == x)):
            break
        if len(path) > limit:
            raise RuntimeError("costs_to_path: no path to a start position")

    return np.asarray(path[::-1], dtype=int)


def optimal_path(E, start_positions, end_positions):
    C = min_costs(E, start_positions)

    return costs_to_path(C, start_positions, end_positions), C


def find_paths(I, wh, wv, idh, idv):
    wt, wb = wh // 2, (wh - 1) // 2
    wl, wr = wv // 2, (wv - 1) // 2
    h, w = I.shape[:2]

    y_h_t = np.arange(1, wh + 1) + idh
    y_h_b = h - idh - wh + np.arange(1, wh + 1)
    x_h = np.arange(1 + idv + wv, w - idv - wv + 1)
    y_v = np.arange(1 + idh + wh, h - idh - wh + 1)
    x_v_l = idv + np.arange(1, wv + 1)
    x_v_r = w - idv - wv + np.arange(1, wv + 1)

    def g(rows, cols):  # contiguous ranges: views, not copies
        rows, cols = _idx(rows), _idx(cols)
        if rows.size == 0 or cols.size == 0:
            return I[:0, :0, :]

        return I[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1, :]

    ET, EB = g(y_h_t, x_h), g(y_h_b, x_h)
    EL, ER = g(y_v, x_v_l), g(y_v, x_v_r)
    CTL, CTR = g(y_h_t, x_v_l), g(y_h_t, x_v_r)
    CBL, CBR = g(y_h_b, x_v_l), g(y_h_b, x_v_r)

    DH = distance(ET, EB)
    DV = distance(EL, ER)
    DC = distance4(CTL, CTR, CBL, CBR)
    DH2 = np.hstack([DC, DH, DC])
    DV2 = np.vstack([DC, DV, DC])

    n = DH2.T.shape[0]
    PH, _ = optimal_path(DH2.T, [[1 + wl, 1 + wt]], [[n - wr, 1 + wt]])
    PH = PH[:, [1, 0]]
    PH = np.stack([idh + PH[:, 0], idv + PH[:, 1]], axis=1)

    n = DV2.shape[0]
    PV, _ = optimal_path(DV2, [[1 + wt, 1 + wl]], [[n - wb, 1 + wl]])
    PV = np.stack([idh + PV[:, 0], idv + PV[:, 1]], axis=1)

    return PH, PV


def blend_tile(I0, PH, PV, wh, wv, idh, idv, y1, y2, x1, x2):
    wt, wb = wh // 2, (wh - 1) // 2
    wl, wr = wv // 2, (wv - 1) // 2
    h0, w0 = I0.shape[:2]

    PH2 = np.stack([PH[:, 1] - idv - wl, PH[:, 0]], axis=1)
    MT = interp_mask_left_vertical(x2 - x1, idh + wh + idh, PH2, idh).T
    PV2 = np.stack([PV[:, 0] - idh - wt, PV[:, 1]], axis=1)
    ML = interp_mask_left_vertical(y2 - y1, idv + wv + idv, PV2, idv)

    y_h_t = _s(1, idh + wh + idh)
    y_h_b = _s(h0 + 1 - idh - wh - idh, h0)
    x_h = _s(1 + idv + wv + idv, w0 - idv - wv - idv)
    y_v = _s(1 + idh + wh + idh, h0 - idh - wh - idh)
    x_v_l = _s(1, idv + wv + idv)
    x_v_r = _s(w0 + 1 - idv - wv - idv, w0)

    MCT = np.hstack([MT[:, MT.shape[1] - wl - idv :], MT[:, : 1 + wr + idv]])[..., None]
    MCL = np.vstack([ML[ML.shape[0] - wt - idh :, :], ML[: 1 + wb + idh, :]])[..., None]
    MET = MT[:, wr + idv + 1 : MT.shape[1] - wl - idv][..., None]
    MEL = ML[wb + idh + 1 : ML.shape[0] - wt - idh, :][..., None]
    for k in range(0, I0.shape[2], _CHUNK):  # planes are independent
        P = I0[:, :, k : k + _CHUNK]  # view: edits land in I0
        # horizontal edge
        ET, EB = P[y_h_t, x_h, :], P[y_h_b, x_h, :]
        EH = (1 - MET) * ET + MET * EB
        P[y_h_t, x_h, :] = EH
        P[y_h_b, x_h, :] = EH

        # vertical edge
        EL, ER = P[y_v, x_v_l, :], P[y_v, x_v_r, :]
        EV = (1 - MEL) * EL + MEL * ER
        P[y_v, x_v_l, :] = EV
        P[y_v, x_v_r, :] = EV

        # corners (taken from the already edge-blended planes, as in MATLAB)
        CTL, CTR = P[y_h_t, x_v_l, :], P[y_h_t, x_v_r, :]
        CBL, CBR = P[y_h_b, x_v_l, :], P[y_h_b, x_v_r, :]
        C = CTL * (1 - MCT) * (1 - MCL) + CTR * (1 - MCT) * MCL + CBL * MCT * (1 - MCL) + CBR * MCT * MCL
        P[y_h_t, x_v_l, :] = C
        P[y_h_t, x_v_r, :] = C
        P[y_h_b, x_v_l, :] = C
        P[y_h_b, x_v_r, :] = C

    # final tile: size (y2-y1) x (x2-x1) -- one smaller than reported area
    return I0[_s(1 + idh + wt, h0 - idh - wb - 1), _s(1 + idv + wl, w0 - idv - wr - 1), :]


def _check_crop(shape_hw, wh, wv, idh, idv, y1, y2, x1, x2):
    wt, wb = wh // 2, (wh - 1) // 2
    wl, wr = wv // 2, (wv - 1) // 2
    h, w = shape_hw
    if y1 - wt - idh < 1 or y2 + wb + idh > h or x1 - wl - idv < 1 or x2 + wr + idv > w:
        raise ValueError("wrong parameters")
    # DEVIATION (invalid input only): when a window is larger than the tile the
    # top/bottom (left/right) windows overlap and MATLAB's index ranges become
    # empty/negative, silently producing a meaningless tile.  Python negative
    # slice indices would wrap instead, so raise a clear error.
    if y2 - y1 - wh - 2 * idh <= 0 or x2 - x1 - wv - 2 * idv <= 0:
        raise ValueError(
            "window_size too large for the selected tile: top/bottom (left/right) "
            f"windows overlap (tile {y2 - y1} x {x2 - x1}, window {wh} x {wv}, interp_dist {idh}/{idv})"
        )

    return _s(y1 - wt - idh, y2 + wb + idh), _s(x1 - wl - idv, x2 + wr + idv)


def create_tile(I, wh, wv, idh, idv, y1, y2, x1, x2, WI=None, inplace=False):
    ys, xs = _check_crop(I.shape[:2], wh, wv, idh, idv, y1, y2, x1, x2)
    I0 = I[ys, xs, :]
    WI0 = I0 if WI is None else WI[ys, xs, :]
    PH, PV = find_paths(WI0, wh, wv, idh, idv)
    if not inplace:
        I0 = I0.copy()

    return blend_tile(I0, PH, PV, wh, wv, idh, idv, y1, y2, x1, x2), (PH, PV)


def cut_tile(I, paths, wh, wv, idh, idv, y1, y2, x1, x2):
    ys, xs = _check_crop(I.shape[:2], wh, wv, idh, idv, y1, y2, x1, x2)
    PH, PV = paths

    return blend_tile(I[ys, xs, :].copy(), PH, PV, wh, wv, idh, idv, y1, y2, x1, x2)


def _as_planes(img):
    img = np.asarray(img)

    return img[:, :, None] if img.ndim == 2 else img


def normalise(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if np.issubdtype(img.dtype, np.floating):
        return img.astype(np.float32)
    raise ValueError("images not of proper data type")


def _stack_images(images, h, w):
    depths = [1 if np.ndim(img) == 2 else np.shape(img)[2] for img in images]
    out = np.empty((h, w, sum(depths)), dtype=np.float32)
    j = 0
    for img, depth in zip(images, depths, strict=True):
        img = np.asarray(img)
        if img.shape[:2] != (h, w):
            raise ValueError("images not of the same size")
        out[:, :, j : j + depth] = _as_planes(normalise(img))
        j += depth

    return out, depths


def restore_dtype(tile, dtype):
    if dtype == np.uint8:
        return np.clip(np.rint(tile * 255.0), 0, 255).astype(np.uint8)
    if dtype == np.uint16:
        return np.clip(np.rint(tile * 65535.0), 0, 65535).astype(np.uint16)

    return tile.astype(dtype)


def resolve_params(shape_hw, window_size, shift, interp_dist, area):
    h, w = shape_hw
    ws = np.atleast_1d(np.asarray(window_size, dtype=float))
    if ws.size == 0 or ws.size > 2 or np.any(ws != np.round(ws)):
        raise ValueError("window_size not set properly")
    ws = np.tile(ws, 2) if ws.size == 1 else ws
    ws = ws.astype(int)

    sh = np.atleast_1d(np.asarray(0 if shift is None or np.size(shift) == 0 else shift, dtype=float))
    if np.any(sh != np.round(sh)) or sh.size not in (1, 4):
        raise ValueError("shift not set properly")
    sh = (np.tile(sh, 4) if sh.size == 1 else sh).astype(int)

    idv_ = np.atleast_1d(
        np.asarray(
            0 if interp_dist is None or np.size(interp_dist) == 0 else interp_dist,
            dtype=float,
        )
    )
    if np.any(idv_ != np.round(idv_)) or idv_.size > 2:
        raise ValueError("interp_dist not set properly")
    idv_ = (np.tile(idv_, 2) if idv_.size == 1 else idv_).astype(int)

    wh, wv = int(ws[1]), int(ws[0])
    wt, wb = wh // 2, (wh - 1) // 2
    wl, wr = wv // 2, (wv - 1) // 2
    idv, idh = int(idv_[0]), int(idv_[1])  # MATLAB: idv=id(1), idh=id(2)

    y1_min, y2_max = 1 + idh + wt, h - idh - wb
    x1_min, x2_max = 1 + idv + wl, w - idv - wr
    if y1_min >= y2_max or x1_min >= x2_max:
        raise ValueError("window_size and/or interp_dist are too big with respect to size of provided images")

    a = np.atleast_1d(np.asarray([] if area is None else area, dtype=float))
    if a.size and np.any(a != np.round(a)):
        raise ValueError("area not set properly")
    if a.size == 0:
        a = [y1_min, x1_min, y2_max - sh[3] - y1_min + 1, x2_max - sh[1] - x1_min + 1]
    elif a.size == 1:
        a = [_round(h / 2 - a[0] / 2) + 1, _round(w / 2 - a[0] / 2) + 1, a[0], a[0]]
    elif a.size == 2:
        a = [_round(h / 2 - a[0] / 2) + 1, _round(w / 2 - a[1] / 2) + 1, a[0], a[1]]
    elif a.size == 4:
        a = [a[0] + 1, a[1] + 1, a[2], a[3]]  # 0-based input -> 1-based
    else:
        raise ValueError("incorrect number of area values")
    a = [int(v) for v in a]
    y1, x1, y2, x2 = a[0], a[1], a[0] + a[2] - 1, a[1] + a[3] - 1
    if y1 < y1_min or y2 > y2_max - sh[3] or x1 < x1_min or x2 > x2_max - sh[1]:
        raise ValueError("area specified incorrectly")
    if y1 >= y2 or x1 >= x2:
        raise ValueError("area not set properly")
    if y1 >= y2 - sh[2] or x1 >= x2 - sh[0]:
        raise ValueError("area or shift not set properly")

    return dict(
        wh=wh,
        wv=wv,
        ws=(int(ws[0]), int(ws[1])),
        shift=[int(v) for v in sh],
        idh=idh,
        idv=idv,
        y1=y1,
        y2=y2,
        x1=x1,
        x2=x2,
    )


def find_tile_detailed(
    images,
    window_size,
    shift,
    interp_dist,
    area=None,
    guidance_images=None,
    joint_search=False,
):
    single = not isinstance(images, (list, tuple))
    imgs = [images] if single else list(images)
    if not imgs:
        raise ValueError("images not provided")
    dtypes = [np.asarray(i).dtype for i in imgs]
    ndims = [np.asarray(i).ndim for i in imgs]
    h, w = np.shape(imgs[0])[:2]

    p = resolve_params((h, w), window_size, shift, interp_dist, area)
    I, ds = _stack_images(imgs, h, w)
    del imgs

    WI = None
    if guidance_images is not None and not (
        hasattr(guidance_images, "__len__") and len(guidance_images) == 0
    ):
        g = [guidance_images] if not isinstance(guidance_images, (list, tuple)) else list(guidance_images)
        try:
            WI, _ = _stack_images(g, h, w)
        except ValueError as exc:
            raise ValueError("guidance_images not of the correct size") from exc
    search = I if WI is None else WI

    sh = p["shift"]
    coordinates2 = [p["y1"], p["y2"] + sh[3], p["x1"], p["x2"] + sh[1]]
    shift2 = [sh[0] + sh[1], sh[2] + sh[3]]
    search_fn = joint_coordinates_for_max_tile if joint_search else automatic_coordinates_for_max_tile
    y1, y2, x1, x2 = search_fn(search, p["ws"], shift2, coordinates2)
    out_area = (y1, x1, y2 - y1 + 1, x2 - x1 + 1)

    T, paths = create_tile(I, p["wh"], p["wv"], p["idh"], p["idv"], y1, y2, x1, x2, WI, inplace=True)

    tiles, j = [], 0
    for dt, nd, d in zip(dtypes, ndims, ds, strict=True):
        tile = restore_dtype(T[:, :, j : j + d], dt)
        j += d
        tiles.append(tile[:, :, 0] if nd == 2 else tile)
    wt, wl = p["wh"] // 2, p["wv"] // 2
    details = dict(
        paths=paths,
        params=p,
        area=(y1, y2, x1, x2),
        crop_origin=(y1 - wt - p["idh"], x1 - wl - p["idv"]),
        crop_size=(
            y2 - y1 + wt + (p["wh"] - 1) // 2 + 2 * p["idh"] + 1,
            x2 - x1 + wl + (p["wv"] - 1) // 2 + 2 * p["idv"] + 1,
        ),
    )

    return (tiles[0] if single else tiles), out_area, details


def find_tile(images, window_size, shift, interp_dist, area=None, guidance_images=None):
    tiles, out_area, _ = find_tile_detailed(images, window_size, shift, interp_dist, area, guidance_images)

    return tiles, out_area, None
