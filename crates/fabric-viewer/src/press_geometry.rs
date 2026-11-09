//! Pressed-sheet geometry and coordinate transforms.
//!
//! Sheet coordinates: `s` along the pull, `a` across, depth away from the viewer.
//! Atlas coordinates: `u` right, `v` down. Vertical pulls map `(s, a) = (v, 1 - u)`
//! (90° counter-clockwise); horizontal pulls map `(u, v)`. Neither mirrors the texture.
//!
//! Photometric data and pad-screen frames both use x right, y up, z towards the viewer.
//! [`SheetMap::data_to_screen`] applies the atlas rotation to photometric vectors.

use crate::lighting::{self, Light};
use egui::{Pos2, Rect, Vec2};

/// How the atlas is laid onto the sheet. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SheetMap {
    /// The pull runs along the atlas rows (`active_grid_axis: vertical`).
    pub pull_vertical: bool,
}

impl SheetMap {
    /// Atlas coordinate of the sheet point `(s, a)`.
    pub fn atlas_uv(self, s: f32, a: f32) -> [f32; 2] {
        if self.pull_vertical {
            [1.0 - a, s]
        } else {
            [s, a]
        }
    }

    /// Rotates a vector of the data frame (`x` right, `y` up) into the screen frame. This is the
    /// rotation by which the texture is displayed.
    pub fn data_to_screen(self, v: [f32; 2]) -> [f32; 2] {
        if self.pull_vertical {
            [-v[1], v[0]]
        } else {
            v
        }
    }

    /// Inverse of [`SheetMap::data_to_screen`].
    pub fn screen_to_data(self, v: [f32; 2]) -> [f32; 2] {
        if self.pull_vertical {
            [v[1], -v[0]]
        } else {
            v
        }
    }

    /// Cloth width across the pull relative to the pull length, from the atlas aspect
    /// (`dimensions_px`): a 1088 x 448 atlas pulled along its rows is 2.43 pull lengths wide.
    pub fn across_length(self, atlas_px: [u32; 2]) -> f32 {
        let (w, h) = (atlas_px[0].max(1) as f32, atlas_px[1].max(1) as f32);
        if self.pull_vertical {
            w / h
        } else {
            h / w
        }
    }
}

/// Two planes from clamps at 0 and 1 to a bar at `beta`, depth `delta`, in clamp-span units.
/// Stretch is `hypot(beta, delta) + hypot(1 - beta, delta)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackageGeometry {
    pub beta: f32,
    pub delta: f32,
    pub lambda: f32,
    /// Fraction of the material length that lies on the first plane.
    pub v_bar: f32,
}

impl PackageGeometry {
    pub fn new(beta: f32, delta: f32) -> Option<Self> {
        if !beta.is_finite() || !delta.is_finite() || !(0.0..=1.0).contains(&beta) || delta < 0.0 {
            return None;
        }
        let l1 = beta.hypot(delta);
        let l2 = (1.0 - beta).hypot(delta);
        let lambda = l1 + l2;
        Some(Self {
            beta,
            delta,
            lambda,
            v_bar: l1 / lambda,
        })
    }

    /// Deepest indentation at bar position `beta` that keeps the stretch at or below `lambda_max`.
    pub fn max_delta(beta: f32, lambda_max: f32) -> f32 {
        if !lambda_max.is_finite() || lambda_max <= 1.0 {
            return 0.0;
        }
        let (mut lo, mut hi) = (0.0, lambda_max);
        for _ in 0..40 {
            let mid = (lo + hi) * 0.5;
            let stretch = beta.hypot(mid) + (1.0 - beta).hypot(mid);
            if stretch <= lambda_max {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// `(pull, depth)` of the point at parameter `t` (0..1) of a plane.
    pub fn point(self, side: usize, t: f32) -> [f32; 2] {
        if side == 0 {
            [self.beta * t, self.delta * t]
        } else {
            [self.beta + (1.0 - self.beta) * t, self.delta * (1.0 - t)]
        }
    }

    /// Position along the material length (0..1) at parameter `t` of a plane. Continuous across
    /// the bar.
    pub fn material_v(self, side: usize, t: f32) -> f32 {
        if side == 0 {
            self.v_bar * t
        } else {
            self.v_bar + (1.0 - self.v_bar) * t
        }
    }

    /// Plane tilt about the across axis; positive normals lean screen-right.
    pub fn tilt(self, side: usize) -> f32 {
        if side == 0 {
            self.delta.atan2(self.beta)
        } else {
            -self.delta.atan2(1.0 - self.beta)
        }
    }
}

/// Plan view or oblique view of the sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Oblique,
    Top,
}

impl ViewMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Oblique => "oblique",
            Self::Top => "top",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Oblique => "Oblique",
            Self::Top => "Top",
        }
    }
}

/// The oblique view looks at the sheet from this angle away from straight above.
pub const OBLIQUE_TILT_DEG: f32 = 30.0;

/// Orthographic projection with pull along screen x. Oblique view tilts the across/depth
/// axes by [`OBLIQUE_TILT_DEG`]; pointer x therefore determines bar position in both views.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    mode: ViewMode,
    origin: Pos2,
    scale: f32,
    across_len: f32,
}

impl Projection {
    /// Fits the whole scene into `canvas`: the full clamp span, the cloth width `across_len`
    /// (relative to the span) and a bar pressed down to `max_depth`.
    pub fn fit(canvas: Rect, mode: ViewMode, across_len: f32, max_depth: f32) -> Self {
        let (sin, cos) = OBLIQUE_TILT_DEG.to_radians().sin_cos();
        let height = match mode {
            ViewMode::Top => across_len,
            ViewMode::Oblique => across_len * sin + max_depth * cos,
        };
        let margin = 0.06;
        let scale = ((canvas.width() / (1.0 + 2.0 * margin))
            .min(canvas.height() / (height + 2.0 * margin)))
        .max(1.0);
        let origin = Pos2::new(
            canvas.center().x - scale * 0.5,
            canvas.center().y - scale * height * 0.5,
        );
        Self {
            mode,
            origin,
            scale,
            across_len,
        }
    }

    pub fn mode(&self) -> ViewMode {
        self.mode
    }

    /// Pixels per clamp span.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Screen position of the sheet point `(pull, depth)` at across fraction `across` (0..1).
    pub fn point(&self, pull: f32, depth: f32, across: f32) -> Pos2 {
        let (sin, cos) = OBLIQUE_TILT_DEG.to_radians().sin_cos();
        let y = match self.mode {
            ViewMode::Top => across * self.across_len,
            ViewMode::Oblique => across * self.across_len * sin + depth * cos,
        };
        Pos2::new(
            self.origin.x + self.scale * pull,
            self.origin.y + self.scale * y,
        )
    }

    /// Pull coordinate under screen `x`. Exact inverse of [`Projection::point`] for any depth.
    pub fn pull_at_x(&self, x: f32) -> f32 {
        (x - self.origin.x) / self.scale
    }

    /// Screen segment of the bar from the top to the bottom edge of the cloth.
    pub fn bar(&self, geometry: &PackageGeometry) -> [Pos2; 2] {
        [0.0, 1.0].map(|a| self.point(geometry.beta, geometry.delta, a))
    }
}

/// One projected cloth plane: screen corners and their texture coordinates, ordered
/// `(t, across) = (0,0), (1,0), (0,1), (1,1)` and drawn as triangles `(0,2,1)` and `(1,2,3)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClothQuad {
    pub pos: [Pos2; 4],
    /// Texture coordinates: atlas `(u, v)`, or continuous tile units for tiled packages.
    pub uv: [[f32; 2]; 4],
    /// Which plane (0 = left of the bar, 1 = right of it).
    pub plane: usize,
}

impl ClothQuad {
    pub const CORNERS: [(f32, f32); 4] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    pub const TRIANGLES: [[u32; 3]; 2] = [[0, 2, 1], [1, 2, 3]];
}

/// Projected cloth silhouette. Material coordinates stay continuous across the bar;
/// the sampler wraps tile coordinates.
pub fn cloth_quads(
    geometry: &PackageGeometry,
    projection: &Projection,
    map: SheetMap,
    tiling: Option<crate::material_package::Tiling>,
) -> [ClothQuad; 2] {
    [0, 1].map(|plane| {
        let mut quad = ClothQuad {
            pos: [Pos2::ZERO; 4],
            uv: [[0.0; 2]; 4],
            plane,
        };
        for (i, (t, across)) in ClothQuad::CORNERS.into_iter().enumerate() {
            let [pull, depth] = geometry.point(plane, t);
            quad.pos[i] = projection.point(pull, depth, across);
            let material = map.atlas_uv(geometry.material_v(plane, t), across);
            quad.uv[i] = tiling.map_or(material, |tiling| tiling.texture_uv(material));
        }
        quad
    })
}

/// Rotates a screen-frame normal about the across axis (screen `y`) by `tilt`.
/// `(0, 0, 1)` becomes `(sin tilt, 0, cos tilt)`.
pub fn tilt_normal(n: [f32; 3], tilt: f32) -> [f32; 3] {
    let (s, c) = tilt.sin_cos();
    [c * n[0] + s * n[2], n[1], -s * n[0] + c * n[2]]
}

/// Screen-space light in a tilted plane's data frame.
/// Apply inverse tilt and atlas rotation to the light instead of rotating every texel normal.
pub fn plane_light_in_data_frame(light: &Light, map: SheetMap, tilt: f32) -> [f32; 3] {
    let screen = lighting::light_dir(light);
    let (s, c) = tilt.sin_cos();
    let untilted = [
        c * screen[0] - s * screen[2],
        screen[1],
        s * screen[0] + c * screen[2],
    ];
    let xy = map.screen_to_data([untilted[0], untilted[1]]);
    [xy[0], xy[1], untilted[2]]
}

/// Screen size of the bar's pointer target in points.
pub const BAR_GRAB_RADIUS: f32 = 14.0;

/// Whether `p` is within [`BAR_GRAB_RADIUS`] of the bar segment.
pub fn near_segment(p: Pos2, segment: [Pos2; 2]) -> bool {
    let d: Vec2 = segment[1] - segment[0];
    let len2 = d.length_sq().max(1.0e-6);
    let t = ((p - segment[0]).dot(d) / len2).clamp(0.0, 1.0);
    (p - (segment[0] + d * t)).length() <= BAR_GRAB_RADIUS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::normal;

    const VERTICAL: SheetMap = SheetMap {
        pull_vertical: true,
    };
    const HORIZONTAL: SheetMap = SheetMap {
        pull_vertical: false,
    };

    fn canvas() -> Rect {
        Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(800.0, 500.0))
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn package_geometry_is_a_fixed_two_plane_contact() {
        let g = PackageGeometry::new(0.4, 0.3).unwrap();
        assert!((g.lambda - (0.4f32.hypot(0.3) + 0.6f32.hypot(0.3))).abs() < 1.0e-6);
        assert_eq!(g.point(0, 0.0), [0.0, 0.0]);
        assert_eq!(g.point(0, 1.0), [0.4, 0.3]);
        assert_eq!(g.point(1, 0.0), [0.4, 0.3]);
        assert_eq!(g.point(1, 1.0), [1.0, 0.0]);
        // The material coordinate is continuous across the bar and covers 0..1.
        assert_eq!(g.material_v(0, 0.0), 0.0);
        assert!((g.material_v(0, 1.0) - g.material_v(1, 0.0)).abs() < 1.0e-6);
        assert!((g.material_v(1, 1.0) - 1.0).abs() < 1.0e-6);
        let flat = PackageGeometry::new(0.5, 0.0).unwrap();
        assert_eq!((flat.lambda, flat.v_bar), (1.0, 0.5));
        assert!(
            PackageGeometry::new(1.2, 0.1).is_none() && PackageGeometry::new(0.5, -0.1).is_none()
        );
        // The deepest indentation exactly reaches the stretch limit.
        let deepest = PackageGeometry::max_delta(0.3, 1.25);
        let at_limit = PackageGeometry::new(0.3, deepest).unwrap().lambda;
        assert!((at_limit - 1.25).abs() < 1.0e-4, "{at_limit}");
        assert_eq!(PackageGeometry::max_delta(0.5, 1.0), 0.0);
    }

    #[test]
    fn plane_tilts_follow_the_bar_position_and_depth() {
        let g = PackageGeometry::new(0.5, 0.1).unwrap();
        assert!((g.tilt(0) - (0.1f32 / 0.5).atan()).abs() < 1.0e-6);
        assert!((g.tilt(1) + (0.1f32 / 0.5).atan()).abs() < 1.0e-6);
        let off = PackageGeometry::new(0.3, 0.2).unwrap();
        assert!((off.tilt(0).to_degrees() - 33.7).abs() < 0.1);
        assert!((off.tilt(1).to_degrees() + 15.9).abs() < 0.1);
        let flat = PackageGeometry::new(0.5, 0.0).unwrap();
        assert_eq!((flat.tilt(0), flat.tilt(1)), (0.0, 0.0));
    }

    /// Signed UV-to-screen area. Both axes are y-down, so positive means rotation, negative mirror.
    fn uv_to_screen_determinant(quad: &ClothQuad) -> f32 {
        let du = [quad.pos[1].x - quad.pos[0].x, quad.pos[1].y - quad.pos[0].y];
        let dt = [quad.pos[2].x - quad.pos[0].x, quad.pos[2].y - quad.pos[0].y];
        let uv_du = [quad.uv[1][0] - quad.uv[0][0], quad.uv[1][1] - quad.uv[0][1]];
        let uv_dt = [quad.uv[2][0] - quad.uv[0][0], quad.uv[2][1] - quad.uv[0][1]];
        // screen = M uv  =>  det M = det(screen basis) / det(uv basis)
        (du[0] * dt[1] - du[1] * dt[0]) / (uv_du[0] * uv_dt[1] - uv_du[1] * uv_dt[0])
    }

    #[test]
    fn the_atlas_is_rotated_onto_the_sheet_never_mirrored() {
        for mode in [ViewMode::Top, ViewMode::Oblique] {
            for (map, across) in [(VERTICAL, 2.4), (HORIZONTAL, 0.4)] {
                let projection = Projection::fit(canvas(), mode, across, 0.4);
                for geometry in [
                    PackageGeometry::new(0.5, 0.0).unwrap(),
                    PackageGeometry::new(0.3, 0.2).unwrap(),
                ] {
                    for quad in cloth_quads(&geometry, &projection, map, None) {
                        assert!(
                            uv_to_screen_determinant(&quad) > 0.0,
                            "{mode:?} {map:?}: the texture is mirrored"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_asymmetric_atlas_lands_in_the_expected_screen_corners() {
        // Vertical pull rotates atlas corners 90° counter-clockwise without mirroring.
        let projection = Projection::fit(canvas(), ViewMode::Top, 2.0, 0.0);
        let flat = PackageGeometry::new(0.5, 0.0).unwrap();
        let quads = cloth_quads(&flat, &projection, VERTICAL, None);
        let screen_of = |u: f32, v: f32| {
            for quad in &quads {
                for i in 0..4 {
                    if (quad.uv[i][0] - u).abs() < 1.0e-6 && (quad.uv[i][1] - v).abs() < 1.0e-6 {
                        return quad.pos[i];
                    }
                }
            }
            panic!("no corner at uv ({u}, {v})");
        };
        let (tl, tr, bl, br) = (
            screen_of(0.0, 0.0),
            screen_of(1.0, 0.0),
            screen_of(0.0, 1.0),
            screen_of(1.0, 1.0),
        );
        assert!(tl.x < br.x && tr.x < br.x, "image rows run left to right");
        assert!(
            tr.y < tl.y,
            "image columns run bottom to top (right edge at the top)"
        );
        assert!((tr.x - tl.x).abs() < 1.0e-3 && (bl.x - br.x).abs() < 1.0e-3);
        assert!(bl.y > tr.y && br.y < bl.y + 1.0e-3 && (br.y - tr.y).abs() < 1.0e-3);
        let quads = cloth_quads(&flat, &projection, HORIZONTAL, None);
        let first = &quads[0];
        assert!(first.pos[1].x > first.pos[0].x && first.pos[2].y > first.pos[0].y);
        assert!(
            (first.uv[2][1] - first.uv[0][1]).abs() > 0.99,
            "across runs down the rows"
        );
    }

    #[test]
    fn tiling_only_rescales_the_material_uv_never_the_silhouette() {
        let projection = Projection::fit(canvas(), ViewMode::Oblique, 2.4, 0.4);
        let geometry = PackageGeometry::new(0.35, 0.25).unwrap();
        let plain = cloth_quads(&geometry, &projection, VERTICAL, None);
        let tiling = crate::material_package::Tiling {
            origin_uv: [0.03, 0.13],
            period_uv: [0.93, 0.66],
        };
        let tiled = cloth_quads(&geometry, &projection, VERTICAL, Some(tiling));
        for (a, b) in plain.iter().zip(&tiled) {
            assert_eq!(a.pos, b.pos);
            for i in 0..4 {
                assert_eq!(b.uv[i], tiling.texture_uv(a.uv[i]));
            }
        }
        // The two planes share one bar edge with identical uv.
        assert_eq!(tiled[0].pos[1], tiled[1].pos[0]);
        assert_eq!(tiled[0].pos[3], tiled[1].pos[2]);
        assert_eq!(tiled[0].uv[1], tiled[1].uv[0]);
        assert_eq!(tiled[0].uv[3], tiled[1].uv[2]);
    }

    #[test]
    fn untiled_uvs_stay_inside_the_atlas() {
        let projection = Projection::fit(canvas(), ViewMode::Top, 1.0, 0.3);
        for beta in [0.1, 0.5, 0.9] {
            let geometry = PackageGeometry::new(beta, 0.2).unwrap();
            for map in [VERTICAL, HORIZONTAL] {
                for quad in cloth_quads(&geometry, &projection, map, None) {
                    for uv in quad.uv {
                        assert!(
                            uv.iter().all(|c| (-1.0e-6..=1.0 + 1.0e-6).contains(c)),
                            "{uv:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_projection_fits_the_canvas_and_inverts_for_dragging() {
        for mode in [ViewMode::Top, ViewMode::Oblique] {
            let canvas = canvas();
            let projection = Projection::fit(canvas, mode, 2.43, 0.41);
            // The whole scene stays inside the canvas for any bar position and depth.
            for beta in [0.05, 0.5, 0.95] {
                let g = PackageGeometry::new(beta, PackageGeometry::max_delta(beta, 1.3)).unwrap();
                for side in 0..2 {
                    for t in [0.0, 1.0] {
                        for across in [0.0, 1.0] {
                            let [pull, depth] = g.point(side, t);
                            let p = projection.point(pull, depth, across);
                            assert!(
                                canvas.expand(0.5).contains(p),
                                "{mode:?} {p:?} outside {canvas:?}"
                            );
                        }
                    }
                }
            }
            // Dragging follows pointer x at every depth.
            for depth in [0.0, 0.1, 0.4] {
                for beta in [0.1, 0.37, 0.8] {
                    let bar = Projection::bar(
                        &projection,
                        &PackageGeometry {
                            beta,
                            delta: depth,
                            lambda: 1.0,
                            v_bar: 0.5,
                        },
                    );
                    let mid = bar[0] + (bar[1] - bar[0]) * 0.5;
                    assert!((projection.pull_at_x(mid.x) - beta).abs() < 1.0e-5);
                    assert!(near_segment(mid, bar));
                    assert!(!near_segment(
                        mid + Vec2::new(BAR_GRAB_RADIUS + 5.0, 0.0),
                        bar
                    ));
                }
            }
        }
    }

    #[test]
    fn the_oblique_view_shows_the_indentation_and_foreshortens_the_across_axis() {
        let top = Projection::fit(canvas(), ViewMode::Top, 1.0, 0.4);
        let oblique = Projection::fit(canvas(), ViewMode::Oblique, 1.0, 0.4);
        let down = |p: &Projection| p.point(0.5, 0.3, 0.5).y - p.point(0.5, 0.0, 0.5).y;
        assert!(down(&top).abs() < 1.0e-4, "the plan view drops the depth");
        assert!(
            down(&oblique) > 0.1 * oblique.scale(),
            "the bar sinks on screen"
        );
        let across = |p: &Projection| p.point(0.5, 0.0, 1.0).y - p.point(0.5, 0.0, 0.0).y;
        assert!(across(&oblique) / oblique.scale() < 0.7 && across(&top) / top.scale() > 0.99);
    }

    #[test]
    fn cloth_aspect_comes_from_the_atlas() {
        assert!((VERTICAL.across_length([1088, 448]) - 1088.0 / 448.0).abs() < 1.0e-6);
        assert!((HORIZONTAL.across_length([1088, 448]) - 448.0 / 1088.0).abs() < 1.0e-6);
        assert_eq!(VERTICAL.across_length([100, 100]), 1.0);
    }

    #[test]
    fn data_and_screen_rotations_are_inverse_proper_rotations() {
        for map in [VERTICAL, HORIZONTAL] {
            for v in [[1.0, 0.0], [0.0, 1.0], [0.3, -0.7]] {
                let back = map.screen_to_data(map.data_to_screen(v));
                assert!((back[0] - v[0]).abs() < 1.0e-6 && (back[1] - v[1]).abs() < 1.0e-6);
            }
        }
        // Vertical pull: image-down (data -y) is screen-right; image-right (data +x) is screen-up.
        assert_eq!(VERTICAL.data_to_screen([0.0, -1.0]), [1.0, 0.0]);
        assert_eq!(VERTICAL.data_to_screen([1.0, 0.0]), [0.0, 1.0]);
        // Determinant +1 (no mirror).
        let (a, b) = (
            VERTICAL.data_to_screen([1.0, 0.0]),
            VERTICAL.data_to_screen([0.0, 1.0]),
        );
        assert_eq!(a[0] * b[1] - a[1] * b[0], 1.0);
    }

    #[test]
    fn a_pad_lamp_at_screen_left_brightens_normals_tilted_toward_screen_left() {
        let left = Light {
            azimuth_deg: 180.0,
            elevation_deg: 30.0,
            intensity: 1.0,
            ambient: 0.0,
        };
        let right = Light {
            azimuth_deg: 0.0,
            ..left
        };
        // Rotate a screen-left normal back into data coordinates.
        for map in [VERTICAL, HORIZONTAL] {
            let to_left = map.screen_to_data([-0.5, 0.0]);
            let n = normal::normalize3([to_left[0], to_left[1], 0.866]);
            let lit_from_left = dot(n, plane_light_in_data_frame(&left, map, 0.0));
            let lit_from_right = dot(n, plane_light_in_data_frame(&right, map, 0.0));
            assert!(
                lit_from_left > lit_from_right + 0.3,
                "{map:?}: {lit_from_left} vs {lit_from_right}"
            );
            let top = Light {
                azimuth_deg: 90.0,
                ..left
            };
            let to_up = map.screen_to_data([0.0, 0.5]);
            let n_up = normal::normalize3([to_up[0], to_up[1], 0.866]);
            assert!(
                dot(n_up, plane_light_in_data_frame(&top, map, 0.0))
                    > dot(n_up, plane_light_in_data_frame(&left, map, 0.0)) + 0.3
            );
        }
        let l = plane_light_in_data_frame(&left, VERTICAL, 0.0);
        assert!(l[1] > 0.8 && l[0].abs() < 1.0e-5, "{l:?}");
    }

    #[test]
    fn plane_light_is_the_inverse_of_rotating_and_tilting_the_normal() {
        let light = Light {
            azimuth_deg: 37.0,
            elevation_deg: 52.0,
            intensity: 1.0,
            ambient: 0.0,
        };
        let screen_light = lighting::light_dir(&light);
        let n = normal::normalize3([0.2, -0.35, 0.9]);
        for map in [VERTICAL, HORIZONTAL] {
            for tilt in [-0.6f32, 0.0, 0.25, 0.7] {
                let xy = map.data_to_screen([n[0], n[1]]);
                let direct = dot(tilt_normal([xy[0], xy[1], n[2]], tilt), screen_light);
                let via_data = dot(n, plane_light_in_data_frame(&light, map, tilt));
                assert!(
                    (direct - via_data).abs() < 1.0e-5,
                    "{map:?} {tilt}: {direct} vs {via_data}"
                );
            }
        }
    }

    #[test]
    fn per_plane_tilt_makes_the_indentation_show_in_the_shading() {
        // Flat texture normals (0,0,1), lamp at the right of the pad: the plane that descends
        // toward the bar (normal leaning right) is brighter than the plane that climbs out.
        let light = Light {
            azimuth_deg: 0.0,
            elevation_deg: 50.0,
            intensity: 1.0,
            ambient: 0.0,
        };
        let g = PackageGeometry::new(0.5, 0.15).unwrap();
        let up = [0.0, 0.0, 1.0];
        for map in [VERTICAL, HORIZONTAL] {
            let left_plane = dot(up, plane_light_in_data_frame(&light, map, g.tilt(0)));
            let right_plane = dot(up, plane_light_in_data_frame(&light, map, g.tilt(1)));
            assert!(
                left_plane > right_plane + 0.2,
                "{left_plane} vs {right_plane}"
            );
            let flat = PackageGeometry::new(0.5, 0.0).unwrap();
            let a = dot(up, plane_light_in_data_frame(&light, map, flat.tilt(0)));
            let b = dot(up, plane_light_in_data_frame(&light, map, flat.tilt(1)));
            assert!((a - b).abs() < 1.0e-6);
        }
        assert_eq!(tilt_normal([0.0, 0.0, 1.0], 0.0), [0.0, 0.0, 1.0]);
        let n = tilt_normal([0.0, 0.0, 1.0], 0.5);
        assert!((n[0] - 0.5f32.sin()).abs() < 1.0e-6 && (n[2] - 0.5f32.cos()).abs() < 1.0e-6);
    }
}
