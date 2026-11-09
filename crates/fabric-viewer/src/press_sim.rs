//! Build [`PressScene`] for interactive views and snapshots, without UI state.

use crate::lighting::Light;
use crate::material_package::{
    ForceFit, MaterialPackage, PressBranch, TexturePlan, NO_PHOTOMETRIC_NOTE,
};
use crate::press_geometry::{
    cloth_quads, plane_light_in_data_frame, PackageGeometry, Projection, SheetMap, ViewMode,
};
use crate::press_render::{ClothScene, OverlayItem, SceneFiles, Shading, Wrap};
use egui::{Color32, Rect};

/// What drives the press.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Control {
    /// Strip tension in N, mapped to a stretch by the force fit.
    Force(f32),
    /// Bar indentation as a fraction of the clamp span.
    Indentation(f32),
}

/// Smallest and largest bar position (fraction of the span).
pub const BETA_RANGE: (f32, f32) = (0.05, 0.95);

/// Result of applying a [`Control`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solved {
    pub geometry: PackageGeometry,
    /// The control actually applied (clamped to what the branch can reach).
    pub control: Control,
    /// Tension from the fit at the resulting stretch, when there is a fit.
    pub fit_force: Option<f32>,
}

/// Apply control within the fit and measured stretch ranges. Indentation needs no fit.
/// Use `previous_lambda` to keep flat fit sections stable.
pub fn solve(
    beta: f32,
    control: Control,
    fit: Option<&ForceFit>,
    lambda_max: f32,
    previous_lambda: f32,
) -> Solved {
    let beta = if beta.is_finite() {
        beta.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let (delta, control) = match (control, fit) {
        (Control::Force(force), Some(fit)) => {
            let (lo, hi) = fit.range();
            let force = if force.is_finite() { force } else { lo };
            let force = force.clamp(lo.min(hi), hi.max(lo));
            let lambda = fit
                .stretch_for_force(force, previous_lambda)
                .min(lambda_max);
            (
                PackageGeometry::max_delta(beta, lambda),
                Control::Force(force),
            )
        }
        (Control::Force(_), None) => (0.0, Control::Indentation(0.0)),
        (Control::Indentation(delta), _) => {
            let delta = if delta.is_finite() { delta } else { 0.0 };
            let delta = delta.clamp(0.0, PackageGeometry::max_delta(beta, lambda_max));
            (delta, Control::Indentation(delta))
        }
    };
    let geometry = PackageGeometry::new(beta, delta).expect("beta and delta are clamped");
    Solved {
        geometry,
        control,
        fit_force: fit.map(|f| f.force_at(geometry.lambda)),
    }
}

/// Everything a Press View frame needs, independent of egui widgets and of the renderer.
#[derive(Debug, Clone)]
pub struct PressScene {
    pub solved: Solved,
    pub plan: TexturePlan,
    pub projection: Projection,
    pub cloth: ClothScene,
    pub overlay: Vec<OverlayItem>,
}

/// The inputs of a scene besides the package and the canvas.
#[derive(Debug, Clone, Copy)]
pub struct SceneRequest {
    pub branch: PressBranch,
    pub control: Control,
    /// Bar position as a fraction of the clamp span.
    pub beta: f32,
    pub previous_lambda: f32,
    pub view: ViewMode,
    /// Relight with this light (the pad is in screen space); `None` shows the colour textures.
    pub relight: Option<Light>,
}

/// Build a scene in canvas points. Return an error if the package lacks required layers.
pub fn build_scene(
    package: &MaterialPackage,
    request: &SceneRequest,
    canvas: Rect,
) -> Result<PressScene, String> {
    let (_, lambda_max) = package.stretch_range(request.branch).ok_or_else(|| {
        format!(
            "No measured {} states in this package.",
            request.branch.name()
        )
    })?;
    let solved = solve(
        request.beta,
        request.control,
        package.force_fit(request.branch),
        lambda_max,
        request.previous_lambda,
    );
    let plan = package.texture_plan(request.branch, solved.geometry.lambda)?;
    let map = SheetMap {
        pull_vertical: package.pull_vertical(),
    };
    let max_depth = PackageGeometry::max_delta(0.5, lambda_max);
    let projection = Projection::fit(
        canvas,
        request.view,
        map.across_length(package.atlas_px()),
        max_depth,
    );
    let quads = cloth_quads(&solved.geometry, &projection, map, package.tiling());
    let wrap = if package.tiling().is_some() {
        Wrap::Repeat
    } else {
        Wrap::Clamp
    };
    let (shading, files) = match request.relight {
        None => (
            Shading::Rgb { t: plan.t },
            SceneFiles::Rgb(plan.textures.clone()),
        ),
        Some(light) => {
            let photometric = package.photometric().map_err(|reason| reason.to_string())?;
            let (Some(albedo), Some(normal)) = (plan.albedo.clone(), plan.normal.clone()) else {
                return Err(NO_PHOTOMETRIC_NOTE.into());
            };
            (
                Shading::Relight {
                    t: plan.t,
                    intensity: light.intensity,
                    ambient: light.ambient,
                    albedo_scale: photometric.albedo_scale,
                    plane_lights: [0, 1]
                        .map(|p| plane_light_in_data_frame(&light, map, solved.geometry.tilt(p))),
                },
                SceneFiles::Relight { albedo, normal },
            )
        }
    };
    let overlay = overlay_items(&solved.geometry, &projection);
    Ok(PressScene {
        solved,
        plan,
        projection,
        cloth: ClothScene {
            quads,
            wrap,
            shading,
            files,
        },
        overlay,
    })
}

/// Clamps (the sheet ends) and the bar.
pub fn overlay_items(geometry: &PackageGeometry, projection: &Projection) -> Vec<OverlayItem> {
    let mut items: Vec<OverlayItem> = [0.0, 1.0]
        .into_iter()
        .map(|pull| OverlayItem::Disc {
            center: projection.point(pull, 0.0, 0.5),
            radius: 4.0,
            color: Color32::LIGHT_GRAY,
        })
        .collect();
    let [a, b] = projection.bar(geometry);
    items.push(OverlayItem::Line {
        a,
        b,
        width: 3.0,
        color: Color32::from_rgb(255, 195, 75),
    });
    items
}

/// Screen position of the bar's top end, for tests.
#[cfg(test)]
pub fn bar_top(scene: &PressScene) -> egui::Pos2 {
    scene.projection.bar(&scene.solved.geometry)[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_package::test_support::demo_project;
    use egui::{Pos2, Vec2};

    fn package(dir: &str) -> MaterialPackage {
        MaterialPackage::load(&demo_project().join(dir).join("warp")).unwrap()
    }

    fn canvas() -> Rect {
        Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 500.0))
    }

    fn request(control: Control) -> SceneRequest {
        SceneRequest {
            branch: PressBranch::Loading,
            control,
            beta: 0.5,
            previous_lambda: 1.0,
            view: ViewMode::Oblique,
            relight: None,
        }
    }

    #[test]
    fn force_control_clamps_to_the_fit_range_and_reaches_the_target() {
        let package = package("material_package_v3_untiled");
        let fit = package.force_fit(PressBranch::Loading).unwrap();
        let (_, lambda_max) = package.stretch_range(PressBranch::Loading).unwrap();
        let (lo, hi) = fit.range();
        let mid = (lo + hi) * 0.5;
        let solved = solve(0.5, Control::Force(mid), Some(fit), lambda_max, 1.0);
        assert!(solved.geometry.delta > 0.0);
        assert!(
            (solved.fit_force.unwrap() - mid).abs() < 0.05,
            "{:?}",
            solved
        );
        // Beyond the range the force is clamped, the stretch never exceeds the measured maximum.
        let over = solve(0.5, Control::Force(hi + 50.0), Some(fit), lambda_max, 1.0);
        assert_eq!(over.control, Control::Force(hi));
        assert!(over.geometry.lambda <= lambda_max + 1.0e-4);
        // Negative tension is not clamped to zero, only to the range of the fit.
        let below = solve(0.5, Control::Force(-3.0), Some(fit), lambda_max, 1.0);
        assert_eq!(below.control, Control::Force(lo));
        // The same tension at a different bar position gives the same stretch.
        let left = solve(
            0.25,
            Control::Force(mid),
            Some(fit),
            lambda_max,
            solved.geometry.lambda,
        );
        assert!((left.geometry.lambda - solved.geometry.lambda).abs() < 1.0e-3);
    }

    #[test]
    fn indentation_works_without_a_fit_and_is_clamped_to_the_measured_stretch() {
        let solved = solve(0.4, Control::Indentation(0.1), None, 1.3, 1.0);
        assert_eq!(solved.control, Control::Indentation(0.1));
        assert!(solved.fit_force.is_none() && solved.geometry.lambda > 1.0);
        let deep = solve(0.4, Control::Indentation(5.0), None, 1.3, 1.0);
        assert!(deep.geometry.lambda <= 1.3 + 1.0e-4 && deep.geometry.delta > 0.2);
        // Garbage input degrades to a flat sheet instead of panicking.
        let nan = solve(f32::NAN, Control::Indentation(f32::NAN), None, 1.3, 1.0);
        assert_eq!((nan.geometry.beta, nan.geometry.delta), (0.5, 0.0));
        // A force request without a fit degrades to a flat sheet too.
        let none = solve(0.5, Control::Force(3.0), None, 1.3, 1.0);
        assert_eq!(none.geometry.delta, 0.0);
    }

    #[test]
    fn switching_branches_keeps_the_control_inside_the_new_range() {
        let package = package("material_package_v3_untiled");
        let loading = package.force_fit(PressBranch::Loading).unwrap().range();
        let release = package.force_fit(PressBranch::Release).unwrap().range();
        let high = loading.1;
        let (_, lambda_max) = package.stretch_range(PressBranch::Release).unwrap();
        let solved = solve(
            0.5,
            Control::Force(high),
            package.force_fit(PressBranch::Release),
            lambda_max,
            1.0,
        );
        let Control::Force(applied) = solved.control else {
            panic!()
        };
        assert!(applied <= release.1 + 1.0e-6 && applied >= release.0 - 1.0e-6);
    }

    #[test]
    fn a_scene_has_two_planes_a_bar_and_the_bracketing_states() {
        let package = package("material_package_v3_untiled");
        let scene = build_scene(&package, &request(Control::Indentation(0.12)), canvas()).unwrap();
        assert_eq!(scene.cloth.quads.len(), 2);
        assert_eq!(scene.overlay.len(), 3);
        assert!(matches!(scene.cloth.shading, Shading::Rgb { .. }));
        assert_eq!(scene.cloth.wrap, Wrap::Clamp);
        let bar = bar_top(&scene);
        assert!((scene.projection.pull_at_x(bar.x) - 0.5).abs() < 1.0e-5);
        // Tiled packages repeat.
        let tiled = self::package("material_package_v3");
        let scene = build_scene(&tiled, &request(Control::Indentation(0.1)), canvas()).unwrap();
        assert_eq!(scene.cloth.wrap, Wrap::Repeat);
    }

    #[test]
    fn relight_works_on_tiled_and_untiled_packages_without_switching_variants() {
        for dir in ["material_package_v3", "material_package_v3_untiled"] {
            let package = package(dir);
            let mut req = request(Control::Indentation(0.1));
            req.relight = Some(Light::default());
            let scene = build_scene(&package, &req, canvas()).unwrap();
            let Shading::Relight {
                albedo_scale,
                plane_lights,
                ..
            } = scene.cloth.shading
            else {
                panic!("{dir}: expected relight shading");
            };
            assert!((albedo_scale - 1.067).abs() < 0.01);
            assert_ne!(
                plane_lights[0], plane_lights[1],
                "the planes are tilted differently"
            );
            assert!(matches!(scene.cloth.files, SceneFiles::Relight { .. }));
        }
    }

    #[test]
    fn relight_without_photometric_layers_explains_why() {
        let dir = std::env::temp_dir().join(format!("press_sim_no_ps_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = demo_project().join("material_package_v3_untiled/warp");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(src.join("package.json")).unwrap()).unwrap();
        manifest.as_object_mut().unwrap().remove("ps");
        for branch in ["loading", "release"] {
            for state in manifest["branches"][branch].as_array_mut().unwrap() {
                let object = state.as_object_mut().unwrap();
                object.remove("albedo");
                object.remove("normal");
                let texture = object["texture"].as_str().unwrap().to_owned();
                std::fs::create_dir_all(dir.join(&texture).parent().unwrap()).unwrap();
                std::fs::copy(src.join(&texture), dir.join(&texture)).unwrap();
            }
        }
        std::fs::write(dir.join("package.json"), manifest.to_string()).unwrap();
        let package = MaterialPackage::load(&dir).unwrap();
        assert_eq!(package.photometric(), Err(NO_PHOTOMETRIC_NOTE));
        let mut req = request(Control::Indentation(0.1));
        req.relight = Some(Light::default());
        assert_eq!(
            build_scene(&package, &req, canvas()).unwrap_err(),
            NO_PHOTOMETRIC_NOTE
        );
        // Plain colour rendering still works.
        req.relight = None;
        assert!(build_scene(&package, &req, canvas()).is_ok());
    }

    /// Compare the demo reference layers against each captured LED photo.
    /// Rotate the pad light into screen space and account for LED gain.
    #[test]
    fn relight_at_a_capture_led_reproduces_the_captured_photo() {
        use crate::press_render::{shade_relight, MipChain, Pixels, TexKind};
        let root = demo_project();
        let dir = root.join("material_package_v3_untiled/warp");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("package.json")).unwrap()).unwrap();
        let ps = &manifest["ps"];
        let azimuths: Vec<f32> = ps["light"]["azimuths_deg"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        // Stored albedo includes exposure gain: use intensity = gains[i] / exposure_gain.
        let exposure_gain = ps["exposure_gain"].as_f64().unwrap() as f32;
        let gains: Vec<f32> = ps["gains"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32 / exposure_gain)
            .collect();
        let elevation = ps["light"]["elevation_deg"].as_f64().unwrap() as f32;
        let bounds = &manifest["atlas"]["reference_bounds_px"];
        let (ox, oy) = (
            bounds[0].as_f64().unwrap().round() as u32,
            bounds[1].as_f64().unwrap().round() as u32,
        );

        let package = MaterialPackage::load(&dir).unwrap();
        assert!(package.pull_vertical());
        let first = &package.states(PressBranch::Loading)[0];
        assert_eq!(first.stretch_ratio, 1.0);
        let load = |path: &std::path::Path| match MipChain::load(path, TexKind::Unorm16)
            .unwrap()
            .levels
            .remove(0)
            .pixels
        {
            Pixels::U16(px) => px,
            Pixels::U8(_) => unreachable!(),
        };
        let albedo = load(first.albedo.as_ref().unwrap());
        let normals = load(first.normal.as_ref().unwrap());
        let [w, h] = package.texture_px();
        let luma = |p: [f32; 3]| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        let unit = |v: u16| v as f32 / 65535.0;

        // Compare raw and 3 × 3-filtered luma; atlas resampling softens fine texture.
        let mae = |pad_azimuth: f32, led: usize, photo: &image::RgbImage| -> (f32, f32) {
            let light = Light {
                azimuth_deg: pad_azimuth,
                elevation_deg: elevation,
                intensity: gains[led],
                ambient: 0.0,
            };
            let mut req = request(Control::Indentation(0.0));
            req.relight = Some(light);
            let scene = build_scene(&package, &req, canvas()).unwrap();
            let Shading::Relight {
                plane_lights,
                albedo_scale,
                intensity,
                ambient,
                ..
            } = scene.cloth.shading
            else {
                panic!()
            };
            assert_eq!(plane_lights[0], plane_lights[1], "a flat sheet has no tilt");
            let mut diff = vec![0.0f32; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) as usize;
                    let a = albedo[i].map(unit);
                    let n = normals[i].map(unit);
                    let out = shade_relight(
                        a,
                        a,
                        n,
                        n,
                        0.0,
                        plane_lights[0],
                        intensity,
                        ambient,
                        albedo_scale,
                    );
                    let p = photo.get_pixel(x + ox, y + oy).0.map(|v| v as f32);
                    diff[i] = luma(out.map(|v| v as f32)) - luma(p);
                }
            }
            // Remove the near-field ramp still present in the photo before comparing texture detail.
            let (mut m, mut r) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
            for y in 0..h {
                for x in 0..w {
                    let basis = [1.0, x as f64, y as f64];
                    let d = diff[(y * w + x) as usize] as f64;
                    for i in 0..3 {
                        r[i] += basis[i] * d;
                        for j in 0..3 {
                            m[i][j] += basis[i] * basis[j];
                        }
                    }
                }
            }
            let ramp = solve3(m, r);
            let mut total = 0.0;
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) as usize;
                    diff[i] -= (ramp[0] + ramp[1] * x as f64 + ramp[2] * y as f64) as f32;
                    total += diff[i].abs();
                }
            }
            // Box filter of the difference equals the difference of the box filtered images.
            let mut filtered = 0.0;
            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    let (mut sum, mut n) = (0.0, 0.0);
                    for (dy, dx) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dy, dx))) {
                        let (px, py) = (x + dx, y + dy);
                        if (0..w as i64).contains(&px) && (0..h as i64).contains(&py) {
                            sum += diff[(py * w as i64 + px) as usize];
                            n += 1.0;
                        }
                    }
                    filtered += (sum / n).abs();
                }
            }
            let count = (w * h) as f32;
            (total / count, filtered / count)
        };
        for (led, &azimuth) in azimuths.iter().enumerate() {
            let photo =
                image::open(root.join(format!("warp_loading/images/img_0000_ps_{}.png", led + 1)))
                    .unwrap()
                    .to_rgb8();
            // Rotate capture azimuth by 90° to match the vertical-pull screen frame.
            let (right, filtered) = mae((azimuth + 90.0).rem_euclid(360.0), led, &photo);
            let (unrotated, _) = mae(azimuth, led, &photo);
            eprintln!(
                "LED {led}: MAE {right:.2}/255, {filtered:.2} filtered (without the rotation {unrotated:.2})"
            );
            assert!(filtered < 3.0, "LED {led}: filtered MAE {filtered}/255");
            assert!(right < 4.0, "LED {led}: MAE {right}/255");
            assert!(
                unrotated > right * 1.4,
                "LED {led}: the pad rotation must matter ({unrotated} vs {right})"
            );
        }
    }

    #[test]
    fn a_stretch_below_the_first_release_state_clamps_the_texture_instead_of_failing() {
        let package = package("material_package_v3_untiled");
        let mut req = request(Control::Indentation(0.0));
        req.branch = PressBranch::Release;
        let scene = build_scene(&package, &req, canvas()).unwrap();
        assert!(scene.plan.clamped, "lambda 1 is below the residual stretch");
        assert_eq!((scene.plan.lo, scene.plan.hi), (0, 0));
    }

    /// Solves the 3x3 system `m * x = r` by Cramer's rule.
    fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
        let det = |a: [[f64; 3]; 3]| {
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        };
        let d = det(m);
        std::array::from_fn(|k| {
            let mut a = m;
            for (row, value) in a.iter_mut().zip(r) {
                row[k] = value;
            }
            det(a) / d
        })
    }
}
