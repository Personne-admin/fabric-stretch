//! Shared Lambertian lighting, matching `fabsim.photometric.shade`.
//!
//! Albedo and output stay linear and display-raw; do not apply sRGB transfer functions.
//! Normals and light use x right, y up, z toward the camera. Azimuth is counter-clockwise
//! from image right. Press View rotates the light into this frame before shading.
//!
//! For capture comparisons, use luma and fold per-LED gain into intensity.
//! Package layers also need the exposure-gain adjustment described in `docs/photometric_stereo.md`.

use egui::{Color32, Sense, Stroke, Vec2};
use serde::{Deserialize, Serialize};

use crate::images::normal;

/// Light description shared by all views. Angles are in degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Light {
    /// Counter-clockwise from image-right, looking at the image.
    pub azimuth_deg: f32,
    /// Angle above the surface plane: 0 = grazing, 90 = straight on.
    pub elevation_deg: f32,
    /// Diffuse multiplier (0 = ambient only).
    pub intensity: f32,
    /// Constant fill added to the diffuse term before scaling the albedo.
    pub ambient: f32,
}

impl Default for Light {
    fn default() -> Self {
        Light {
            azimuth_deg: 135.0,
            elevation_deg: 45.0,
            intensity: 1.0,
            ambient: 0.1,
        }
    }
}

/// Unit direction towards the light in the camera / data frame.
pub fn light_dir(light: &Light) -> [f32; 3] {
    normal::light_direction(light.azimuth_deg, light.elevation_deg)
}

/// Shade linear, rescaled albedo with unit normal and light vectors in the same frame.
/// Returns display-raw bytes, rounded after clamping.
pub fn shade_pixel(
    albedo: [f32; 3],
    n: [f32; 3],
    light: [f32; 3],
    intensity: f32,
    ambient: f32,
) -> [u8; 3] {
    let gain = ambient + intensity * normal::lambert(n, light);
    albedo.map(|a| ((a * gain).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
}

/// Radius of the pad disc in points.
const PAD_RADIUS: f32 = 26.0;

/// Pad offset (screen y down) to azimuth/elevation. Centre is 90°, rim is 0°.
/// Keep the current azimuth when the pointer is too close to the centre.
pub fn pad_to_angles(offset: Vec2, radius: f32, current_azimuth: f32) -> (f32, f32) {
    let azimuth = if offset.length() < 1.0 {
        current_azimuth
    } else {
        (-offset.y).atan2(offset.x).to_degrees().rem_euclid(360.0)
    };
    let r = (offset.length() / radius).clamp(0.0, 1.0);
    (azimuth, 90.0 * (1.0 - r))
}

/// Inverse of [`pad_to_angles`]: where the light marker sits on the pad.
pub fn angles_to_pad(azimuth_deg: f32, elevation_deg: f32, radius: f32) -> Vec2 {
    let r = (1.0 - elevation_deg.clamp(0.0, 90.0) / 90.0) * radius;
    let a = azimuth_deg.to_radians();
    Vec2::new(a.cos() * r, -a.sin() * r)
}

/// Light-pad controls. Returns whether the light changed.
pub fn light_pad(ui: &mut egui::Ui, light: &mut Light) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let size = Vec2::splat(PAD_RADIUS * 2.0 + 4.0);
        let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
        let center = response.rect.center();
        if response.is_pointer_button_down_on() {
            if let Some(pos) = response.interact_pointer_pos() {
                let (az, el) = pad_to_angles(pos - center, PAD_RADIUS, light.azimuth_deg);
                if (az, el) != (light.azimuth_deg, light.elevation_deg) {
                    light.azimuth_deg = az;
                    light.elevation_deg = el;
                    changed = true;
                }
            }
        }
        let visuals = ui.visuals();
        painter.circle_filled(center, PAD_RADIUS, visuals.extreme_bg_color);
        painter.circle_stroke(center, PAD_RADIUS, visuals.widgets.noninteractive.fg_stroke);
        painter.circle_stroke(
            center,
            PAD_RADIUS * 0.5,
            Stroke::new(0.5, visuals.widgets.noninteractive.bg_stroke.color),
        );
        let marker = center + angles_to_pad(light.azimuth_deg, light.elevation_deg, PAD_RADIUS);
        painter.line_segment([center, marker], Stroke::new(1.0, Color32::from_gray(120)));
        painter.circle_filled(marker, 4.5, Color32::from_rgb(255, 200, 40));
        response.on_hover_text(
            "Light direction as seen on screen: drag around the disc for azimuth, \
             towards the centre for a higher light",
        );

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label("elev");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut light.elevation_deg)
                            .range(0.0..=90.0)
                            .speed(0.5)
                            .suffix("\u{b0}"),
                    )
                    .changed();
                ui.label("azim");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut light.azimuth_deg)
                            .range(0.0..=360.0)
                            .speed(1.0)
                            .suffix("\u{b0}"),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("intensity");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut light.intensity)
                            .range(0.0..=4.0)
                            .speed(0.01)
                            .fixed_decimals(2),
                    )
                    .changed();
                ui.label("ambient");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut light.ambient)
                            .range(0.0..=1.0)
                            .speed(0.005)
                            .fixed_decimals(2),
                    )
                    .changed();
            });
        });
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn default_light_matches_the_spec() {
        let l = Light::default();
        assert_eq!(
            (l.azimuth_deg, l.elevation_deg, l.intensity, l.ambient),
            (135.0, 45.0, 1.0, 0.1)
        );
    }

    #[test]
    fn light_dir_uses_the_camera_frame_with_y_up() {
        let at = |az, el| {
            light_dir(&Light {
                azimuth_deg: az,
                elevation_deg: el,
                ..Light::default()
            })
        };
        assert!(
            approx(at(0.0, 0.0), [1.0, 0.0, 0.0]),
            "0 deg is image-right"
        );
        assert!(approx(at(90.0, 0.0), [0.0, 1.0, 0.0]), "90 deg is image-up");
        assert!(approx(at(180.0, 0.0), [-1.0, 0.0, 0.0]));
        assert!(approx(at(0.0, 90.0), [0.0, 0.0, 1.0]), "90 deg elevation");
    }

    #[test]
    fn linear_shading_is_display_raw() {
        // Flat surface, overhead light, no ambient: output == albedo.
        let up = [0.0, 0.0, 1.0];
        assert_eq!(
            shade_pixel([0.5, 0.25, 1.0], up, up, 1.0, 0.0),
            [128, 64, 255]
        );
        // 45 degree light on a flat surface: 0.5 * cos(45) in linear, no sRGB.
        let l = normal::light_direction(0.0, 45.0);
        let expected = (0.5 * 45f32.to_radians().cos() * 255.0 + 0.5) as u8;
        assert_eq!(shade_pixel([0.5; 3], up, l, 1.0, 0.0)[0], expected);
        assert_eq!(
            shade_pixel([1.0; 3], up, [0.0, 0.0, -1.0], 1.0, 0.2),
            [51; 3]
        );
        // A surface facing away from the light stays black without ambient.
        assert_eq!(
            shade_pixel([1.0; 3], up, [0.0, 0.0, -1.0], 1.0, 0.0),
            [0; 3]
        );
        // Overexposure clamps instead of wrapping.
        assert_eq!(shade_pixel([1.0; 3], up, up, 3.0, 0.5), [255; 3]);
    }

    #[test]
    fn pad_round_trips_angles() {
        for (az, el) in [(0.0, 45.0), (90.0, 30.0), (200.0, 10.0), (315.0, 80.0)] {
            let offset = angles_to_pad(az, el, PAD_RADIUS);
            let (az2, el2) = pad_to_angles(offset, PAD_RADIUS, 0.0);
            assert!((az - az2).abs() < 1e-3, "azimuth {az} -> {az2}");
            assert!((el - el2).abs() < 1e-3, "elevation {el} -> {el2}");
        }
    }

    #[test]
    fn pad_screen_up_is_image_up() {
        // A pointer above the centre (negative screen y) is azimuth 90.
        let (az, el) = pad_to_angles(Vec2::new(0.0, -PAD_RADIUS * 0.5), PAD_RADIUS, 0.0);
        assert!((az - 90.0).abs() < 1e-3);
        assert!((el - 45.0).abs() < 1e-3);
        // At the centre the previous azimuth is kept and the light is overhead.
        let (az, el) = pad_to_angles(Vec2::ZERO, PAD_RADIUS, 123.0);
        assert_eq!((az, el), (123.0, 90.0));
        // Beyond the rim the elevation clamps to grazing.
        let (_, el) = pad_to_angles(Vec2::new(PAD_RADIUS * 3.0, 0.0), PAD_RADIUS, 0.0);
        assert_eq!(el, 0.0);
    }

    #[test]
    fn legacy_relight_settings_payloads_still_load() {
        // Legacy flat light keys are ignored.
        let light: Light = serde_json::from_str(r#"{"azimuth_deg":10.0}"#).unwrap();
        assert_eq!(light.azimuth_deg, 10.0);
        assert_eq!(light.elevation_deg, Light::default().elevation_deg);
        let light: Light = serde_json::from_str(r#"{"enabled":true,"intensity":2.0}"#).unwrap();
        assert_eq!(light.intensity, 2.0);
    }
}
