//! CPU relighting and a background worker for the Image Viewer.
//! Uses [`crate::lighting`] with decoded 8-bit normals from the image cache.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use super::normal;
use crate::lighting::{light_dir, shade_pixel, Light};

/// Relight cache key. Sampling filters only affect display and do not require recomputing pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct RelightSignature {
    pub albedo_key: PathBuf,
    pub normal_key: PathBuf,
    /// Reflectance represented by an albedo byte of 255.
    pub albedo_scale: f32,
    pub light: Light,
}

/// Scaled nearest-neighbour index, pinning the far edge to the last source texel.
pub fn nearest_src_coord(dst_x: u32, dst_len: u32, src_len: u32) -> u32 {
    if dst_len <= 1 || src_len <= 1 {
        return 0;
    }
    if dst_x + 1 == dst_len {
        src_len - 1
    } else {
        ((dst_x as u64 * src_len as u64) / dst_len as u64).min(src_len as u64 - 1) as u32
    }
}

/// Relight linear RGBA albedo with a scaled nearest-neighbour normal map.
/// Return the albedo's size and alpha. An empty normal map uses +Z; empty albedo stays empty.
pub fn relight(
    albedo: &image::RgbaImage,
    albedo_scale: f32,
    normal: &image::RgbaImage,
    light: &Light,
) -> image::RgbaImage {
    let aw = albedo.width();
    let ah = albedo.height();
    if aw == 0 || ah == 0 {
        return image::RgbaImage::new(0, 0);
    }
    let l = light_dir(light);
    let nw = normal.width();
    let nh = normal.height();
    let flat_normal = nw == 0 || nh == 0; // no usable normal data -> +Z everywhere
    let to_linear = albedo_scale / 255.0;

    let mut out = image::RgbaImage::new(aw, ah);
    for (x, y, px) in out.enumerate_pixels_mut() {
        let a = albedo.get_pixel(x, y);
        let n = if flat_normal {
            [0.0, 0.0, 1.0]
        } else {
            let nx = nearest_src_coord(x, aw, nw);
            let ny = nearest_src_coord(y, ah, nh);
            let n = normal.get_pixel(nx, ny);
            normal::decode_normal(n[0], n[1], n[2])
        };
        let shaded = shade_pixel(
            [
                a[0] as f32 * to_linear,
                a[1] as f32 * to_linear,
                a[2] as f32 * to_linear,
            ],
            n,
            l,
            light.intensity,
            light.ambient,
        );
        px[0] = shaded[0];
        px[1] = shaded[1];
        px[2] = shaded[2];
        px[3] = a[3];
    }
    out
}

/// A single relight request handed to the worker. Holds refcounted decoded
/// pixels, so no disk reads happen on the worker.
pub struct RelightJob {
    /// Monotonic generation; the UI ignores any result older than the newest
    /// submitted job.
    pub generation: u64,
    /// Check the signature before applying a result; users may have reverted the light mid-job.
    pub signature: RelightSignature,
    pub albedo: Arc<image::RgbaImage>,
    pub normal: Arc<image::RgbaImage>,
}

/// The finished relight, delivered back to the UI thread.
pub struct RelightResult {
    pub generation: u64,
    pub signature: RelightSignature,
    pub image: image::RgbaImage,
}

/// Background relighting. Drain queued jobs to the latest request before starting work.
#[derive(Debug)]
pub struct RelightWorker {
    request_tx: Option<Sender<RelightJob>>,
    result_rx: Receiver<RelightResult>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RelightWorker {
    pub fn new() -> Self {
        let (request_tx, request_rx) = channel::<RelightJob>();
        let (result_tx, result_rx) = channel::<RelightResult>();
        let handle = std::thread::Builder::new()
            .name("fabsim-relight".to_owned())
            .spawn(move || {
                while let Ok(mut job) = request_rx.recv() {
                    while let Ok(next) = request_rx.try_recv() {
                        job = next;
                    }
                    let image = relight(
                        &job.albedo,
                        job.signature.albedo_scale,
                        &job.normal,
                        &job.signature.light,
                    );
                    let result = RelightResult {
                        generation: job.generation,
                        signature: job.signature,
                        image,
                    };
                    if result_tx.send(result).is_err() {
                        break; // worker dropped
                    }
                }
            })
            .expect("failed to spawn relight worker thread");
        RelightWorker {
            request_tx: Some(request_tx),
            result_rx,
            handle: Some(handle),
        }
    }

    /// Submit a new generation. Returns `false` if the worker has exited; stale results are discarded.
    pub fn submit(
        &mut self,
        generation: u64,
        signature: RelightSignature,
        albedo: Arc<image::RgbaImage>,
        normal: Arc<image::RgbaImage>,
    ) -> bool {
        let Some(tx) = &self.request_tx else {
            return false;
        };
        tx.send(RelightJob {
            generation,
            signature,
            albedo,
            normal,
        })
        .is_ok()
    }

    /// Non-blocking poll for a finished result.
    pub fn try_recv_result(&mut self) -> Option<RelightResult> {
        self.result_rx.try_recv().ok()
    }
}

impl Default for RelightWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RelightWorker {
    fn drop(&mut self) {
        self.request_tx = None; // close channel -> worker recv errors -> exits
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_albedo(color: [u8; 4], w: u32, h: u32) -> image::RgbaImage {
        image::RgbaImage::from_pixel(w, h, image::Rgba(color))
    }

    fn normal_plus_z(w: u32, h: u32) -> image::RgbaImage {
        // (128, 128, 255) encodes +Z (to within a fraction of a degree).
        image::RgbaImage::from_pixel(w, h, image::Rgba([128, 128, 255, 255]))
    }

    fn light(azimuth_deg: f32, elevation_deg: f32, intensity: f32, ambient: f32) -> Light {
        Light {
            azimuth_deg,
            elevation_deg,
            intensity,
            ambient,
        }
    }

    fn straight_down() -> Light {
        light(0.0, 90.0, 1.0, 0.0)
    }

    #[test]
    fn relight_straight_on_preserves_white() {
        let albedo = flat_albedo([255, 255, 255, 255], 2, 2);
        let out = relight(&albedo, 1.0, &normal_plus_z(2, 2), &straight_down());
        let p = out.get_pixel(1, 1);
        assert!(p[0] > 250 && p[1] > 250 && p[2] > 250, "got {p:?}");
        assert_eq!(p[3], 255);
    }

    #[test]
    fn relight_grazing_light_darkens_to_ambient() {
        let albedo = flat_albedo([255, 255, 255, 255], 2, 2);
        let out = relight(
            &albedo,
            1.0,
            &normal_plus_z(2, 2),
            &light(0.0, 0.0, 1.0, 0.0),
        );
        let p = out.get_pixel(0, 0);
        assert!(
            p[0] < 8 && p[1] < 8 && p[2] < 8,
            "expected black, got {p:?}"
        );
    }

    #[test]
    fn relight_half_brightness_is_linear_half() {
        let albedo = flat_albedo([255, 255, 255, 255], 2, 2);
        // Elevation 30 deg -> 60 deg from the normal, so lambert = 0.5.
        let out = relight(
            &albedo,
            1.0,
            &normal_plus_z(2, 2),
            &light(0.0, 30.0, 1.0, 0.0),
        );
        let p = out.get_pixel(0, 0);
        // Linear, display-raw: half of 255 is ~128 (an sRGB re-encode would
        // give ~188).
        assert!((p[0] as i32 - 128).abs() <= 3, "got {p:?}");
    }

    #[test]
    fn albedo_scale_restores_the_recorded_range() {
        // A stored byte of 128 with scale 2 is reflectance ~1.
        let albedo = flat_albedo([128, 128, 128, 255], 1, 1);
        let out = relight(&albedo, 2.0, &normal_plus_z(1, 1), &straight_down());
        assert!(out.get_pixel(0, 0)[0] >= 253);
        // Overexposure clamps rather than wrapping.
        let bright = flat_albedo([255, 255, 255, 255], 1, 1);
        let out = relight(&bright, 4.0, &normal_plus_z(1, 1), &straight_down());
        assert_eq!(out.get_pixel(0, 0)[0], 255);
    }

    #[test]
    fn azimuth_90_brightens_the_top_tilted_normal_not_the_bottom_tilted_one() {
        let albedo = flat_albedo([200, 200, 200, 255], 2, 1);
        let mut normal = image::RgbaImage::new(2, 1);
        normal.put_pixel(0, 0, image::Rgba([128, 230, 200, 255])); // tilts image-up
        normal.put_pixel(1, 0, image::Rgba([128, 26, 200, 255])); // tilts image-down
        let out = relight(&albedo, 1.0, &normal, &light(90.0, 30.0, 1.0, 0.0));
        assert!(out.get_pixel(0, 0)[0] > out.get_pixel(1, 0)[0] + 40);
    }

    /// Re-rendering a photometric-stereo solution under one of its own LED
    /// directions (intensity 1, ambient 0) reproduces the captured photo.
    #[test]
    fn relight_under_the_capture_led_reproduces_the_photo() {
        let (w, h) = (24u32, 16u32);
        let led = light(91.9, 30.0, 1.0, 0.0);
        let l = light_dir(&led);
        let mut albedo = image::RgbaImage::new(w, h);
        let mut normal = image::RgbaImage::new(w, h);
        let mut photo = Vec::new();
        for y in 0..h {
            for x in 0..w {
                // A smooth bump field and a textured reflectance in [0, 0.8].
                let nx = 0.4 * (x as f32 * 0.5).sin();
                let ny = 0.4 * (y as f32 * 0.7).cos();
                let n = normal::normalize3([nx, ny, 1.0]);
                let rho = 0.2 + 0.6 * ((x * 7 + y * 3) % 10) as f32 / 10.0;
                let scale = 0.8;
                let enc = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u8;
                normal.put_pixel(x, y, image::Rgba([enc(n[0]), enc(n[1]), enc(n[2]), 255]));
                let byte = (rho / scale * 255.0).round() as u8;
                albedo.put_pixel(x, y, image::Rgba([byte, byte, byte, 255]));
                let ndl = (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]).max(0.0);
                photo.push((rho * ndl * 255.0).round());
            }
        }
        let out = relight(&albedo, 0.8, &normal, &led);
        let mae: f32 = out
            .pixels()
            .zip(&photo)
            .map(|(p, want)| (p[0] as f32 - want).abs())
            .sum::<f32>()
            / photo.len() as f32;
        assert!(mae < 3.0, "re-render MAE {mae}/255");
    }

    /// Products of one demo run (`fabsim ps`): manifest plus the run directory.
    struct DemoRun {
        dir: std::path::PathBuf,
        manifest: serde_json::Value,
    }

    impl DemoRun {
        fn open(name: &str) -> Self {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples/demo")
                .join(name);
            let manifest =
                serde_json::from_slice(&std::fs::read(dir.join("ps/ps_manifest.json")).unwrap())
                    .unwrap();
            DemoRun { dir, manifest }
        }

        fn scale(&self) -> f32 {
            self.manifest["albedo_scale"].as_f64().unwrap() as f32
        }

        /// Azimuth, elevation and radiance gain of capture LED `led` (0-based).
        fn led(&self, led: usize) -> (f32, f32, f32) {
            let light = &self.manifest["light"];
            (
                light["azimuths_deg"][led].as_f64().unwrap() as f32,
                light["elevation_deg"].as_f64().unwrap() as f32,
                self.manifest["gains"][led].as_f64().unwrap() as f32,
            )
        }

        fn frame_ids(&self) -> Vec<u64> {
            let frames = self.manifest["frames"].as_array().unwrap();
            frames
                .iter()
                .map(|f| f["frame_id"].as_u64().unwrap())
                .collect()
        }

        fn load(&self, relative: String) -> image::DynamicImage {
            image::open(self.dir.join(relative)).unwrap()
        }
    }

    /// Rec. 601 luma of an RGB triple: what the solver fits (see `fabsim.photometric`).
    fn luma(rgb: [f32; 3]) -> f32 {
        0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    }

    /// Compare captured LED photos in luma: their colour cast differs from the diffuse albedo.
    /// Fold each LED's gain into intensity and use zero ambient.
    #[test]
    fn relight_of_the_demo_project_matches_its_captured_photos() {
        for run_name in ["warp_loading", "warp_release"] {
            let run = DemoRun::open(run_name);
            for id in run.frame_ids() {
                let albedo = run.load(format!("ps/img_{id:04}_albedo.png")).to_rgba8();
                let normal = run.load(format!("ps/img_{id:04}_normal.png")).to_rgba8();
                for led in 0..3 {
                    let (azimuth, elevation, gain) = run.led(led);
                    let out = relight(
                        &albedo,
                        run.scale(),
                        &normal,
                        &light(azimuth, elevation, gain, 0.0),
                    );
                    let photo = run
                        .load(format!("images/img_{id:04}_ps_{}.png", led + 1))
                        .to_rgb8();
                    assert_eq!(out.dimensions(), photo.dimensions());
                    let mae: f32 = out
                        .pixels()
                        .zip(photo.pixels())
                        .map(|(a, b)| {
                            let f = |p: [u8; 3]| luma(p.map(f32::from));
                            (f([a[0], a[1], a[2]]) - f(b.0)).abs()
                        })
                        .sum::<f32>()
                        / (out.width() * out.height()) as f32;
                    assert!(mae < 3.0, "{run_name} frame {id} LED {led}: MAE {mae}/255");
                }
            }
        }
    }

    /// Check the display error introduced by reducing 16-bit normals to 8-bit.
    #[test]
    fn eight_bit_normals_cost_less_than_half_a_level() {
        let run = DemoRun::open("warp_loading");
        let id = run.frame_ids()[0];
        let albedo = run.load(format!("ps/img_{id:04}_albedo.png")).to_rgba8();
        let normal16 = run.load(format!("ps/img_{id:04}_normal.png")).to_rgb16();
        assert!(
            normal16.pixels().any(|p| p[0] % 257 != 0),
            "the fixture must really hold 16-bit normals"
        );
        let normal8 = image::DynamicImage::ImageRgb16(normal16.clone()).to_rgba8();
        let scale = run.scale();
        let mut worst_mae = 0.0f32;
        for led in 0..3 {
            let (azimuth, elevation, gain) = run.led(led);
            let lit = light(azimuth, elevation, gain, 0.0);
            let l = light_dir(&lit);
            let out = relight(&albedo, scale, &normal8, &lit);
            let mut total = 0.0f32;
            for (x, y, px) in out.enumerate_pixels() {
                let n16 = normal16.get_pixel(x, y);
                let n = normal::normalize3(n16.0.map(|v| v as f32 * (2.0 / 65535.0) - 1.0));
                let diffuse = gain * normal::lambert(n, l);
                let a = albedo.get_pixel(x, y);
                for c in 0..3 {
                    let exact = (a[c] as f32 / 255.0 * scale * diffuse).clamp(0.0, 1.0) * 255.0;
                    total += (px[c] as f32 - exact).abs();
                }
            }
            worst_mae = worst_mae.max(total / (out.width() * out.height() * 3) as f32);
        }
        assert!(worst_mae < 0.5, "8-bit normals cost {worst_mae}/255");
    }

    #[test]
    fn neutral_normal_sentinel_lights_flat() {
        // The neutral sentinel should light as flat +Z.
        let albedo = flat_albedo([255, 255, 255, 255], 3, 3);
        let normal = image::RgbaImage::from_pixel(3, 3, image::Rgba([128, 128, 128, 255]));
        let out = relight(&albedo, 1.0, &normal, &straight_down());
        for (x, y, p) in out.enumerate_pixels() {
            assert!(
                p[0] > 250 && p[1] > 250 && p[2] > 250,
                "({x},{y}): got {p:?}"
            );
        }
    }

    #[test]
    fn a_zero_sized_normal_map_lights_flat() {
        let albedo = flat_albedo([255, 255, 255, 255], 2, 2);
        let out = relight(&albedo, 1.0, &image::RgbaImage::new(0, 0), &straight_down());
        assert_eq!(out.get_pixel(1, 1)[0], 255);
    }

    #[test]
    fn nearest_coord_covers_both_endpoints() {
        for x in 0..5 {
            assert_eq!(nearest_src_coord(x, 5, 5), x);
        }
        // Upsampling must include both normal-map endpoints.
        assert_eq!(nearest_src_coord(0, 10, 3), 0);
        assert_eq!(
            nearest_src_coord(9, 10, 3),
            2,
            "far edge must reach last texel"
        );
        // Downscale (normal larger): both endpoints still covered.
        assert_eq!(nearest_src_coord(0, 2, 5), 0);
        assert_eq!(
            nearest_src_coord(1, 2, 5),
            4,
            "far edge must reach last texel"
        );
        assert_eq!(nearest_src_coord(0, 1, 7), 0);
        assert_eq!(nearest_src_coord(3, 8, 1), 0);
    }

    #[test]
    fn relight_boundary_sampling_up_and_down() {
        // Upsampling must reach both edge texels.
        let albedo = flat_albedo([255, 255, 255, 255], 5, 1);
        let mut normal = image::RgbaImage::new(3, 1);
        normal.put_pixel(0, 0, image::Rgba([127, 127, 255, 255])); // +Z: bright
        normal.put_pixel(2, 0, image::Rgba([0, 127, 127, 255])); // -X: dark
        let out = relight(&albedo, 1.0, &normal, &straight_down());
        assert!(
            out.get_pixel(0, 0)[0] > 250,
            "left edge -> first normal texel"
        );
        assert!(
            out.get_pixel(4, 0)[0] < 10,
            "right edge -> last normal texel"
        );

        let albedo = flat_albedo([255, 255, 255, 255], 3, 1);
        let normal = normal_plus_z(1, 1);
        let out = relight(&albedo, 1.0, &normal, &straight_down());
        for x in 0..3 {
            assert!(out.get_pixel(x, 0)[0] > 250, "uniform at x={x}");
        }
    }

    #[test]
    fn relight_scaled_sampling_is_nearest_not_tiled() {
        // Scaled sampling should split bright/dark halves; modulo sampling would repeat them.
        let albedo = flat_albedo([255, 255, 255, 255], 6, 1);
        let mut normal = image::RgbaImage::new(2, 1);
        normal.put_pixel(0, 0, image::Rgba([127, 127, 255, 255])); // +Z
        normal.put_pixel(1, 0, image::Rgba([0, 127, 127, 255])); // -X
        let out = relight(&albedo, 1.0, &normal, &straight_down());

        let bright = out.get_pixel(0, 0);
        let dark = out.get_pixel(5, 0);
        let dark_anti_tiling = out.get_pixel(4, 0);
        assert!(bright[0] > 250, "left must stay bright, got {bright:?}");
        assert!(dark[0] < 10, "right must darken, got {dark:?}");
        assert!(
            dark_anti_tiling[0] < 10,
            "x=4 must sample the right normal (scaled, not tiled), got {dark_anti_tiling:?}"
        );
    }

    #[test]
    fn relight_handles_zero_sized_inputs() {
        let empty = image::RgbaImage::new(0, 0);
        let normal = normal_plus_z(2, 2);
        let out = relight(&empty, 1.0, &normal, &straight_down());
        assert_eq!(out.width(), 0);
        assert_eq!(out.height(), 0);

        // Zero-sized normal map -> flat +Z lighting, no panic.
        let albedo = flat_albedo([200, 100, 50, 255], 2, 2);
        let out = relight(&albedo, 1.0, &empty, &straight_down());
        let p = out.get_pixel(1, 1);
        assert!(p[3] == 255 && p[0] > 195, "flat-lit, got {p:?}");
    }

    #[test]
    fn relight_scaled_sampling_across_resolutions() {
        // A single normal texel lights the whole albedo uniformly.
        let albedo = flat_albedo([255, 255, 255, 255], 4, 4);
        let normal = normal_plus_z(1, 1);
        let out = relight(&albedo, 1.0, &normal, &straight_down());
        let first = *out.get_pixel(0, 0);
        for (x, y, p) in out.enumerate_pixels() {
            assert_eq!(p.0, first.0, "({x},{y}) must match the single normal");
        }
    }

    #[test]
    fn worker_latest_result_wins() {
        let mut worker = RelightWorker::new();
        let albedo = Arc::new(flat_albedo([255, 255, 255, 255], 8, 8));
        let normal = Arc::new(normal_plus_z(8, 8));
        // Submit many jobs rapidly; the newest generation must be delivered.
        let sig = |ambient: f32| RelightSignature {
            albedo_key: PathBuf::from("/alb"),
            normal_key: PathBuf::from("/nrm"),
            albedo_scale: 1.0,
            light: light(0.0, 90.0, 1.0, ambient),
        };
        for generation in 0..10 {
            worker.submit(
                generation,
                sig(generation as f32 / 10.0),
                albedo.clone(),
                normal.clone(),
            );
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut seen = Vec::new();
        let mut latest_bright = false;
        while std::time::Instant::now() < deadline {
            while let Some(result) = worker.try_recv_result() {
                let p = result.image.get_pixel(0, 0);
                if result.generation == 9 {
                    latest_bright = p[0] > 220;
                }
                // Results must carry the signature they were submitted for.
                assert_eq!(result.signature, sig(result.generation as f32 / 10.0));
                seen.push(result.generation);
            }
            if seen.contains(&9) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            seen.contains(&9),
            "the latest generation must be delivered, saw {seen:?}"
        );
        assert!(latest_bright, "the generation-9 result must be bright");
    }

    #[test]
    fn worker_shuts_down_cleanly() {
        let worker = RelightWorker::new();
        drop(worker);
    }
}
