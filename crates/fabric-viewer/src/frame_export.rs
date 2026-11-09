//! PNG/video export matching the Image Viewer. Overlay painting is duplicated in
//! `ui::image_viewer::overlay`; keep both implementations in sync.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use egui::Color32;
use fabric_data::tracking::registration::RegistrationTransform;
use fabric_data::tracking::stabilize::ANCHOR_MIN_CONFIDENCE;
use fabric_data::tracking::TrackedPoint;
use image::{Rgb, RgbImage};

use crate::tracked::{OverlayPaintData, OverlayStyle};
use crate::ui::image_viewer::overlay::{
    COLOR_CURRENT, COLOR_DISP, COLOR_GHOST, COLOR_LINE, COLOR_LOW_CONF, COLOR_NEUTRAL_BG,
    COLOR_RAW, COLOR_REF_BADGE, COLOR_TRAJ, COLOR_TRANSFORMED,
};

pub struct ExportFrame {
    pub image_path: PathBuf,
    pub transform: RegistrationTransform,
    pub overlay: Option<OverlayPaintData>,
}

pub struct ExportOptions {
    pub fps: f32,
    /// Export-owned directory. Replace existing `frame_NNNNN.png` files
    /// to avoid leftover frames from longer exports.
    pub frames_dir: PathBuf,
    pub video: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportReport {
    pub frames: usize,
    pub frames_dir: PathBuf,
    pub video: Option<PathBuf>,
    pub video_note: Option<String>,
}

#[derive(Debug)]
pub struct ExportHandle {
    pub total: usize,
    done: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<ExportReport, String>>>>,
}

impl ExportHandle {
    pub fn progress(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn take_result(&self) -> Option<Result<ExportReport, String>> {
        self.result.lock().ok()?.take()
    }
}

pub fn spawn(frames: Vec<ExportFrame>, options: ExportOptions) -> ExportHandle {
    let handle = ExportHandle {
        total: frames.len(),
        done: Arc::new(AtomicUsize::new(0)),
        cancel: Arc::new(AtomicBool::new(false)),
        result: Arc::new(Mutex::new(None)),
    };

    let (done, cancel, result) = (
        handle.done.clone(),
        handle.cancel.clone(),
        handle.result.clone(),
    );

    std::thread::spawn(move || {
        let outcome = run_export(&frames, &options, &done, &cancel);
        if let Ok(mut slot) = result.lock() {
            *slot = Some(outcome);
        }
    });
    handle
}

fn run_export(
    frames: &[ExportFrame],
    options: &ExportOptions,
    done: &AtomicUsize,
    cancel: &AtomicBool,
) -> Result<ExportReport, String> {
    if frames.is_empty() {
        return Err("no frames to export".into());
    }
    prepare_frames_dir(&options.frames_dir)?;

    for (i, frame) in frames.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("export cancelled".into());
        }

        let src = image::open(&frame.image_path)
            .map_err(|e| format!("{}: {e}", frame.image_path.display()))?
            .to_rgb8();

        let out = render_frame(&src, frame.transform, frame.overlay.as_ref());
        let path = options.frames_dir.join(format!("frame_{i:05}.png"));
        out.save(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;

        done.store(i + 1, Ordering::Relaxed);
    }
    let (video, video_note) = match &options.video {
        None => (None, None),
        Some(path) => match encode_video(&options.frames_dir, frames.len(), options.fps, path) {
            Ok(()) => (Some(path.clone()), None),
            Err(note) => (None, Some(note)),
        },
    };

    Ok(ExportReport {
        frames: frames.len(),
        frames_dir: options.frames_dir.clone(),
        video,
        video_note,
    })
}

/// Create the frames directory and remove `frame_NNNNN.png` files left by an
/// earlier (longer or cancelled) export, so they can never end up in the video.
fn prepare_frames_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_frame = name
            .strip_prefix("frame_")
            .and_then(|rest| rest.strip_suffix(".png"))
            .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()));
        if is_frame {
            std::fs::remove_file(entry.path())
                .map_err(|e| format!("{}: {e}", entry.path().display()))?;
        }
    }
    Ok(())
}

fn encode_video(
    frames_dir: &Path,
    frame_count: usize,
    fps: f32,
    video: &Path,
) -> Result<(), String> {
    let output = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate"])
        .arg(format!("{}", fps.max(0.1)))
        .arg("-i")
        .arg(frames_dir.join("frame_%05d.png"))
        .args(["-frames:v"])
        .arg(frame_count.to_string())
        .args([
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(video)
        .output()
        .map_err(|e| format!("ffmpeg could not be started ({e}); frames were written"))?;

    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn rgb(c: Color32) -> [f32; 3] {
    [c.r() as f32, c.g() as f32, c.b() as f32]
}

pub fn render_frame(
    src: &RgbImage,
    transform: RegistrationTransform,
    overlay: Option<&OverlayPaintData>,
) -> RgbImage {
    let (w, h) = src.dimensions();
    let bg = Rgb([
        COLOR_NEUTRAL_BG.r(),
        COLOR_NEUTRAL_BG.g(),
        COLOR_NEUTRAL_BG.b(),
    ]);
    let mut out = if transform.is_identity() {
        src.clone()
    } else {
        let mut out = RgbImage::from_pixel(w, h, bg);
        let inverse = transform.inverse();
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = inverse.apply(x as f64 + 0.5, y as f64 + 0.5);
                if let Some(px) = sample_bilinear(src, sx as f32, sy as f32) {
                    out.put_pixel(x, y, px);
                }
            }
        }
        out
    };
    if let Some(overlay) = overlay {
        let k = (w.max(h) as f32 / 1000.0).max(1.0);
        let mut painter = Painter { img: &mut out, k };
        paint_overlay(&mut painter, transform, overlay);
    }
    out
}

fn sample_bilinear(src: &RgbImage, x: f32, y: f32) -> Option<Rgb<u8>> {
    let (w, h) = src.dimensions();
    if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
        return None;
    }

    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let at = |ix: f32, iy: f32| {
        src.get_pixel(
            ix.clamp(0.0, w as f32 - 1.0) as u32,
            iy.clamp(0.0, h as f32 - 1.0) as u32,
        )
    };
    let (p00, p10, p01, p11) = (
        at(x0, y0),
        at(x0 + 1.0, y0),
        at(x0, y0 + 1.0),
        at(x0 + 1.0, y0 + 1.0),
    );
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - tx) + p10[c] as f32 * tx;
        let bottom = p01[c] as f32 * (1.0 - tx) + p11[c] as f32 * tx;
        out[c] = (top * (1.0 - ty) + bottom * ty).round() as u8;
    }
    Some(Rgb(out))
}

struct Painter<'a> {
    img: &'a mut RgbImage,
    k: f32,
}

impl Painter<'_> {
    fn blend(&mut self, x: i64, y: i64, color: [f32; 3], coverage: f32) {
        let (w, h) = self.img.dimensions();
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 || coverage <= 0.0 {
            return;
        }
        let a = coverage.clamp(0.0, 1.0);
        let px = self.img.get_pixel_mut(x as u32, y as u32);
        for c in 0..3 {
            px[c] = (px[c] as f32 * (1.0 - a) + color[c] * a).round() as u8;
        }
    }

    fn shade(
        &mut self,
        min: (f32, f32),
        max: (f32, f32),
        color: [f32; 3],
        coverage: impl Fn(f32, f32) -> f32,
    ) {
        let (x0, y0) = (min.0.floor() as i64 - 1, min.1.floor() as i64 - 1);
        let (x1, y1) = (max.0.ceil() as i64 + 1, max.1.ceil() as i64 + 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let c = coverage(x as f32 + 0.5, y as f32 + 0.5);
                self.blend(x, y, color, c);
            }
        }
    }

    fn disc(&mut self, center: (f32, f32), radius: f32, color: Color32) {
        let r = radius * self.k;
        self.shade(
            (center.0 - r, center.1 - r),
            (center.0 + r, center.1 + r),
            rgb(color),
            |x, y| r - ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt() + 0.5,
        );
    }

    fn ring(&mut self, center: (f32, f32), radius: f32, width: f32, color: Color32) {
        let (r, half) = (radius * self.k, width * self.k * 0.5);
        self.shade(
            (center.0 - r - half, center.1 - r - half),
            (center.0 + r + half, center.1 + r + half),
            rgb(color),
            |x, y| {
                let d = ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt();
                half - (d - r).abs() + 0.5
            },
        );
    }

    fn line(&mut self, a: (f32, f32), b: (f32, f32), width: f32, color: Color32) {
        let half = width * self.k * 0.5;
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len2 = dx * dx + dy * dy;
        self.shade(
            (a.0.min(b.0) - half, a.1.min(b.1) - half),
            (a.0.max(b.0) + half, a.1.max(b.1) + half),
            rgb(color),
            |x, y| {
                let t = if len2 > 0.0 {
                    (((x - a.0) * dx + (y - a.1) * dy) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let d = ((x - (a.0 + t * dx)).powi(2) + (y - (a.1 + t * dy)).powi(2)).sqrt();
                half - d + 0.5
            },
        );
    }

    fn triangle(&mut self, p: [(f32, f32); 3], color: Color32) {
        let area = (p[1].0 - p[0].0) * (p[2].1 - p[0].1) - (p[1].1 - p[0].1) * (p[2].0 - p[0].0);
        if area.abs() < 1.0e-6 {
            return;
        }
        let min = (
            p.iter().map(|q| q.0).fold(f32::MAX, f32::min),
            p.iter().map(|q| q.1).fold(f32::MAX, f32::min),
        );
        let max = (
            p.iter().map(|q| q.0).fold(f32::MIN, f32::max),
            p.iter().map(|q| q.1).fold(f32::MIN, f32::max),
        );
        self.shade(min, max, rgb(color), |x, y| {
            let w = |a: (f32, f32), b: (f32, f32)| {
                ((b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0)) / area
            };
            let (w0, w1, w2) = (w(p[1], p[2]), w(p[2], p[0]), w(p[0], p[1]));
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                1.0
            } else {
                0.0
            }
        });
    }

    fn arrow(&mut self, from: (f32, f32), to: (f32, f32), color: Color32) {
        if from == to {
            return;
        }
        self.line(from, to, 1.5, color);
        let (dx, dy) = (to.0 - from.0, to.1 - from.1);
        let len = (dx * dx + dy * dy).sqrt();
        let dir = (dx / len, dy / len);
        let normal = (-dir.1, dir.0);
        let base = (to.0 - dir.0 * 8.0 * self.k, to.1 - dir.1 * 8.0 * self.k);
        let wing = 4.5 * self.k;
        self.triangle(
            [
                (base.0 + normal.0 * wing, base.1 + normal.1 * wing),
                (base.0 - normal.0 * wing, base.1 - normal.1 * wing),
                to,
            ],
            color,
        );
    }

    fn polyline(&mut self, points: &[(f32, f32)], width: f32, color: Color32) {
        for pair in points.windows(2) {
            self.line(pair[0], pair[1], width, color);
        }
    }
}

fn paint_overlay(p: &mut Painter, transform: RegistrationTransform, overlay: &OverlayPaintData) {
    let raw = |pt: &TrackedPoint| (pt.x as f32, pt.y as f32);
    let reg = |pt: &TrackedPoint| {
        let (x, y) = transform.apply(pt.x, pt.y);
        (x as f32, y as f32)
    };
    match overlay.style {
        OverlayStyle::Hidden => return,
        OverlayStyle::Points => {
            for track in &overlay.tracks {
                let pos = reg(&track.current);
                if track
                    .current
                    .confidence
                    .is_none_or(|c| c >= ANCHOR_MIN_CONFIDENCE)
                {
                    p.disc(pos, 2.4, COLOR_CURRENT);
                } else {
                    p.ring(pos, 2.6, 1.0, COLOR_LOW_CONF);
                }
            }
            return;
        }
        OverlayStyle::Full => {}
    }
    let flags = &overlay.flags;

    if flags.show_trajectories && flags.history_length > 0 {
        for track in &overlay.tracks {
            if track.history.is_empty() {
                continue;
            }
            let mut pts: Vec<(f32, f32)> = track.history.iter().map(reg).collect();
            pts.push(reg(&track.current));
            p.polyline(&pts, 1.0, COLOR_TRAJ);
        }
    }
    if flags.show_reference_ghost {
        if let Some(reference) = &overlay.reference_points {
            for (_, pt) in reference {
                p.ring(raw(pt), 4.0, 1.2, COLOR_GHOST);
            }
        }
    }
    if flags.show_displacement_vectors || flags.show_displacement_lines {
        for track in &overlay.tracks {
            let Some(reference) = &track.reference else {
                continue;
            };
            let (from, to) = (raw(reference), reg(&track.current));
            if flags.show_displacement_vectors {
                p.arrow(from, to, COLOR_DISP);
            } else {
                p.line(from, to, 1.0, COLOR_LINE);
            }
        }
    }

    if flags.show_reference_locations {
        for track in &overlay.tracks {
            if let Some(reference) = &track.reference {
                let c = raw(reference);
                p.line((c.0 - 4.0, c.1), (c.0 + 4.0, c.1), 1.0, COLOR_GHOST);
                p.line((c.0, c.1 - 4.0), (c.0, c.1 + 4.0), 1.0, COLOR_GHOST);
            }
        }
    }
    if flags.show_raw_points {
        for track in &overlay.tracks {
            p.disc(raw(&track.current), 3.0, COLOR_RAW);
        }
    }
    if flags.show_transformed_points {
        for track in &overlay.tracks {
            p.ring(reg(&track.current), 5.0, 1.0, COLOR_TRANSFORMED);
        }
    }
    if flags.show_points {
        for track in &overlay.tracks {
            let pos = reg(&track.current);
            if track.is_reference {
                p.ring(pos, 6.5, 1.2, COLOR_REF_BADGE);
            }
            p.disc(pos, 3.2, COLOR_CURRENT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracked::{OverlaySettings, OverlayTrack};
    use fabric_data::tracking::TrackId;

    fn flat(w: u32, h: u32, v: u8) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb([v, v, v]))
    }

    fn point(x: f64, y: f64, confidence: Option<f64>) -> TrackedPoint {
        TrackedPoint { x, y, confidence }
    }

    fn overlay(
        style: OverlayStyle,
        tracks: Vec<OverlayTrack>,
        flags: OverlaySettings,
    ) -> OverlayPaintData {
        OverlayPaintData {
            tracks,
            reference_points: None,
            selected_track: None,
            flags,
            style,
            transform: RegistrationTransform::IDENTITY,
        }
    }

    fn track(id: &str, current: TrackedPoint, reference: Option<TrackedPoint>) -> OverlayTrack {
        OverlayTrack {
            id: TrackId::from(id.to_string()),
            current,
            reference,
            is_reference: false,
            history: Vec::new(),
        }
    }

    const AMBER: [u8; 3] = [255, 200, 40];

    #[test]
    fn identity_without_overlay_is_the_source_frame() {
        let mut src = flat(32, 24, 90);
        src.put_pixel(5, 6, Rgb([1, 2, 3]));
        assert_eq!(
            render_frame(&src, RegistrationTransform::IDENTITY, None),
            src
        );
    }

    #[test]
    fn stabilization_moves_the_image_and_exposes_background() {
        let mut src = flat(40, 30, 200);
        src.put_pixel(10, 10, Rgb([255, 0, 0]));
        let t = RegistrationTransform::Translation { tx: 5.0, ty: 3.0 };
        let out = render_frame(&src, t, None);
        assert_eq!(
            out.get_pixel(15, 13).0,
            [255, 0, 0],
            "content moves by the transform"
        );
        assert_eq!(
            out.get_pixel(2, 2).0,
            [34, 34, 34],
            "uncovered area is background"
        );
        assert_eq!(out.get_pixel(20, 20).0, [200, 200, 200]);
    }

    #[test]
    fn hidden_style_draws_the_stabilized_image_only() {
        let src = flat(40, 30, 120);
        let t = RegistrationTransform::Translation { tx: 4.0, ty: 0.0 };
        let hidden = overlay(
            OverlayStyle::Hidden,
            vec![track("a", point(20.0, 15.0, Some(1.0)), None)],
            OverlaySettings::default(),
        );
        assert_eq!(
            render_frame(&src, t, Some(&hidden)),
            render_frame(&src, t, None)
        );
    }

    #[test]
    fn points_style_puts_dots_where_the_image_content_moved() {
        let src = flat(60, 40, 60);
        let t = RegistrationTransform::Translation { tx: 7.0, ty: -2.0 };
        let data = overlay(
            OverlayStyle::Points,
            vec![
                track("a", point(20.0, 20.0, Some(0.9)), None),
                track("b", point(40.0, 25.0, Some(0.1)), None),
            ],
            OverlaySettings::default(),
        );
        let out = render_frame(&src, t, Some(&data));
        assert_eq!(
            out.get_pixel(27, 18).0,
            AMBER,
            "confident point at the transformed position"
        );
        assert_ne!(
            out.get_pixel(20, 20).0,
            AMBER,
            "nothing at the untransformed position"
        );
        // low confidence is hollow: the centre keeps the image, the rim is coloured
        assert_eq!(
            out.get_pixel(47, 23).0,
            render_frame(&src, t, None).get_pixel(47, 23).0
        );
        assert_ne!(out.get_pixel(49, 23).0, [60, 60, 60]);
    }

    #[test]
    fn full_style_draws_displacement_lines_and_point_positions() {
        let src = flat(80, 50, 40);
        let flags = OverlaySettings {
            show_displacement_lines: true,
            show_points: true,
            show_reference_ghost: false,
            show_trajectories: false,
            ..Default::default()
        };
        let data = overlay(
            OverlayStyle::Full,
            vec![track(
                "a",
                point(60.0, 25.0, Some(1.0)),
                Some(point(20.0, 25.0, Some(1.0))),
            )],
            flags,
        );
        let out = render_frame(&src, RegistrationTransform::IDENTITY, Some(&data));
        assert_eq!(out.get_pixel(60, 25).0, AMBER, "current point marker");
        assert_ne!(
            out.get_pixel(40, 25).0,
            [40, 40, 40],
            "displacement line between reference and current"
        );
        assert_eq!(
            out.get_pixel(40, 10).0,
            [40, 40, 40],
            "nothing off the line"
        );
        let mut no_points = data.clone();
        no_points.flags.show_points = false;
        let out2 = render_frame(&src, RegistrationTransform::IDENTITY, Some(&no_points));
        assert_ne!(out2.get_pixel(40, 25).0, [40, 40, 40]);
        assert_ne!(out2.get_pixel(60, 25).0, AMBER);
    }

    #[test]
    fn overlay_style_follows_the_stabilize_switch_matrix() {
        use crate::tracked::TrackingSession;
        let expect = [
            ((false, false), OverlayStyle::Full),
            ((true, false), OverlayStyle::Hidden),
            ((true, true), OverlayStyle::Points),
            ((false, true), OverlayStyle::Points),
        ];
        for ((enabled, show_points), style) in expect {
            let mut session = TrackingSession::default();
            session.stabilize.enabled = enabled;
            session.stabilize.show_points = show_points;
            assert_eq!(
                session.overlay_style(),
                style,
                "enabled={enabled} show_points={show_points}"
            );
        }
    }

    fn write_frames(dir: &Path, n: u32) -> Vec<ExportFrame> {
        std::fs::create_dir_all(dir).unwrap();
        (0..n)
            .map(|i| {
                let path = dir.join(format!("src_{i}.png"));
                flat(64, 48, 50 + i as u8 * 20).save(&path).unwrap();
                ExportFrame {
                    image_path: path,
                    transform: RegistrationTransform::Translation {
                        tx: i as f64,
                        ty: 0.0,
                    },
                    overlay: Some(overlay(
                        OverlayStyle::Points,
                        vec![track("a", point(30.0, 20.0, Some(1.0)), None)],
                        OverlaySettings::default(),
                    )),
                }
            })
            .collect()
    }

    fn wait(handle: &ExportHandle) -> Result<ExportReport, String> {
        for _ in 0..600 {
            if let Some(result) = handle.take_result() {
                return result;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        Err("timeout".into())
    }

    #[test]
    fn export_writes_numbered_png_frames() {
        let dir = std::env::temp_dir().join(format!("fabsim-export-{}-frames", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let handle = spawn(
            write_frames(&dir.join("src"), 5),
            ExportOptions {
                fps: 10.0,
                frames_dir: dir.join("out"),
                video: None,
            },
        );
        assert_eq!(handle.total, 5);
        let report = wait(&handle).unwrap();
        assert_eq!(report.frames, 5);
        assert_eq!(handle.progress(), 5);
        for i in 0..5 {
            let frame = image::open(dir.join(format!("out/frame_{i:05}.png"))).unwrap();
            assert_eq!((frame.width(), frame.height()), (64, 48));
        }
        assert!(report.video.is_none() && report.video_note.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_shorter_export_removes_the_stale_frames_of_an_earlier_one() {
        let dir = std::env::temp_dir().join(format!("fabsim-export-{}-stale", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let out = dir.join("out");
        let options = |frames_dir: &Path| ExportOptions {
            fps: 10.0,
            frames_dir: frames_dir.to_owned(),
            video: None,
        };
        wait(&spawn(write_frames(&dir.join("src"), 6), options(&out))).unwrap();
        std::fs::write(out.join("keep.txt"), "not a frame").unwrap();
        wait(&spawn(write_frames(&dir.join("src"), 3), options(&out))).unwrap();
        assert!(out.join("frame_00002.png").exists());
        assert!(!out.join("frame_00003.png").exists(), "stale frame removed");
        assert!(!out.join("frame_00005.png").exists(), "stale frame removed");
        assert!(out.join("keep.txt").exists(), "other files are left alone");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_encodes_a_video_when_ffmpeg_is_available() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("fabsim-export-{}-video", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let video = dir.join("clip.mp4");
        let handle = spawn(
            write_frames(&dir.join("src"), 6),
            ExportOptions {
                fps: 12.0,
                frames_dir: dir.join("out"),
                video: Some(video.clone()),
            },
        );
        let report = wait(&handle).unwrap();
        assert_eq!(
            report.video.as_deref(),
            Some(video.as_path()),
            "{:?}",
            report.video_note
        );
        assert!(std::fs::metadata(&video).unwrap().len() > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_source_image_is_reported_not_skipped() {
        let dir =
            std::env::temp_dir().join(format!("fabsim-export-{}-missing", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let handle = spawn(
            vec![ExportFrame {
                image_path: dir.join("nope.png"),
                transform: RegistrationTransform::IDENTITY,
                overlay: None,
            }],
            ExportOptions {
                fps: 10.0,
                frames_dir: dir.join("out"),
                video: None,
            },
        );
        assert!(wait(&handle).unwrap_err().contains("nope.png"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
