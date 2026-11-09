//! Tests of the Image Viewer: layout math, overlay geometry, panel structure
//! (headless egui) and the relight request logic.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use egui::{Pos2, Rect, Vec2};
use fabric_data::tracking::registration::RegistrationTransform;
use fabric_data::tracking::{FrameId, TrackId, TrackedPoint, TrackingSequence};
use fabric_data::Run;

use super::advanced::advanced_tracking_section;
use super::canvas::{comparison_rects, fit_scale, RegDisplay};
use super::export::{export_frames, export_mode_label};
use super::overlay::{build_overlay, history_for};
use super::relight::{RelightSources, RelightState};
use super::replay::{replay_row, ReplayTarget};
use super::tracking_panel::{calibration_px_per_mm, tracking_panel};
use super::*;
use crate::lighting::Light;
use crate::project::{frame_ref_for, ProjectSlot};
use crate::settings::RelightSettings;
use crate::tracked::{OverlayStyle, TrackingSession};
fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect::from_min_size(Pos2::new(x, y), Vec2::new(w, h))
}

#[test]
fn fit_scale_preserves_aspect() {
    // 2:1 image in a 1:2 box fits the width: scale = box_w / img_w.
    let s = fit_scale((200, 100), rect(0.0, 0.0, 100.0, 400.0));
    assert!((s - 0.5).abs() < 1e-6);
    // 1:2 image in a 2:1 box fits the height.
    let s = fit_scale((100, 200), rect(0.0, 0.0, 400.0, 100.0));
    assert!((s - 0.5).abs() < 1e-6);
    // Degenerate inputs never divide by zero or produce NaN.
    let s = fit_scale((0, 0), rect(0.0, 0.0, 10.0, 10.0));
    assert_eq!(s, 1.0);
    let s = fit_scale((10, 10), rect(0.0, 0.0, 0.0, 0.0));
    assert_eq!(s, 1.0);
}

#[test]
fn side_by_side_rects_split_in_half() {
    let r = rect(100.0, 50.0, 200.0, 100.0);
    let (primary, secondary) = comparison_rects(ComparisonMode::SideBySide, true, r, 0.5);
    assert_eq!(primary.min.x, 100.0);
    assert_eq!(primary.max.x, 200.0);
    assert_eq!(secondary.min.x, 200.0);
    assert_eq!(secondary.max.x, 300.0);
    assert_eq!(primary.height(), 100.0);
    assert_eq!(secondary.height(), 100.0);
}

#[test]
fn split_wipe_rects_follow_divider() {
    let r = rect(0.0, 0.0, 100.0, 50.0);
    let (primary, secondary) = comparison_rects(ComparisonMode::SplitWipe, true, r, 0.3);
    assert_eq!(primary, r, "primary uses the full canvas");
    assert!(
        (secondary.min.x - 30.0).abs() < 1e-4,
        "got {}",
        secondary.min.x
    );
    assert!(
        (secondary.max.x - 100.0).abs() < 1e-4,
        "got {}",
        secondary.max.x
    );

    let (primary, secondary) = comparison_rects(ComparisonMode::SplitWipe, false, r, 0.3);
    assert_eq!(primary, r);
    assert_eq!(
        secondary, r,
        "without a secondary both fall back to the canvas"
    );
}

#[test]
fn single_mode_uses_full_canvas() {
    let r = rect(5.0, 5.0, 60.0, 40.0);
    let (primary, secondary) = comparison_rects(ComparisonMode::Single, true, r, 0.5);
    assert_eq!(primary, r);
    assert_eq!(secondary, r);
}

#[test]
fn relight_state_notes_are_human_readable() {
    assert!(RelightState::NoNormal.note().is_some());
    assert!(RelightState::NoAlbedo("no PS".into())
        .note()
        .is_some_and(|n| n.contains("no PS")));
    assert!(RelightState::SourceError("boom".into()).note().is_some());
    assert!(RelightState::Active.note().is_none());
    assert!(RelightState::Disabled.note().is_none());
}

#[test]
fn reg_display_maps_image_to_screen() {
    // Identity transform: the display is a pure scale + offset.
    let view = RegDisplay {
        scale: 2.0,
        rect: rect(100.0, 50.0, 200.0, 100.0),
        transform: RegistrationTransform::IDENTITY,
    };
    let p = TrackedPoint::new(10.0, 20.0);
    let screen = view.reg(&p);
    assert!((screen.x - 120.0).abs() < 1e-5);
    assert!((screen.y - 90.0).abs() < 1e-5);
    let raw = view.raw(&p);
    assert!((raw.x - 120.0).abs() < 1e-5);
    assert!((raw.y - 90.0).abs() < 1e-5);
}

#[test]
fn reg_display_applies_translation_before_zoom() {
    // Translation (5, -3) in image pixels: the registered point lands at
    // raw + translation, scaled by `scale`.
    let view = RegDisplay {
        scale: 2.0,
        rect: rect(0.0, 0.0, 400.0, 400.0),
        transform: RegistrationTransform::Translation { tx: 5.0, ty: -3.0 },
    };
    let p = TrackedPoint::new(10.0, 20.0);
    let screen = view.reg(&p);
    assert!((screen.x - 30.0).abs() < 1e-5, "got {}", screen.x);
    assert!((screen.y - 34.0).abs() < 1e-5, "got {}", screen.y);
    let raw = view.raw(&p);
    assert!((raw.x - 20.0).abs() < 1e-5);
    assert!((raw.y - 40.0).abs() < 1e-5);
}

#[test]
fn overlay_history_takes_last_n_frames() {
    use fabric_data::tracking::{TrackingFrame, TrackingSequence};
    let frames: Vec<TrackingFrame> = (0..6)
        .map(|i| {
            TrackingFrame::new(
                FrameId(i),
                vec![(TrackId::new("a"), TrackedPoint::new(i as f64, i as f64))],
            )
        })
        .collect();
    let seq = TrackingSequence::new(frames);
    let history = history_for(&seq, FrameId(5), &TrackId::new("a"), 0.0, 3);
    assert_eq!(history.len(), 3);
    let xs: Vec<f64> = history.iter().map(|p| p.x).collect();
    assert_eq!(xs, vec![2.0, 3.0, 4.0]);
    let history = history_for(&seq, FrameId(1), &TrackId::new("a"), 0.0, 3);
    let xs: Vec<f64> = history.iter().map(|p| p.x).collect();
    assert_eq!(xs, vec![0.0]);
}

#[test]
fn calibration_requires_real_metadata() {
    // Missing calibration stays unknown.
    let run = Run {
        dir: Path::new("/tmp").to_owned(),
        session: fabric_data::Session::new(
            "s",
            "t",
            fabric_data::Direction::Warp,
            fabric_data::Branch::Loading,
            0,
        ),
        frames: fabric_data::FramesFile::default(),
        force: fabric_data::ForceFile::default(),
        load_notes: vec![],
    };
    assert_eq!(calibration_px_per_mm(&run), None);
    let mut run = run;
    run.session
        .extra
        .insert("pixels_per_mm".into(), serde_json::json!(100.0));
    assert_eq!(calibration_px_per_mm(&run), Some(100.0));
}

/// A 3x3 grid at rows y = 0/100/200 (frame 0), moved by `shift` with
/// `stretch` extra pixels of downward extension per row (frame `id`).
fn stab_grid(id: u32, shift: (f64, f64), stretch: f64) -> fabric_data::tracking::TrackingFrame {
    use fabric_data::tracking::TrackingFrame;
    let mut points = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            points.push((
                TrackId::new(format!("r{row}c{col}")),
                TrackedPoint::with_confidence(
                    col as f64 * 50.0 + shift.0,
                    row as f64 * 100.0 + shift.1 + row as f64 * stretch,
                    1.0,
                ),
            ));
        }
    }
    TrackingFrame::new(FrameId(id), points)
}

fn stab_sequence() -> TrackingSequence {
    TrackingSequence::new(vec![
        stab_grid(0, (0.0, 0.0), 0.0),
        stab_grid(1, (20.0, 10.0), 30.0),
        stab_grid(2, (14.0, 26.0), 60.0),
    ])
}

#[test]
fn stabilized_anchor_points_hold_one_screen_position() {
    // Downscaled display must not move the stabilization anchor.
    use fabric_data::tracking::stabilize::stabilization_for_frame;
    let sequence = stab_sequence();
    let display_rect = rect(120.0, 40.0, 1280.0 * 0.35, 1024.0 * 0.35);
    let screen_of = |frame: u32, track: &str| -> Pos2 {
        let stab = stabilization_for_frame(&sequence, FrameId(0), FrameId(frame)).unwrap();
        let view = RegDisplay {
            scale: 0.35,
            rect: display_rect,
            transform: stab.transform,
        };
        view.reg(
            sequence
                .point(&FrameId(frame), &TrackId::new(track))
                .unwrap(),
        )
    };

    let anchor0 = screen_of(0, "r0c1");
    for frame in [1, 2] {
        let anchor = screen_of(frame, "r0c1");
        assert!(
            anchor.distance(anchor0) < 1e-4,
            "anchored point moved to {anchor:?} on frame {frame}"
        );
    }
    // The lower fabric keeps moving down the screen, by the extension
    // scaled into display pixels (2 rows x 30 px x 0.35 = 21 pt).
    let bottom0 = screen_of(0, "r2c1");
    let bottom1 = screen_of(1, "r2c1");
    let bottom2 = screen_of(2, "r2c1");
    assert!(
        (bottom1.y - bottom0.y - 21.0).abs() < 1e-4,
        "{}",
        bottom1.y - bottom0.y
    );
    assert!(bottom2.y > bottom1.y, "extension keeps growing downwards");
    assert!((bottom1.x - bottom0.x).abs() < 1e-4, "no sideways drift");
}

#[test]
fn stabilized_image_is_drawn_offset_by_the_inverse_displacement() {
    // Image corners and overlay points must use the same offset and scale.
    use fabric_data::tracking::stabilize::stabilization_for_frame;
    let sequence = stab_sequence();
    let stab = stabilization_for_frame(&sequence, FrameId(0), FrameId(1)).unwrap();
    let view = RegDisplay {
        scale: 0.5,
        rect: rect(10.0, 20.0, 640.0, 512.0),
        transform: stab.transform,
    };
    let corner = view.reg_xy(0.0, 0.0);
    assert!((corner.x - (10.0 - 0.5 * 20.0)).abs() < 1e-4, "{corner:?}");
    assert!((corner.y - (20.0 - 0.5 * 10.0)).abs() < 1e-4, "{corner:?}");
    let far = view.reg_xy(1280.0, 1024.0);
    assert!((far.x - corner.x - 640.0).abs() < 1e-4);
    assert!((far.y - corner.y - 512.0).abs() < 1e-4);
}

#[test]
fn stabilization_off_renders_exactly_the_raw_image() {
    let view = RegDisplay {
        scale: 0.35,
        rect: rect(120.0, 40.0, 448.0, 358.4),
        transform: RegistrationTransform::IDENTITY,
    };
    let p = TrackedPoint::new(640.0, 512.0);
    assert_eq!(view.reg(&p), view.raw(&p));
    assert_eq!(view.reg_xy(0.0, 0.0), view.rect.min);
    assert!(view.transform.is_identity(), "the fast image path is used");
}

#[test]
fn reference_frame_is_never_offset() {
    use fabric_data::tracking::stabilize::stabilization_for_frame;
    let sequence = stab_sequence();
    let stab = stabilization_for_frame(&sequence, FrameId(0), FrameId(0)).unwrap();
    assert!(stab.transform.is_identity(), "frame 0 renders unchanged");
}

#[test]
fn point_overlay_is_lightweight_and_ignores_the_legacy_scope() {
    // Simple points ignore stale history/reference filters.
    use crate::tracked::{TrackScope, TrackingSession};
    let dir = std::env::temp_dir().join(format!("fabsim_ov_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tracks.json");
    std::fs::write(
        &path,
        br#"{"frames":[
            {"frame_id":0,"points":[{"id":"a","x":0.0,"y":0.0},{"id":"b","x":10.0,"y":300.0}]},
            {"frame_id":1,"points":[{"id":"a","x":5.0,"y":5.0},{"id":"b","x":15.0,"y":340.0}]}
        ]}"#,
    )
    .unwrap();
    let key = Path::new("/project/sample/warp_loading");
    let mut tracking = TrackingSession::default();
    tracking.import_file(key, &path).unwrap();
    tracking.overlay.scope = TrackScope::ReferenceOnly;
    tracking.set_reference_tracks(key, BTreeSet::from([TrackId::new("a")]));

    let run = tracking.run_state(key).unwrap();
    let full = build_overlay(
        &tracking,
        run,
        FrameId(1),
        RegistrationTransform::IDENTITY,
        OverlayStyle::Full,
    )
    .unwrap();
    assert_eq!(full.tracks.len(), 1, "the legacy scope still applies");
    assert!(full.reference_points.is_some());

    let points = build_overlay(
        &tracking,
        run,
        FrameId(1),
        RegistrationTransform::IDENTITY,
        OverlayStyle::Points,
    )
    .unwrap();
    assert_eq!(points.tracks.len(), 2, "every tracked point is drawn");
    assert!(points.reference_points.is_none(), "no ghost data is built");
    assert!(
        points.tracks.iter().all(|t| t.history.is_empty()),
        "no trajectory history is built"
    );
    assert_eq!(points.style, OverlayStyle::Points);
    std::fs::remove_dir_all(&dir).ok();
}

/// Every string painted by one headless egui frame, so a test can assert
/// which controls a panel actually rendered.
fn painted_text(output: &egui::FullOutput) -> String {
    fn walk(shape: &egui::Shape, out: &mut String) {
        match shape {
            egui::Shape::Text(text) => {
                out.push_str(text.galley.text());
                out.push('\n');
            }
            egui::Shape::Vec(shapes) => {
                for s in shapes {
                    walk(s, out);
                }
            }
            _ => {}
        }
    }
    let mut out = String::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut out);
    }
    out
}

fn headless_frame(mut build: impl FnMut(&mut egui::Ui)) -> String {
    let ctx = egui::Context::default();
    let mut text = String::new();
    // Two frames: collapsing headers settle their open state on the first.
    for _ in 0..2 {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 1200.0))),
            ..Default::default()
        };
        let output = ctx.run_ui(input, |ui| build(ui));
        text = painted_text(&output);
    }
    text
}

fn tracking_fixture(tag: &str) -> (TrackingSession, PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("fabsim_panel_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tracks.json");
    std::fs::write(
        &path,
        br#"{"frames":[
            {"frame_id":0,"points":[{"id":"a","x":0.0,"y":0.0},{"id":"b","x":10.0,"y":300.0}]},
            {"frame_id":1,"points":[{"id":"a","x":5.0,"y":5.0},{"id":"b","x":15.0,"y":340.0}]}
        ]}"#,
    )
    .unwrap();
    let key = PathBuf::from("/project/sample/warp_loading");
    let mut tracking = TrackingSession::default();
    tracking.import_file(&key, &path).unwrap();
    (tracking, key, dir)
}

fn empty_run() -> Run {
    Run {
        dir: Path::new("/tmp").to_owned(),
        session: fabric_data::Session::new(
            "s",
            "t",
            fabric_data::Direction::Warp,
            fabric_data::Branch::Loading,
            0,
        ),
        frames: fabric_data::FramesFile::default(),
        force: fabric_data::ForceFile::default(),
        load_notes: vec![],
    }
}

/// Controls that must never appear in the default (non-advanced) panel.
const ADVANCED_MARKERS: [&str; 8] = [
    "min confidence",
    "reference tracks:",
    "mode:",
    "canvas:",
    "diagnostics:",
    "overlay:",
    "Export tracks",
    "Select top row",
];

/// Playback moved out of the tracking section entirely.
const PLAYBACK_MARKERS: [&str; 3] = ["FPS", "Loop", "Play"];

#[test]
fn default_tracking_panel_is_simple_and_hides_the_workbench() {
    let (mut tracking, key, dir) = tracking_fixture("simple");
    let run = empty_run();
    let text = headless_frame(|ui| {
        tracking_panel(ui, &mut tracking, &key, &run, FrameId(0), false);
    });
    for expected in [
        "Tracking",
        "Tracking: loaded",
        "Reference frame:",
        "Use current",
        "Stabilize using CoWTracker",
        "Show tracked points",
        "Load tracking...",
        "Unload",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    assert!(
        !text.contains("Advanced tracking & registration"),
        "the workbench must be completely absent by default:\n{text}"
    );
    for marker in ADVANCED_MARKERS {
        assert!(!text.contains(marker), "advanced control {marker:?} leaked");
    }
    for marker in PLAYBACK_MARKERS {
        assert!(
            !text.contains(marker),
            "playback control {marker:?} still lives in the tracking panel"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn advanced_preference_adds_the_collapsed_workbench() {
    let (mut tracking, key, dir) = tracking_fixture("advanced");
    let run = empty_run();
    let text = headless_frame(|ui| {
        tracking_panel(ui, &mut tracking, &key, &run, FrameId(0), true);
    });
    assert!(
        text.contains("Advanced tracking & registration"),
        "the workbench must be reachable when the preference is on:\n{text}"
    );
    // Legacy controls stay collapsed when enabled.
    for marker in ADVANCED_MARKERS {
        assert!(
            !text.contains(marker),
            "{marker:?} should stay behind the collapsed header"
        );
    }
    for expected in ["Stabilize using CoWTracker", "Show tracked points"] {
        assert!(text.contains(expected), "missing {expected:?}");
    }
    for marker in PLAYBACK_MARKERS {
        assert!(!text.contains(marker), "playback leaked into tracking");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn playback_controls_live_with_the_replay_timeline() {
    // Playback must remain available without loaded tracks.
    let mut selection = FrameSelection::default();
    let tracking = TrackingSession::default();
    let mut playback = PlaybackState::default();
    assert!(!tracking.any_loaded(), "no tracking in this fixture");
    let at = |frame, n_frames| ReplayTarget {
        project: 0,
        run: 0,
        frame,
        n_frames,
    };
    let text = headless_frame(|ui| {
        replay_row(ui, &[], &mut selection, &mut playback, at(3, 20));
    });
    for expected in ["Frame 4 / 20", "Play", "FPS", "Loop", "timeline"] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }

    playback.playing = true;
    let text = headless_frame(|ui| {
        replay_row(ui, &[], &mut selection, &mut playback, at(3, 20));
    });
    assert!(text.contains("Pause"), "playing state shows Pause:\n{text}");
    assert!(playback.playing, "rendering must not toggle play");

    // A run without frames draws no transport instead of a broken one.
    let text = headless_frame(|ui| {
        replay_row(ui, &[], &mut selection, &mut playback, at(0, 0));
    });
    assert!(!text.contains("FPS"), "no frames -> no transport:\n{text}");
}

#[test]
fn the_legacy_workbench_still_renders_every_control() {
    // Expanding the advanced section exposes the legacy controls.
    let (mut tracking, key, dir) = tracking_fixture("workbench");
    let run = empty_run();
    let text = headless_frame(|ui| {
        advanced_tracking_section(ui, &mut tracking, &key, &run, FrameId(0));
    });
    for expected in [
        "path:",
        "Export tracks",
        "reference tracks:",
        "Select top row",
        "Use selected",
        "Clear reference",
        "mode:",
        "min confidence",
        "overlay:",
        "All valid",
        "Reference only",
        "canvas:",
        "trajectories",
        "diagnostics:",
        "raw points",
        "reference ghost",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    for marker in PLAYBACK_MARKERS {
        assert!(!text.contains(marker), "playback leaked into the workbench");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn export_mode_label_covers_the_switch_matrix() {
    assert_eq!(export_mode_label(true, false), "stabilized, no points");
    assert_eq!(
        export_mode_label(true, true),
        "stabilized, with tracked points"
    );
    assert_eq!(
        export_mode_label(false, true),
        "registered, with tracked points"
    );
    assert!(export_mode_label(false, false).contains("full overlay"));
}

/// Render switch combinations on `$FABSIM_DATA_DIR/fabric_002/warp_loading`.
/// Set `FABSIM_EXPORT_DEBUG_DIR` to retain images for inspection.
#[test]
#[ignore = "needs FABSIM_DATA_DIR real data"]
fn real_run_renders_the_four_switch_modes() {
    let data = std::env::var_os("FABSIM_DATA_DIR").expect("FABSIM_DATA_DIR must be set");
    let project = fabric_data::Project::discover(Path::new(&data).join("fabric_002")).unwrap();
    let slot = ProjectSlot::new("fabric_002".into(), project);
    let ridx = slot
        .project
        .runs
        .iter()
        .position(|r| r.dir.ends_with("warp_loading") && r.dir.join("tracks.json").exists())
        .expect("fabric_002/warp_loading/tracks.json");
    let run = &slot.project.runs[ridx];
    let run_key = slot.run_keys[ridx].clone();
    let mut tracking = TrackingSession::default();
    tracking.auto_discover(&run_key).ok();
    assert!(
        tracking.run_state(&run_key).is_some(),
        "tracks.json must import"
    );
    let settings = ViewerSettings::default();
    let last = run.frames.frames.len() - 1;
    let render = |tracking: &TrackingSession, index: usize| {
        let frames = export_frames(tracking, &settings, run, &slot.entries[ridx].ps, &run_key);
        let f = &frames[index];
        let src = image::open(&f.image_path).unwrap().to_rgb8();
        (
            src.clone(),
            crate::frame_export::render_frame(&src, f.transform, f.overlay.as_ref()),
            f.transform.is_identity(),
            f.overlay.is_some(),
        )
    };
    let mut results = Vec::new();
    for (enabled, show_points, name) in [
        (true, false, "stab_nopoints"),
        (true, true, "stab_points"),
        (false, true, "raw_points"),
        (false, false, "raw_full"),
    ] {
        tracking.stabilize.enabled = enabled;
        tracking.stabilize.show_points = show_points;
        tracking.overlay.show_displacement_lines = !enabled && !show_points;
        tracking.overlay.show_displacement_vectors = false;
        let (src, out, identity, has_overlay) = render(&tracking, last);
        if let Ok(dir) = std::env::var("FABSIM_EXPORT_DEBUG_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            out.save(std::path::Path::new(&dir).join(format!("{name}.png")))
                .unwrap();
        }
        results.push((name, src, out, identity, has_overlay));
    }
    let get = |n: &str| results.iter().find(|r| r.0 == n).unwrap();
    assert!(
        !get("stab_nopoints").3,
        "last frame must be moved by stabilization"
    );
    assert_ne!(get("stab_nopoints").2, get("stab_nopoints").1);
    assert_eq!(
        get("raw_points").3,
        get("raw_full").3,
        "same transform without stabilization"
    );
    assert!(get("stab_points").4 && get("raw_points").4 && get("raw_full").4);
    let differs = |a: &str, b: &str| get(a).2 != get(b).2;
    assert!(
        differs("stab_nopoints", "stab_points"),
        "points drawn over the stabilized image"
    );
    assert!(
        differs("raw_points", "raw_full"),
        "full overlay adds displacement lines"
    );
    let drawn = |n: &str| {
        get(n)
            .2
            .pixels()
            .zip(get(n).1.pixels())
            .filter(|(a, b)| a != b)
            .count()
    };
    assert!(
        drawn("raw_full") > drawn("raw_points"),
        "full overlay draws more than points"
    );
}

fn relit_entry(ctx: &egui::Context, signature: RelightSignature) -> relight::RelightEntry {
    let options = egui::TextureOptions::LINEAR;
    let image = image::RgbaImage::new(2, 2);
    relight::RelightEntry {
        signature,
        texture: ctx.load_texture("test", egui::ColorImage::example(), options),
        image,
        options,
    }
}

fn signature(azimuth_deg: f32) -> RelightSignature {
    RelightSignature {
        albedo_key: PathBuf::from("/a"),
        normal_key: PathBuf::from("/n"),
        albedo_scale: 1.0,
        light: Light {
            azimuth_deg,
            ..Light::default()
        },
    }
}

#[test]
fn relight_revert_invalidates_superseded_job() {
    // Reverting from in-flight B to displayed A must invalidate B's result.
    let (sig_a, sig_b) = (signature(45.0), signature(120.0));
    let ctx = egui::Context::default();
    let mut state = ImageViewerState {
        relight: Some(relit_entry(&ctx, sig_a.clone())),
        relight_pending: true,
        relight_generation: 7,
        relight_requested: Some(sig_b.clone()),
        ..ImageViewerState::default()
    };

    assert!(
        state.invalidate_superseded_inflight(&sig_a),
        "displayed texture is current"
    );
    assert!(!state.relight_pending, "superseded job dropped");
    assert_eq!(state.relight_requested, None);
    assert_ne!(state.relight_generation, 7, "a stale B result is discarded");
    assert!(state.relight.is_some(), "the displayed texture survives");

    // An undisplayed signature must not cancel current work.
    let mut state2 = ImageViewerState {
        relight_pending: true,
        relight_generation: 3,
        relight_requested: Some(sig_b),
        ..ImageViewerState::default()
    };
    assert!(!state2.invalidate_superseded_inflight(&sig_a));
    assert!(state2.relight_pending);
    assert_eq!(state2.relight_generation, 3);
}

#[test]
fn the_texture_filter_is_not_part_of_the_relight_signature() {
    // Changing the sampler reuploads the texture without reshading.
    let ctx = egui::Context::default();
    let sig = signature(10.0);
    let mut state = ImageViewerState {
        relight: Some(relit_entry(&ctx, sig.clone())),
        ..ImageViewerState::default()
    };
    state.poll_relight(&ctx, egui::TextureOptions::NEAREST);
    assert_eq!(
        state.relight.as_ref().unwrap().options,
        egui::TextureOptions::NEAREST
    );
    assert!(state.invalidate_superseded_inflight(&sig));
    assert!(!state.relight_pending, "no new job was started");
}

fn relight_on() -> RelightSettings {
    RelightSettings {
        enabled: true,
        light: Light::default(),
    }
}

#[test]
fn relight_needs_albedo_and_normal_and_says_why() {
    let mut cache = ImageCache::new();
    let mut state = ImageViewerState::default();

    let off = state.request_relight(
        &mut cache,
        &RelightSettings::default(),
        &RelightSources::default(),
    );
    assert_eq!(off, RelightState::Disabled);

    // No PS layers at all: the hint names the command that creates them.
    let none = state.request_relight(&mut cache, &relight_on(), &RelightSources::default());
    assert!(
        matches!(&none, RelightState::NoAlbedo(r) if r.contains("fabsim ps")),
        "{none:?}"
    );

    // The displayed (diffuse) image is never a substitute for the albedo.
    let only_normal = RelightSources {
        normal: Some(PathBuf::from("/n.png")),
        run_has_ps: true,
        ..RelightSources::default()
    };
    let state_only_normal = state.request_relight(&mut cache, &relight_on(), &only_normal);
    assert!(matches!(state_only_normal, RelightState::NoAlbedo(_)));

    let only_albedo = RelightSources {
        albedo: Some((PathBuf::from("/a.png"), 1.0)),
        ..RelightSources::default()
    };
    assert_eq!(
        state.request_relight(&mut cache, &relight_on(), &only_albedo),
        RelightState::NoNormal
    );
}

#[test]
fn relight_waits_for_both_sources_then_reports_a_missing_one() {
    let dir = std::env::temp_dir().join(format!("fabsim_iv_relight_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let sources = RelightSources {
        albedo: Some((dir.join("albedo.png"), 1.5)),
        normal: Some(dir.join("normal.png")),
        run_has_ps: true,
    };
    let mut cache = ImageCache::new();
    let mut state = ImageViewerState::default();
    cache.request(sources.albedo.as_ref().unwrap().0.clone());
    cache.request(sources.normal.clone().unwrap());
    cache.flush_requests();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while cache.has_pending() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(2));
        cache.finish_pending();
    }
    // Cache missing-file errors rather than retrying on every repaint.
    let result = state.request_relight(&mut cache, &relight_on(), &sources);
    assert!(matches!(result, RelightState::SourceError(_)), "{result:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn compare_controls_stay_reachable_in_single_mode() {
    // Comparison controls must remain visible in Single mode.
    let roles = vec![
        ("diffuse".to_owned(), "d.png".to_owned()),
        ("ps_1".to_owned(), "p.png".to_owned()),
    ];
    let mut settings = ViewerSettings::default();
    settings.images.comparison = ComparisonMode::Single;
    let mut state = ImageViewerState::default();
    let model = |roles: &[(String, String)]| {
        // Give each simulated draw its own role list.
        (roles.to_vec(), "diffuse")
    };
    let (owned, primary) = model(&roles);
    let text = headless_frame(|ui| {
        controls_row(
            ui,
            &mut settings,
            &mut state,
            ControlsModel {
                roles: &owned,
                primary_role: primary,
                can_relight: false,
                run_has_ps: false,
            },
        );
    });
    for expected in [
        "Compare:",
        "Single",
        "Side-by-side",
        "Split / wipe",
        "Relight",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    assert!(!text.contains("2nd:"), "the 2nd-role picker needs a mode");
    assert!(!text.contains("Reset"), "Reset duplicated Fit and is gone");

    let one = vec![("diffuse".to_owned(), "d.png".to_owned())];
    let text = headless_frame(|ui| {
        controls_row(
            ui,
            &mut settings,
            &mut state,
            ControlsModel {
                roles: &one,
                primary_role: "diffuse",
                can_relight: false,
                run_has_ps: false,
            },
        );
    });
    assert!(!text.contains("Compare:"));
}

#[test]
fn the_relight_pad_appears_only_with_relight_on_and_sources() {
    let roles = vec![("diffuse".to_owned(), "d.png".to_owned())];
    let mut state = ImageViewerState::default();
    let mut render = |enabled: bool, can_relight: bool| {
        let mut settings = ViewerSettings::default();
        settings.images.relight.enabled = enabled;
        headless_frame(|ui| {
            controls_row(
                ui,
                &mut settings,
                &mut state,
                ControlsModel {
                    roles: &roles,
                    primary_role: "diffuse",
                    can_relight,
                    run_has_ps: can_relight,
                },
            );
        })
    };
    assert!(!render(false, true).contains("ambient"));
    assert!(
        render(true, true).contains("ambient"),
        "pad with ambient control"
    );
    assert!(
        !render(true, false).contains("ambient"),
        "no pad without sources"
    );
}

/// One-frame fixture: black diffuse, optional albedo (128, scale 2) and camera-facing normals.
fn write_project(tag: &str, with_ps: bool) -> (PathBuf, ProjectSlot) {
    let dir = std::env::temp_dir().join(format!("fabsim_iv_tab_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let run = dir.join("warp_loading");
    std::fs::create_dir_all(run.join("images")).unwrap();
    std::fs::write(
        run.join("session.json"),
        r#"{"sample_id":"s","test_id":"t","direction":"warp","branch":"loading","count":1}"#,
    )
    .unwrap();
    std::fs::write(
        run.join("frames.json"),
        r#"{"frames":[{"frame_id":0,"step_index":0,"displacement_mm":0.0,
            "image_diffuse":"img_0000_diffuse.png"}]}"#,
    )
    .unwrap();
    std::fs::write(run.join("force.json"), r#"{"forces":[]}"#).unwrap();
    let solid = |rgb: [u8; 3]| image::RgbImage::from_pixel(4, 4, image::Rgb(rgb));
    solid([0, 0, 0])
        .save(run.join("images/img_0000_diffuse.png"))
        .unwrap();
    if with_ps {
        let ps = run.join("ps");
        std::fs::create_dir_all(&ps).unwrap();
        solid([128, 128, 128])
            .save(ps.join("img_0000_albedo.png"))
            .unwrap();
        solid([128, 128, 255])
            .save(ps.join("img_0000_normal.png"))
            .unwrap();
        std::fs::write(
            ps.join("ps_manifest.json"),
            r#"{"format":"fabsim-ps-frames","version":1,"albedo_scale":2.0,
                "frames":[{"frame_id":0,"albedo":"img_0000_albedo.png",
                           "normal":"img_0000_normal.png"}]}"#,
        )
        .unwrap();
    }
    let project = fabric_data::Project::discover(&dir).unwrap();
    (dir.clone(), ProjectSlot::new("p".into(), project))
}

/// Run the whole tab for up to ~2 s until `done` holds.
fn drive_tab(
    slots: &[ProjectSlot],
    settings: &mut ViewerSettings,
    state: &mut ImageViewerState,
    mut done: impl FnMut(&ImageViewerState) -> bool,
) {
    let ctx = egui::Context::default();
    let mut cache = ImageCache::new();
    let mut selection = FrameSelection {
        selected: frame_ref_for(slots, 0, 0, 0),
        ..FrameSelection::default()
    };
    let mut tracking = TrackingSession::default();
    let mut playback = PlaybackState::default();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 900.0))),
            ..Default::default()
        };
        let _ = ctx.run_ui(input, |ui| {
            cache.poll(ui.ctx());
            let mut cx = ImageViewerCtx {
                slots,
                selection: &mut selection,
                settings,
                image_cache: &mut cache,
                tracking: &mut tracking,
                playback: &mut playback,
            };
            image_viewer_tab(ui, state, &mut cx);
        });
        if done(state) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("the tab did not reach the expected state in time");
}

#[test]
fn relight_shades_the_ps_albedo_even_while_the_black_diffuse_is_displayed() {
    let (dir, slot) = write_project("relight", true);
    let mut settings = ViewerSettings::default();
    assert_eq!(settings.images.preferred_role.as_deref(), Some("diffuse"));
    settings.images.relight = RelightSettings {
        enabled: true,
        light: Light {
            azimuth_deg: 0.0,
            elevation_deg: 90.0,
            intensity: 1.0,
            ambient: 0.0,
        },
    };
    let mut state = ImageViewerState::default();
    drive_tab(&[slot], &mut settings, &mut state, |s| {
        s.relight.is_some() && !s.relight_pending
    });
    let entry = state.relight.as_ref().unwrap();
    assert_eq!(entry.signature.albedo_scale, 2.0, "scale from the manifest");
    assert!(entry
        .signature
        .albedo_key
        .ends_with("ps/img_0000_albedo.png"));
    // The bright result must come from albedo; diffuse is black in this fixture.
    let p = entry.image.get_pixel(1, 1);
    assert!(p[0] >= 250 && p[1] >= 250 && p[2] >= 250, "got {p:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_run_without_ps_layers_is_not_relit_and_keeps_showing_the_photo() {
    let (dir, slot) = write_project("no_ps", false);
    let mut settings = ViewerSettings::default();
    settings.images.relight.enabled = true;
    let mut state = ImageViewerState::default();
    drive_tab(&[slot], &mut settings, &mut state, |s| {
        s.last_image.is_some()
    });
    assert!(state.relight.is_none() && !state.relight_pending);
    assert!(
        settings.images.relight.enabled,
        "the setting is not cleared"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_canvas_keeps_the_last_image_while_the_next_one_decodes() {
    let (dir, slot) = write_project("last", false);
    let mut settings = ViewerSettings::default();
    let mut state = ImageViewerState::default();
    drive_tab(&[slot], &mut settings, &mut state, |s| {
        s.last_image.is_some()
    });
    let kept = state.last_image.as_ref().unwrap();
    assert_eq!(kept.size, (4, 4));
    std::fs::remove_dir_all(&dir).ok();
}
