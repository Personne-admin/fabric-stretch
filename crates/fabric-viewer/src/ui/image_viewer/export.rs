//! Video export from the Image Viewer: the export row (progress, cancel,
//! status) and the per-frame inputs handed to `crate::frame_export`.

use std::path::Path;

use fabric_data::tracking::FrameId;
use fabric_data::Run;

use super::overlay::frame_render_inputs;
use super::ImageViewerState;
use crate::images::{frame_roles, resolve_role, PsLayers};
use crate::project::ProjectSlot;
use crate::settings::ViewerSettings;
use crate::tracked::TrackingSession;
use egui::RichText;

/// What an export will contain for the current stabilize / show-points switches.
pub(super) fn export_mode_label(enabled: bool, show_points: bool) -> &'static str {
    match (enabled, show_points) {
        (true, false) => "stabilized, no points",
        (true, true) => "stabilized, with tracked points",
        (false, true) => "registered, with tracked points",
        (false, false) => "registered, full overlay (displacement lines, point positions)",
    }
}

/// Export PNG frames and MP4 using the current tracking switches and preferred role.
/// Relighting and comparison layouts are excluded.
pub(super) fn export_row(
    ui: &mut egui::Ui,
    state: &mut ImageViewerState,
    tracking: &TrackingSession,
    fps: f32,
    settings: &ViewerSettings,
    slot: &ProjectSlot,
    run_index: usize,
) {
    if let Some(handle) = &state.export {
        if let Some(result) = handle.take_result() {
            state.export_status = Some(match result {
                Ok(report) => match (&report.video, &report.video_note) {
                    (Some(video), _) => format!(
                        "Exported {} frames to {} and {}",
                        report.frames,
                        report.frames_dir.display(),
                        video.display()
                    ),
                    (None, Some(note)) => format!(
                        "Exported {} frames to {} ({note})",
                        report.frames,
                        report.frames_dir.display()
                    ),
                    (None, None) => format!(
                        "Exported {} frames to {}",
                        report.frames,
                        report.frames_dir.display()
                    ),
                },
                Err(error) => format!("Export failed: {error}"),
            });
            state.export = None;
        }
    }
    ui.horizontal_wrapped(|ui| {
        if let Some(handle) = &state.export {
            let done = handle.progress();
            ui.add(
                egui::ProgressBar::new(done as f32 / handle.total.max(1) as f32)
                    .desired_width(180.0)
                    .text(format!("Exporting {done}/{}", handle.total)),
            );
            if ui.button("Cancel").clicked() {
                handle.cancel();
            }
            ui.ctx().request_repaint();
        } else {
            let mode = export_mode_label(tracking.stabilize.enabled, tracking.stabilize.show_points);
            if ui
                .button("Export video...")
                .on_hover_text(format!(
                    "Write every frame of this run as PNG plus an MP4 (needs ffmpeg), drawn as on screen: {mode}. Relight and comparison are not exported."
                ))
                .clicked()
            {
                start_export(state, tracking, fps, settings, slot, run_index);
            }
            ui.label(RichText::new(mode).small().weak());
        }
        if let Some(status) = &state.export_status {
            ui.label(RichText::new(status).small());
        }
    });
}

/// Every frame of `run` with the image and overlay the canvas shows for the current switches.
/// Frames without an image are left out; selection and hover highlights are never exported.
pub(super) fn export_frames(
    tracking: &TrackingSession,
    settings: &ViewerSettings,
    run: &Run,
    ps: &PsLayers,
    run_key: &Path,
) -> Vec<crate::frame_export::ExportFrame> {
    let mut frames = Vec::new();
    for frame in &run.frames.frames {
        let roles = frame_roles(frame, ps);
        let Some((_, filename)) = resolve_role(
            roles.iter().map(|(r, f)| (r.as_str(), f.as_str())),
            settings.images.preferred_role.as_deref(),
        ) else {
            continue;
        };
        let (transform, mut overlay) =
            frame_render_inputs(tracking, run_key, FrameId(frame.frame_id));
        if let Some(overlay) = overlay.as_mut() {
            overlay.selected_track = None;
        }
        frames.push(crate::frame_export::ExportFrame {
            image_path: run.images_dir().join(filename),
            transform,
            overlay,
        });
    }
    frames
}

fn start_export(
    state: &mut ImageViewerState,
    tracking: &TrackingSession,
    fps: f32,
    settings: &ViewerSettings,
    slot: &ProjectSlot,
    run_index: usize,
) {
    let run = &slot.project.runs[run_index];
    let name = format!(
        "{}_{}.mp4",
        slot.name.replace(' ', "_"),
        run.dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "run".into())
    );
    let Some(video) = rfd::FileDialog::new()
        .add_filter("MP4 video", &["mp4"])
        .set_file_name(&name)
        .save_file()
    else {
        return;
    };
    let frames = export_frames(
        tracking,
        settings,
        run,
        &slot.entries[run_index].ps,
        &slot.run_keys[run_index],
    );
    if frames.is_empty() {
        state.export_status = Some("Export failed: this run has no frame images".into());
        return;
    }
    let frames_dir = video.with_file_name(format!(
        "{}_frames",
        video
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "export".into())
    ));
    state.export_status = None;
    state.export = Some(crate::frame_export::spawn(
        frames,
        crate::frame_export::ExportOptions {
            fps,
            frames_dir,
            video: Some(video),
        },
    ));
}
