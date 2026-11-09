//! Tracking import, reference and stabilization controls. Legacy controls are opt-in.

use std::path::{Path, PathBuf};

use egui::RichText;
use fabric_data::tracking::FrameId;
use fabric_data::Run;

use super::advanced::advanced_tracking_section;
use crate::tracked::TrackingSession;
use crate::ui::style;

pub(super) fn tracking_panel(
    ui: &mut egui::Ui,
    tracking: &mut TrackingSession,
    run_key: &Path,
    run: &Run,
    frame_id: FrameId,
    advanced: bool,
) {
    egui::CollapsingHeader::new(RichText::new("Tracking").strong())
        .default_open(true)
        .show(ui, |ui| {
            simple_tracking_section(ui, tracking, run_key, frame_id);
            // Keep legacy controls collapsed even after enabling the workbench.
            if advanced {
                egui::CollapsingHeader::new("Advanced tracking & registration")
                    .default_open(false)
                    .show(ui, |ui| {
                        advanced_tracking_section(ui, tracking, run_key, run, frame_id);
                    });
            }
        });
}

/// The everyday tracking UI: what is loaded, which frame anchors the replay,
/// the two stabilization switches and load/unload. No registration knobs.
fn simple_tracking_section(
    ui: &mut egui::Ui,
    tracking: &mut TrackingSession,
    run_key: &Path,
    frame_id: FrameId,
) {
    // Copy read-only state before controls mutate the session.
    let has_sequence = tracking
        .run_state(run_key)
        .and_then(|r| r.sequence.as_ref())
        .is_some();
    let error = tracking.run_state(run_key).and_then(|r| r.error.clone());
    let import_path = tracking
        .run_state(run_key)
        .and_then(|r| r.import_path.clone());
    let import_warnings = tracking
        .run_state(run_key)
        .map(|r| r.import_warnings.clone())
        .unwrap_or_default();
    let alignment_warning = tracking
        .run_state(run_key)
        .and_then(|r| r.alignment_warning.clone());
    let frame_ids: Vec<FrameId> = tracking
        .run_state(run_key)
        .map(|r| tracking.frame_ids(r))
        .unwrap_or_default();
    let track_count = tracking
        .run_state(run_key)
        .and_then(|r| r.sequence.as_ref())
        .map_or(0, |s| s.track_ids().len());
    let current_ref = tracking
        .run_state(run_key)
        .and_then(|r| r.state.reference_frame);

    let status = if has_sequence {
        format!(
            "Tracking: loaded · {} points · {} frames",
            track_count,
            frame_ids.len()
        )
    } else if import_path.is_some() {
        "Tracking: file loaded but unusable (no frames)".to_owned()
    } else {
        "Tracking: not loaded for this run".to_owned()
    };
    let status_label = ui.label(RichText::new(status).small().weak());
    if let Some(path) = &import_path {
        status_label.on_hover_text(format!("source: {}", path.display()));
    }
    if let Some(err) = &error {
        ui.colored_label(style::ERROR, err);
    }
    if let Some(warning) = &alignment_warning {
        ui.colored_label(style::WARN, warning).on_hover_text(
            "Tracking frame ids do not line up with this run's \
                 frames.json. Fix the export or choose a matching file \
                 before trusting any per-frame analysis.",
        );
    }
    if !import_warnings.is_empty() {
        ui.label(
            RichText::new(format!("{} import warning(s)", import_warnings.len()))
                .small()
                .weak(),
        )
        .on_hover_text(import_warnings.join("\n"));
    }

    // Keep ambiguous discovery candidates until the user chooses one.
    let (auto_candidates, mut choice) = tracking
        .run_state(run_key)
        .map(|r| (r.auto_candidates.clone(), r.candidate_choice.clone()))
        .unwrap_or_default();
    let mut load_chosen = false;
    if !auto_candidates.is_empty() {
        let file_name = |p: &Path| {
            p.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(format!(
                    "{} tracking files found next to this run - choose one:",
                    auto_candidates.len()
                ))
                .small(),
            );
            egui::ComboBox::from_id_salt("tracking_auto_candidate")
                .selected_text(choice.as_deref().map(file_name).unwrap_or_default())
                .width(180.0)
                .show_ui(ui, |ui| {
                    for candidate in &auto_candidates {
                        if ui
                            .selectable_label(
                                choice.as_ref() == Some(candidate),
                                file_name(candidate),
                            )
                            .clicked()
                        {
                            choice = Some(candidate.clone());
                        }
                    }
                });
            load_chosen = ui
                .add_enabled(choice.is_some(), egui::Button::new("Load chosen"))
                .on_hover_text("import the selected file; the others are ignored")
                .clicked();
        });
        if let Some(run) = tracking.run_state_mut(run_key) {
            run.candidate_choice = choice.clone();
        }
    }
    if let (true, Some(path)) = (load_chosen, choice) {
        let _ = tracking.import_file(run_key, &path);
    }

    if has_sequence {
        ui.horizontal_wrapped(|ui| {
            ui.label("Reference frame:");
            egui::ComboBox::from_id_salt("tracking_ref_frame")
                .selected_text(current_ref.map_or_else(|| "—".to_owned(), |f| f.to_string()))
                .width(70.0)
                .show_ui(ui, |ui| {
                    for fid in &frame_ids {
                        if ui
                            .selectable_label(Some(*fid) == current_ref, fid.to_string())
                            .clicked()
                        {
                            if let Some(r) = tracking.run_state_mut(run_key) {
                                r.state.reference_frame = Some(*fid);
                            }
                        }
                    }
                });
            if ui
                .button("Use current")
                .on_hover_text("anchor the replay on the currently displayed frame")
                .clicked()
            {
                if let Some(r) = tracking.run_state_mut(run_key) {
                    r.state.reference_frame = Some(frame_id);
                }
            }
        });

        stabilize_switches(ui, tracking, run_key, frame_id);
    }

    let mut load_path: Option<PathBuf> = None;
    ui.horizontal_wrapped(|ui| {
        if ui.button("Load tracking...").clicked() {
            if let Some(file) = rfd::FileDialog::new()
                .add_filter("tracking", &fabric_data::tracking::TRACKING_EXTENSIONS)
                .pick_file()
            {
                load_path = Some(file);
            }
        }
        if (has_sequence || import_path.is_some()) && ui.button("Unload").clicked() {
            tracking.clear_run(run_key);
        }
    });
    if let Some(path) = load_path {
        let _ = tracking.import_file(run_key, &path);
    }
}

fn stabilize_switches(
    ui: &mut egui::Ui,
    tracking: &mut TrackingSession,
    run_key: &Path,
    frame_id: FrameId,
) {
    ui.checkbox(
        &mut tracking.stabilize.enabled,
        "Stabilize using CoWTracker",
    );
    ui.checkbox(&mut tracking.stabilize.show_points, "Show tracked points");
    if !tracking.stabilize.enabled {
        return;
    }

    let stabilization = tracking
        .run_state(run_key)
        .and_then(|r| tracking.stabilization_for(r, frame_id));
    match &stabilization {
        Some(stab) if stab.is_estimated() => {
            let (dx, dy) = stab.offset();
            ui.label(
                RichText::new(format!(
                    "{} anchors · offset ({dx:+.1}, {dy:+.1}) px",
                    stab.anchor_pairs
                ))
                .small()
                .weak(),
            );
        }
        _ => {
            ui.label(
                RichText::new("no anchor tracks on this frame")
                    .small()
                    .color(style::WARN),
            );
        }
    }
}

/// Inspector readout for the clicked/selected tracked point. Only shows mm
/// when a genuine pixel/mm calibration is present in the session metadata.
pub(super) fn selected_point_row(
    ui: &mut egui::Ui,
    tracking: &TrackingSession,
    run: &Run,
    run_key: &Path,
    frame_id: FrameId,
) {
    let Some(track_id) = tracking.selected_track.clone() else {
        return;
    };
    let Some(run_state) = tracking.run_state(run_key) else {
        return;
    };
    let Some(seq) = run_state.sequence.as_ref() else {
        return;
    };
    let Some(current) = seq.point(&frame_id, &track_id) else {
        return;
    };
    let reference = tracking
        .reference_for(run_state)
        .and_then(|rf| rf.get(&track_id).copied());
    let tracked_frames = tracking.trajectory(run_state, &track_id).len();

    ui.separator();
    ui.label(RichText::new(format!("selected point: {track_id}")).strong());
    ui.horizontal_wrapped(|ui| {
        match reference {
            Some(rp) => {
                let dx = current.x - rp.x;
                let dy = current.y - rp.y;
                let mag = dx.hypot(dy);
                ui.label(
                    RichText::new(format!(
                        "ref ({:.1}, {:.1})  current ({:.1}, {:.1})",
                        rp.x, rp.y, current.x, current.y
                    ))
                    .small(),
                );
                ui.label(
                    RichText::new(format!("dx {dx:.1} px · dy {dy:.1} px · |d| {mag:.1} px"))
                        .small(),
                );
                // Use only measured pixel/mm calibration.
                if let Some(px_per_mm) = calibration_px_per_mm(run) {
                    ui.label(
                        RichText::new(format!("≈ {:.2} mm", mag / px_per_mm))
                            .small()
                            .weak(),
                    );
                }
            }
            None => {
                ui.label(
                    RichText::new(format!(
                        "current ({:.1}, {:.1}) · no reference position",
                        current.x, current.y
                    ))
                    .small(),
                );
            }
        }
        ui.label(
            RichText::new(format!("· {tracked_frames} frames tracked"))
                .small()
                .weak(),
        );
    });
}

/// Pixel/mm calibration from session metadata, if recorded.
pub(super) fn calibration_px_per_mm(run: &Run) -> Option<f64> {
    for key in ["pixels_per_mm", "px_per_mm", "calibration_px_per_mm"] {
        if let Some(v) = run.session.extra.get(key) {
            let f = v.as_f64().or_else(|| v.as_i64().map(|i| i as f64));
            if let Some(f) = f {
                if f.is_finite() && f > 0.0 {
                    return Some(f);
                }
            }
        }
    }
    None
}
