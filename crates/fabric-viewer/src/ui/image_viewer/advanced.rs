//! Opt-in legacy registration tools and CSV export. Persisted registration choices still use them.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use egui::RichText;
use fabric_data::tracking::registration::RegistrationMode;
use fabric_data::tracking::{FrameId, TrackId};
use fabric_data::Run;

use super::tracking_panel::selected_point_row;
use crate::tracked::{CanvasMode, TrackScope, TrackingSession};
use crate::ui::style;

pub(super) fn advanced_tracking_section(
    ui: &mut egui::Ui,
    tracking: &mut TrackingSession,
    run_key: &Path,
    run: &Run,
    frame_id: FrameId,
) {
    let mut note: Option<String> = None;
    let has_sequence = tracking
        .run_state(run_key)
        .and_then(|r| r.sequence.as_ref())
        .is_some();
    let import_path = tracking
        .run_state(run_key)
        .and_then(|r| r.import_path.clone());
    let frame_ids: Vec<FrameId> = tracking
        .run_state(run_key)
        .map(|r| tracking.frame_ids(r))
        .unwrap_or_default();
    let ref_count = tracking
        .run_state(run_key)
        .map_or(0, |r| r.state.reference_tracks.len());
    let mode = tracking
        .run_state(run_key)
        .map_or(RegistrationMode::Off, |r| r.state.mode);
    let conf = tracking
        .run_state(run_key)
        .map_or(0.0, |r| r.state.min_confidence);
    let axis_angle = tracking
        .run_state(run_key)
        .map_or(0.0, |r| r.state.loading_axis_angle_deg());

    // Keep path edits in a buffer until Enter or Load; rebuilding from the import path loses typing.
    let mut path_text = tracking
        .run_state(run_key)
        .and_then(|r| r.path_edit.clone())
        .unwrap_or_else(|| {
            import_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
    let mut load_path: Option<PathBuf> = None;
    let mut edited = false;
    ui.horizontal_wrapped(|ui| {
        ui.label("path:");
        let edit = ui.add(
            egui::TextEdit::singleline(&mut path_text)
                .desired_width(220.0)
                .hint_text("path to tracking.json / tracking.csv / tracks.tsv"),
        );
        edited = edit.changed();
        let trimmed = path_text.trim().to_owned();
        let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if !trimmed.is_empty() && (ui.button("Load").clicked() || enter) {
            load_path = Some(PathBuf::from(trimmed));
        }
    });
    if edited {
        if let Some(run) = tracking.run_state_mut(run_key) {
            run.path_edit = Some(path_text);
        }
    }
    if let Some(path) = load_path {
        let _ = tracking.import_file(run_key, &path);
    }

    if !has_sequence {
        ui.label(
            RichText::new("load a tracking file to use the registration controls")
                .small()
                .weak(),
        );
        return;
    }
    if frame_ids.is_empty() {
        ui.label(RichText::new("sequence has no frames").small().weak());
        return;
    }

    ui.horizontal_wrapped(|ui| {
        if let Some(path) = &import_path {
            if let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) {
                if ui.button(format!("Load \"{name}\" for all runs")).clicked() {
                    let (loaded, _) = tracking.load_similar_for_all_runs(path);
                    note = Some(format!("loaded for {loaded} more run(s)"));
                }
            }
        }
        if ui.button("Export tracks (CSV)...").clicked() {
            if let Some(target) = rfd::FileDialog::new()
                .add_filter("CSV", &["csv"])
                .set_file_name("tracks_registered.csv")
                .save_file()
            {
                match tracking.export_registered_csv(run_key, run, &target) {
                    Ok(()) => {
                        note = Some(format!(
                            "exported registered tracks to {}",
                            target.display()
                        ));
                    }
                    Err(err) => {
                        note = Some(err);
                    }
                }
            }
        }
    });

    ui.separator();

    ui.horizontal_wrapped(|ui| {
        ui.label("reference tracks:");
        if ui.button("Select top row").clicked() {
            tracking.select_top_row(run_key);
        }
        if ui.button("Use selected").clicked() {
            let tracks: BTreeSet<TrackId> = if !tracking.multi_select.is_empty() {
                tracking.multi_select.clone()
            } else if let Some(t) = &tracking.selected_track {
                BTreeSet::from([t.clone()])
            } else {
                BTreeSet::new()
            };
            tracking.set_reference_tracks(run_key, tracks);
        }
        if ui
            .button("Clear reference")
            .on_hover_text("use every valid track for registration")
            .clicked()
        {
            tracking.set_reference_tracks(run_key, BTreeSet::new());
        }
        if ui
            .button("Reset reference frame")
            .on_hover_text("reference frame = first tracking frame")
            .clicked()
        {
            if let Some(first) = frame_ids.first() {
                if let Some(r) = tracking.run_state_mut(run_key) {
                    r.state.reference_frame = Some(*first);
                }
            }
        }
        ui.label(
            RichText::new(if ref_count == 0 {
                "all valid".to_owned()
            } else {
                format!("{ref_count} reference track(s)")
            })
            .small()
            .weak(),
        );
    });

    ui.horizontal_wrapped(|ui| {
        ui.label("mode:");
        egui::ComboBox::from_id_salt("tracking_mode")
            .selected_text(mode.label())
            .width(180.0)
            .show_ui(ui, |ui| {
                for m in RegistrationMode::ALL {
                    if ui.selectable_label(m == mode, m.label()).clicked() {
                        if let Some(r) = tracking.run_state_mut(run_key) {
                            r.state.mode = m;
                        }
                    }
                }
            });

        let mut conf_edit = conf;
        if ui
            .add(egui::Slider::new(&mut conf_edit, 0.0..=1.0).text("min confidence"))
            .changed()
        {
            if let Some(r) = tracking.run_state_mut(run_key) {
                r.state.min_confidence = conf_edit;
            }
        }
    });

    if matches!(
        mode,
        RegistrationMode::LoadingAxisOnly | RegistrationMode::CrossAxisOnly
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.label("loading axis:");
            let mut angle = axis_angle;
            if ui
                .add(egui::Slider::new(&mut angle, 0.0..=360.0).text("angle [°]"))
                .changed()
            {
                if let Some(r) = tracking.run_state_mut(run_key) {
                    r.state.set_loading_axis_angle_deg(angle);
                }
            }
            let (ax, ay) = tracking
                .run_state(run_key)
                .map_or((1.0, 0.0), |r| r.state.loading_axis());
            ui.label(RichText::new(format!("({ax:.2}, {ay:.2})")).small().weak());
        });
    }

    ui.horizontal_wrapped(|ui| {
        ui.label("overlay:");
        ui.radio_value(
            &mut tracking.overlay.scope,
            TrackScope::AllValid,
            "All valid",
        );
        ui.radio_value(
            &mut tracking.overlay.scope,
            TrackScope::ReferenceOnly,
            "Reference only",
        );
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("canvas:");
        for mode in CanvasMode::ALL {
            ui.radio_value(&mut tracking.overlay.canvas_mode, mode, mode.label());
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut tracking.overlay.show_points, "points");
        ui.checkbox(&mut tracking.overlay.show_ids, "IDs");
        ui.checkbox(&mut tracking.overlay.show_trajectories, "trajectories");
        ui.checkbox(
            &mut tracking.overlay.show_displacement_vectors,
            "displacement",
        );
        ui.add(egui::Slider::new(&mut tracking.overlay.history_length, 1..=60).text("history"));
    });
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("diagnostics:").small().weak());
        ui.checkbox(&mut tracking.overlay.show_raw_points, "raw points");
        ui.checkbox(&mut tracking.overlay.show_transformed_points, "transformed");
        ui.checkbox(
            &mut tracking.overlay.show_reference_ghost,
            "reference ghost",
        );
        ui.checkbox(&mut tracking.overlay.show_displacement_lines, "disp. lines");
        ui.checkbox(
            &mut tracking.overlay.show_reference_locations,
            "ref. locations",
        );
    });
    ui.label(
        RichText::new("the overlay switches above apply while \"Show tracked points\" is off")
            .small()
            .weak(),
    );

    let result = tracking
        .run_state(run_key)
        .and_then(|r| tracking.result_for(r, frame_id));
    if let Some(result) = result {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(result.mode_used.label()).small().weak());
            ui.label(
                RichText::new(format!(
                    "valid {}/{}",
                    result.valid_count, result.reference_count
                ))
                .small(),
            );
            if let Some(rms) = result.rms_residual_px {
                ui.label(RichText::new(format!("RMS {rms:.2} px")).small());
            }
            if let Some(fb) = &result.fallback {
                ui.colored_label(style::WARN, fb);
            }
        });
        if !result.warnings.is_empty() {
            ui.label(
                RichText::new(format!("{} warning(s)", result.warnings.len()))
                    .small()
                    .weak(),
            )
            .on_hover_text(result.warnings.join("\n"));
        }
    } else if !mode.is_off() {
        ui.label(
            RichText::new("no registration for this frame (missing reference or points)")
                .small()
                .weak(),
        );
    }

    if let Some(note) = note {
        ui.label(RichText::new(note).small().weak());
    }

    selected_point_row(ui, tracking, run, run_key, frame_id);
}
