//! Run/frame inspector and validation details.

use std::path::PathBuf;

use egui::{Color32, RichText};
use fabric_data::Run;

use super::style::{severity_badge, ERROR, WARN};
use crate::images::{frame_roles, PsLayers};
use crate::project::{frame_ref_for, resolve_frame_ref, ProjectSlot};
use crate::selection::{FrameRef, FrameSelection};

/// Inspector tab state: which run is being inspected, independent of the
/// frame selection so zero-frame runs can still be surfaced.
#[derive(Debug, Default)]
pub struct InspectorState {
    /// Inspected run keyed by project and run paths. Zero-frame runs are reachable via the selector.
    inspected_run: Option<(PathBuf, PathBuf)>,
    /// The last frame the inspector followed (the pinned selection, falling
    /// back to the hovered frame only while nothing is pinned).
    last_followed: Option<FrameRef>,
}

impl InspectorState {
    /// Follow pinned selection, not transient hover, so the chosen run stays stable while scrubbing.
    pub fn follow(&mut self, followed: Option<&FrameRef>, slots: &[ProjectSlot]) {
        let followed = followed.cloned();
        if followed == self.last_followed {
            return;
        }
        if let Some(fr) = &followed {
            if resolve_frame_ref(slots, fr).is_some() {
                self.inspected_run = Some((fr.project_root.clone(), fr.run_dir.clone()));
            }
        }
        self.last_followed = followed;
    }

    /// The inspected run as `(project, run)` indices, when it is still open
    /// and enabled.
    pub fn resolve(&self, slots: &[ProjectSlot]) -> Option<(usize, usize)> {
        let (root, run_dir) = self.inspected_run.as_ref()?;
        let (pidx, slot) = slots
            .iter()
            .enumerate()
            .find(|(_, s)| s.enabled && s.root_key == *root)?;
        let ridx = slot.run_keys.iter().position(|k| k == run_dir)?;
        Some((pidx, ridx))
    }

    fn inspect(&mut self, slots: &[ProjectSlot], pidx: usize, ridx: usize) {
        self.inspected_run = Some((
            slots[pidx].root_key.clone(),
            slots[pidx].run_keys[ridx].clone(),
        ));
    }
}

pub fn inspector_tab(
    ui: &mut egui::Ui,
    slots: &[ProjectSlot],
    selection: &mut FrameSelection,
    state: &mut InspectorState,
) {
    if slots.is_empty() {
        ui.label("No projects open.");
        return;
    }

    run_selector(ui, slots, selection, state);
    let followed = selection
        .selected
        .clone()
        .or_else(|| selection.hovered.clone());
    state.follow(followed.as_ref(), slots);

    let Some((pidx, ridx)) = state.resolve(slots) else {
        ui.label("Select a run to see its details.");
        return;
    };
    let Some(slot) = slots.get(pidx) else {
        return;
    };
    let Some(entry) = slot.entries.get(ridx) else {
        return;
    };
    let Some(run) = slot.project.runs.get(ridx) else {
        return;
    };

    let n_frames = run.frames.frames.len();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::CollapsingHeader::new("Frame")
                .default_open(true)
                .show(ui, |ui| {
                    if n_frames == 0 {
                        ui.colored_label(WARN, "run has no frames");
                    } else {
                        let active_fidx = selection
                            .active()
                            .and_then(|fr| resolve_frame_ref(slots, fr))
                            .filter(|&(p, r, _)| p == pidx && r == ridx)
                            .map(|(_, _, f)| f)
                            .unwrap_or(0)
                            .min(n_frames - 1);
                        let mut fidx = active_fidx;
                        let slider = egui::Slider::new(&mut fidx, 0..=n_frames - 1)
                            .text("frame")
                            .show_value(true);
                        if ui.add(slider).changed() {
                            if let Some(fr) = frame_ref_for(slots, pidx, ridx, fidx) {
                                selection.selected = Some(fr);
                                selection.hovered = None;
                                selection.pinned = false;
                            }
                        }
                        let frame = &run.frames.frames[fidx];
                        frame_grid(ui, run, frame);
                    }
                });

            egui::CollapsingHeader::new("Run").show(ui, |ui| {
                ui.label(RichText::new(&entry.curve.test_id).small().weak());
                ui.add_space(4.0);
                session_grid(ui, run);
            });

            egui::CollapsingHeader::new("Images").show(ui, |ui| {
                if n_frames == 0 {
                    ui.label("run has no frames to reference images from");
                } else {
                    let fidx = selection
                        .active()
                        .and_then(|fr| resolve_frame_ref(slots, fr))
                        .filter(|&(p, r, _)| p == pidx && r == ridx)
                        .map(|(_, _, f)| f)
                        .unwrap_or(0)
                        .min(n_frames - 1);
                    images_grid(ui, run, &entry.ps, &run.frames.frames[fidx]);
                }
            });

            egui::CollapsingHeader::new("Validation").show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Severity:");
                    ui.label(severity_badge(entry.validation.severity()));
                });
                let issues = entry.validation.issues();
                if issues.is_empty() {
                    ui.label("no issues found");
                }
                for issue in &issues {
                    ui.colored_label(WARN, format!("- {issue}"));
                }
                ui.add_space(4.0);
                for line in entry.validation.to_string().lines() {
                    ui.label(RichText::new(line).small());
                }
            });
        });
}

fn run_selector(
    ui: &mut egui::Ui,
    slots: &[ProjectSlot],
    selection: &mut FrameSelection,
    state: &mut InspectorState,
) {
    let selected_hit = selection
        .active()
        .and_then(|fr| resolve_frame_ref(slots, fr));
    let inspected = state.resolve(slots);
    let current = inspected
        .and_then(|(p, r)| slots.get(p)?.entries.get(r))
        .map_or_else(
            || {
                selected_hit
                    .and_then(|(p, r, _)| slots.get(p)?.entries.get(r))
                    .map_or_else(|| "select a run".to_owned(), |e| e.legend_name.clone())
            },
            |e| e.legend_name.clone(),
        );

    let mut pick: Option<(usize, usize)> = None;
    egui::ComboBox::from_id_salt("run_selector")
        .width(ui.available_width().min(320.0))
        .selected_text(current)
        .show_ui(ui, |ui| {
            for (pidx, slot) in slots.iter().enumerate() {
                if !slot.enabled {
                    continue;
                }
                for (ridx, entry) in slot.entries.iter().enumerate() {
                    let is_sel = inspected == Some((pidx, ridx));
                    ui.horizontal(|ui| {
                        if ui.selectable_label(is_sel, &entry.legend_name).clicked() {
                            pick = Some((pidx, ridx));
                        }
                        ui.label(severity_badge(entry.validation.severity()));
                    });
                }
            }
        });
    if let Some((p, r)) = pick {
        state.inspect(slots, p, r);
        if let Some(fr) = frame_ref_for(slots, p, r, 0) {
            selection.selected = Some(fr);
            selection.hovered = None;
            selection.pinned = false;
        }
    }

    let load_errors: Vec<String> = slots
        .iter()
        .flat_map(|s| {
            s.project
                .load_errors
                .iter()
                .map(move |(dir, err)| format!("{}: {}: {err}", s.name, dir.display()))
        })
        .collect();
    for msg in load_errors {
        ui.colored_label(ERROR, msg);
    }
}

fn frame_grid(ui: &mut egui::Ui, run: &Run, frame: &fabric_data::Frame) {
    egui::Grid::new("frame_grid").num_columns(2).show(ui, |ui| {
        let mut row = |k: &str, v: String| {
            ui.label(RichText::new(k).weak());
            ui.label(v);
            ui.end_row();
        };
        row("frame_id", frame.frame_id.to_string());
        row("displacement_mm", format!("{}", frame.displacement_mm));
        row(
            "line_load_N_per_m",
            frame
                .line_load_n_per_m
                .map_or_else(|| "-".into(), |v| format!("{v:.3}")),
        );
        row("relaxation hold", frame.is_relaxation_hold.to_string());

        match run.force_for_frame(frame) {
            Some(rec) => {
                row(
                    "force record",
                    format!("index {} ({})", rec.index, rec.status),
                );
                row(
                    "force_n",
                    rec.force_n
                        .map_or_else(|| "null".into(), |v| format!("{v}")),
                );
                if let Some(raw) = &rec.raw {
                    row("raw", raw.clone());
                }
            }
            None => {
                ui.label(RichText::new("force record").weak());
                ui.colored_label(ERROR, "unresolved link");
                ui.end_row();
            }
        }
    });
}

fn images_grid(ui: &mut egui::Ui, run: &Run, ps: &PsLayers, frame: &fabric_data::Frame) {
    let images_dir = run.images_dir();
    let roles = frame_roles(frame, ps);
    if roles.is_empty() {
        ui.label(RichText::new("frame has no image roles").weak());
        return;
    }
    for (role, filename) in &roles {
        let exists = images_dir.join(filename).is_file();
        let (mark, color) = if exists {
            ("ok", Color32::from_rgb(96, 189, 115))
        } else {
            ("MISSING", ERROR)
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(role).weak());
            ui.label(RichText::new(filename).small());
            ui.colored_label(color, mark);
        });
    }
}

fn session_grid(ui: &mut egui::Ui, run: &Run) {
    let s = &run.session;
    ui.label(RichText::new("dir").weak());
    ui.add(
        egui::Label::new(RichText::new(wrap_path(&run.dir)).small())
            .wrap()
            .selectable(true),
    );
    ui.add_space(2.0);

    egui::Grid::new("session_grid")
        .num_columns(2)
        .show(ui, |ui| {
            let mut row = |k: &str, v: String| {
                ui.label(RichText::new(k).weak());
                ui.label(v);
                ui.end_row();
            };
            if let Some(ts) = &s.timestamp {
                row("timestamp", ts.clone());
            }
            row("count", s.count.to_string());
            if let Some(v) = s.frames_written {
                row("frames_written", v.to_string());
            }
            if let Some(v) = s.step_size {
                row("step_size", v.to_string());
            }
            if let Some(v) = s.motor_step_um {
                row("motor_step_um", format!("{v}"));
            }
            if let Some(v) = s.clamp_width_m {
                row("clamp_width_m", format!("{v}"));
            }
            if let Some(v) = s.max_stretch_mm {
                row("max_stretch_mm", format!("{v}"));
            }
            if let Some(v) = s.max_force_n {
                row("max_force_n", format!("{v}"));
            }
            if let Some(v) = s.relax_enabled {
                row("relax_enabled", v.to_string());
            }
            if let Some(reason) = &s.abort_reason {
                ui.label(RichText::new("abort_reason").weak());
                ui.colored_label(ERROR, reason);
                ui.end_row();
            }
        });
}

fn wrap_path(path: &std::path::Path) -> String {
    path.to_string_lossy()
        .chars()
        .flat_map(|c| {
            if c == '/' || c == '\\' {
                vec![c, '\u{200b}']
            } else {
                vec![c]
            }
        })
        .collect()
}
