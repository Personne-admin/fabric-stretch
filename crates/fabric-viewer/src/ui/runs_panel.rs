//! Run visibility, filters and solo controls, sharing flags with Curves.

use std::collections::HashMap;

use egui::{Color32, RichText};
use fabric_data::{Branch, Direction};

use super::style::direction_color;
use crate::project::{ProjectSlot, RunEntry};

/// Which runs the Runs tab lists (a view filter; it does not change plot
/// visibility).
#[derive(Debug, Clone, Copy)]
pub struct RunsFilter {
    pub warp: bool,
    pub weft: bool,
    pub bias: bool,
    pub other_dir: bool,
    pub loading: bool,
    pub release: bool,
    pub other_branch: bool,
}

impl Default for RunsFilter {
    fn default() -> Self {
        RunsFilter {
            warp: true,
            weft: true,
            bias: true,
            other_dir: true,
            loading: true,
            release: true,
            other_branch: true,
        }
    }
}

pub fn runs_tab(ui: &mut egui::Ui, slots: &mut [ProjectSlot], filter: &mut RunsFilter) {
    if slots.is_empty() || slots.iter().all(|s| !s.enabled) {
        ui.label("No enabled projects.");
        return;
    }

    ui.horizontal_wrapped(|ui| {
        if ui.small_button("all").clicked() {
            for slot in slots.iter_mut().filter(|s| s.enabled) {
                for e in &mut slot.entries {
                    e.visible = true;
                }
            }
        }
        if ui.small_button("none").clicked() {
            for slot in slots.iter_mut().filter(|s| s.enabled) {
                for e in &mut slot.entries {
                    e.visible = false;
                }
            }
        }
        ui.separator();
        ui.label(RichText::new("list:").weak().small());
        let chip = |ui: &mut egui::Ui, label: &str, on: &mut bool, color: Option<Color32>| {
            let mut text = RichText::new(label);
            if let Some(c) = color {
                text = text.color(c);
            }
            if ui.selectable_label(*on, text).clicked() {
                *on = !*on;
            }
        };
        chip(
            ui,
            "warp",
            &mut filter.warp,
            Some(direction_color(&Direction::Warp)),
        );
        chip(
            ui,
            "weft",
            &mut filter.weft,
            Some(direction_color(&Direction::Weft)),
        );
        chip(
            ui,
            "bias",
            &mut filter.bias,
            Some(direction_color(&Direction::Bias)),
        );
        chip(ui, "other", &mut filter.other_dir, None);
        chip(ui, "loading", &mut filter.loading, None);
        chip(ui, "release", &mut filter.release, None);
        chip(ui, "branch?", &mut filter.other_branch, None);
    });
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let enabled: Vec<usize> = slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.enabled)
                .map(|(i, _)| i)
                .collect();
            for slot_idx in enabled {
                project_section(ui, slots, slot_idx, filter);
            }
        });
}

fn project_section(
    ui: &mut egui::Ui,
    slots: &mut [ProjectSlot],
    slot_idx: usize,
    filter: &RunsFilter,
) {
    // Copy header data before the closure mutates slots.
    let name = slots[slot_idx].name.clone();
    let visible_entries: Vec<usize> = slots[slot_idx]
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| filter.matches(e))
        .map(|(i, _)| i)
        .collect();
    if visible_entries.is_empty() {
        return;
    }
    let any_visible = slots[slot_idx].entries.iter().any(|e| e.visible);
    let visible_count = slots[slot_idx].entries.iter().filter(|e| e.visible).count();
    let entry_count = slots[slot_idx].entries.len();

    // Use a stable project ID: the label changes when filtering the run count.
    egui::CollapsingHeader::new(format!("{name} ({})", visible_entries.len()))
        .id_salt(("runs_project", &slots[slot_idx].root_key))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .selectable_label(any_visible, RichText::new(&name).strong())
                    .on_hover_text("Toggle all runs of this project")
                    .clicked()
                {
                    for e in &mut slots[slot_idx].entries {
                        e.visible = !any_visible;
                    }
                }
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!("{visible_count}/{entry_count} visible"))
                        .small()
                        .weak(),
                );
            });

            for direction in ["warp", "weft", "bias", "other"] {
                let dir_entries: Vec<usize> = visible_entries
                    .iter()
                    .copied()
                    .filter(|&i| {
                        direction_matches(
                            &slots[slot_idx].entries[i].curve.key.direction,
                            direction,
                        )
                    })
                    .collect();
                if dir_entries.is_empty() {
                    continue;
                }
                egui::CollapsingHeader::new(direction.to_string())
                    .default_open(true)
                    .show(ui, |ui| {
                        for branch in ["loading", "release", "other"] {
                            let branch_entries: Vec<usize> = dir_entries
                                .iter()
                                .copied()
                                .filter(|&i| {
                                    branch_matches(
                                        &slots[slot_idx].entries[i].curve.key.branch,
                                        branch,
                                    )
                                })
                                .collect();
                            if branch_entries.is_empty() {
                                continue;
                            }
                            let branch_any_visible = branch_entries
                                .iter()
                                .any(|&i| slots[slot_idx].entries[i].visible);
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .selectable_label(
                                        branch_any_visible,
                                        RichText::new(branch).weak(),
                                    )
                                    .clicked()
                                {
                                    for &i in &branch_entries {
                                        slots[slot_idx].entries[i].visible = !branch_any_visible;
                                    }
                                }
                                ui.add_space(4.0);
                                // Disambiguate repeated sample IDs with the run directory.
                                let disambiguate: Vec<bool> = {
                                    let mut counts: HashMap<&str, usize> = HashMap::new();
                                    for &i in &branch_entries {
                                        *counts
                                            .entry(
                                                slots[slot_idx].entries[i]
                                                    .curve
                                                    .key
                                                    .sample_id
                                                    .as_str(),
                                            )
                                            .or_insert(0) += 1;
                                    }
                                    branch_entries
                                        .iter()
                                        .map(|&i| {
                                            counts[slots[slot_idx].entries[i]
                                                .curve
                                                .key
                                                .sample_id
                                                .as_str()]
                                                > 1
                                        })
                                        .collect()
                                };
                                for (k, &i) in branch_entries.iter().enumerate() {
                                    run_checkbox(ui, slots, slot_idx, i, disambiguate[k]);
                                }
                            });
                        }
                    });
            }
        });
}

fn direction_matches(d: &Direction, label: &str) -> bool {
    matches!(
        (d, label),
        (Direction::Warp, "warp")
            | (Direction::Weft, "weft")
            | (Direction::Bias, "bias")
            | (Direction::Other(_), "other")
    )
}

fn branch_matches(b: &Branch, label: &str) -> bool {
    matches!(
        (b, label),
        (Branch::Loading, "loading") | (Branch::Release, "release") | (Branch::Other(_), "other")
    )
}

/// Row label: sample_id, with the run directory name appended when several
/// runs share the same sample_id within one branch (re-measurements).
fn run_label(slot: &ProjectSlot, idx: usize, disambiguate: bool) -> String {
    let key = &slot.entries[idx].curve.key;
    if !disambiguate {
        return key.sample_id.clone();
    }
    let dir_name = slot
        .project
        .runs
        .get(idx)
        .and_then(|r| r.dir.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("{} ({dir_name})", key.sample_id)
}

fn run_checkbox(
    ui: &mut egui::Ui,
    slots: &mut [ProjectSlot],
    slot_idx: usize,
    run_idx: usize,
    disambiguate: bool,
) {
    let label = run_label(&slots[slot_idx], run_idx, disambiguate);
    let legend_name = slots[slot_idx].entries[run_idx].legend_name.clone();
    let color = direction_color(&slots[slot_idx].entries[run_idx].curve.key.direction);
    let mut visible = slots[slot_idx].entries[run_idx].visible;
    let response = ui.checkbox(&mut visible, RichText::new(&label).color(color));
    slots[slot_idx].entries[run_idx].visible = visible;
    response.context_menu(|ui| {
        ui.set_min_width(160.0);
        ui.label(RichText::new(&legend_name).small().weak());
        if ui.button("Solo (hide others)").clicked() {
            for (si, slot) in slots.iter_mut().enumerate() {
                for (ri, e) in slot.entries.iter_mut().enumerate() {
                    e.visible = si == slot_idx && ri == run_idx;
                }
            }
            ui.close();
        }
        if ui.button("Hide run").clicked() {
            slots[slot_idx].entries[run_idx].visible = false;
            ui.close();
        }
        if ui.button("Show all").clicked() {
            for slot in slots.iter_mut().filter(|s| s.enabled) {
                for e in &mut slot.entries {
                    e.visible = true;
                }
            }
            ui.close();
        }
    });
}

impl RunsFilter {
    fn matches(&self, entry: &RunEntry) -> bool {
        let d = match &entry.curve.key.direction {
            Direction::Warp => self.warp,
            Direction::Weft => self.weft,
            Direction::Bias => self.bias,
            Direction::Other(_) => self.other_dir,
        };
        let b = match &entry.curve.key.branch {
            Branch::Loading => self.loading,
            Branch::Release => self.release,
            Branch::Other(_) => self.other_branch,
        };
        d && b
    }
}
