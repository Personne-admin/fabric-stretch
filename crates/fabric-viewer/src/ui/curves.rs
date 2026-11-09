//! The Curves dock tab: force/line-load plots with hover/click selection,
//! a compact toolbar, and an optional hover-image popup.

use std::collections::HashSet;
use std::path::PathBuf;

use egui::{Color32, Id};
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, PlotMemory, PlotPoints, Points};
use fabric_data::{Branch, Frame, Run};

use super::plot_math::{nearest_point, plot_y_values, smooth_curve_points};
use super::style::direction_color;
use crate::images::{frame_roles, resolve_role, ImageCache, PsLayers};
use crate::project::{frame_ref_for, resolve_frame_ref, ProjectSlot};
use crate::selection::{FrameRef, FrameSelection};
use crate::settings::{PlotQuantity, ViewerSettings};

const PLOT_ID: &str = "curves_plot";

/// Curves tab state that outlives a frame.
#[derive(Debug, Default)]
pub struct PlotState {
    /// Latch the frame identity at right-click; while open, the pointer is over the menu.
    pub context_target: Option<FrameRef>,
}

/// Legend IDs hidden by run visibility flags. Seed egui_plot memory from these each frame.
pub fn legend_hidden_ids(slots: &[ProjectSlot]) -> HashSet<egui::Id> {
    let mut hidden = HashSet::new();
    for slot in slots.iter().filter(|s| s.enabled) {
        for e in &slot.entries {
            if !e.visible {
                hidden.insert(e.legend_id);
            }
        }
    }
    hidden
}

/// The Curves tab: compact toolbar on top, the plot filling the rest.
pub fn curves_tab(
    ui: &mut egui::Ui,
    slots: &mut [ProjectSlot],
    selection: &mut FrameSelection,
    settings: &mut ViewerSettings,
    image_cache: &mut ImageCache,
    plot_state: &mut PlotState,
) {
    if slots.is_empty() {
        ui.centered_and_justified(|ui| {
            ui.label("Open a project folder to plot its runs.");
        });
        return;
    }
    if slots.iter().all(|s| !s.enabled) {
        ui.centered_and_justified(|ui| {
            ui.label("No enabled projects — enable one from the Projects menu.");
        });
        return;
    }

    toolbar(ui, settings);

    // Copy settings before the toolbar and legend need mutable access to slots.
    let quantity = settings.plot.quantity;
    let mirror = settings.plot.mirror_release;
    let show_relax = settings.plot.show_relax;
    let smooth = settings.plot.smooth;
    let show_legend = settings.plot.show_legend;
    let show_grid = settings.plot.show_grid;
    let line_width = settings.plot.line_width;
    let selected_line_width = settings.plot.selected_line_width;
    let marker_radius = settings.plot.marker_radius;
    let hover_preview = settings.plot.hover_preview;
    let hover_preview_max_px = settings.plot.hover_preview_max_px;
    let preferred_role = settings.images.preferred_role.clone();

    let plot_id = Id::new(PLOT_ID);

    // Overwrite cached legend visibility from run flags, even with the legend hidden.
    // Read back this frame's legend clicks afterwards.
    let hidden_set = legend_hidden_ids(slots);
    if let Some(mut mem) = PlotMemory::load(ui.ctx(), plot_id) {
        mem.hidden_items = hidden_set.iter().copied().collect();
        mem.store(ui.ctx(), plot_id);
    }

    let selected_hit = resolve_hit(slots, selection.selected.as_ref());
    let hovered_hit = resolve_hit(slots, selection.hovered.as_ref());

    let mut plot = Plot::new("curves")
        .id(plot_id)
        .show_grid(show_grid)
        .x_axis_label(if mirror {
            "stretch [mm]"
        } else {
            "displacement (travel within run) [mm]"
        })
        .y_axis_label(quantity.label());
    if show_legend {
        plot = plot.legend(Legend::default().hidden_items(hidden_set.iter().copied()));
    }

    let plot_response = plot.show(ui, |plot_ui| {
        for (pidx, slot) in slots.iter().enumerate() {
            if !slot.enabled {
                continue;
            }
            for (ridx, entry) in slot.entries.iter().enumerate() {
                let curve = &entry.curve;
                let is_release = curve.key.branch == Branch::Release;
                let start = if mirror && is_release {
                    slot.release_start_mm(ridx)
                } else {
                    0.0
                };
                let x_of = |d: f64| {
                    if mirror && is_release {
                        curve.mirrored_displacement(start, d)
                    } else {
                        d
                    }
                };

                let y_by_frame = plot_y_values(curve, quantity, show_relax, smooth);
                let mut series: Vec<[f64; 2]> = curve
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| show_relax || !p.is_relaxation_hold)
                    .filter_map(|(i, p)| Some([x_of(p.displacement_mm), y_by_frame[i]?]))
                    .collect();
                if smooth {
                    series = smooth_curve_points(&series);
                }

                let color = direction_color(&curve.key.direction);
                let is_selected_run = selected_hit.is_some_and(|(p, r, _)| p == pidx && r == ridx);
                let width = if is_selected_run {
                    selected_line_width
                } else {
                    line_width
                };
                let mut line = Line::new(entry.legend_name.clone(), PlotPoints::from(series))
                    .color(color)
                    .width(width);
                if is_release {
                    line = line.style(LineStyle::Dashed { length: 8.0 });
                }
                // Register hidden runs too: removing them from the legend would lose their hidden state.
                plot_ui.line(line);

                if show_relax {
                    let relax: Vec<[f64; 2]> = curve
                        .points
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.is_relaxation_hold)
                        .filter_map(|(i, p)| Some([x_of(p.displacement_mm), y_by_frame[i]?]))
                        .collect();
                    if !relax.is_empty() {
                        // Share the line's legend name so markers hide with it.
                        plot_ui.points(
                            Points::new(entry.legend_name.clone(), PlotPoints::from(relax))
                                .color(color)
                                .shape(MarkerShape::Diamond)
                                .radius(marker_radius),
                        );
                    }
                }

                // Hidden runs must not leave selection markers.
                if entry.visible {
                    if let Some((_, _, fidx)) =
                        selected_hit.filter(|&(p, r, _)| p == pidx && r == ridx)
                    {
                        if let Some(p) = curve.points.get(fidx) {
                            if let Some(y) = y_by_frame.get(fidx).copied().flatten() {
                                plot_ui.points(
                                    Points::new(
                                        entry.legend_name.clone(),
                                        PlotPoints::from(vec![[x_of(p.displacement_mm), y]]),
                                    )
                                    .color(Color32::WHITE)
                                    .shape(MarkerShape::Circle)
                                    .radius(6.0),
                                );
                            }
                        }
                    }

                    if let Some((_, _, fidx)) =
                        hovered_hit.filter(|&(p, r, _)| p == pidx && r == ridx)
                    {
                        if let Some(p) = curve.points.get(fidx) {
                            if let Some(y) = y_by_frame.get(fidx).copied().flatten() {
                                plot_ui.points(
                                    Points::new(
                                        entry.legend_name.clone(),
                                        PlotPoints::from(vec![[x_of(p.displacement_mm), y]]),
                                    )
                                    .color(Color32::from_rgb(240, 170, 60))
                                    .shape(MarkerShape::Square)
                                    .filled(false)
                                    .radius(7.0),
                                );
                            }
                        }
                    }
                }
            }
        }
    });

    // Copy legend clicks back to run flags. Skip disabled projects.
    if let Some(mem) = PlotMemory::load(ui.ctx(), plot_id) {
        for slot in slots.iter_mut().filter(|s| s.enabled) {
            for e in &mut slot.entries {
                e.visible = !mem.hidden_items.contains(&e.legend_id);
            }
        }
    }

    // Hover: update `selection.hovered`, clear it when the pointer leaves
    // the plot or drifts away from every curve.
    let hover_hit = plot_response.response.hover_pos().and_then(|pos| {
        nearest_point(
            slots,
            quantity,
            mirror,
            show_relax,
            smooth,
            pos,
            &plot_response.transform,
        )
    });
    match hover_hit {
        Some(hit) => {
            selection.hovered = frame_ref_for(slots, hit.0, hit.1, hit.2);
            if hover_preview {
                hover_preview_popup(
                    ui.ctx(),
                    &plot_response,
                    slots,
                    hit,
                    preferred_role.as_deref(),
                    image_cache,
                    hover_preview_max_px,
                );
            }
        }
        None => selection.hovered = None,
    }

    if plot_response.response.clicked() {
        if let Some(pos) = plot_response.response.interact_pointer_pos() {
            if let Some((pidx, ridx, fidx)) = nearest_point(
                slots,
                quantity,
                mirror,
                show_relax,
                smooth,
                pos,
                &plot_response.transform,
            ) {
                if let Some(fr) = frame_ref_for(slots, pidx, ridx, fidx) {
                    selection.selected = Some(fr);
                    selection.pinned = false;
                }
            }
        }
    }

    // Latch at right-click; an open menu moves the pointer off the plot.
    if plot_response.response.secondary_clicked() {
        plot_state.context_target = plot_response
            .response
            .interact_pointer_pos()
            .and_then(|pos| {
                nearest_point(
                    slots,
                    quantity,
                    mirror,
                    show_relax,
                    smooth,
                    pos,
                    &plot_response.transform,
                )
            })
            .and_then(|(p, r, f)| frame_ref_for(slots, p, r, f));
    }
    plot_response
        .response
        .context_menu(|ui| plot_context_menu(ui, slots, selection, &plot_state.context_target));
}

fn toolbar(ui: &mut egui::Ui, settings: &mut ViewerSettings) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Plot:");
        ui.radio_value(
            &mut settings.plot.quantity,
            PlotQuantity::ForceN,
            "Force [N]",
        );
        ui.radio_value(
            &mut settings.plot.quantity,
            PlotQuantity::LineLoad,
            "Line load [N/m]",
        );
        ui.separator();
        ui.checkbox(&mut settings.plot.mirror_release, "Mirror release")
            .on_hover_text(
                "Release runs record travel from the stretched position; mirroring plots \
             them against stretch so loading + release form a hysteresis loop.",
            );
        ui.checkbox(&mut settings.plot.show_relax, "Relaxation holds");
        ui.checkbox(&mut settings.plot.smooth, "Smooth")
            .on_hover_text(
                "Smooth load-cell micro-jumps with a local Gaussian linear filter and draw \
                 the result with monotone cubic interpolation. Off shows the raw measurements.",
            );
        ui.separator();
        ui.checkbox(&mut settings.plot.show_legend, "Legend");
        ui.checkbox(&mut settings.plot.show_grid, "Grid");
        ui.separator();
    });
}

fn resolve_hit(slots: &[ProjectSlot], fr: Option<&FrameRef>) -> Option<(usize, usize, usize)> {
    resolve_frame_ref(slots, fr?)
}

/// Optional hover popup showing the hovered frame's image at the pointer.
fn hover_preview_popup(
    ctx: &egui::Context,
    plot_response: &egui_plot::PlotResponse<()>,
    slots: &[ProjectSlot],
    hit: (usize, usize, usize),
    preferred_role: Option<&str>,
    image_cache: &mut ImageCache,
    max_px: f32,
) {
    let (pidx, ridx, fidx) = hit;
    let Some(slot) = slots.get(pidx) else { return };
    let Some(run) = slot.project.runs.get(ridx) else {
        return;
    };
    let Some(frame) = run.frames.frames.get(fidx) else {
        return;
    };
    let ps = slot.entries.get(ridx).map(|e| &e.ps);
    let Some(path) = preview_image_path(run, ps, frame, preferred_role) else {
        return;
    };
    let frame_id = frame.frame_id;
    let legend_name = slot
        .entries
        .get(ridx)
        .map(|e| e.legend_name.clone())
        .unwrap_or_default();

    image_cache.request_and_repaint(ctx, &path);
    let Some(texture) = image_cache.texture(&path) else {
        return; // still decoding; the image viewer will show it once ready
    };
    plot_response.response.clone().on_hover_ui_at_pointer(|ui| {
        ui.label(
            egui::RichText::new(format!("{legend_name} - frame {frame_id}"))
                .small()
                .weak(),
        );
        ui.add(egui::Image::from_texture(texture).max_size(egui::vec2(max_px, max_px)));
    });
}

/// Preferred-role image path, including virtual PS roles.
pub fn preview_image_path(
    run: &Run,
    ps: Option<&PsLayers>,
    frame: &Frame,
    preferred: Option<&str>,
) -> Option<PathBuf> {
    let roles = frame_roles(frame, ps.unwrap_or(&PsLayers::default()));
    let (_, filename) = resolve_role(
        roles.iter().map(|(r, f)| (r.as_str(), f.as_str())),
        preferred,
    )?;
    Some(run.images_dir().join(filename))
}

fn plot_context_menu(
    ui: &mut egui::Ui,
    slots: &mut [ProjectSlot],
    selection: &mut FrameSelection,
    context_target: &Option<FrameRef>,
) {
    ui.set_min_width(180.0);
    // The target project may have closed while the menu was open.
    let target = context_target
        .as_ref()
        .and_then(|fr| resolve_frame_ref(slots, fr));
    if let Some((pidx, ridx, fidx)) = target {
        let entry = &slots[pidx].entries[ridx];
        let name = entry.legend_name.clone();
        let frame_id = slots[pidx].project.runs[ridx].frames.frames[fidx].frame_id;
        ui.label(egui::RichText::new(name).small().weak());
        if ui
            .button(format!("Select run at frame {frame_id}"))
            .clicked()
        {
            if let Some(fr) = frame_ref_for(slots, pidx, ridx, fidx) {
                selection.selected = Some(fr);
                selection.pinned = false;
            }
            ui.close();
        }
        if ui.button("Hide run").clicked() {
            slots[pidx].entries[ridx].visible = false;
            ui.close();
        }
        if ui.button("Solo run (hide others)").clicked() {
            for (pi, slot) in slots.iter_mut().enumerate() {
                for (ri, e) in slot.entries.iter_mut().enumerate() {
                    e.visible = (pi, ri) == (pidx, ridx);
                }
            }
            ui.close();
        }
        ui.separator();
    }
    if ui.button("Show all runs").clicked() {
        for slot in slots.iter_mut().filter(|s| s.enabled) {
            for e in &mut slot.entries {
                e.visible = true;
            }
        }
        ui.close();
    }
    if ui.button("Hide all runs").clicked() {
        for slot in slots.iter_mut().filter(|s| s.enabled) {
            for e in &mut slot.entries {
                e.visible = false;
            }
        }
        ui.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hover_preview_honours_virtual_ps_roles() {
        let dir = std::env::temp_dir().join(format!("fabsim_curves_prev_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let run_dir = dir.join("warp_loading");
        let ps_dir = run_dir.join("ps_output");
        std::fs::create_dir_all(&ps_dir).unwrap();
        std::fs::write(ps_dir.join("img_0000_albedo.png"), b"x").unwrap();
        std::fs::write(
            run_dir.join("session.json"),
            r#"{"sample_id":"s","test_id":"t","direction":"warp","branch":"loading","count":1}"#,
        )
        .unwrap();
        std::fs::write(
            run_dir.join("frames.json"),
            r#"{"frames":[{"frame_id":0,"displacement_mm":0.0,"image_diffuse":"d.png"}]}"#,
        )
        .unwrap();
        std::fs::write(run_dir.join("force.json"), r#"{"forces":[]}"#).unwrap();
        let project = fabric_data::Project::discover(&dir).unwrap();
        let run = &project.runs[0];
        let ps = PsLayers::scan(&run.dir);
        let frame = &run.frames.frames[0];

        let albedo = preview_image_path(run, Some(&ps), frame, Some("albedo")).unwrap();
        assert!(albedo.ends_with("images/../ps_output/img_0000_albedo.png"));
        // Without a PS index, fall back to declared diffuse.
        let fallback = preview_image_path(run, None, frame, Some("albedo")).unwrap();
        assert!(fallback.ends_with("images/d.png"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
