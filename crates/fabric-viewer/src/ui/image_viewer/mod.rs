//! Image Viewer tab, using the shared [`FrameSelection`].
//! `canvas::RegDisplay` applies the same registration/stabilization to images and overlays.
//! Relighting runs on a worker; `tracking_panel` hosts the controls, with legacy options in `advanced`.

mod advanced;
mod canvas;
mod export;
mod grid_preview;
pub(crate) mod overlay;
mod relight;
mod replay;
mod tracking_panel;

#[cfg(test)]
mod tests;

use egui::{Color32, RichText, Vec2};
use fabric_data::tracking::{registration::RegistrationTransform, FrameId};
use fabric_data::{Frame, Run};

use self::canvas::{canvas, PaintImage, Pane};
use self::export::export_row;
use self::grid_preview::{Grid, PreviewCache};
use self::overlay::frame_render_inputs;
use self::relight::{RelightEntry, RelightSources, RelightState};
use self::replay::{replay_row, ReplayTarget};
use self::tracking_panel::tracking_panel;
use crate::images::relight::{RelightSignature, RelightWorker};
use crate::images::{find_normal_role, frame_roles, resolve_role, ImageCache, ImageStatus};
use crate::lighting::light_pad;
use crate::playback::PlaybackState;
use crate::project::{resolve_frame_ref, ProjectSlot};
use crate::selection::{FrameRef, FrameSelection};
use crate::settings::{
    ComparisonMode, FitMode, ImagePaneSettings, ImageStabilization, Interpolation, ViewerSettings,
};
use crate::tracked::TrackingSession;
use crate::ui::style;
use std::sync::Arc;

/// Frames decoded ahead of the current one while playing.
const PREFETCH_AHEAD: usize = 3;

/// The last image that was ready to show, kept (with its texture alive) so
/// the canvas does not blank while the next frame decodes.
struct LastImage {
    texture: egui::TextureHandle,
    size: (u32, u32),
}

impl std::fmt::Debug for LastImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LastImage")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

/// Live state of the Image Viewer tab (view transform, comparison choice,
/// relight job, export).
#[derive(Debug)]
pub struct ImageViewerState {
    /// Zoom; `None` means Fit.
    pub scale: Option<f32>,
    pub pan: egui::Vec2,
    wipe_dividers: Vec<f32>,
    wipe_dragging: Option<usize>,
    previews: PreviewCache,
    pub synced_fit: Option<FitMode>,
    /// One-shot: re-apply the fit mode even if it did not change.
    pub fit_force: bool,
    last_image: Option<LastImage>,
    relight: Option<RelightEntry>,
    relight_pending: bool,
    relight_requested: Option<RelightSignature>,
    relight_generation: u64,
    relight_worker: RelightWorker,
    export: Option<crate::frame_export::ExportHandle>,
    export_status: Option<String>,
}

impl Default for ImageViewerState {
    fn default() -> Self {
        ImageViewerState {
            scale: None,
            pan: egui::Vec2::ZERO,
            wipe_dividers: Vec::new(),
            wipe_dragging: None,
            previews: PreviewCache::default(),
            synced_fit: None,
            fit_force: false,
            last_image: None,
            relight: None,
            relight_pending: false,
            relight_requested: None,
            relight_generation: 0,
            relight_worker: RelightWorker::new(),
            export: None,
            export_status: None,
        }
    }
}

/// Everything the tab reads or edits besides its own [`ImageViewerState`].
pub struct ImageViewerCtx<'a> {
    pub slots: &'a [ProjectSlot],
    pub selection: &'a mut FrameSelection,
    pub settings: &'a mut ViewerSettings,
    pub image_cache: &'a mut ImageCache,
    pub tracking: &'a mut TrackingSession,
    pub playback: &'a mut PlaybackState,
}

fn centered_message(ui: &mut egui::Ui, text: &str) {
    ui.centered_and_justified(|ui| {
        ui.label(text);
    });
}

/// Apply a fit mode to both the viewer transform and the persisted setting so
/// toolbar, keyboard and Preferences all agree.
fn apply_fit(state: &mut ImageViewerState, settings: &mut ViewerSettings, mode: FitMode) {
    state.scale = match mode {
        FitMode::Fit => None,
        FitMode::OneToOne => Some(1.0),
    };
    state.pan = Vec2::ZERO;
    state.synced_fit = Some(mode);
    settings.images.fit = mode;
}

/// Start decoding the primary images of the next frames while playing.
fn prefetch(
    ui: &egui::Ui,
    image_cache: &mut ImageCache,
    slot: &ProjectSlot,
    run_index: usize,
    frame_index: usize,
    preferred_role: Option<&str>,
) {
    let run = &slot.project.runs[run_index];
    let ps = &slot.entries[run_index].ps;
    for frame in run
        .frames
        .frames
        .iter()
        .skip(frame_index + 1)
        .take(PREFETCH_AHEAD)
    {
        let roles = frame_roles(frame, ps);
        let picked = resolve_role(
            roles.iter().map(|(r, f)| (r.as_str(), f.as_str())),
            preferred_role,
        );
        if let Some((_, filename)) = picked {
            image_cache.request_and_repaint(ui.ctx(), run.images_dir().join(filename));
        }
    }
}

pub fn image_viewer_tab(ui: &mut egui::Ui, state: &mut ImageViewerState, cx: &mut ImageViewerCtx) {
    let slots = cx.slots;
    let settings = &mut *cx.settings;
    let image_cache = &mut *cx.image_cache;
    let tracking = &mut *cx.tracking;

    // Apply sampler changes from either toolbar or Preferences.
    let texture_options = settings.images.interpolation.texture_options();
    image_cache.set_texture_options(texture_options);
    state.poll_relight(ui.ctx(), texture_options);
    state.previews.poll(ui.ctx(), texture_options);

    // Reapply fit on an explicit click, even when the mode did not change.
    if state.fit_force || state.synced_fit != Some(settings.images.fit) {
        state.scale = match settings.images.fit {
            FitMode::Fit => None,
            FitMode::OneToOne => Some(1.0),
        };
        state.pan = Vec2::ZERO;
        state.synced_fit = Some(settings.images.fit);
        state.fit_force = false;
    }

    if slots.is_empty() {
        centered_message(ui, "Open a project folder to view its frame images.");
        return;
    }

    // Hover wins unless pinned or follow-hover is disabled.
    let frame_ref = if settings.images.hover_follow {
        cx.selection.active().cloned()
    } else {
        cx.selection.selected.clone()
    };
    let Some(frame_ref) = frame_ref else {
        centered_message(
            ui,
            "Select a run or click a curve point to see its frame images.",
        );
        return;
    };
    let Some((pidx, ridx, fidx)) = resolve_frame_ref(slots, &frame_ref) else {
        centered_message(
            ui,
            "The selected frame is no longer available (project closed or changed).",
        );
        return;
    };
    let slot = &slots[pidx];
    let run = &slot.project.runs[ridx];
    let ps = &slot.entries[ridx].ps;
    let Some(frame) = run.frames.frames.get(fidx) else {
        return;
    };

    // Keep playback outside collapsible tracking sections.
    replay_row(
        ui,
        slots,
        cx.selection,
        cx.playback,
        ReplayTarget {
            project: pidx,
            run: ridx,
            frame: fidx,
            n_frames: run.frames.frames.len(),
        },
    );
    ui.separator();

    let roles: Vec<(String, String)> = frame_roles(frame, ps);
    let role_pairs = || roles.iter().map(|(r, f)| (r.as_str(), f.as_str()));
    let Some((primary_role, primary_filename)) =
        resolve_role(role_pairs(), settings.images.preferred_role.as_deref())
    else {
        centered_message(ui, "frame has no images");
        return;
    };
    let images_dir = run.images_dir();
    let primary_path = images_dir.join(primary_filename);
    let used_fallback = settings
        .images
        .preferred_role
        .as_deref()
        .is_none_or(|p| p != primary_role);

    // Relight from PS albedo/normals, independent of the displayed role.
    let find_role = |wanted: &str| {
        roles
            .iter()
            .find(|(role, _)| role == wanted)
            .map(|(_, filename)| filename.as_str())
    };
    let sources = RelightSources {
        albedo: find_role("albedo").map(|f| (images_dir.join(f), ps.albedo_scale_for(f))),
        normal: find_normal_role(role_pairs())
            .and_then(|role| find_role(role).map(|f| images_dir.join(f))),
        run_has_ps: !ps.is_empty(),
    };

    controls_row(
        ui,
        settings,
        state,
        ControlsModel {
            roles: &roles,
            primary_role,
            can_relight: sources.albedo.is_some() && sources.normal.is_some(),
            run_has_ps: sources.run_has_ps,
        },
    );

    // New requests schedule repaint even when made after this frame's cache poll.
    let primary_status = image_cache.request_and_repaint(ui.ctx(), &primary_path);
    if settings.images.relight.enabled {
        for path in [
            sources.albedo.as_ref().map(|(p, _)| p),
            sources.normal.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            image_cache.request_and_repaint(ui.ctx(), path);
        }
    }
    if cx.playback.playing {
        prefetch(
            ui,
            image_cache,
            slot,
            ridx,
            fidx,
            settings.images.preferred_role.as_deref(),
        );
    }
    let primary_size = image_cache.size(&primary_path);
    let primary_texture = image_cache.texture(&primary_path).cloned();
    if let (Some(texture), Some(size)) = (&primary_texture, primary_size) {
        state.last_image = Some(LastImage {
            texture: texture.clone(),
            size,
        });
    }

    let relight_state = state.request_relight(image_cache, &settings.images.relight, &sources);
    if matches!(relight_state, RelightState::Busy) {
        ui.ctx().request_repaint();
    }
    let use_relight = relight_state.shows_relit(state.relight.is_some());

    // What the canvas shows: the relit image, the decoded primary, the
    // previous frame while this one decodes, or a status message.
    let mut stale = false;
    let mut message = None;
    let displayed: Option<PaintImage> = if use_relight {
        state.relight.as_ref().map(|e| PaintImage {
            texture: e.texture.id(),
            size: (e.image.width(), e.image.height()),
        })
    } else if let (Some(texture), Some(size)) = (&primary_texture, primary_size) {
        Some(PaintImage {
            texture: texture.id(),
            size,
        })
    } else {
        match &primary_status {
            ImageStatus::Error(err) => {
                message = Some((format!("image failed to load: {err}"), true));
                None
            }
            status => {
                let text = if *status == ImageStatus::Pending {
                    "loading image..."
                } else {
                    "preparing texture..."
                };
                message = Some((text.to_owned(), false));
                stale = state.last_image.is_some();
                state.last_image.as_ref().map(|last| PaintImage {
                    texture: last.texture.id(),
                    size: last.size,
                })
            }
        }
    };

    if let ImageStatus::Error(err) = &primary_status {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(style::ERROR, format!("primary image failed to load: {err}"));
            if ui
                .button("Retry")
                .on_hover_text("Forget the cached failure and try loading again")
                .clicked()
            {
                image_cache.invalidate(&primary_path);
                image_cache.request_and_repaint(ui.ctx(), &primary_path);
            }
        });
    }
    if let Some(note) = relight_state.note() {
        ui.label(RichText::new(note).small().weak());
    }

    metadata_row(
        ui,
        FrameInfo {
            run,
            frame,
            frame_ref: &frame_ref,
            role: primary_role,
            filename: primary_filename,
            size: primary_size,
            used_fallback,
            relit: use_relight,
        },
    );

    let run_key = slot.run_keys[ridx].as_path();
    let frame_id = FrameId(frame.frame_id);

    // Discover once per run so Unload sticks; check frame IDs once per import.
    let _ = tracking.auto_discover(run_key);
    if tracking.needs_alignment(run_key) {
        let run_frame_ids: Vec<u32> = run.frames.frames.iter().map(|f| f.frame_id).collect();
        tracking.refresh_alignment(run_key, &run_frame_ids);
    }

    // Show legacy registration controls only when enabled in Preferences.
    tracking_panel(
        ui,
        tracking,
        run_key,
        run,
        frame_id,
        settings.ui.advanced_tracking,
    );
    export_row(ui, state, tracking, cx.playback.fps, settings, slot, ridx);

    // Read tracking controls after drawing them so changes apply in this frame.
    let (tracking_transform, mut overlay) = frame_render_inputs(tracking, run_key, frame_id);
    let tracking_run = tracking.run_state(run_key);
    let comparison = settings.images.comparison;
    let configs = if comparison == ComparisonMode::Single {
        &[][..]
    } else {
        settings.images.comparison_panes.as_slice()
    };
    let primary_config = &settings.images.primary_pane;
    let all_configs = std::iter::once(primary_config).chain(configs);
    let wants_grid = all_configs
        .clone()
        .any(|c| c.stabilization == ImageStabilization::Grid);
    let grid = wants_grid
        .then(|| tracking_run.and_then(|r| Grid::for_frame(r, frame_id)))
        .flatten()
        .map(Arc::new);
    let wants_top = all_configs
        .clone()
        .any(|c| c.stabilization == ImageStabilization::Top);
    let top = wants_top
        .then(|| tracking_run.and_then(|r| tracking.stabilization_for(r, frame_id)))
        .flatten()
        .map(|s| s.transform);
    state.previews.set_active_count(configs.len() + 1);

    let mut panes = Vec::with_capacity(configs.len() + 1);
    for (i, config) in all_configs.enumerate() {
        let primary = i == 0;
        let comparison_role = || {
            config
                .role
                .as_ref()
                .and_then(|choice| roles.iter().find(|(role, _)| role == choice))
                .or_else(|| {
                    config
                        .role
                        .is_none()
                        .then(|| {
                            roles
                                .iter()
                                .find(|(role, _)| role != primary_role)
                                .or_else(|| roles.first())
                        })
                        .flatten()
                })
        };
        let (role, image, decoded, note) = if primary {
            let decoded = if use_relight {
                state.relight.as_ref().and_then(|entry| {
                    let wanted = state.relight_requested.as_ref()?;
                    (entry.signature.albedo_key == wanted.albedo_key
                        && entry.signature.normal_key == wanted.normal_key)
                        .then(|| entry.image.clone())
                })
            } else {
                image_cache.decoded(&primary_path)
            };
            (primary_role, displayed.clone(), decoded, message.clone())
        } else if let Some((role, filename)) = comparison_role() {
            let path = images_dir.join(filename);
            let status = image_cache.request_and_repaint(ui.ctx(), &path);
            let image =
                image_cache
                    .texture(&path)
                    .zip(image_cache.size(&path))
                    .map(|(tex, size)| PaintImage {
                        texture: tex.id(),
                        size,
                    });
            let note = match status {
                ImageStatus::Error(error) => Some((error, true)),
                _ if image.is_none() => Some(("Loading image…".into(), false)),
                _ => None,
            };
            (role.as_str(), image, image_cache.decoded(&path), note)
        } else {
            (
                config.role.as_deref().unwrap_or("Image"),
                None,
                None,
                Some(("Role unavailable for this frame".into(), true)),
            )
        };
        let mut note = note;
        let mut image = image;
        let mut transform = match config.stabilization {
            ImageStabilization::Tracking => tracking_transform,
            ImageStabilization::Raw | ImageStabilization::Grid => RegistrationTransform::IDENTITY,
            ImageStabilization::Top => top.unwrap_or(RegistrationTransform::IDENTITY),
        };
        if config.stabilization == ImageStabilization::Top && top.is_none() && image.is_some() {
            note = Some(("Top tracking anchors unavailable".into(), true));
        }
        let grid_mode = config.stabilization == ImageStabilization::Grid;
        let blend = config.tile_check && config.blend_edges;
        if grid_mode && grid.is_none() {
            image = None;
            note = Some(("A complete tracked grid is required".into(), true));
        } else if (grid_mode || blend) && image.is_some() {
            let ready = decoded.and_then(|decoded| {
                state.previews.request(
                    decoded,
                    if grid_mode { grid.clone() } else { None },
                    blend,
                    ui.ctx(),
                )
            });
            image = ready.map(|(texture, size)| PaintImage { texture, size });
            if image.is_none() && note.is_none() {
                note = Some(("Preparing preview…".into(), false));
            }
        }
        if primary {
            if stale || config.tile_check || (grid_mode && !tracking.stabilize.show_points) {
                overlay = None;
            } else if let Some(overlay) = &mut overlay {
                overlay.transform = transform;
                if grid_mode {
                    overlay.tracks.retain_mut(|track| {
                        let Some([x, y]) = grid
                            .as_ref()
                            .and_then(|grid| grid.rectified_point(&track.id))
                        else {
                            return false;
                        };
                        track.current.x = x;
                        track.current.y = y;
                        true
                    });
                }
            }
        }
        if primary && stale {
            transform = RegistrationTransform::IDENTITY;
        }
        let role = if primary && use_relight {
            "Relit"
        } else {
            role
        };
        let title = format!(
            "{role} · {}{}",
            config.stabilization.label(),
            if config.tile_check {
                " · Tile check"
            } else {
                ""
            }
        );
        panes.push(Pane {
            image,
            title,
            message: note,
            transform,
            tiled: config.tile_check,
        });
    }
    let (hovered_track, clicked_track) = canvas(ui, state, &panes, comparison, overlay);
    tracking.hovered_track = hovered_track;

    // Ctrl/Shift toggles tracks in the candidate reference set.
    if let Some(track) = clicked_track {
        let (ctrl, shift) = ui.ctx().input(|i| (i.modifiers.ctrl, i.modifiers.shift));
        if (ctrl || shift) && !tracking.multi_select.remove(&track) {
            tracking.multi_select.insert(track.clone());
        }
        tracking.selected_track = Some(track);
    }

    let ctx = ui.ctx();
    if !ctx.egui_wants_keyboard_input() {
        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            apply_fit(state, settings, FitMode::Fit);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
            apply_fit(state, settings, FitMode::OneToOne);
        }
    }
}

/// What the controls row needs to know about the current frame.
struct ControlsModel<'a> {
    roles: &'a [(String, String)],
    primary_role: &'a str,
    can_relight: bool,
    run_has_ps: bool,
}

fn controls_row(
    ui: &mut egui::Ui,
    settings: &mut ViewerSettings,
    state: &mut ImageViewerState,
    model: ControlsModel<'_>,
) {
    let ControlsModel {
        roles,
        primary_role,
        can_relight,
        run_has_ps,
    } = model;
    ui.horizontal_wrapped(|ui| {
        ui.label("Role:");
        egui::ComboBox::from_id_salt("image_role")
            .selected_text(primary_role)
            .width(ui.available_width().min(140.0))
            .show_ui(ui, |ui| {
                for (role, _) in roles {
                    if ui.selectable_label(*role == primary_role, role).clicked() {
                        settings.images.preferred_role = Some(role.clone());
                    }
                }
            });

        ui.separator();
        if ui
            .button("Fit")
            .on_hover_text("scale to fit the canvas (F)")
            .clicked()
        {
            apply_fit(state, settings, FitMode::Fit);
        }
        if ui
            .button("1:1")
            .on_hover_text("one image pixel = one point (1)")
            .clicked()
        {
            apply_fit(state, settings, FitMode::OneToOne);
        }
        ui.separator();
        for (mode, label) in [
            (Interpolation::Linear, "Linear"),
            (Interpolation::Nearest, "Nearest"),
        ] {
            if ui
                .selectable_label(settings.images.interpolation == mode, label)
                .clicked()
            {
                settings.images.interpolation = mode;
            }
        }

        ui.separator();
        ui.label("Compare:");
        for mode in ComparisonMode::ALL {
            ui.radio_value(&mut settings.images.comparison, mode, mode.label());
        }
        egui::ComboBox::from_id_salt("primary_stabilization")
            .selected_text(settings.images.primary_pane.stabilization.label())
            .show_ui(ui, |ui| {
                for mode in ImageStabilization::ALL {
                    ui.selectable_value(
                        &mut settings.images.primary_pane.stabilization,
                        mode,
                        mode.label(),
                    );
                }
            });
        egui::containers::menu::MenuButton::new("View settings")
            .config(
                egui::containers::menu::MenuConfig::default()
                    .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
            )
            .ui(ui, |ui| {
                tile_controls(ui, &mut settings.images.primary_pane);
            });
        // Disable relighting without both PS layers.
        ui.separator();
        let relight = &mut settings.images.relight;
        ui.add_enabled(
            can_relight || relight.enabled,
            egui::Checkbox::new(&mut relight.enabled, "Relight"),
        )
        .on_hover_text(if can_relight {
            "Shade the PS albedo with the PS normals under the light below"
        } else if run_has_ps {
            "This frame has no PS albedo / normal layers"
        } else {
            "No photometric-stereo layers for this run: generate them with `fabsim ps <project>`"
        });
        if relight.enabled && can_relight {
            light_pad(ui, &mut relight.light);
        }
    });
    if settings.images.comparison != ComparisonMode::Single {
        egui::CollapsingHeader::new("Comparison views")
            .default_open(true)
            .show(ui, |ui| {
                let panes = &mut settings.images.comparison_panes;
                if ui.button("+ View").clicked() {
                    panes.push(ImagePaneSettings::default());
                }
                let mut remove = None;
                egui::ScrollArea::vertical()
                    .max_height(160.0)
                    .show(ui, |ui| {
                        for (i, pane) in panes.iter_mut().enumerate() {
                            ui.push_id(i, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(format!("View {}", i + 2));
                                    egui::ComboBox::from_id_salt("role")
                                        .selected_text(
                                            pane.role.as_deref().unwrap_or("Alternate role"),
                                        )
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut pane.role,
                                                None,
                                                "Alternate role",
                                            );
                                            for (role, _) in roles {
                                                ui.selectable_value(
                                                    &mut pane.role,
                                                    Some(role.clone()),
                                                    role,
                                                );
                                            }
                                        });
                                    egui::ComboBox::from_id_salt("stabilization")
                                        .selected_text(pane.stabilization.label())
                                        .show_ui(ui, |ui| {
                                            for mode in ImageStabilization::ALL {
                                                ui.selectable_value(
                                                    &mut pane.stabilization,
                                                    mode,
                                                    mode.label(),
                                                );
                                            }
                                        });
                                    tile_controls(ui, pane);
                                    if ui.small_button("Remove").clicked() {
                                        remove = Some(i);
                                    }
                                })
                            });
                        }
                    });
                if let Some(i) = remove {
                    panes.remove(i);
                }
            });
    }
}

fn tile_controls(ui: &mut egui::Ui, pane: &mut ImagePaneSettings) {
    ui.checkbox(&mut pane.tile_check, "Tile check")
        .on_hover_text("Repeat the image in a 3 × 3 grid to inspect its seams");
    if pane.tile_check {
        ui.checkbox(&mut pane.blend_edges, "Blend edges")
            .on_hover_text("Crossfade opposite edges before repeating the image");
    }
}

/// Facts about the displayed frame for the metadata line.
struct FrameInfo<'a> {
    run: &'a Run,
    frame: &'a Frame,
    frame_ref: &'a FrameRef,
    role: &'a str,
    filename: &'a str,
    size: Option<(u32, u32)>,
    used_fallback: bool,
    relit: bool,
}

fn metadata_row(ui: &mut egui::Ui, info: FrameInfo<'_>) {
    let FrameInfo {
        run,
        frame,
        frame_ref,
        role,
        filename,
        size,
        used_fallback,
        relit,
    } = info;
    let mut text = format!(
        "{} · frame {} · {} px · displacement {:.2} mm",
        frame_ref
            .run_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        frame.frame_id,
        size.map_or_else(|| "?".into(), |(w, h)| format!("{w}\u{00d7}{h}")),
        frame.displacement_mm,
    );
    if let Some(force) = run.force_n_for_frame(frame) {
        text.push_str(&format!(" · force {force:.2} N"));
    }
    if let Some(line_load) = frame.line_load_n_per_m {
        text.push_str(&format!(" · line load {line_load:.2} N/m"));
    }
    text.push_str(&format!(
        " · {} {}",
        run.session.direction, run.session.branch
    ));
    if frame.is_relaxation_hold {
        text.push_str(" · RELAXATION HOLD");
    }
    let color = if used_fallback {
        style::WARN
    } else {
        Color32::from_gray(170)
    };
    ui.label(RichText::new(text).small().color(color))
        .on_hover_text(format!(
            "{}\nrole: {} (fallback: {})\nfile: {}",
            run.dir.display(),
            role,
            if used_fallback { "yes" } else { "no" },
            filename,
        ));
    if relit {
        ui.label(
            RichText::new("relit (PS albedo + normals)")
                .small()
                .color(Color32::from_rgb(96, 189, 115)),
        );
    }
}
