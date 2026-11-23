//! Image canvas and picking. [`RegDisplay`] keeps images and overlays in the same coordinates.

use egui::{Color32, Pos2, Rect, Sense, StrokeKind, Vec2};
use fabric_data::tracking::registration::RegistrationTransform;
use fabric_data::tracking::TrackId;

use super::overlay::{nearest_overlay_track, paint_overlay};
use super::ImageViewerState;
use crate::settings::ComparisonMode;
use crate::tracked::{CanvasMode, OverlayPaintData};
use crate::ui::style;

/// A texture plus its source size, ready to paint.
#[derive(Debug, Clone)]
pub(super) struct PaintImage {
    pub texture: egui::TextureId,
    pub size: (u32, u32),
}

pub(super) struct Pane {
    pub image: Option<PaintImage>,
    pub title: String,
    pub message: Option<(String, bool)>,
    pub transform: RegistrationTransform,
    pub tiled: bool,
}

impl Pane {
    fn image_for_layout(&self) -> Option<PaintImage> {
        let mut image = self.image.clone()?;
        if self.tiled {
            image.size = (image.size.0 * 3, image.size.1 * 3);
        }
        Some(image)
    }
}

/// The fit viewport and visible clip for each configuration.
pub(super) fn pane_rects(
    mode: ComparisonMode,
    rect: Rect,
    count: usize,
    dividers: &[f32],
) -> Vec<(Rect, Rect)> {
    if count <= 1 || mode == ComparisonMode::Single {
        return vec![(rect, rect)];
    }
    if mode == ComparisonMode::SplitWipe {
        let x = |i: usize| {
            let fraction = if i == 0 {
                0.0
            } else if i == count {
                1.0
            } else {
                dividers
                    .get(i - 1)
                    .copied()
                    .unwrap_or(i as f32 / count as f32)
            };
            rect.min.x + rect.width() * fraction
        };
        return (0..count)
            .map(|i| {
                let clip = Rect::from_min_max(
                    Pos2::new(x(i), rect.min.y),
                    Pos2::new(x(i + 1), rect.max.y),
                );
                (rect, clip)
            })
            .collect();
    }
    let columns = count.min(((rect.width() / 230.0).floor() as usize).max(2));
    let rows = count.div_ceil(columns);
    let cell = rect.size() / Vec2::new(columns as f32, rows as f32);
    (0..count)
        .map(|i| {
            let origin = rect.min + Vec2::new((i % columns) as f32, (i / columns) as f32) * cell;
            let pane = Rect::from_min_size(origin, cell);
            (pane, pane)
        })
        .collect()
}

#[cfg(test)]
pub(super) fn comparison_rects(
    mode: ComparisonMode,
    show_secondary: bool,
    rect: Rect,
    wipe_pos: f32,
) -> (Rect, Rect) {
    let layout = pane_rects(mode, rect, if show_secondary { 2 } else { 1 }, &[wipe_pos]);
    (layout[0].0, layout.get(1).map_or(rect, |pane| pane.1))
}

/// Pure layout helper: fit scale of an image into a rect, preserving aspect.
pub(super) fn fit_scale(size: (u32, u32), fit_rect: Rect) -> f32 {
    let (w, h) = (size.0 as f32, size.1 as f32);
    if w <= 0.0 || h <= 0.0 || fit_rect.width() <= 0.0 || fit_rect.height() <= 0.0 {
        1.0
    } else {
        (fit_rect.width() / w).min(fit_rect.height() / h)
    }
}

/// The zoom in effect for `img` in `fit_rect`: the user's zoom, else Fit.
fn effective_scale(img: &PaintImage, fit_rect: Rect, state: &ImageViewerState) -> f32 {
    state.scale.unwrap_or_else(|| fit_scale(img.size, fit_rect))
}

fn image_rect_for(img: &PaintImage, fit_rect: Rect, state: &ImageViewerState) -> Rect {
    let scale = effective_scale(img, fit_rect, state);
    let disp = Vec2::new(img.size.0 as f32 * scale, img.size.1 as f32 * scale);
    Rect::from_center_size(fit_rect.center() + state.pan, disp)
}

fn full_uv() -> Rect {
    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0))
}

/// Source-image to screen transform shared by the image mesh, the overlay and
/// all diagnostics: `screen = rect.min + scale * T(image_point)`.
pub(super) struct RegDisplay {
    pub scale: f32,
    pub rect: Rect,
    pub transform: RegistrationTransform,
}

impl RegDisplay {
    /// Un-registered (raw) image point -> screen.
    pub fn raw(&self, p: &fabric_data::tracking::TrackedPoint) -> Pos2 {
        Pos2::new(
            self.rect.min.x + self.scale * p.x as f32,
            self.rect.min.y + self.scale * p.y as f32,
        )
    }

    /// Registered image point -> screen (applies the registration transform).
    pub fn reg(&self, p: &fabric_data::tracking::TrackedPoint) -> Pos2 {
        self.reg_xy(p.x, p.y)
    }

    pub fn reg_xy(&self, x: f64, y: f64) -> Pos2 {
        let (rx, ry) = self.transform.apply(x, y);
        Pos2::new(
            self.rect.min.x + self.scale * rx as f32,
            self.rect.min.y + self.scale * ry as f32,
        )
    }
}

/// Paint using `view`'s source-to-screen mapping, including registration rotation.
fn paint_registered(painter: &egui::Painter, img: &PaintImage, view: &RegDisplay, clip: Rect) {
    if view.transform.is_identity() {
        let min = view.reg_xy(0.0, 0.0);
        let max = view.reg_xy(img.size.0 as f64, img.size.1 as f64);
        let rect = Rect::from_min_max(min, max);
        painter
            .with_clip_rect(clip)
            .image(img.texture, rect, full_uv(), Color32::WHITE);
        return;
    }
    // Preserve original UVs while transforming quad corners.
    let (w, h) = (img.size.0 as f64, img.size.1 as f64);
    let corners = [
        (0.0, 0.0, 0.0, 0.0),
        (w, 0.0, 1.0, 0.0),
        (w, h, 1.0, 1.0),
        (0.0, h, 0.0, 1.0),
    ];
    let mut mesh = egui::Mesh::with_texture(img.texture);
    for (px, py, u, v) in corners {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: view.reg_xy(px, py),
            uv: Pos2::new(u as f32, v as f32),
            color: Color32::WHITE,
        });
    }
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    painter.with_clip_rect(clip).add(egui::Shape::mesh(mesh));
}

/// Pointer-anchored wheel zoom. Use the primary pane's fit basis for side-by-side layouts.
fn zoom_at_pointer(
    ui: &egui::Ui,
    response: &egui::Response,
    state: &mut ImageViewerState,
    primary: &PaintImage,
    primary_rect: Rect,
) {
    if !response.hovered() {
        return;
    }
    let (scroll_y, zoom_factor) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
    if zoom_factor == 1.0 && scroll_y == 0.0 {
        return;
    }
    let base = effective_scale(primary, primary_rect, state);
    let new_scale = (base * zoom_factor * (scroll_y * 0.0015).exp()).clamp(0.01, 64.0);
    if let Some(pointer) = response.hover_pos() {
        // Keep the image point under the cursor fixed while scaling.
        let image_point = (pointer - primary_rect.center() - state.pan) / base;
        state.pan = pointer - primary_rect.center() - image_point * new_scale;
    }
    state.scale = Some(new_scale);
}

/// A split divider moves within its neighbouring strips. Other drags pan the canvas.
fn drag(
    ui: &egui::Ui,
    response: &egui::Response,
    state: &mut ImageViewerState,
    rect: Rect,
    wipe_active: bool,
) {
    let nearest = |pos: Pos2| {
        state
            .wipe_dividers
            .iter()
            .enumerate()
            .map(|(i, fraction)| (i, (pos.x - rect.min.x - rect.width() * fraction).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|&(_, distance)| distance < 12.0)
            .map(|(index, _)| index)
    };
    if !wipe_active {
        state.wipe_dragging = None;
    } else {
        if response.hover_pos().and_then(nearest).is_some() || state.wipe_dragging.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
        if response.drag_started_by(egui::PointerButton::Primary) {
            state.wipe_dragging = ui.input(|i| i.pointer.press_origin()).and_then(nearest);
        }
    }
    if response.dragged_by(egui::PointerButton::Primary) {
        if let Some(i) = state.wipe_dragging {
            if let Some(pos) = response.interact_pointer_pos() {
                let before = i.checked_sub(1).map_or(0.0, |i| state.wipe_dividers[i]);
                let after = state.wipe_dividers.get(i + 1).copied().unwrap_or(1.0);
                let gap = (4.0 / rect.width()).min((after - before) / 2.0);
                state.wipe_dividers[i] =
                    ((pos.x - rect.min.x) / rect.width()).clamp(before + gap, after - gap);
            }
        } else {
            state.pan += response.drag_delta();
        }
    }
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        state.wipe_dragging = None;
    }
}

fn paint_tiles(painter: &egui::Painter, image: &PaintImage, view: &RegDisplay, clip: Rect) {
    let tile_size = view.rect.size() / 3.0;
    for row in 0..3 {
        for col in 0..3 {
            let origin = view.rect.min + Vec2::new(col as f32, row as f32) * tile_size;
            let tile = Rect::from_min_size(origin, tile_size);
            let tile_view = RegDisplay {
                scale: view.scale,
                rect: tile,
                transform: view.transform,
            };
            paint_registered(painter, image, &tile_view, clip.intersect(tile));
        }
    }
}

/// Draw the configurations and return the primary pane's hovered and clicked tracks.
pub(super) fn canvas(
    ui: &mut egui::Ui,
    state: &mut ImageViewerState,
    panes: &[Pane],
    comparison: ComparisonMode,
    overlay: Option<OverlayPaintData>,
) -> (Option<TrackId>, Option<TrackId>) {
    let (response, painter) = ui.allocate_painter(
        ui.available_size().max(Vec2::splat(80.0)),
        Sense::click_and_drag(),
    );
    let rect = response.rect;
    let painter = painter.with_clip_rect(rect);
    painter.rect_filled(rect, 4.0, Color32::from_gray(24));
    let split = comparison == ComparisonMode::SplitWipe && panes.len() > 1;
    if state.wipe_dividers.len() + 1 != panes.len() && split {
        state.wipe_dividers = (1..panes.len())
            .map(|i| i as f32 / panes.len() as f32)
            .collect();
        state.wipe_dragging = None;
    }
    drag(ui, &response, state, rect, split);
    let layout = pane_rects(comparison, rect, panes.len(), &state.wipe_dividers);
    let images: Vec<_> = panes.iter().map(Pane::image_for_layout).collect();
    let basis = response
        .hover_pos()
        .and_then(|pos| layout.iter().position(|(_, clip)| clip.contains(pos)))
        .filter(|&i| images[i].is_some())
        .or_else(|| images.iter().position(Option::is_some));
    if let Some(i) = basis {
        let zoom_basis = if split {
            images[0].as_ref().unwrap_or(images[i].as_ref().unwrap())
        } else {
            images[i].as_ref().unwrap()
        };
        zoom_at_pointer(ui, &response, state, zoom_basis, layout[i].0);
        let limit =
            image_rect_for(zoom_basis, layout[i].0, state).size() * 0.5 + layout[i].0.size() * 0.5;
        state.pan = state.pan.clamp(-limit, limit);
    }
    let shared_scale = if split {
        images
            .iter()
            .flatten()
            .next()
            .map(|img| effective_scale(img, rect, state))
    } else {
        None
    };
    let mut tracks = (None, None);
    for (i, ((pane, image), &(viewport, clip))) in
        panes.iter().zip(&images).zip(&layout).enumerate()
    {
        let pane_painter = painter.with_clip_rect(clip);
        if let Some((image, layout_image)) = pane.image.as_ref().zip(image.as_ref()) {
            let scale =
                shared_scale.unwrap_or_else(|| effective_scale(layout_image, viewport, state));
            let disp = Rect::from_center_size(
                viewport.center() + state.pan,
                Vec2::new(layout_image.size.0 as f32, layout_image.size.1 as f32) * scale,
            );
            let view = RegDisplay {
                scale,
                rect: disp,
                transform: pane.transform,
            };
            let pane_overlay = overlay.as_ref().filter(|_| i == 0);
            if pane_overlay.is_some_and(|o| o.flags.canvas_mode.clips_to_reference()) {
                pane_painter.rect_filled(disp, 0.0, super::overlay::COLOR_NEUTRAL_BG);
            }
            if pane.tiled {
                paint_tiles(&pane_painter, image, &view, clip);
            } else {
                paint_registered(&pane_painter, image, &view, clip);
            }
            if let Some(overlay) = pane_overlay {
                if overlay.flags.canvas_mode == CanvasMode::ValidBoundary {
                    pane_painter.rect_stroke(
                        disp,
                        0.0,
                        egui::Stroke::new(1.0, Color32::from_gray(120)),
                        StrokeKind::Inside,
                    );
                }
                let pickable = overlay.style.is_pickable();
                let pick = |pos: Pos2, radius| {
                    (pickable && clip.contains(pos))
                        .then(|| nearest_overlay_track(overlay, &view, pos, radius))
                        .flatten()
                };
                let hovered = response.hover_pos().and_then(|pos| pick(pos, 14.0));
                let clicked = if response.clicked() && state.wipe_dragging.is_none() {
                    response
                        .interact_pointer_pos()
                        .and_then(|pos| pick(pos, 16.0))
                } else {
                    None
                };
                paint_overlay(&pane_painter, &view, overlay, &hovered);
                tracks = (hovered, clicked);
            }
        }
        if let Some((message, warning)) = &pane.message {
            pane_painter.text(
                if pane.image.is_some() {
                    clip.right_top() + Vec2::new(-8.0, 6.0)
                } else {
                    clip.center()
                },
                if pane.image.is_some() {
                    egui::Align2::RIGHT_TOP
                } else {
                    egui::Align2::CENTER_CENTER
                },
                message,
                egui::FontId::proportional(12.0),
                if *warning {
                    style::ERROR
                } else {
                    Color32::from_gray(165)
                },
            );
        }
        if panes.len() > 1 {
            pane_painter.text(
                clip.left_top() + Vec2::new(6.0, 5.0),
                egui::Align2::LEFT_TOP,
                &pane.title,
                egui::FontId::proportional(12.0),
                Color32::from_gray(220),
            );
        }
        if split && i > 0 {
            painter.line_segment(
                [clip.left_top(), clip.left_bottom()],
                egui::Stroke::new(1.5, Color32::from_rgb(240, 170, 60)),
            );
        } else if panes.len() > 1 && !split {
            painter.rect_stroke(
                clip,
                0.0,
                egui::Stroke::new(1.0, Color32::from_gray(85)),
                StrokeKind::Inside,
            );
        }
    }
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, Color32::from_gray(90)),
        StrokeKind::Inside,
    );
    tracks
}
