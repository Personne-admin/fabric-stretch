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

/// What the canvas draws for one frame.
pub(super) struct CanvasInput<'a> {
    /// The image to paint; `None` shows `message` instead.
    pub primary: Option<&'a PaintImage>,
    /// Status text: the reason there is no image, or a "loading" note over a
    /// stale one.
    pub message: Option<(String, bool)>,
    /// The painted image belongs to an earlier frame (the new one is still
    /// decoding); the overlay is not drawn over it.
    pub stale: bool,
    pub secondary: Option<&'a PaintImage>,
    pub comparison: ComparisonMode,
    pub overlay: Option<OverlayPaintData>,
}

/// Pure layout helper: the drawing rects for primary and secondary under each
/// comparison mode. In Single mode both rects are the full canvas.
pub(super) fn comparison_rects(
    comparison: ComparisonMode,
    show_secondary: bool,
    rect: Rect,
    wipe_pos: f32,
) -> (Rect, Rect) {
    match (comparison, show_secondary) {
        (ComparisonMode::SideBySide, true) => {
            let mid = rect.min.x + rect.width() * 0.5;
            (
                Rect::from_min_max(rect.min, Pos2::new(mid, rect.max.y)),
                Rect::from_min_max(Pos2::new(mid, rect.min.y), rect.max),
            )
        }
        (ComparisonMode::SplitWipe, true) => (
            rect,
            Rect::from_min_max(
                Pos2::new(rect.min.x + rect.width() * wipe_pos, rect.min.y),
                rect.max,
            ),
        ),
        _ => (rect, rect),
    }
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

/// Primary-button drag: grabs the split-wipe divider when the press started
/// near it (and then only moves the divider), otherwise pans the image.
fn drag(
    ui: &egui::Ui,
    response: &egui::Response,
    state: &mut ImageViewerState,
    rect: Rect,
    wipe_active: bool,
) {
    const GRAB_RADIUS: f32 = 12.0;
    if !wipe_active {
        state.wipe_dragging = false;
    } else if response.drag_started_by(egui::PointerButton::Primary) {
        let divider_x = rect.min.x + rect.width() * state.wipe_pos;
        state.wipe_dragging = ui
            .input(|i| i.pointer.press_origin())
            .is_some_and(|pos| (pos.x - divider_x).abs() < GRAB_RADIUS);
    }
    if response.dragged_by(egui::PointerButton::Primary) {
        if state.wipe_dragging {
            if let Some(pos) = response.interact_pointer_pos() {
                state.wipe_pos = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
            }
        } else {
            state.pan += response.drag_delta();
        }
    }
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        state.wipe_dragging = false;
    }
}

/// Draw the canvas; return `(hovered_track, clicked_track)`.
pub(super) fn canvas(
    ui: &mut egui::Ui,
    state: &mut ImageViewerState,
    input: CanvasInput<'_>,
) -> (Option<TrackId>, Option<TrackId>) {
    let CanvasInput {
        primary,
        message,
        stale,
        secondary,
        comparison,
        overlay,
    } = input;
    let overlay = overlay.filter(|_| !stale);

    let avail = ui.available_size();
    let (response, painter) =
        ui.allocate_painter(avail.max(Vec2::splat(80.0)), Sense::click_and_drag());
    let rect = response.rect;
    painter.rect_filled(rect, 4.0, Color32::from_gray(24));
    let painter = painter.with_clip_rect(rect);

    // Centred when there is no image at all, a small corner badge over a
    // stale one.
    let status_text = |painter: &egui::Painter, corner: bool| {
        if let Some((msg, warn)) = &message {
            let (pos, align, size) = if corner {
                (
                    rect.right_top() + Vec2::new(-8.0, 6.0),
                    egui::Align2::RIGHT_TOP,
                    12.0,
                )
            } else {
                (rect.center(), egui::Align2::CENTER_CENTER, 14.0)
            };
            painter.text(
                pos,
                align,
                msg,
                egui::FontId::proportional(size),
                if *warn {
                    style::ERROR
                } else {
                    Color32::from_gray(150)
                },
            );
        }
    };
    let Some(primary) = primary else {
        status_text(&painter, false);
        return (None, None);
    };

    let transform = overlay
        .as_ref()
        .map(|o| o.transform)
        .unwrap_or(RegistrationTransform::IDENTITY);

    let show_secondary = comparison != ComparisonMode::Single && secondary.is_some();
    let (primary_rect, secondary_rect) =
        comparison_rects(comparison, show_secondary, rect, state.wipe_pos);

    zoom_at_pointer(ui, &response, state, primary, primary_rect);
    drag(
        ui,
        &response,
        state,
        rect,
        comparison == ComparisonMode::SplitWipe && show_secondary,
    );

    // Bound pan so some image remains visible.
    let primary_disp = image_rect_for(primary, primary_rect, state);
    let max_pan = primary_disp.size() * 0.5 + primary_rect.size() * 0.5;
    state.pan = state.pan.clamp(-max_pan, max_pan);
    let primary_disp = image_rect_for(primary, primary_rect, state);

    let primary_scale = effective_scale(primary, primary_rect, state);
    let primary_view = RegDisplay {
        scale: primary_scale,
        rect: primary_disp,
        transform,
    };
    let primary_region = if comparison == ComparisonMode::SideBySide && show_secondary {
        primary_rect
    } else {
        rect
    };

    // Paint neutral backgrounds before images so they cannot cover the comparison pane.
    // Clip registered images to reference bounds.
    let clips_to_reference = overlay
        .as_ref()
        .is_some_and(|o| o.flags.canvas_mode.clips_to_reference());

    let primary_clip = if comparison == ComparisonMode::SplitWipe && show_secondary {
        Rect::from_min_max(
            rect.min,
            Pos2::new(rect.min.x + rect.width() * state.wipe_pos, rect.max.y),
        )
    } else {
        primary_region
    };

    if clips_to_reference {
        painter.with_clip_rect(primary_clip).rect_filled(
            primary_disp,
            0.0,
            super::overlay::COLOR_NEUTRAL_BG,
        );
    }

    if let Some(secondary) = secondary.filter(|_| show_secondary) {
        let (sec_disp, sec_scale, sec_clip) = if comparison == ComparisonMode::SplitWipe {
            let right = Rect::from_min_max(
                Pos2::new(rect.min.x + rect.width() * state.wipe_pos, rect.min.y),
                rect.max,
            );
            (primary_disp, primary_scale, right)
        } else {
            let d = image_rect_for(secondary, secondary_rect, state);
            let s = effective_scale(secondary, secondary_rect, state);
            let clip = if comparison == ComparisonMode::SideBySide {
                secondary_rect
            } else {
                rect
            };
            (d, s, clip)
        };
        let sec_view = RegDisplay {
            scale: sec_scale,
            rect: sec_disp,
            transform,
        };
        if clips_to_reference {
            painter.with_clip_rect(sec_clip).rect_filled(
                sec_disp,
                0.0,
                super::overlay::COLOR_NEUTRAL_BG,
            );
        }
        paint_registered(&painter, secondary, &sec_view, sec_clip);
    }

    paint_registered(&painter, primary, &primary_view, primary_clip);

    if comparison == ComparisonMode::SplitWipe && show_secondary {
        let x = rect.min.x + rect.width() * state.wipe_pos;
        painter.line_segment(
            [Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)],
            egui::Stroke::new(1.5, Color32::from_rgb(240, 170, 60)),
        );
    }

    if show_secondary {
        let label = |pos: Pos2, text: &str| {
            painter.text(
                pos,
                egui::Align2::LEFT_TOP,
                text,
                egui::FontId::proportional(11.0),
                Color32::from_gray(200),
            );
        };
        label(Pos2::new(rect.min.x + 6.0, rect.min.y + 4.0), "primary");
        if comparison == ComparisonMode::SideBySide {
            label(
                Pos2::new(rect.center().x + 6.0, rect.min.y + 4.0),
                "secondary",
            );
        }
    }

    if let Some(overlay) = &overlay {
        if overlay.flags.canvas_mode == CanvasMode::ValidBoundary {
            painter.rect_stroke(
                primary_disp,
                0.0,
                egui::Stroke::new(1.0, Color32::from_gray(120)),
                StrokeKind::Inside,
            );
        }
    }

    let (hovered, clicked) = if let Some(overlay) = &overlay {
        let mut hovered = None;
        // Pick only visible points in the primary pane.
        let pointer_in_primary = overlay.style.is_pickable()
            && response
                .hover_pos()
                .is_some_and(|pos| primary_region.contains(pos));
        if pointer_in_primary {
            if let Some(pos) = response.hover_pos() {
                hovered = nearest_overlay_track(overlay, &primary_view, pos, 14.0);
            }
        }
        // Clip the overlay to the primary pane.
        let overlay_painter = painter.with_clip_rect(primary_region);
        paint_overlay(&overlay_painter, &primary_view, overlay, &hovered);

        let clicked = if response.clicked() && !state.wipe_dragging && overlay.style.is_pickable() {
            response.interact_pointer_pos().and_then(|pos| {
                if primary_region.contains(pos) {
                    nearest_overlay_track(overlay, &primary_view, pos, 16.0)
                } else {
                    None
                }
            })
        } else {
            None
        };
        (hovered, clicked)
    } else {
        (None, None)
    };

    if stale {
        status_text(&painter, true);
    }
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, Color32::from_gray(90)),
        StrokeKind::Inside,
    );

    (hovered, clicked)
}
