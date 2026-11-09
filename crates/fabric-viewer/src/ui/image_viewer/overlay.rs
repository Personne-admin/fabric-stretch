//! Tracked-point overlay for the canvas. CPU video export shares these colours and paint inputs.

use std::path::Path;

use egui::{Color32, Pos2, Vec2};
use fabric_data::tracking::registration::RegistrationTransform;
use fabric_data::tracking::stabilize::Stabilization;
use fabric_data::tracking::{FrameId, TrackId, TrackedPoint, TrackingSequence};

use super::canvas::RegDisplay;
use crate::tracked::{
    OverlayPaintData, OverlayStyle, OverlayTrack, TrackScope, TrackedRunState, TrackingSession,
};

/// Image transform and overlay shared by the canvas and video export.
pub(crate) fn frame_render_inputs(
    tracking: &TrackingSession,
    run_key: &Path,
    frame_id: FrameId,
) -> (RegistrationTransform, Option<OverlayPaintData>) {
    // Use raw coordinates if stabilization is off or the frame has no anchors.
    let stabilization: Option<Stabilization> = if tracking.stabilize.enabled {
        tracking
            .run_state(run_key)
            .and_then(|r| tracking.stabilization_for(r, frame_id))
    } else {
        None
    };
    // Share one registration result across images and overlay.
    let reg_transform: RegistrationTransform = tracking
        .run_state(run_key)
        .and_then(|r| tracking.result_for(r, frame_id))
        .map(|res| res.to_reference())
        .unwrap_or(RegistrationTransform::IDENTITY);
    // The stabilization offset replaces the registration transform while the mode is on.
    let transform = match &stabilization {
        Some(stab) => stab.transform,
        None => reg_transform,
    };
    let overlay = tracking
        .run_state(run_key)
        .and_then(|r| build_overlay(tracking, r, frame_id, transform, tracking.overlay_style()));
    (transform, overlay)
}

/// Build overlay data, or `None` without tracks for this frame.
/// The simple style skips trajectory and reference-history work.
pub(crate) fn build_overlay(
    tracking: &TrackingSession,
    run: &TrackedRunState,
    frame_id: FrameId,
    transform: RegistrationTransform,
    style: OverlayStyle,
) -> Option<OverlayPaintData> {
    let sequence = run.sequence.as_ref()?;
    let full = style == OverlayStyle::Full;
    let reference = tracking.reference_for(run);
    let overlay = &tracking.overlay;

    let points = tracking.points_in(run, frame_id);
    if points.is_empty() {
        return None;
    }
    let mut tracks = Vec::new();
    for (id, point) in points {
        let is_reference = tracking.is_reference_track(run, id);
        // Simple points ignore the legacy reference-only filter.
        let in_scope = !full
            || match overlay.scope {
                TrackScope::AllValid => true,
                TrackScope::ReferenceOnly => is_reference,
            };
        if !in_scope {
            continue;
        }
        let reference_pos = reference.as_ref().and_then(|r| r.get(id)).copied();
        let history = if full {
            history_for(
                sequence,
                frame_id,
                id,
                run.state.min_confidence,
                overlay.history_length,
            )
        } else {
            Vec::new()
        };
        tracks.push(OverlayTrack {
            id: id.clone(),
            current: *point,
            reference: reference_pos,
            is_reference,
            history,
        });
    }
    if tracks.is_empty() {
        return None;
    }
    let reference_points = if full {
        reference.map(|r| r.points().to_vec())
    } else {
        None
    };
    Some(OverlayPaintData {
        tracks,
        reference_points,
        selected_track: tracking.selected_track.clone(),
        flags: overlay.clone(),
        style,
        transform,
    })
}

/// Up to `len` previous valid positions of `track` (image space), in frame
/// order, for trajectory history.
pub(super) fn history_for(
    sequence: &TrackingSequence,
    current_frame_id: FrameId,
    track: &TrackId,
    min_confidence: f64,
    len: usize,
) -> Vec<TrackedPoint> {
    let Some(idx) = sequence.frame_index_of(&current_frame_id) else {
        return Vec::new();
    };
    let start = idx.saturating_sub(len);
    (start..idx)
        .filter_map(|i| {
            sequence
                .frames()
                .get(i)
                .and_then(|f| f.get(track))
                .filter(|p| p.is_valid(min_confidence))
                .copied()
        })
        .collect()
}

pub(crate) const COLOR_CURRENT: Color32 = Color32::from_rgb(255, 200, 40); // amber
pub(crate) const COLOR_RAW: Color32 = Color32::from_rgb(220, 90, 90); // dim red
pub(crate) const COLOR_TRANSFORMED: Color32 = Color32::from_rgb(255, 240, 150); // hollow
pub(crate) const COLOR_GHOST: Color32 = Color32::from_rgb(80, 190, 255); // cyan
pub(crate) const COLOR_DISP: Color32 = Color32::from_rgb(120, 230, 150); // green
pub(crate) const COLOR_LINE: Color32 = Color32::from_rgb(150, 150, 160); // gray
pub(crate) const COLOR_TRAJ: Color32 = Color32::from_rgb(120, 120, 130);
pub(crate) const COLOR_REF_BADGE: Color32 = Color32::from_rgb(230, 140, 255); // magenta
const COLOR_SELECTED: Color32 = Color32::WHITE;
const COLOR_HOVER: Color32 = Color32::from_rgb(110, 220, 255); // light blue
pub(crate) const COLOR_NEUTRAL_BG: Color32 = Color32::from_gray(34);
pub(crate) const COLOR_LOW_CONF: Color32 = Color32::from_rgb(210, 110, 110);

/// A small arrowhead line from `from` to `to`. Degenerate (zero-length)
/// segments draw nothing rather than producing NaN geometry.
fn arrow(painter: &egui::Painter, from: Pos2, to: Pos2, color: Color32) {
    if from == to {
        return;
    }
    painter.line_segment([from, to], egui::Stroke::new(1.5, color));
    let dir = (to - from).normalized();
    let normal = egui::vec2(-dir.y, dir.x);
    let tip = to;
    let base = to - dir * 8.0;
    let wing = 4.5;
    let p1 = base + normal * wing;
    let p2 = base - normal * wing;
    painter.add(egui::Shape::convex_polygon(
        vec![p1, p2, tip],
        color,
        egui::Stroke::NONE,
    ));
}

/// Draw low-confidence points hollow, using the stabilization confidence threshold.
const LOW_CONFIDENCE: f64 = fabric_data::tracking::stabilize::ANCHOR_MIN_CONFIDENCE;

/// Dots at displayed track positions, hollow for low confidence.
/// No history or diagnostic vectors.
fn paint_tracked_points(
    painter: &egui::Painter,
    view: &RegDisplay,
    overlay: &OverlayPaintData,
    hovered: &Option<TrackId>,
) {
    for track in &overlay.tracks {
        let pos = view.reg(&track.current);
        let confident = track.current.confidence.is_none_or(|c| c >= LOW_CONFIDENCE);
        if confident {
            painter.circle_filled(pos, 2.4, COLOR_CURRENT);
        } else {
            painter.circle_stroke(pos, 2.6, egui::Stroke::new(1.0, COLOR_LOW_CONF));
        }
        let is_selected = overlay.selected_track.as_ref() == Some(&track.id);
        let is_hovered = hovered.as_ref() == Some(&track.id);
        if is_hovered || is_selected {
            painter.circle_stroke(
                pos,
                5.5,
                egui::Stroke::new(
                    1.2,
                    if is_hovered {
                        COLOR_HOVER
                    } else {
                        COLOR_SELECTED
                    },
                ),
            );
        }
    }
}

pub(super) fn paint_overlay(
    painter: &egui::Painter,
    view: &RegDisplay,
    overlay: &OverlayPaintData,
    hovered: &Option<TrackId>,
) {
    match overlay.style {
        OverlayStyle::Hidden => return,
        OverlayStyle::Points => {
            paint_tracked_points(painter, view, overlay, hovered);
            return;
        }
        OverlayStyle::Full => {}
    }
    let flags = &overlay.flags;

    if flags.show_trajectories && flags.history_length > 0 {
        for track in &overlay.tracks {
            if track.history.is_empty() {
                continue;
            }
            let mut pts: Vec<Pos2> = track.history.iter().map(|p| view.reg(p)).collect();
            pts.push(view.reg(&track.current));
            if pts.len() >= 2 {
                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0, COLOR_TRAJ)));
            }
        }
    }

    // Reference ghost points (the reference frame's targets, drawn in the
    // reference frame's own space).
    if flags.show_reference_ghost {
        if let Some(reference) = &overlay.reference_points {
            for (_, p) in reference {
                painter.circle_stroke(view.raw(p), 4.0, egui::Stroke::new(1.2, COLOR_GHOST));
            }
        }
    }

    if flags.show_displacement_vectors || flags.show_displacement_lines {
        for track in &overlay.tracks {
            let Some(ref_p) = &track.reference else {
                continue;
            };
            let from = view.raw(ref_p);
            let to = view.reg(&track.current);
            if flags.show_displacement_vectors {
                arrow(painter, from, to, COLOR_DISP);
            } else {
                painter.line_segment([from, to], egui::Stroke::new(1.0, COLOR_LINE));
            }
        }
    }

    if flags.show_reference_locations {
        for track in &overlay.tracks {
            if let Some(ref_p) = &track.reference {
                let pos = view.raw(ref_p);
                painter.line_segment(
                    [pos - Vec2::new(4.0, 0.0), pos + Vec2::new(4.0, 0.0)],
                    egui::Stroke::new(1.0, COLOR_GHOST),
                );
                painter.line_segment(
                    [pos - Vec2::new(0.0, 4.0), pos + Vec2::new(0.0, 4.0)],
                    egui::Stroke::new(1.0, COLOR_GHOST),
                );
            }
        }
    }

    if flags.show_raw_points {
        for track in &overlay.tracks {
            painter.circle_filled(view.raw(&track.current), 3.0, COLOR_RAW);
        }
    }

    if flags.show_transformed_points {
        for track in &overlay.tracks {
            painter.circle_stroke(
                view.reg(&track.current),
                5.0,
                egui::Stroke::new(1.0, COLOR_TRANSFORMED),
            );
        }
    }

    if flags.show_points {
        for track in &overlay.tracks {
            let pos = view.reg(&track.current);
            let is_selected = overlay.selected_track.as_ref() == Some(&track.id);
            let is_hovered = hovered.as_ref() == Some(&track.id);
            let color = if is_hovered {
                COLOR_HOVER
            } else if is_selected {
                COLOR_SELECTED
            } else {
                COLOR_CURRENT
            };
            if track.is_reference {
                // Reference tracks are visually distinct (magenta badge).
                painter.circle_stroke(pos, 6.5, egui::Stroke::new(1.2, COLOR_REF_BADGE));
            }
            painter.circle_filled(pos, 3.2, color);
            if is_hovered {
                painter.circle_stroke(pos, 6.0, egui::Stroke::new(1.5, COLOR_HOVER));
            }
        }
    }

    if flags.show_ids {
        let font = egui::FontId::proportional(10.0);
        for track in &overlay.tracks {
            let pos = view.reg(&track.current);
            painter.text(
                pos + Vec2::new(7.0, -7.0),
                egui::Align2::LEFT_TOP,
                track.id.to_string(),
                font.clone(),
                Color32::from_gray(220),
            );
        }
    }
}

/// Nearest overlay track to `pos` in screen space, within `radius_px`.
pub(super) fn nearest_overlay_track(
    overlay: &OverlayPaintData,
    view: &RegDisplay,
    pos: Pos2,
    radius_px: f32,
) -> Option<TrackId> {
    let mut best: Option<(f32, &TrackId)> = None;
    for track in &overlay.tracks {
        let p = view.reg(&track.current);
        let d = p.distance(pos);
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, &track.id));
        }
    }
    best.filter(|(d, _)| *d <= radius_px)
        .map(|(_, id)| id.clone())
}
