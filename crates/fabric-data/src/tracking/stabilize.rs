//! Hold the upper tracked band still during replay. Anchoring all points would compensate
//! away part of the actual fabric extension.
//!
//! Choose anchors by reference y within [`ANCHOR_BAND_FRACTION`] of the tracked extent,
//! not the whole image. Negate their median displacement for a translation-only render offset.
//! Source coordinates and measurements remain unchanged.

use super::registration::{median, RegistrationTransform};
use super::{FrameId, TrackId, TrackedPoint, TrackingFrame, TrackingSequence};

/// Fraction of the reference frame's tracked y extent that forms the upper
/// anchor band (top 25 %).
pub const ANCHOR_BAND_FRACTION: f64 = 0.25;

/// Anchor confidence threshold. If too few points pass, retry with finite coordinates only.
pub const ANCHOR_MIN_CONFIDENCE: f64 = 0.5;

/// Fewest anchor tracks a band is allowed to have before it is widened.
pub const MIN_ANCHOR_TRACKS: usize = 3;

/// The upper tracked band of the reference frame: the positions the
/// stabilized view holds still.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorBand {
    reference_frame: FrameId,
    points: Vec<(TrackId, TrackedPoint)>,
    band_max_y: f64,
    widened: bool,
    ignored_confidence: bool,
}

impl AnchorBand {
    /// Choose anchors in the top [`ANCHOR_BAND_FRACTION`] of the tracked extent.
    /// Drop confidence filtering if needed (`ignored_confidence`); widen to the uppermost
    /// `max(MIN_ANCHOR_TRACKS, 25%)` points if the band is sparse (`widened`).
    /// Return `None` without any usable points.
    pub fn from_reference(reference: &TrackingFrame) -> Option<AnchorBand> {
        let confident: Vec<(TrackId, TrackedPoint)> = reference
            .valid_points(ANCHOR_MIN_CONFIDENCE)
            .map(|(id, p)| (id.clone(), *p))
            .collect();
        let (mut candidates, ignored_confidence) = if confident.len() >= MIN_ANCHOR_TRACKS {
            (confident, false)
        } else {
            let all: Vec<(TrackId, TrackedPoint)> = reference
                .iter()
                .filter(|(_, p)| p.coords_valid())
                .map(|(id, p)| (id.clone(), *p))
                .collect();
            // Only call it a fallback when it actually adds points.
            let ignored = all.len() > confident.len();
            if all.is_empty() {
                return None;
            }
            (all, ignored)
        };
        if candidates.is_empty() {
            return None;
        }

        let min_y = candidates
            .iter()
            .map(|(_, p)| p.y)
            .fold(f64::INFINITY, f64::min);
        let max_y = candidates
            .iter()
            .map(|(_, p)| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        let band_max_y = min_y + (max_y - min_y) * ANCHOR_BAND_FRACTION;
        let in_band: Vec<(TrackId, TrackedPoint)> = candidates
            .iter()
            .filter(|(_, p)| p.y <= band_max_y)
            .cloned()
            .collect();
        if in_band.len() >= MIN_ANCHOR_TRACKS || in_band.len() == candidates.len() {
            return Some(AnchorBand {
                reference_frame: reference.frame_id(),
                band_max_y,
                points: in_band,
                widened: false,
                ignored_confidence,
            });
        }

        // Thin band: take the uppermost points instead of giving up.
        candidates.sort_by(|a, b| {
            a.1.y
                .partial_cmp(&b.1.y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let quarter = (candidates.len() as f64 * ANCHOR_BAND_FRACTION).ceil() as usize;
        let take = candidates.len().min(MIN_ANCHOR_TRACKS.max(quarter));
        let points: Vec<(TrackId, TrackedPoint)> = candidates.into_iter().take(take).collect();
        let band_max_y = points.last().map_or(band_max_y, |(_, p)| p.y);
        Some(AnchorBand {
            reference_frame: reference.frame_id(),
            points,
            band_max_y,
            widened: true,
            ignored_confidence,
        })
    }

    /// The frame the anchor positions come from.
    pub fn reference_frame(&self) -> FrameId {
        self.reference_frame
    }

    /// The anchor `(track, reference position)` pairs.
    pub fn points(&self) -> &[(TrackId, TrackedPoint)] {
        &self.points
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Lower y bound of the anchor band in image pixels (points at or above
    /// this line anchor the view).
    pub fn band_max_y(&self) -> f64 {
        self.band_max_y
    }

    /// `true` when the top-fraction band was too thin and the uppermost
    /// points were used instead.
    pub fn widened(&self) -> bool {
        self.widened
    }

    /// `true` when the confidence threshold had to be dropped to find
    /// anchors.
    pub fn ignored_confidence(&self) -> bool {
        self.ignored_confidence
    }
}

/// One frame's stabilization: the measured anchor displacement and the
/// render-time transform that cancels it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stabilization {
    pub reference_frame: FrameId,
    pub frame: FrameId,
    /// The transform to apply when rendering: `Translation(-dx, -dy)`, or
    /// `Identity` when there is nothing to compensate.
    pub transform: RegistrationTransform,
    /// Median x displacement of the anchor tracks, current minus reference.
    pub anchor_dx: f64,
    /// Median y displacement of the anchor tracks, current minus reference.
    pub anchor_dy: f64,
    /// Anchor tracks that were valid in both frames and entered the median.
    pub anchor_pairs: usize,
    /// Anchor tracks available on the reference frame.
    pub anchor_total: usize,
    /// Median absolute deviation of the anchor displacements from the median
    /// displacement, in pixels: how rigidly the band actually moved.
    pub anchor_spread_px: f64,
}

impl Stabilization {
    /// The "no compensation possible" result for a frame.
    fn none(reference_frame: FrameId, frame: FrameId, anchor_total: usize) -> Self {
        Stabilization {
            reference_frame,
            frame,
            transform: RegistrationTransform::IDENTITY,
            anchor_dx: 0.0,
            anchor_dy: 0.0,
            anchor_pairs: 0,
            anchor_total,
            anchor_spread_px: 0.0,
        }
    }

    /// The render offset in image pixels (the inverse anchor displacement).
    pub fn offset(&self) -> (f64, f64) {
        self.transform.translation()
    }

    /// Whether the offset is identity (reference frame or no usable anchors).
    pub fn is_identity(&self) -> bool {
        self.transform.is_identity()
    }

    /// `true` when an anchor displacement was actually measured.
    pub fn is_estimated(&self) -> bool {
        self.anchor_pairs > 0
    }
}

/// Estimate inverse anchor displacement. The reference frame uses identity.
pub fn stabilize_frame(anchors: &AnchorBand, current: &TrackingFrame) -> Stabilization {
    let reference_frame = anchors.reference_frame();
    let frame = current.frame_id();
    let anchor_total = anchors.len();

    let pairs = anchor_pairs(anchors, current, ANCHOR_MIN_CONFIDENCE);
    let pairs = if pairs.len() < MIN_ANCHOR_TRACKS {
        // Retry without confidence filtering, matching anchor-band fallback.
        let relaxed = anchor_pairs(anchors, current, f64::NEG_INFINITY);
        if relaxed.len() > pairs.len() {
            relaxed
        } else {
            pairs
        }
    } else {
        pairs
    };
    if pairs.is_empty() {
        return Stabilization::none(reference_frame, frame, anchor_total);
    }

    let mut dxs: Vec<f64> = pairs.iter().map(|(dx, _)| *dx).collect();
    let mut dys: Vec<f64> = pairs.iter().map(|(_, dy)| *dy).collect();
    let dx = median(&mut dxs);
    let dy = median(&mut dys);
    if !dx.is_finite() || !dy.is_finite() {
        return Stabilization::none(reference_frame, frame, anchor_total);
    }
    let mut deviations: Vec<f64> = pairs
        .iter()
        .map(|(px, py)| ((px - dx).powi(2) + (py - dy).powi(2)).sqrt())
        .collect();
    let spread = median(&mut deviations);

    // Keep reference frames on the identity render path.
    let transform = if dx == 0.0 && dy == 0.0 {
        RegistrationTransform::IDENTITY
    } else {
        RegistrationTransform::Translation { tx: -dx, ty: -dy }
    };
    Stabilization {
        reference_frame,
        frame,
        transform,
        anchor_dx: dx,
        anchor_dy: dy,
        anchor_pairs: pairs.len(),
        anchor_total,
        anchor_spread_px: if spread.is_finite() { spread } else { 0.0 },
    }
}

/// Build anchors and stabilize the requested frame. `None` for missing frames or unusable reference.
pub fn stabilization_for_frame(
    sequence: &TrackingSequence,
    reference_frame: FrameId,
    current_frame: FrameId,
) -> Option<Stabilization> {
    let reference = sequence.get_frame(&reference_frame)?;
    let current = sequence.get_frame(&current_frame)?;
    let anchors = AnchorBand::from_reference(reference)?;
    Some(stabilize_frame(&anchors, current))
}

/// `(dx, dy)` of every anchor track that is valid in `current` at
/// `min_confidence`.
fn anchor_pairs(
    anchors: &AnchorBand,
    current: &TrackingFrame,
    min_confidence: f64,
) -> Vec<(f64, f64)> {
    anchors
        .points()
        .iter()
        .filter_map(|(id, reference)| {
            let point = current.get(id)?;
            if !point.is_valid(min_confidence) || !reference.coords_valid() {
                return None;
            }
            Some((point.x - reference.x, point.y - reference.y))
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn point(x: f64, y: f64) -> TrackedPoint {
        TrackedPoint::with_confidence(x, y, 1.0)
    }

    fn frame(id: u32, pts: &[(&str, f64, f64)]) -> TrackingFrame {
        TrackingFrame::new(
            FrameId(id),
            pts.iter()
                .map(|(n, x, y)| (TrackId::new(*n), point(*x, *y))),
        )
    }

    /// A 3-column, 4-row grid (rows at y = 0, 100, 200, 300).
    fn grid(id: u32, shift: (f64, f64), stretch_per_row: f64) -> TrackingFrame {
        let mut pts: Vec<(TrackId, TrackedPoint)> = Vec::new();
        for row in 0..4 {
            for col in 0..3 {
                let x = col as f64 * 50.0 + shift.0;
                let y = row as f64 * 100.0 + shift.1 + row as f64 * stretch_per_row;
                pts.push((TrackId::new(format!("r{row}c{col}")), point(x, y)));
            }
        }
        TrackingFrame::new(FrameId(id), pts)
    }

    fn band_of(frame: &TrackingFrame) -> AnchorBand {
        AnchorBand::from_reference(frame).expect("reference frame has points")
    }

    fn ids(band: &AnchorBand) -> Vec<String> {
        let mut names: Vec<String> = band.points().iter().map(|(id, _)| id.to_string()).collect();
        names.sort();
        names
    }

    #[test]
    fn anchor_band_is_the_upper_quarter_of_the_tracked_extent() {
        // Rows at y = 0, 100, 200, 300 -> band_max_y = 0 + 0.25 * 300 = 75,
        // so only the top row (y = 0) anchors the view.
        let band = band_of(&grid(0, (0.0, 0.0), 0.0));
        assert!(
            (band.band_max_y() - 75.0).abs() < TOL,
            "{}",
            band.band_max_y()
        );
        assert_eq!(band.len(), 3);
        assert_eq!(ids(&band), vec!["r0c0", "r0c1", "r0c2"]);
        assert!(!band.widened());
        assert!(!band.ignored_confidence());
        assert_eq!(band.reference_frame(), FrameId(0));
    }

    #[test]
    fn anchor_band_offset_does_not_change_which_tracks_anchor() {
        // Use the tracked extent, even when the grid starts far below image y = 0.
        let band = band_of(&grid(0, (10.0, 600.0), 0.0));
        assert_eq!(ids(&band), vec!["r0c0", "r0c1", "r0c2"]);
        assert!((band.band_max_y() - 675.0).abs() < TOL);
    }

    #[test]
    fn thin_band_widens_to_the_uppermost_points_instead_of_failing() {
        // One lone point at the top: the 25 % band holds a single track, so
        // the uppermost max(3, 25 %) points are used instead.
        let f = frame(
            0,
            &[
                ("top", 0.0, 0.0),
                ("a", 0.0, 900.0),
                ("b", 10.0, 910.0),
                ("c", 20.0, 920.0),
                ("d", 30.0, 1000.0),
            ],
        );
        let band = band_of(&f);
        assert!(band.widened());
        assert_eq!(band.len(), MIN_ANCHOR_TRACKS);
        assert_eq!(ids(&band), vec!["a", "b", "top"]);
    }

    #[test]
    fn single_row_of_tracks_anchors_on_itself() {
        let f = frame(0, &[("a", 0.0, 5.0), ("b", 10.0, 5.0), ("c", 20.0, 5.0)]);
        let band = band_of(&f);
        assert_eq!(band.len(), 3, "a degenerate extent keeps every track");
        assert!(!band.widened());
    }

    #[test]
    fn low_confidence_points_are_kept_out_of_the_band() {
        let f = TrackingFrame::new(
            FrameId(0),
            vec![
                (
                    TrackId::new("a"),
                    TrackedPoint::with_confidence(0.0, 0.0, 0.9),
                ),
                (
                    TrackId::new("b"),
                    TrackedPoint::with_confidence(10.0, 0.0, 0.9),
                ),
                (
                    TrackId::new("c"),
                    TrackedPoint::with_confidence(20.0, 0.0, 0.9),
                ),
                // Low confidence, inside the band geometrically.
                (
                    TrackId::new("bad"),
                    TrackedPoint::with_confidence(30.0, 1.0, 0.01),
                ),
                (
                    TrackId::new("low"),
                    TrackedPoint::with_confidence(0.0, 300.0, 0.9),
                ),
            ],
        );
        let band = band_of(&f);
        assert_eq!(ids(&band), vec!["a", "b", "c"]);
        assert!(!band.ignored_confidence());
    }

    #[test]
    fn all_low_confidence_falls_back_to_finite_coordinates() {
        // Unusually scaled confidence scores should fall back to finite points.
        let f = TrackingFrame::new(
            FrameId(0),
            vec![
                (
                    TrackId::new("a"),
                    TrackedPoint::with_confidence(0.0, 0.0, 0.2),
                ),
                (
                    TrackId::new("b"),
                    TrackedPoint::with_confidence(10.0, 0.0, 0.2),
                ),
                (
                    TrackId::new("c"),
                    TrackedPoint::with_confidence(20.0, 0.0, 0.2),
                ),
                (
                    TrackId::new("d"),
                    TrackedPoint::with_confidence(0.0, 300.0, 0.2),
                ),
            ],
        );
        let band = band_of(&f);
        assert!(band.ignored_confidence());
        assert_eq!(ids(&band), vec!["a", "b", "c"]);
    }

    #[test]
    fn frame_without_usable_points_has_no_band() {
        let f = TrackingFrame::new(
            FrameId(0),
            vec![(TrackId::new("a"), TrackedPoint::new(f64::NAN, 1.0))],
        );
        assert!(AnchorBand::from_reference(&f).is_none());
        assert!(AnchorBand::from_reference(&TrackingFrame::new(FrameId(0), vec![])).is_none());
    }

    #[test]
    fn reference_frame_is_rendered_unchanged() {
        let reference = grid(0, (0.0, 0.0), 0.0);
        let stab = stabilize_frame(&band_of(&reference), &reference);
        assert!(stab.is_identity(), "the reference frame is never offset");
        assert_eq!(stab.offset(), (0.0, 0.0));
        assert_eq!(stab.anchor_dx, 0.0);
        assert_eq!(stab.anchor_dy, 0.0);
        assert_eq!(stab.anchor_pairs, 3);
        assert_eq!(stab.anchor_total, 3);
        assert!(stab.is_estimated());
    }

    #[test]
    fn transform_is_the_negated_anchor_median_displacement() {
        let reference = grid(0, (0.0, 0.0), 0.0);
        // Whole specimen moved (+31, +12); the lower rows additionally stretch
        // 20 px per row downwards.
        let current = grid(1, (31.0, 12.0), 20.0);
        let stab = stabilize_frame(&band_of(&reference), &current);
        assert!((stab.anchor_dx - 31.0).abs() < TOL);
        assert!((stab.anchor_dy - 12.0).abs() < TOL);
        let (ox, oy) = stab.offset();
        assert!((ox + 31.0).abs() < TOL, "dx offset {ox}");
        assert!((oy + 12.0).abs() < TOL, "dy offset {oy}");
        assert!(stab.anchor_spread_px < TOL, "a rigid band has no spread");
    }

    #[test]
    fn stabilized_anchor_points_land_back_on_their_reference_positions() {
        let reference = grid(0, (0.0, 0.0), 0.0);
        let current = grid(7, (31.0, 12.0), 20.0);
        let band = band_of(&reference);
        let stab = stabilize_frame(&band, &current);
        for (id, ref_point) in band.points() {
            let now = current.get(id).unwrap();
            let (sx, sy) = stab.transform.apply(now.x, now.y);
            assert!((sx - ref_point.x).abs() < TOL, "{id}: x {sx}");
            assert!((sy - ref_point.y).abs() < TOL, "{id}: y {sy}");
        }
    }

    #[test]
    fn lower_fabric_still_moves_down_after_stabilization() {
        // Remove global motion while preserving lower-row extension.
        let reference = grid(0, (0.0, 0.0), 0.0);
        let current = grid(7, (31.0, 12.0), 20.0);
        let stab = stabilize_frame(&band_of(&reference), &current);
        let mut previous = 0.0;
        for row in 0..4 {
            let id = TrackId::new(format!("r{row}c1"));
            let before = reference.get(&id).unwrap();
            let after = current.get(&id).unwrap();
            let (_, sy) = stab.transform.apply(after.x, after.y);
            let moved = sy - before.y;
            assert!(
                moved >= previous - TOL,
                "row {row} must not move less than the row above it ({moved} < {previous})"
            );
            previous = moved;
        }
        let top = reference.get(&TrackId::new("r0c1")).unwrap();
        let top_now = current.get(&TrackId::new("r0c1")).unwrap();
        let (_, top_y) = stab.transform.apply(top_now.x, top_now.y);
        assert!((top_y - top.y).abs() < TOL);
        let bottom = reference.get(&TrackId::new("r3c1")).unwrap();
        let bottom_now = current.get(&TrackId::new("r3c1")).unwrap();
        let (_, bottom_y) = stab.transform.apply(bottom_now.x, bottom_now.y);
        assert!((bottom_y - bottom.y - 60.0).abs() < TOL, "{bottom_y}");
    }

    #[test]
    fn gross_outlier_anchor_cannot_move_the_view() {
        let reference = frame(
            0,
            &[
                ("a", 0.0, 0.0),
                ("b", 10.0, 0.0),
                ("c", 20.0, 0.0),
                ("d", 30.0, 0.0),
                ("far", 0.0, 400.0),
            ],
        );
        let current = frame(
            1,
            &[
                ("a", 5.0, 3.0),
                ("b", 15.0, 3.0),
                ("c", 25.0, 3.0),
                ("d", 900.0, -900.0), // tracker blew up on one anchor
                ("far", 5.0, 60.0),
            ],
        );
        let stab = stabilize_frame(&band_of(&reference), &current);
        assert_eq!(stab.anchor_total, 4);
        assert!((stab.anchor_dx - 5.0).abs() < TOL, "dx {}", stab.anchor_dx);
        assert!((stab.anchor_dy - 3.0).abs() < TOL, "dy {}", stab.anchor_dy);
        // MAD spread should ignore a single jumping track.
        assert!(
            stab.anchor_spread_px < TOL,
            "spread {}",
            stab.anchor_spread_px
        );
    }

    #[test]
    fn anchors_lost_in_a_frame_are_skipped() {
        let reference = grid(0, (0.0, 0.0), 0.0);
        // Only one of the three anchors survives into this frame; the
        // reference position of "r0c1" is (50, 0), so it moved (7, 9).
        let current = TrackingFrame::new(
            FrameId(4),
            vec![
                (TrackId::new("r0c1"), point(57.0, 9.0)),
                (TrackId::new("r3c0"), point(500.0, 500.0)),
            ],
        );
        let stab = stabilize_frame(&band_of(&reference), &current);
        assert_eq!(stab.anchor_pairs, 1);
        assert_eq!(stab.anchor_total, 3);
        assert!((stab.anchor_dx - 7.0).abs() < TOL, "dx {}", stab.anchor_dx);
        assert!((stab.anchor_dy - 9.0).abs() < TOL, "dy {}", stab.anchor_dy);
    }

    #[test]
    fn a_frame_without_any_anchor_is_left_alone() {
        let reference = grid(0, (0.0, 0.0), 0.0);
        let current = TrackingFrame::new(
            FrameId(9),
            vec![(TrackId::new("unrelated"), point(1.0, 2.0))],
        );
        let stab = stabilize_frame(&band_of(&reference), &current);
        assert!(stab.is_identity(), "no anchor -> render the raw image");
        assert!(!stab.is_estimated());
        assert_eq!(stab.anchor_pairs, 0);
    }

    #[test]
    fn non_finite_current_points_never_offset_the_view() {
        let reference = grid(0, (0.0, 0.0), 0.0);
        let current = TrackingFrame::new(
            FrameId(2),
            vec![
                (TrackId::new("r0c0"), TrackedPoint::new(f64::NAN, 1.0)),
                (TrackId::new("r0c1"), TrackedPoint::new(1.0, f64::INFINITY)),
                (TrackId::new("r0c2"), TrackedPoint::new(f64::NAN, f64::NAN)),
            ],
        );
        let stab = stabilize_frame(&band_of(&reference), &current);
        assert!(stab.is_identity());
        assert_eq!(stab.anchor_pairs, 0);
    }

    #[test]
    fn sequence_helper_resolves_both_frames() {
        let sequence =
            TrackingSequence::new(vec![grid(0, (0.0, 0.0), 0.0), grid(1, (4.0, 6.0), 10.0)]);
        let stab = stabilization_for_frame(&sequence, FrameId(0), FrameId(1)).unwrap();
        assert_eq!(stab.reference_frame, FrameId(0));
        assert_eq!(stab.frame, FrameId(1));
        assert!((stab.offset().0 + 4.0).abs() < TOL);
        assert!((stab.offset().1 + 6.0).abs() < TOL);
        assert!(stabilization_for_frame(&sequence, FrameId(0), FrameId(5)).is_none());
        assert!(stabilization_for_frame(&sequence, FrameId(5), FrameId(1)).is_none());
    }

    #[test]
    fn stabilization_never_rotates_or_scales() {
        let sequence =
            TrackingSequence::new(vec![grid(0, (0.0, 0.0), 0.0), grid(1, (4.0, 6.0), 10.0)]);
        let stab = stabilization_for_frame(&sequence, FrameId(0), FrameId(1)).unwrap();
        assert!(
            matches!(stab.transform, RegistrationTransform::Translation { .. }),
            "translation-only by construction, got {:?}",
            stab.transform
        );
        let (ax, ay) = stab.transform.apply(0.0, 0.0);
        let (bx, by) = stab.transform.apply(100.0, 40.0);
        assert!(((bx - ax) - 100.0).abs() < TOL);
        assert!(((by - ay) - 40.0).abs() < TOL);
    }
}
