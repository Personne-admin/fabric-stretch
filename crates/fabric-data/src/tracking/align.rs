//! Detect frame-ID mismatches, including 0-based versus 1-based exports.
//! Report likely offsets rather than applying them silently.

use std::collections::HashSet;

use super::TrackingSequence;

/// How well the tracking frame ids line up with the run's frame ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameAlignmentStatus {
    /// The tracking ids match the run ids (>= 80% overlap, no better
    /// shifted match).
    Exact,
    /// Shifting IDs by one improves overlap, suggesting 0-based versus 1-based numbering.
    OffsetByOne,
    /// Some ids overlap, but well under half of the frames line up.
    LowOverlap,
    /// No tracked frame id exists in the run at all.
    None,
}

impl FrameAlignmentStatus {
    /// Whether a user-facing warning should be raised for this state.
    pub fn is_warning(self) -> bool {
        !matches!(self, FrameAlignmentStatus::Exact)
    }
}

/// The full alignment readout: status, raw overlap and a human-readable note.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameAlignment {
    pub status: FrameAlignmentStatus,
    /// Raw number of tracking ids present in the run's frame ids.
    pub overlap: usize,
    pub tracking_count: usize,
    pub run_count: usize,
    pub note: String,
}

/// Check frame-ID overlap against the run. A ±1 shift wins only if it improves overlap
/// and covers at least 80% of the smaller set. Otherwise ≥80% is Exact, positive overlap
/// is LowOverlap, and zero is None.
pub fn align_tracking_frames(tracking: &TrackingSequence, run_frames: &[u32]) -> FrameAlignment {
    let tracking_ids: Vec<u32> = tracking.frame_ids().map(|f| f.as_u32()).collect();
    let tracking_set: HashSet<u32> = tracking_ids.iter().copied().collect();
    let run_set: HashSet<u32> = run_frames.iter().copied().collect();

    let tracking_count = tracking_ids.len();
    let run_count = run_frames.len();
    let min_count = tracking_count.min(run_count).max(1);

    let overlap_with = |ids: &HashSet<u32>| ids.intersection(&run_set).count();
    let plain = overlap_with(&tracking_set);

    // Shifted id spaces: tracking - 1 (1-based -> 0-based) and tracking + 1
    // (0-based -> 1-based).
    let shift_neg: HashSet<u32> = tracking_set
        .iter()
        .filter_map(|t| t.checked_sub(1))
        .collect();
    let shift_pos: HashSet<u32> = tracking_set.iter().map(|t| t + 1).collect();
    let shifted_neg = overlap_with(&shift_neg);
    let shifted_pos = overlap_with(&shift_pos);
    let best_shifted = shifted_neg.max(shifted_pos);
    let shift_note = if shifted_neg > shifted_pos {
        "tracking frame ids look 1-based while the run is 0-based"
    } else if shifted_pos > shifted_neg {
        "tracking frame ids look 0-based while the run is 1-based"
    } else {
        ""
    };

    let ratio = |n: usize| n as f64 / min_count as f64;

    let (status, note) = if best_shifted > plain && ratio(best_shifted) >= 0.8 {
        (
            FrameAlignmentStatus::OffsetByOne,
            format!(
                "{shift_note}: shifted overlap {best_shifted}/{min_count} beats the \
                 plain overlap {plain}/{min_count}"
            ),
        )
    } else if ratio(plain) >= 0.8 {
        (
            FrameAlignmentStatus::Exact,
            format!("frame ids align: {plain} tracking frame(s) overlap the run's frames"),
        )
    } else if plain > 0 {
        (
            FrameAlignmentStatus::LowOverlap,
            format!("low overlap between tracking and run frame ids ({plain}/{min_count})"),
        )
    } else {
        (
            FrameAlignmentStatus::None,
            "no overlap between tracking and run frame ids - check the tracker's \
             frame numbering"
                .to_owned(),
        )
    };

    FrameAlignment {
        status,
        overlap: plain,
        tracking_count,
        run_count,
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::{FrameId, TrackId, TrackedPoint, TrackingFrame};

    fn seq(ids: &[u32]) -> TrackingSequence {
        TrackingSequence::new(ids.iter().map(|id| {
            TrackingFrame::new(
                FrameId(*id),
                vec![(TrackId::new("a"), TrackedPoint::new(1.0, 2.0))],
            )
        }))
    }

    #[test]
    fn exact_match_is_exact() {
        let run = vec![0, 1, 2, 3];
        let a = align_tracking_frames(&seq(&[0, 1, 2, 3]), &run);
        assert_eq!(a.status, FrameAlignmentStatus::Exact);
        assert!(!a.status.is_warning());
        assert_eq!(a.overlap, 4);
    }

    #[test]
    fn tracking_subset_of_run_is_exact() {
        let run = vec![0, 1, 2, 3, 4, 5];
        let a = align_tracking_frames(&seq(&[1, 2, 3]), &run);
        assert_eq!(a.status, FrameAlignmentStatus::Exact, "{}", a.note);
    }

    #[test]
    fn one_based_tracking_is_detected() {
        // Tracking 1..=4 vs run 0..=3: shifting tracking down by one matches
        // the run exactly, which must beat the plain (near-full) overlap.
        let run = vec![0, 1, 2, 3];
        let a = align_tracking_frames(&seq(&[1, 2, 3, 4]), &run);
        assert_eq!(a.status, FrameAlignmentStatus::OffsetByOne, "{}", a.note);
        assert!(a.status.is_warning());
        assert!(a.note.contains("1-based"));
        assert_eq!(a.overlap, 3, "plain overlap is one short");
    }

    #[test]
    fn zero_based_tracking_against_one_based_run_is_detected() {
        let run = vec![1, 2, 3, 4];
        let a = align_tracking_frames(&seq(&[0, 1, 2, 3]), &run);
        assert_eq!(a.status, FrameAlignmentStatus::OffsetByOne, "{}", a.note);
        assert!(a.note.contains("0-based"));
    }

    #[test]
    fn partially_overlapping_ids_are_low_overlap() {
        let run = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let a = align_tracking_frames(&seq(&[0, 1, 2, 100, 101, 102, 103, 104]), &run);
        assert_eq!(a.status, FrameAlignmentStatus::LowOverlap, "{}", a.note);
        assert!(a.status.is_warning());
    }

    #[test]
    fn disjoint_ids_are_none() {
        let run = vec![10, 11, 12, 13];
        let a = align_tracking_frames(&seq(&[0, 1, 2]), &run);
        assert_eq!(a.status, FrameAlignmentStatus::None, "{}", a.note);
        assert!(a.status.is_warning());
        assert_eq!(a.overlap, 0);
    }

    #[test]
    fn empty_inputs_report_none_without_panicking() {
        let a = align_tracking_frames(&seq(&[]), &[0, 1, 2]);
        assert_eq!(a.status, FrameAlignmentStatus::None);
        let a = align_tracking_frames(&seq(&[0, 1]), &[]);
        assert_eq!(a.status, FrameAlignmentStatus::None);
    }
}
