//! Registered-track CSV, one row per point and frame.
//!
//! Raw coordinates stay in image pixels; registered coordinates apply the current-to-reference
//! transform (identity without registration). Export the actual mode after fallback, accepted
//! pair count and RMS residual alongside optional displacement/force metadata. Missing metadata
//! is left empty. Quote IDs as CSV fields.
//!
//! The header in [`registered_tracks_csv`] defines the column order.

use std::borrow::Cow;
use std::collections::HashMap;

use super::registration::{compute_registration, ReferenceFrame, RegistrationState};
use super::{FrameId, TrackingSequence};

/// Per-run frame metadata attached to exported rows when available.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunFrameMeta {
    pub displacement_mm: Option<f64>,
    pub force_n: Option<f64>,
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map_or_else(String::new, |x| format!("{x:.4}"))
}

/// A CSV field, quoted and escaped when it contains a delimiter, a quote or a
/// line break.
fn csv_field(text: &str) -> Cow<'_, str> {
    if text.contains([',', '"', '\n', '\r']) {
        Cow::Owned(format!("\"{}\"", text.replace('"', "\"\"")))
    } else {
        Cow::Borrowed(text)
    }
}

/// Serialize `sequence` under the current registration `state` as a CSV
/// document. `frame_meta` maps each run frame id to its displacement/force;
/// frames absent from it export empty displacement/force cells.
pub fn registered_tracks_csv(
    sequence: &TrackingSequence,
    state: &RegistrationState,
    frame_meta: &[(FrameId, RunFrameMeta)],
) -> String {
    let meta: HashMap<FrameId, RunFrameMeta> = frame_meta.iter().copied().collect();
    let reference = state.reference_frame.and_then(|id| {
        sequence
            .get_frame(&id)
            .map(|f| ReferenceFrame::from_frame(f, &state.reference_tracks, state.min_confidence))
    });

    let mut out = String::from(
        "frame_id,track_id,raw_x,raw_y,registered_x,registered_y,confidence,\
         registration_mode_used,valid_reference_tracks,rms_reference_error_px,\
         displacement_mm,force_n\n",
    );
    for frame in sequence.frames() {
        let result = match &reference {
            Some(reference) if !state.mode.is_off() => {
                Some(compute_registration(reference, frame, state))
            }
            _ => None,
        };
        let (mode_used, valid, rms) = match &result {
            Some(r) => (r.mode_used.code(), r.valid_count, r.rms_residual_px),
            None => ("off", 0, None),
        };
        let m = meta.get(&frame.frame_id()).copied().unwrap_or_default();
        for (tid, p) in frame.points() {
            if !p.coords_valid() {
                continue;
            }
            let (rx, ry) = match &result {
                Some(r) => r.to_reference().apply(p.x, p.y),
                None => (p.x, p.y),
            };
            let conf = p.confidence.map_or_else(String::new, |c| format!("{c}"));
            out.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{}\n",
                frame.frame_id(),
                csv_field(tid.as_str()),
                p.x,
                p.y,
                rx,
                ry,
                conf,
                mode_used,
                valid,
                fmt_opt(rms),
                fmt_opt(m.displacement_mm),
                fmt_opt(m.force_n),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::registration::{RegistrationMode, RegistrationState};
    use crate::tracking::{
        CsvTrackingImporter, TrackId, TrackedPoint, TrackingFrame, TrackingImporter,
    };

    fn seq() -> TrackingSequence {
        TrackingSequence::new(vec![
            TrackingFrame::new(
                FrameId(0),
                vec![
                    (
                        TrackId::new("a"),
                        TrackedPoint::with_confidence(10.0, 20.0, 0.99),
                    ),
                    (TrackId::new("b"), TrackedPoint::new(30.0, 40.0)),
                ],
            ),
            TrackingFrame::new(
                FrameId(1),
                vec![(
                    TrackId::new("a"),
                    TrackedPoint::with_confidence(15.0, 17.0, 0.9),
                )],
            ),
        ])
    }

    fn meta() -> Vec<(FrameId, RunFrameMeta)> {
        vec![
            (
                FrameId(0),
                RunFrameMeta {
                    displacement_mm: Some(0.0),
                    force_n: Some(1.5),
                },
            ),
            (
                FrameId(1),
                RunFrameMeta {
                    displacement_mm: Some(1.0),
                    force_n: None,
                },
            ),
        ]
    }

    #[test]
    fn off_mode_exports_raw_positions() {
        let csv = registered_tracks_csv(&seq(), &RegistrationState::default(), &meta());
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4, "header + 3 points");
        assert!(lines[0].starts_with("frame_id,track_id,raw_x"));
        assert!(lines[1].contains("0,a,10.0000,20.0000,10.0000,20.0000"));
        assert!(
            lines[1].contains("0.99,off,0,,"),
            "off: no pairs, no rms; got: {}",
            lines[1]
        );
        assert!(lines[1].contains("0.0000,1.5000"), "displacement and force");
        assert!(lines[3].contains("1,a,"));
        assert!(
            lines[3].contains("1.0000,"),
            "displacement present, force empty: {}",
            lines[3]
        );
    }

    #[test]
    fn translation_mode_applies_the_transform() {
        // Frame 1 drift is (+5, -3); registered coordinates should subtract it.
        let state = RegistrationState {
            mode: RegistrationMode::TranslationXY,
            reference_frame: Some(FrameId(0)),
            ..Default::default()
        };
        let csv = registered_tracks_csv(&seq(), &state, &meta());
        let lines: Vec<&str> = csv.lines().collect();
        // Frame 0 is the reference itself: drift 0, registered == raw.
        assert!(lines[1].contains("0,a,10.0000,20.0000,10.0000,20.0000"));
        assert!(lines[1].contains("translation_xy"));
        // Frame 1 "a": raw (15,17) registered back to (10,20) = raw - drift.
        let frame1 = lines[3];
        assert!(
            frame1.contains("1,a,15.0000,17.0000,10.0000,20.0000"),
            "registered position returns to the reference: {frame1}"
        );
        assert!(
            frame1.contains("translation_xy,1,0.0000"),
            "mode, 1 valid pair, zero rms: {frame1}"
        );
    }

    #[test]
    fn missing_reference_falls_back_to_off_columns() {
        // A non-off mode with no reference frame exports raw positions and
        // off/0/empty registration columns.
        let state = RegistrationState {
            mode: RegistrationMode::Rigid,
            reference_frame: None,
            ..Default::default()
        };
        let csv = registered_tracks_csv(&seq(), &state, &meta());
        assert!(csv.lines().nth(1).unwrap().contains("off,0,,"));
    }

    #[test]
    fn track_ids_are_quoted_and_round_trip_through_the_csv_importer() {
        let ids = ["plain", "a,b", "say \"hi\"", "x,\"y\",z"];
        let seq = TrackingSequence::new(vec![TrackingFrame::new(
            FrameId(0),
            ids.iter().enumerate().map(|(i, id)| {
                (
                    TrackId::new(id),
                    TrackedPoint::new(i as f64, 2.0 * i as f64),
                )
            }),
        )]);
        let csv = registered_tracks_csv(&seq, &RegistrationState::default(), &[]);
        assert!(csv.contains("\"a,b\""), "{csv}");
        assert!(csv.contains("\"say \"\"hi\"\"\""), "{csv}");

        let back = CsvTrackingImporter
            .import_reader(&mut std::io::Cursor::new(csv.into_bytes()))
            .unwrap();
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                back.sequence.point(&FrameId(0), &TrackId::new(id)),
                Some(&TrackedPoint::new(i as f64, 2.0 * i as f64)),
                "{id}"
            );
        }
    }
}
