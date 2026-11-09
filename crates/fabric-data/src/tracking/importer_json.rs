//! JSON tracking import. Minimal input:
//!
//! ```json
//! {"frames": [{"frame_id": 0, "points": [{"id": "a", "x": 1.0, "y": 2.0}]}]}
//! ```
//!
//! Frame IDs must be non-negative integers (integral floats accepted), matching `frames.json`.
//! Duplicate frames keep the last occurrence. Points may also be an ID-keyed object.
//! Accept `id`/`track_id`/`label`, optional `confidence`/`conf`, and ignore unknown fields
//! including format/version. Skip invalid frames or points with warnings.

use std::fmt;
use std::io::BufRead;

use serde_json::{Map, Value};

use super::{
    FrameId, TrackId, TrackedPoint, TrackingFrame, TrackingImport, TrackingImportError,
    TrackingImporter, TrackingSequence,
};

/// Skip malformed points with warnings. Invalid JSON or a missing/empty `frames` array
/// is an import error.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonTrackingImporter;

fn number_as_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
}

/// A frame id from a JSON number; only non-negative integers that fit a
/// `u32` qualify, so `-1`, `1.7` and `1e10` never alias a real frame.
fn frame_id_of(v: &Value) -> Option<FrameId> {
    if let Some(u) = v.as_u64() {
        return u32::try_from(u).ok().map(FrameId);
    }
    let f = v.as_f64()?;
    (f.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&f)).then_some(FrameId(f as u32))
}

/// A point from its `x`/`y`/`confidence` keys, or why it is unusable.
fn parse_point(obj: &Map<String, Value>) -> Result<TrackedPoint, &'static str> {
    let x = obj.get("x").and_then(number_as_f64).ok_or("missing x")?;
    let y = obj.get("y").and_then(number_as_f64).ok_or("missing y")?;
    let confidence = ["confidence", "conf"]
        .iter()
        .find_map(|k| obj.get(*k).and_then(number_as_f64))
        .filter(|c| c.is_finite());
    let point = TrackedPoint { x, y, confidence };
    if point.coords_valid() {
        Ok(point)
    } else {
        Err("has non-finite coordinates")
    }
}

impl TrackingImporter for JsonTrackingImporter {
    fn name(&self) -> &'static str {
        "json"
    }

    fn import_reader(
        &self,
        reader: &mut dyn BufRead,
    ) -> Result<TrackingImport, TrackingImportError> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|e| TrackingImportError::Parse(format!("invalid JSON: {e}")))?;
        let root = root.as_object().ok_or_else(|| {
            TrackingImportError::Parse("top-level value must be a JSON object".to_owned())
        })?;

        let frames_value = root
            .get("frames")
            .ok_or_else(|| TrackingImportError::Parse("missing \"frames\" array".to_owned()))?;
        let frames_value = frames_value
            .as_array()
            .ok_or_else(|| TrackingImportError::Parse("\"frames\" must be an array".to_owned()))?;

        let mut warnings = Vec::new();
        let mut frames: Vec<TrackingFrame> = Vec::new();

        for (fi, frame_value) in frames_value.iter().enumerate() {
            let Some(frame_obj) = frame_value.as_object() else {
                warnings.push(format!("frame {fi}: not an object, skipped"));
                continue;
            };
            let Some(frame_id) = frame_obj.get("frame_id").and_then(frame_id_of) else {
                warnings.push(format!(
                    "frame {fi}: missing or non-integer frame_id, skipped"
                ));
                continue;
            };
            let Some(points_value) = frame_obj.get("points") else {
                frames.push(TrackingFrame::new(
                    frame_id,
                    Vec::<(TrackId, TrackedPoint)>::new(),
                ));
                continue;
            };

            let mut entries: Vec<(TrackId, TrackedPoint)> = Vec::new();
            let mut skip = |label: &dyn fmt::Display, reason: &str| {
                warnings.push(format!("frame {frame_id}: point {label} {reason}, skipped"));
            };
            if let Some(point_array) = points_value.as_array() {
                for (pi, pv) in point_array.iter().enumerate() {
                    let Some(obj) = pv.as_object() else {
                        skip(&pi, "not an object");
                        continue;
                    };
                    let id = ["id", "track_id", "label"]
                        .iter()
                        .find_map(|k| obj.get(*k).and_then(Value::as_str));
                    let Some(id) = id else {
                        skip(&pi, "missing id");
                        continue;
                    };
                    match parse_point(obj) {
                        Ok(point) => entries.push((TrackId::new(id), point)),
                        Err(reason) => skip(&pi, reason),
                    }
                }
            } else if let Some(point_map) = points_value.as_object() {
                // Compact form: { "<id>": { "x": .., "y": .., "confidence": .. } }
                for (id, pv) in point_map {
                    match pv.as_object().ok_or("not an object").and_then(parse_point) {
                        Ok(point) => entries.push((TrackId::new(id), point)),
                        Err(reason) => skip(id, reason),
                    }
                }
            } else {
                warnings.push(format!(
                    "frame {frame_id}: \"points\" must be an array or object, frame kept empty"
                ));
            }
            frames.push(TrackingFrame::new(frame_id, entries));
        }

        Ok(TrackingImport {
            sequence: TrackingSequence::new(frames),
            warnings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn import(text: &str) -> Result<TrackingImport, TrackingImportError> {
        let mut cursor = std::io::Cursor::new(text.as_bytes().to_vec());
        JsonTrackingImporter.import_reader(&mut cursor)
    }

    #[test]
    fn round_trips_documented_format() {
        let json = r#"{
            "format": "fabsim-tracks",
            "version": 1,
            "frames": [
                {"frame_id": 0, "points": [
                    {"id": "a", "x": 1.0, "y": 2.0, "confidence": 0.99},
                    {"id": "b", "x": 3.0, "y": 4.0}
                ]},
                {"frame_id": 1, "points": [
                    {"track_id": "a", "x": 1.5, "y": 2.5, "conf": 0.5}
                ]}
            ]
        }"#;
        let result = import(json).unwrap();
        assert_eq!(result.sequence.len(), 2);
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("b")),
            Some(&TrackedPoint::new(3.0, 4.0))
        );
        assert_eq!(
            f0.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(1.0, 2.0, 0.99))
        );
        let f1 = result.sequence.get_frame(&FrameId(1)).unwrap();
        assert_eq!(
            f1.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(1.5, 2.5, 0.5))
        );
        assert_eq!(result.sequence.track_ids().len(), 2);
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    #[test]
    fn compact_map_form_is_accepted() {
        let json = r#"{
            "frames": [
                {"frame_id": 5, "points": {
                    "a": {"x": 1.0, "y": 2.0},
                    "b": {"x": 3.0, "y": 4.0, "confidence": 0.8}
                }}
            ]
        }"#;
        let result = import(json).unwrap();
        let f = result.sequence.get_frame(&FrameId(5)).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(
            f.get(&TrackId::new("a")),
            Some(&TrackedPoint::new(1.0, 2.0))
        );
    }

    #[test]
    fn missing_points_keep_empty_frame() {
        let json = r#"{"frames":[{"frame_id": 3}]}"#;
        let result = import(json).unwrap();
        assert_eq!(result.sequence.len(), 1);
        assert!(result.sequence.get_frame(&FrameId(3)).unwrap().is_empty());
    }

    #[test]
    fn bad_points_become_warnings_not_errors() {
        let json = r#"{"frames":[{"frame_id":0,"points":[
            {"x": 1.0, "y": 2.0},
            {"id": "a", "y": 2.0},
            {"id": "b", "x": 1.0, "y": "not-a-number"},
            {"id": "c", "x": 1.0, "y": 2.0},
            {"id": "d", "x": "nan", "y": 2.0}
        ]}]}"#;
        let result = import(json).unwrap();
        let f = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f.len(), 1, "only c survives");
        assert!(!result.warnings.is_empty(), "skips must be reported");
        assert_eq!(result.sequence.track_ids()[0].as_str(), "c");
    }

    #[test]
    fn frame_ids_must_be_non_negative_integers() {
        let json = r#"{"frames":[
            {"frame_id": -1, "points": []},
            {"frame_id": 1.7, "points": []},
            {"frame_id": 4294967296, "points": []},
            {"frame_id": "2", "points": []},
            {"frame_id": 3.0, "points": []},
            {"frame_id": 0, "points": []}
        ]}"#;
        let result = import(json).unwrap();
        let ids: Vec<u32> = result.sequence.frame_ids().map(|f| f.as_u32()).collect();
        assert_eq!(ids, vec![3, 0]);
        assert_eq!(result.warnings.len(), 4, "{:?}", result.warnings);
    }

    #[test]
    fn duplicate_frame_ids_keep_the_last_frame() {
        let json = r#"{"frames":[
            {"frame_id": 1, "points": [{"id": "a", "x": 1.0, "y": 1.0}]},
            {"frame_id": 1, "points": [{"id": "a", "x": 2.0, "y": 2.0}]},
            {"frame_id": 2, "points": [{"id": "a", "x": 3.0, "y": 3.0}]}
        ]}"#;
        let seq = import(json).unwrap().sequence;
        assert_eq!(seq.len(), 2);
        let a = TrackId::new("a");
        assert_eq!(
            seq.point(&FrameId(1), &a),
            Some(&TrackedPoint::new(2.0, 2.0))
        );
        assert_eq!(
            seq.point(&FrameId(2), &a),
            Some(&TrackedPoint::new(3.0, 3.0))
        );
    }

    #[test]
    fn out_of_range_number_literal_is_a_parse_error() {
        // serde_json rejects `1e999` before per-point validation; CSV can skip that row instead.
        let json = r#"{"frames":[{"frame_id":0,"points":[
            {"id": "a", "x": 1.0, "y": 2.0},
            {"id": "b", "x": 1e999, "y": 2.0}
        ]}]}"#;
        assert!(import(json).is_err());
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(import("not json at all {").is_err());
        assert!(import("[1,2,3]").is_err(), "top level must be an object");
        assert!(import("{}").is_err(), "missing frames array");
    }

    #[test]
    fn missing_id_aliases_and_extra_keys() {
        let json = r#"{"frames":[{"frame_id":0,"points":[
            {"label": "L", "x": 1.0, "y": 2.0, "unexpected": "ignored"}
        ]}]}"#;
        let result = import(json).unwrap();
        let f = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(
            f.get(&TrackId::new("L")),
            Some(&TrackedPoint::new(1.0, 2.0))
        );
    }

    #[test]
    fn adapter_output_with_provenance_loads() {
        // Adapter output includes provenance and extra fields; ignore them on import.
        let json = r#"{
            "format": "fabsim-tracks",
            "version": 1,
            "provenance": {
                "tool": "cowtracker_to_fabsim",
                "source": "/path/to/run",
                "role": "diffuse",
                "coordinate_convention": "pixel_center",
                "reference_frame": 0,
                "source_dimensions": [1280, 1024],
                "inference_dimensions": [560, 336],
                "frame_mapping": [
                    {"sequence_index": 0, "frame_id": 0, "rotated_position": 0},
                    {"sequence_index": 1, "frame_id": 1, "rotated_position": 1}
                ]
            },
            "frames": [
                {"frame_id": 0, "points": [
                    {"id": "grid_0", "x": 100.125, "y": 80.5, "confidence": 0.931},
                    {"id": "grid_1", "x": 300.25, "y": 200.75, "confidence": 0.12}
                ]},
                {"frame_id": 1, "points": [
                    {"id": "grid_0", "x": 102.875, "y": 85.0, "confidence": 0.9},
                    {"id": "grid_1", "x": 302.5, "y": 205.5, "confidence": 0.1}
                ]}
            ]
        }"#;
        let result = import(json).unwrap();
        assert_eq!(result.sequence.len(), 2);
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("grid_0")),
            Some(&TrackedPoint::with_confidence(100.125, 80.5, 0.931))
        );
        // Low confidence is retained, never dropped.
        assert_eq!(
            f0.get(&TrackId::new("grid_1")),
            Some(&TrackedPoint::with_confidence(300.25, 200.75, 0.12))
        );
        // Frame ids are the fabsim ids, not the sequence position.
        assert_eq!(
            result
                .sequence
                .frame_ids()
                .map(|f| f.as_u32())
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }
}
