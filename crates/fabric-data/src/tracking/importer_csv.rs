//! Long-table tracking import. Headerless order is `frame_id, track_id, x, y, confidence`.
//!
//! Headers accept these aliases:
//! - frame: `frame_id`, `frame`, `frame_index`, `fid`
//! - track: `track_id`, `track`, `id`, `label`, `point_id`, `tid`
//! - x: `x`, `px`, `x_px`, `x_pixel`, `u`, `raw_x`
//! - y: `y`, `py`, `y_px`, `y_pixel`, `v`, `raw_y`
//! - confidence: `confidence`, `conf`, `quality`, `c`
//!
//! Ignore extra columns. Blank confidence means unreported; quality values are not rescaled.
//! Skip blank/comment lines and warn on malformed rows. Last duplicate `(frame, track)` wins.
//! Quoted fields support doubled quotes. Frame IDs should match the run.
//!
//! [`CsvTrackingImporter`] uses commas; [`LongTableTrackingImporter`] detects comma, tab or
//! semicolon. Both accept the adapter's CSV output, not a native CoWTracker schema.

use std::io::BufRead;

use super::{
    FrameId, TrackId, TrackedPoint, TrackingFrame, TrackingImport, TrackingImportError,
    TrackingImporter, TrackingSequence,
};

/// The generic CSV importer (comma-delimited, see module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct CsvTrackingImporter;

/// Long-table importer with header aliases and comma/tab/semicolon detection.
/// Accepts adapter CSV output; this is not a native CoWTracker schema.
#[derive(Debug, Clone, Copy, Default)]
pub struct LongTableTrackingImporter;

/// Column roles discovered from the header (or the fixed default order).
#[derive(Default)]
struct Columns {
    frame: Option<usize>,
    track: Option<usize>,
    x: Option<usize>,
    y: Option<usize>,
    confidence: Option<usize>,
}

impl TrackingImporter for CsvTrackingImporter {
    fn name(&self) -> &'static str {
        "csv"
    }

    fn import_reader(
        &self,
        reader: &mut dyn BufRead,
    ) -> Result<TrackingImport, TrackingImportError> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        parse_long_table(&text, Some(','))
    }
}

impl TrackingImporter for LongTableTrackingImporter {
    fn name(&self) -> &'static str {
        "long-table"
    }

    fn import_reader(
        &self,
        reader: &mut dyn BufRead,
    ) -> Result<TrackingImport, TrackingImportError> {
        let mut text = String::new();
        reader.read_to_string(&mut text)?;
        parse_long_table(&text, None)
    }
}

/// Pick the delimiter from the first data row: whichever of `,`, `\t`, `;`
/// appears most often. Ties prefer the earlier candidate (comma first).
fn detect_delimiter(text: &str) -> char {
    let Some(first_data) = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
    else {
        return ',';
    };
    [',', '\t', ';']
        .into_iter()
        .map(|d| (d, first_data.chars().filter(|&c| c == d).count()))
        .max_by_key(|(_, n)| *n)
        .map(|(d, _)| d)
        .unwrap_or(',')
}

/// A minimal RFC-4180-style line splitter: quoted fields may contain the
/// delimiter, quotes are escaped by doubling.
fn split_line(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            c if c == delimiter && !in_quotes => {
                fields.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    fields.push(current);
    fields
}

fn header_columns(header: &[String]) -> Columns {
    let mut cols = Columns::default();
    for (i, name) in header.iter().enumerate() {
        let n = name
            .trim()
            .trim_start_matches('\u{feff}')
            .to_ascii_lowercase();
        match n.as_str() {
            "frame_id" | "frame" | "fid" | "frame_index" => cols.frame.get_or_insert(i),
            "track_id" | "id" | "label" | "track" | "tid" | "point_id" => {
                cols.track.get_or_insert(i)
            }
            "x" | "px" | "u" | "x_px" | "x_pixel" | "raw_x" => cols.x.get_or_insert(i),
            "y" | "py" | "v" | "y_px" | "y_pixel" | "raw_y" => cols.y.get_or_insert(i),
            "confidence" | "conf" | "c" | "quality" => cols.confidence.get_or_insert(i),
            _ => continue,
        };
    }
    cols
}

/// Recognize headers by frame/x/y aliases without numeric cells.
/// A track ID such as `c` in a headerless row must not be mistaken for a column name.
fn parse_header(fields: &[String]) -> Option<Columns> {
    let cols = header_columns(fields);
    let (frame, x, y) = (cols.frame?, cols.x?, cols.y?);
    let numeric = [frame, x, y]
        .iter()
        .any(|&i| fields[i].trim().parse::<f64>().is_ok());
    (!numeric).then_some(cols)
}

/// Shared long-table parser. `forced_delimiter = None` auto-detects it.
fn parse_long_table(
    text: &str,
    forced_delimiter: Option<char>,
) -> Result<TrackingImport, TrackingImportError> {
    let delimiter = forced_delimiter.unwrap_or_else(|| detect_delimiter(text));
    let mut warnings = Vec::new();
    let mut rows: Vec<(FrameId, TrackId, TrackedPoint)> = Vec::new();

    let mut first = true;
    let mut cols = Columns::default();
    for (line_no, line) in text.lines().enumerate() {
        let line_no = line_no + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = split_line(trimmed, delimiter);
        if first {
            first = false;
            if let Some(header) = parse_header(&fields) {
                cols = header;
                continue; // never treat the header as a data row
            }
            cols = Columns {
                frame: Some(0),
                track: Some(1),
                x: Some(2),
                y: Some(3),
                confidence: if fields.len() > 4 { Some(4) } else { None },
            };
        }

        let parse =
            |i: usize| -> Option<f64> { fields.get(i).and_then(|s| s.trim().parse::<f64>().ok()) };
        let idx = |i: Option<usize>| i.filter(|&i| i < fields.len());

        let frame_idx = idx(cols.frame);
        let track_idx = idx(cols.track);
        let x_idx = idx(cols.x);
        let y_idx = idx(cols.y);
        let conf_idx = idx(cols.confidence);

        let Some(fi) = frame_idx else {
            warnings.push(format!("line {line_no}: missing frame_id, skipped"));
            continue;
        };
        let Some(ti) = track_idx else {
            warnings.push(format!("line {line_no}: missing track_id, skipped"));
            continue;
        };
        let Some(xi) = x_idx else {
            warnings.push(format!("line {line_no}: missing x, skipped"));
            continue;
        };
        let Some(yi) = y_idx else {
            warnings.push(format!("line {line_no}: missing y, skipped"));
            continue;
        };

        let Some(frame_val) = fields[fi].trim().parse::<u32>().ok() else {
            warnings.push(format!("line {line_no}: non-numeric frame_id, skipped"));
            continue;
        };
        let track = fields[ti].trim();
        if track.is_empty() {
            warnings.push(format!("line {line_no}: empty track_id, skipped"));
            continue;
        }
        let Some(x) = parse(xi) else {
            warnings.push(format!("line {line_no}: non-numeric x, skipped"));
            continue;
        };
        let Some(y) = parse(yi) else {
            warnings.push(format!("line {line_no}: non-numeric y, skipped"));
            continue;
        };
        let confidence = conf_idx.and_then(parse).filter(|c| c.is_finite());
        let point = TrackedPoint { x, y, confidence };
        if !point.coords_valid() {
            warnings.push(format!("line {line_no}: non-finite coordinates, skipped"));
            continue;
        }
        rows.push((FrameId(frame_val), TrackId::new(track), point));
    }

    // Keep first-seen frame order and the last value for duplicate `(frame, track)` rows.
    let mut frame_rows: std::collections::HashMap<FrameId, Vec<(TrackId, TrackedPoint)>> =
        std::collections::HashMap::new();
    let mut order: Vec<FrameId> = Vec::new();
    for (fid, tid, point) in rows {
        let bucket = frame_rows.entry(fid).or_default();
        if bucket.is_empty() {
            order.push(fid);
        }
        bucket.push((tid, point));
    }
    let frames: Vec<TrackingFrame> = order
        .into_iter()
        .map(|fid| TrackingFrame::new(fid, frame_rows.remove(&fid).unwrap()))
        .collect();

    Ok(TrackingImport {
        sequence: TrackingSequence::new(frames),
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn import(text: &str) -> Result<TrackingImport, TrackingImportError> {
        let mut cursor = std::io::Cursor::new(text.as_bytes().to_vec());
        CsvTrackingImporter.import_reader(&mut cursor)
    }

    fn import_cow(text: &str) -> Result<TrackingImport, TrackingImportError> {
        let mut cursor = std::io::Cursor::new(text.as_bytes().to_vec());
        LongTableTrackingImporter.import_reader(&mut cursor)
    }

    #[test]
    fn round_trips_documented_format_with_header() {
        let csv =
            "frame_id,track_id,x,y,confidence\n0,a,1.0,2.0,0.99\n0,b,3.0,4.0,\n1,a,1.5,2.5,0.5\n";
        let result = import(csv).unwrap();
        assert_eq!(result.sequence.len(), 2);
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(1.0, 2.0, 0.99))
        );
        assert_eq!(
            f0.get(&TrackId::new("b")),
            Some(&TrackedPoint::new(3.0, 4.0))
        );
        let f1 = result.sequence.get_frame(&FrameId(1)).unwrap();
        assert_eq!(
            f1.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(1.5, 2.5, 0.5))
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    #[test]
    fn headerless_rows_use_fixed_column_order() {
        let csv = "0,a,10.0,20.0,0.9\n0,b,30.0,40.0,\n";
        let result = import(csv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(10.0, 20.0, 0.9))
        );
    }

    #[test]
    fn header_aliases_are_mapped() {
        let csv = "frame,label,u,v,conf\n0,A,1,2,0.8\n";
        let result = import(csv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(
            f0.get(&TrackId::new("A")),
            Some(&TrackedPoint::with_confidence(1.0, 2.0, 0.8))
        );
    }

    #[test]
    fn bad_rows_are_skipped_with_warnings() {
        let csv = "frame_id,track_id,x,y\n0,a,1.0,2.0\nnot_a_number,a,1.0,2.0\n0,b,nope,2.0\n0,c,3.0,4.0\n\n# comment\n0,d,5.0,6.0\n";
        let result = import(csv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        let ids: Vec<&str> = f0.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(ids, vec!["a", "c", "d"]);
        assert!(!result.warnings.is_empty(), "skips must be reported");
    }

    #[test]
    fn duplicate_frame_track_keeps_last() {
        let csv = "frame_id,track_id,x,y\n0,a,1.0,1.0\n0,a,9.0,9.0\n";
        let result = import(csv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 1);
        assert_eq!(
            f0.get(&TrackId::new("a")),
            Some(&TrackedPoint::new(9.0, 9.0))
        );
    }

    #[test]
    fn quoted_fields_are_parsed() {
        let csv = "frame_id,track_id,x,y\n0,\"a,b\",1.0,2.0\n";
        let result = import(csv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(
            f0.get(&TrackId::new("a,b")),
            Some(&TrackedPoint::new(1.0, 2.0))
        );
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err = CsvTrackingImporter
            .import_path(std::path::Path::new("/nonexistent/nope.csv"))
            .unwrap_err();
        assert!(matches!(err, TrackingImportError::Io(_)));
    }

    #[test]
    fn headerless_first_row_with_alias_track_id_is_data() {
        let csv = "0,c,1.0,2.0\n0,d,3.0,4.0\n";
        let result = import(csv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("c")),
            Some(&TrackedPoint::new(1.0, 2.0))
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    #[test]
    fn long_table_profile_maps_long_table_aliases() {
        // Extra adapter CSV columns are ignored; aliases such as `frame_index` and `quality` also parse.
        let csv = "frame_index,point_id,x,y,quality\n1,a,10.0,20.0,0.9\n1,b,30.0,40.0,0.7\n2,a,11.0,21.0,0.9\n";
        let result = import_cow(csv).unwrap();
        assert_eq!(result.sequence.len(), 2);
        let f1 = result.sequence.get_frame(&FrameId(1)).unwrap();
        assert_eq!(f1.len(), 2);
        assert_eq!(
            f1.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(10.0, 20.0, 0.9))
        );
        assert_eq!(
            f1.get(&TrackId::new("b")),
            Some(&TrackedPoint::with_confidence(30.0, 40.0, 0.7))
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    #[test]
    fn long_table_profile_detects_tab_delimited_tsv() {
        let tsv = "frame_id\ttrack_id\tx\ty\tconfidence\n0\ta\t1.0\t2.0\t0.8\n0\tb\t3.0\t4.0\t\n";
        let result = import_cow(tsv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(1.0, 2.0, 0.8))
        );
        assert_eq!(
            f0.get(&TrackId::new("b")),
            Some(&TrackedPoint::new(3.0, 4.0))
        );
    }

    #[test]
    fn long_table_profile_detects_semicolon_delimited() {
        let text = "frame;track;x;y\n0;a;1.0;2.0\n0;b;3.0;4.0\n";
        let result = import_cow(text).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("b")),
            Some(&TrackedPoint::new(3.0, 4.0))
        );
    }

    #[test]
    fn long_table_profile_headerless_tsv_uses_fixed_order() {
        let tsv = "0\ta\t1.0\t2.0\t0.9\n0\tb\t3.0\t4.0\t\n";
        let result = import_cow(tsv).unwrap();
        let f0 = result.sequence.get_frame(&FrameId(0)).unwrap();
        assert_eq!(f0.len(), 2);
        assert_eq!(
            f0.get(&TrackId::new("a")),
            Some(&TrackedPoint::with_confidence(1.0, 2.0, 0.9))
        );
    }
}
