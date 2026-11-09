//! Tracked-point import, alignment, registration and stabilization.
//!
//! Frame and track IDs are identities, not indices. Coordinates remain source-image pixels
//! (x right, y down). [`importer_for_path`] chooses JSON, CSV or long-table parsing;
//! [`align_tracking_frames`] checks numbering against the run.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod align;
pub mod export;
pub mod importer_csv;
pub mod importer_json;
pub mod registration;
pub mod stabilize;

pub use align::{align_tracking_frames, FrameAlignment, FrameAlignmentStatus};
pub use export::{registered_tracks_csv, RunFrameMeta};
pub use importer_csv::{CsvTrackingImporter, LongTableTrackingImporter};
pub use importer_json::JsonTrackingImporter;
pub use stabilize::{stabilization_for_frame, stabilize_frame, AnchorBand, Stabilization};

/// Errors raised while importing a tracked-point file.
#[derive(Debug, thiserror::Error)]
pub enum TrackingImportError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
}

/// Imported sequence and warnings for skipped rows/points. A malformed row need not discard the run.
#[derive(Debug, Clone)]
pub struct TrackingImport {
    pub sequence: TrackingSequence,
    pub warnings: Vec<String>,
}

/// Parse a tracking format into [`TrackingSequence`].
pub trait TrackingImporter {
    /// Short stable profile name (for UI/diagnostics, e.g. "csv" or
    /// "long-table").
    fn name(&self) -> &'static str;

    /// Parse from an open buffered reader.
    fn import_reader(
        &self,
        reader: &mut dyn BufRead,
    ) -> Result<TrackingImport, TrackingImportError>;

    /// Convenience wrapper that opens `path` and imports it.
    fn import_path(&self, path: &Path) -> Result<TrackingImport, TrackingImportError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        self.import_reader(&mut reader)
    }
}

/// File-name extensions the tracking importers accept when dispatched.
pub const TRACKING_EXTENSIONS: [&str; 4] = ["json", "csv", "txt", "tsv"];

/// Case-insensitive discovery names. `cowtracker.*` is a legacy alias;
/// the adapter normally writes `tracks.json`.
const TRACKING_FILE_NAMES: [&str; 9] = [
    "tracks.json",
    "tracking.json",
    "tracked_points.json",
    "tracks.csv",
    "tracking.csv",
    "cowtracker.csv",
    "tracks.tsv",
    "tracking.tsv",
    "cowtracker.tsv",
];

/// Choose JSON for `.json`, long-table for `.tsv` or names containing `cowtracker`,
/// and comma CSV for other `.csv` files. Sniff `.txt` and unknown extensions: a leading
/// JSON object selects JSON; otherwise use long-table.
pub fn importer_for_path(path: &Path) -> Box<dyn TrackingImporter> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "json" {
        return Box::new(JsonTrackingImporter);
    }
    if name.contains("cowtracker") || ext == "tsv" || ext == "txt" {
        // `.tsv`/CowTracker-named files are tabular by construction; `.txt`
        // may still be JSON so it is sniffed below.
        if ext == "txt" && looks_like_json(path) {
            return Box::new(JsonTrackingImporter);
        }
        return Box::new(LongTableTrackingImporter);
    }
    if ext == "csv" {
        return Box::new(CsvTrackingImporter);
    }
    if looks_like_json(path) {
        Box::new(JsonTrackingImporter)
    } else {
        Box::new(LongTableTrackingImporter)
    }
}

/// Convenience wrapper: import `path` with the automatically dispatched
/// importer (see [`importer_for_path`]).
pub fn import_tracking_path(path: &Path) -> Result<TrackingImport, TrackingImportError> {
    importer_for_path(path).import_path(path)
}

/// Check for a leading JSON object by reading only the first few kilobytes.
fn looks_like_json(path: &Path) -> bool {
    use std::io::Read;
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = Vec::new();
    if file.take(4096).read_to_end(&mut head).is_err() {
        return false;
    }
    head.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'{')
}

/// Search `run_dir`, `images/` and `tracking/` for conventional names.
/// Return sorted, deduplicated candidates; the caller handles ambiguity.
pub fn conventional_tracking_candidates(run_dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let dirs = [
        run_dir.to_owned(),
        run_dir.join("images"),
        run_dir.join("tracking"),
    ];
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let lower = name.to_ascii_lowercase();
            if TRACKING_FILE_NAMES.iter().any(|n| *n == lower) {
                found.push(path);
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

/// Opaque track identity shared across frames, independent of point-list order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TrackId(Arc<str>);

impl TrackId {
    pub fn new(id: impl AsRef<str>) -> Self {
        TrackId(id.as_ref().into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for TrackId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TrackId {
    fn from(s: &str) -> Self {
        TrackId::new(s)
    }
}

impl From<String> for TrackId {
    fn from(s: String) -> Self {
        TrackId::new(s)
    }
}

/// Stable identity of one captured frame, matching `frames.json` `frame_id`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FrameId(pub u32);

impl FrameId {
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for FrameId {
    fn from(v: u32) -> Self {
        FrameId(v)
    }
}

/// Image-pixel observation. Confidence is expected in [0, 1]; absent confidence
/// passes [`TrackedPoint::is_valid`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackedPoint {
    pub x: f64,
    pub y: f64,
    pub confidence: Option<f64>,
}

impl TrackedPoint {
    pub fn new(x: f64, y: f64) -> Self {
        TrackedPoint {
            x,
            y,
            confidence: None,
        }
    }

    pub fn with_confidence(x: f64, y: f64, confidence: f64) -> Self {
        TrackedPoint {
            x,
            y,
            confidence: Some(confidence),
        }
    }

    /// `true` when the coordinates are finite (NaN/inf tracked points are
    /// never usable, regardless of confidence).
    pub fn coords_valid(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Finite coordinates meeting `min_confidence`, or with no confidence reported.
    pub fn is_valid(&self, min_confidence: f64) -> bool {
        self.coords_valid()
            && self
                .confidence
                .is_none_or(|c| c >= min_confidence && c.is_finite())
    }
}

/// Points in import order, indexed by track ID. Last duplicate wins.
#[derive(Debug, Clone)]
pub struct TrackingFrame {
    frame_id: FrameId,
    points: Vec<(TrackId, TrackedPoint)>,
    index: HashMap<TrackId, usize>,
}

impl TrackingFrame {
    pub fn new(
        frame_id: FrameId,
        points: impl IntoIterator<Item = (TrackId, TrackedPoint)>,
    ) -> Self {
        let mut entries: Vec<(TrackId, TrackedPoint)> = Vec::new();
        let mut index: HashMap<TrackId, usize> = HashMap::new();
        for (id, point) in points {
            if let Some(&prev) = index.get(&id) {
                entries[prev] = (id, point);
            } else {
                let pos = entries.len();
                index.insert(id.clone(), pos);
                entries.push((id, point));
            }
        }
        TrackingFrame {
            frame_id,
            points: entries,
            index,
        }
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// All `(track_id, point)` pairs in import order.
    pub fn points(&self) -> &[(TrackId, TrackedPoint)] {
        &self.points
    }

    /// O(1) point lookup by stable track id.
    pub fn get(&self, id: &TrackId) -> Option<&TrackedPoint> {
        self.index.get(id).map(|&i| &self.points[i].1)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TrackId, &TrackedPoint)> {
        self.points.iter().map(|(id, p)| (id, p))
    }

    /// Points passing [`TrackedPoint::is_valid`].
    pub fn valid_points(
        &self,
        min_confidence: f64,
    ) -> impl Iterator<Item = (&TrackId, &TrackedPoint)> {
        self.points
            .iter()
            .filter(move |item| item.1.is_valid(min_confidence))
            .map(|(id, p)| (id, p))
    }
}

/// Frames in acquisition order with ID lookup. Track order follows first appearance.
#[derive(Debug, Clone)]
pub struct TrackingSequence {
    frames: Vec<TrackingFrame>,
    frame_index: HashMap<FrameId, usize>,
    track_order: Vec<TrackId>,
}

impl TrackingSequence {
    /// Keep the last duplicate frame in the first occurrence's position.
    pub fn new(frames: impl IntoIterator<Item = TrackingFrame>) -> Self {
        let mut out: Vec<TrackingFrame> = Vec::new();
        let mut index: HashMap<FrameId, usize> = HashMap::new();
        for frame in frames {
            match index.get(&frame.frame_id()) {
                Some(&prev) => out[prev] = frame,
                None => {
                    index.insert(frame.frame_id(), out.len());
                    out.push(frame);
                }
            }
        }
        // Derive track order after deduplicating frames.
        let mut seen_tracks: HashSet<&TrackId> = HashSet::new();
        let mut track_order: Vec<TrackId> = Vec::new();
        for (id, _) in out.iter().flat_map(TrackingFrame::iter) {
            if seen_tracks.insert(id) {
                track_order.push(id.clone());
            }
        }
        TrackingSequence {
            frames: out,
            frame_index: index,
            track_order,
        }
    }

    pub fn frames(&self) -> &[TrackingFrame] {
        &self.frames
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The frame with this id, O(1).
    pub fn get_frame(&self, id: &FrameId) -> Option<&TrackingFrame> {
        self.frame_index.get(id).map(|&i| &self.frames[i])
    }

    /// Position of `id` in [`TrackingSequence::frames`].
    pub fn frame_index_of(&self, id: &FrameId) -> Option<usize> {
        self.frame_index.get(id).copied()
    }

    pub fn first_frame(&self) -> Option<&TrackingFrame> {
        self.frames.first()
    }

    /// Every track id that appears in any frame, in first-seen order.
    pub fn track_ids(&self) -> &[TrackId] {
        &self.track_order
    }

    /// The point of `track` in `frame`, if that frame tracks it.
    pub fn point(&self, frame: &FrameId, track: &TrackId) -> Option<&TrackedPoint> {
        self.get_frame(frame)?.get(track)
    }

    /// A track's trajectory: valid points across frames, in frame order.
    /// Missing or invalid frames simply do not contribute an entry.
    pub fn trajectory(&self, track: &TrackId, min_confidence: f64) -> Vec<(FrameId, TrackedPoint)> {
        self.frames
            .iter()
            .filter_map(|f| {
                f.get(track)
                    .filter(|p| p.is_valid(min_confidence))
                    .map(|p| (f.frame_id(), *p))
            })
            .collect()
    }

    /// Frame ids in acquisition order.
    pub fn frame_ids(&self) -> impl Iterator<Item = FrameId> + '_ {
        self.frames.iter().map(|f| f.frame_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u32, pts: &[(&str, f64, f64, Option<f64>)]) -> TrackingFrame {
        TrackingFrame::new(
            FrameId(id),
            pts.iter().map(|(n, x, y, c)| {
                let p = match c {
                    Some(c) => TrackedPoint::with_confidence(*x, *y, *c),
                    None => TrackedPoint::new(*x, *y),
                };
                (TrackId::new(*n), p)
            }),
        )
    }

    #[test]
    fn identities_are_stable_across_ordering_changes() {
        // Re-ordering the point list must not change identity/lookup.
        let a = frame(
            1,
            &[
                ("t1", 1.0, 2.0, None),
                ("t2", 3.0, 4.0, None),
                ("t3", 5.0, 6.0, None),
            ],
        );
        assert_eq!(
            a.get(&TrackId::new("t2")),
            Some(&TrackedPoint::new(3.0, 4.0))
        );
        assert_eq!(a.get(&TrackId::new("missing")), None);
    }

    #[test]
    fn duplicate_track_in_frame_keeps_last() {
        let f = TrackingFrame::new(
            FrameId(0),
            vec![
                (TrackId::new("a"), TrackedPoint::new(1.0, 1.0)),
                (TrackId::new("a"), TrackedPoint::new(9.0, 9.0)),
            ],
        );
        assert_eq!(f.len(), 1);
        assert_eq!(
            f.get(&TrackId::new("a")),
            Some(&TrackedPoint::new(9.0, 9.0))
        );
    }

    #[test]
    fn sequence_dedupes_frames_and_indexes_by_id() {
        let seq = TrackingSequence::new(vec![frame(1, &[("a", 0.0, 0.0, None)])]);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.get_frame(&FrameId(1)).unwrap().frame_id(), FrameId(1));
        assert_eq!(seq.frame_index_of(&FrameId(1)), Some(0));
        assert!(seq.get_frame(&FrameId(2)).is_none());
        assert_eq!(seq.frame_index_of(&FrameId(2)), None);
    }

    /// Which of the frames' marker points `x` ends up under each frame id.
    fn xs(seq: &TrackingSequence) -> Vec<(u32, f64)> {
        seq.frames()
            .iter()
            .map(|f| (f.frame_id().as_u32(), f.points()[0].1.x))
            .collect()
    }

    #[test]
    fn duplicate_frame_ids_keep_last_and_index_stays_consistent() {
        // [1, 1, 2]: the second "1" replaces the first in place.
        let seq = TrackingSequence::new(vec![
            frame(1, &[("a", 10.0, 0.0, None)]),
            frame(1, &[("a", 11.0, 0.0, None)]),
            frame(2, &[("a", 20.0, 0.0, None)]),
        ]);
        assert_eq!(xs(&seq), vec![(1, 11.0), (2, 20.0)]);
        assert_eq!(seq.frame_index_of(&FrameId(1)), Some(0));
        assert_eq!(seq.frame_index_of(&FrameId(2)), Some(1));
        assert_eq!(seq.get_frame(&FrameId(1)).unwrap().points()[0].1.x, 11.0);
        assert_eq!(seq.get_frame(&FrameId(2)).unwrap().points()[0].1.x, 20.0);

        // [1, 2, 1]: the late duplicate must not point past the end.
        let seq = TrackingSequence::new(vec![
            frame(1, &[("a", 10.0, 0.0, None)]),
            frame(2, &[("a", 20.0, 0.0, None)]),
            frame(1, &[("a", 12.0, 0.0, None)]),
        ]);
        assert_eq!(xs(&seq), vec![(1, 12.0), (2, 20.0)]);
        assert_eq!(seq.frame_index_of(&FrameId(1)), Some(0));
        assert_eq!(seq.get_frame(&FrameId(1)).unwrap().points()[0].1.x, 12.0);

        // [0, 0]: a single frame, indexed at 0.
        let seq = TrackingSequence::new(vec![
            frame(0, &[("a", 1.0, 0.0, None)]),
            frame(0, &[("a", 2.0, 0.0, None)]),
        ]);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.frame_index_of(&FrameId(0)), Some(0));
        assert_eq!(seq.get_frame(&FrameId(0)).unwrap().points()[0].1.x, 2.0);
    }

    #[test]
    fn track_order_ignores_replaced_duplicate_frames() {
        let seq = TrackingSequence::new(vec![
            frame(1, &[("gone", 0.0, 0.0, None)]),
            frame(1, &[("kept", 0.0, 0.0, None)]),
        ]);
        let ids: Vec<&str> = seq.track_ids().iter().map(|t| t.as_str()).collect();
        assert_eq!(ids, vec!["kept"]);
    }

    #[test]
    fn track_order_is_first_seen() {
        let seq = TrackingSequence::new(vec![
            frame(1, &[("b", 0.0, 0.0, None), ("a", 1.0, 1.0, None)]),
            frame(2, &[("c", 2.0, 2.0, None), ("b", 3.0, 3.0, None)]),
        ]);
        let ids: Vec<&str> = seq.track_ids().iter().map(|t| t.as_str()).collect();
        assert_eq!(ids, vec!["b", "a", "c"]);
    }

    #[test]
    fn point_lookup_and_absent_points() {
        let seq = TrackingSequence::new(vec![frame(1, &[("a", 1.0, 2.0, None)])]);
        assert_eq!(
            seq.point(&FrameId(1), &TrackId::new("a")),
            Some(&TrackedPoint::new(1.0, 2.0))
        );
        assert_eq!(seq.point(&FrameId(1), &TrackId::new("z")), None);
        assert_eq!(seq.point(&FrameId(9), &TrackId::new("a")), None);
    }

    #[test]
    fn trajectory_skips_missing_and_invalid() {
        let seq = TrackingSequence::new(vec![
            frame(1, &[("a", 1.0, 1.0, Some(0.9))]),
            frame(2, &[("b", 5.0, 5.0, None)]),      // a lost here
            frame(3, &[("a", 3.0, 3.0, Some(0.1))]), // below threshold
            frame(4, &[("a", 4.0, 4.0, Some(0.8))]),
        ]);
        let traj = seq.trajectory(&TrackId::new("a"), 0.5);
        let ids: Vec<u32> = traj.iter().map(|(f, _)| f.as_u32()).collect();
        assert_eq!(ids, vec![1, 4]);
    }

    #[test]
    fn validity_rejects_nan_and_low_confidence() {
        let mut p = TrackedPoint::with_confidence(f64::NAN, 2.0, 1.0);
        assert!(!p.is_valid(0.0));
        p.x = 1.0;
        assert!(p.is_valid(0.0));
        let low = TrackedPoint::with_confidence(1.0, 2.0, 0.5);
        assert!(!low.is_valid(0.9), "0.5 < 0.9 must be rejected");
        assert!(low.is_valid(0.4), "0.5 >= 0.4 must be accepted");
        let no_conf = TrackedPoint::new(1.0, 1.0);
        assert!(no_conf.is_valid(0.9), "missing confidence never fails");
        let inf = TrackedPoint::new(1.0, f64::INFINITY);
        assert!(!inf.is_valid(0.0));
    }

    #[test]
    fn import_trait_path_and_reader_agree() {
        use std::io::Cursor;
        let json = br#"{"frames":[{"frame_id":7,"points":[{"id":"a","x":1,"y":2}]}]}"#;
        let importer = JsonTrackingImporter;
        let from_reader = importer
            .import_reader(&mut Cursor::new(json.to_vec()))
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tracks.json");
        std::fs::write(&path, json).unwrap();
        let from_path = importer.import_path(&path).unwrap();
        assert_eq!(from_reader.sequence.len(), from_path.sequence.len());
        assert_eq!(
            from_reader
                .sequence
                .get_frame(&FrameId(7))
                .unwrap()
                .frame_id(),
            FrameId(7)
        );
    }

    fn write_text(dir: &Path, name: &str, text: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }

    fn tmp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn dispatch_picks_importer_by_extension_and_name() {
        let tmp = tmp_dir();
        let dir = tmp.path();
        let j = write_text(dir, "a.json", r#"{"frames":[]}"#);
        let c = write_text(dir, "a.csv", "frame_id,track_id,x,y\n");
        let t = write_text(dir, "a.tsv", "frame_id\ttrack_id\tx\ty\n");
        let cow = write_text(dir, "cowtracker.csv", "frame,x,y,id\n");
        let cow_upper = write_text(dir, "CowTracker.csv", "frame,x,y,id\n");
        let txt = write_text(dir, "a.txt", "frame\tx\ty\tid\n");
        let txt_json = write_text(dir, "b.txt", r#"{"frames":[]}"#);
        let noext = write_text(dir, "tracks", "frame_id,track_id,x,y\n");
        let noext_json = write_text(dir, "t2", r#"{"frames":[]}"#);

        assert_eq!(importer_for_path(&j).name(), "json");
        assert_eq!(importer_for_path(&c).name(), "csv");
        assert_eq!(importer_for_path(&t).name(), "long-table");
        assert_eq!(importer_for_path(&cow).name(), "long-table");
        assert_eq!(importer_for_path(&cow_upper).name(), "long-table");
        assert_eq!(importer_for_path(&txt).name(), "long-table");
        assert_eq!(importer_for_path(&txt_json).name(), "json");
        assert_eq!(importer_for_path(&noext).name(), "long-table");
        assert_eq!(importer_for_path(&noext_json).name(), "json");
    }

    #[test]
    fn json_sniffing_only_looks_at_the_head_of_the_file() {
        let tmp = tmp_dir();
        let dir = tmp.path();
        // Valid JSON start followed by bytes that are not UTF-8.
        let mut blob = b"  \n{\"frames\":[]}".to_vec();
        blob.extend(std::iter::repeat_n(0xff, 100_000));
        let json_like = dir.join("blob.dat");
        std::fs::write(&json_like, blob).unwrap();
        assert_eq!(importer_for_path(&json_like).name(), "json");

        let table = write_text(dir, "table.dat", "frame_id,track_id,x,y\n");
        assert_eq!(importer_for_path(&table).name(), "long-table");
        let missing = dir.join("missing.dat");
        assert_eq!(importer_for_path(&missing).name(), "long-table");
    }

    #[test]
    fn import_tracking_path_loads_via_dispatch() {
        let tmp = tmp_dir();
        let dir = tmp.path();
        let tsv = write_text(dir, "run.tsv", "frame_id\ttrack_id\tx\ty\n1\ta\t1.0\t2.0\n");
        let outcome = import_tracking_path(&tsv).unwrap();
        assert_eq!(outcome.sequence.len(), 1);
        assert_eq!(outcome.sequence.get_frame(&FrameId(1)).unwrap().len(), 1);
        let json = write_text(
            dir,
            "run.json",
            r#"{"frames":[{"frame_id":2,"points":[]}]}"#,
        );
        let outcome = import_tracking_path(&json).unwrap();
        assert!(outcome.sequence.get_frame(&FrameId(2)).is_some());
    }

    #[test]
    fn conventional_candidates_scan_run_dir_and_subdirs() {
        let tmp = tmp_dir();
        let dir = tmp.path();
        let run = dir.join("run");
        std::fs::create_dir_all(run.join("images")).unwrap();
        std::fs::create_dir_all(run.join("tracking")).unwrap();
        write_text(&run, "tracks.json", "{}");
        write_text(&run.join("images"), "tracking.csv", "x");
        write_text(&run.join("tracking"), "CowTracker.csv", "x");
        write_text(&run, "points.csv", "x");
        write_text(&run, "tracks.dat", "x");

        let candidates = conventional_tracking_candidates(&run);
        let mut names: Vec<String> = candidates
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec!["CowTracker.csv", "tracking.csv", "tracks.json"],
            "conventional names found, others ignored"
        );
        assert!(conventional_tracking_candidates(dir).is_empty());
    }

    #[test]
    fn cowtracker_named_files_still_dispatch_to_long_table() {
        // Legacy filenames containing "cowtracker" keep dispatching to the
        // generic long-table importer (case-insensitively).
        let tmp = tmp_dir();
        let dir = tmp.path();
        for name in ["cowtracker.csv", "CowTracker.csv", "COWTRACKER.TSV"] {
            let p = write_text(dir, name, "frame_id,track_id,x,y\n0,a,1.0,2.0\n");
            let importer = importer_for_path(&p);
            assert_eq!(importer.name(), "long-table", "{name}");
            assert!(
                importer.import_path(&p).is_ok(),
                "{name} must import via the long-table profile"
            );
        }
    }
}
