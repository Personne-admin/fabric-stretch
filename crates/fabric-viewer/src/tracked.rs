//! Per-run tracking imports, settings and persistence. Math lives in `fabric_data::tracking`.
//! State uses canonical run paths so reordering projects does not change its identity.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use fabric_data::tracking::registration::{
    compute_registration, ReferenceFrame, RegistrationMode, RegistrationResult, RegistrationState,
};
use fabric_data::tracking::stabilize::{stabilization_for_frame, Stabilization};
use fabric_data::tracking::{FrameId, TrackId, TrackedPoint, TrackingSequence};
use serde::{Deserialize, Serialize};

use crate::playback::PlaybackState;

pub const DEFAULT_HISTORY: usize = 8;

/// How the canvas treats the registered image and the (fixed) reference
/// frame's bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanvasMode {
    /// The registered image is drawn un-clipped (may extend past the
    /// reference frame's bounds).
    Normal,
    /// Clip to reference bounds; fill uncovered regions without clamping image edges.
    ReferenceBounds,
    /// Reference bounds plus a subtle outline marking the valid image
    /// region.
    ValidBoundary,
}

impl CanvasMode {
    pub const ALL: [CanvasMode; 3] = [
        CanvasMode::Normal,
        CanvasMode::ReferenceBounds,
        CanvasMode::ValidBoundary,
    ];

    pub fn label(self) -> &'static str {
        match self {
            CanvasMode::Normal => "Normal",
            CanvasMode::ReferenceBounds => "Reference bounds",
            CanvasMode::ValidBoundary => "Valid boundary",
        }
    }

    /// Clipping to the reference bounds is used by both
    /// [`CanvasMode::ReferenceBounds`] and [`CanvasMode::ValidBoundary`].
    pub fn clips_to_reference(self) -> bool {
        !matches!(self, CanvasMode::Normal)
    }
}

/// Which tracks the overlay shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackScope {
    /// Every track with a valid point in the current frame.
    AllValid,
    /// Only the tracks in the registration reference set.
    ReferenceOnly,
}

/// Overlay and canvas preferences (global, persisted).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OverlaySettings {
    pub scope: TrackScope,
    /// Trajectory history length in frames.
    pub history_length: usize,
    /// Main overlay: current (registered) point markers.
    pub show_points: bool,
    pub show_ids: bool,
    pub show_trajectories: bool,
    pub show_displacement_vectors: bool,
    /// Diagnostics: raw (un-registered) current points.
    pub show_raw_points: bool,
    /// Separate diagnostic toggle for transformed points.
    pub show_transformed_points: bool,
    /// Diagnostics: the reference frame's ghost points.
    pub show_reference_ghost: bool,
    /// Diagnostics: displacement lines from reference location to current.
    pub show_displacement_lines: bool,
    /// Diagnostics: reference location markers.
    pub show_reference_locations: bool,
    pub canvas_mode: CanvasMode,
}

impl Default for OverlaySettings {
    fn default() -> Self {
        OverlaySettings {
            scope: TrackScope::AllValid,
            history_length: DEFAULT_HISTORY,
            show_points: true,
            show_ids: false,
            show_trajectories: true,
            show_displacement_vectors: false,
            show_raw_points: false,
            show_transformed_points: false,
            show_reference_ghost: true,
            show_displacement_lines: false,
            show_reference_locations: false,
            canvas_mode: CanvasMode::ReferenceBounds,
        }
    }
}

/// The whole CoWTracker stabilization mode, as two switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StabilizeSettings {
    /// Hold the upper tracked fabric band still while replaying.
    pub enabled: bool,
    /// Draw points with the image's transform.
    pub show_points: bool,
}

/// What the canvas paints on top of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverlayStyle {
    /// The legacy diagnostic overlay driven by [`OverlaySettings`].
    Full,
    /// The simple tracked-point overlay of the stabilization mode.
    Points,
    /// Nothing (stabilized replay with the point overlay switched off).
    Hidden,
}

impl OverlayStyle {
    /// Allow picking only when points are visible.
    pub fn is_pickable(self) -> bool {
        !matches!(self, OverlayStyle::Hidden)
    }
}

/// One track's overlay data for a single frame, fully owned so the canvas can
/// paint without borrowing the session.
#[derive(Debug, Clone)]
pub struct OverlayTrack {
    pub id: TrackId,
    /// The current frame's raw (un-registered) position.
    pub current: TrackedPoint,
    /// The reference frame's position for this track, if tracked there.
    pub reference: Option<TrackedPoint>,
    /// Whether this track is part of the registration reference set.
    pub is_reference: bool,
    /// Previous frames' positions (image space), for trajectory history.
    pub history: Vec<TrackedPoint>,
}

/// Everything the canvas needs to paint the tracked overlay for one frame.
#[derive(Debug, Clone)]
pub struct OverlayPaintData {
    pub tracks: Vec<OverlayTrack>,
    /// The reference frame's full (filtered) point set, for ghost markers.
    pub reference_points: Option<Vec<(TrackId, TrackedPoint)>>,
    pub selected_track: Option<TrackId>,
    pub flags: OverlaySettings,
    /// Which overlay the canvas paints (the legacy diagnostics, the simple
    /// tracked points, or nothing).
    pub style: OverlayStyle,
    /// Stabilization offset when enabled, otherwise registration. Shared by the image and overlay.
    pub transform: fabric_data::tracking::registration::RegistrationTransform,
}

/// Everything the viewer keeps per run once tracking data is loaded.
#[derive(Debug, Default)]
pub struct TrackedRunState {
    /// Stable canonical run directory (the state key).
    pub run_key: PathBuf,
    pub sequence: Option<TrackingSequence>,
    /// Saved import path; reload the data in a later session.
    pub import_path: Option<PathBuf>,
    pub import_warnings: Vec<String>,
    pub error: Option<String>,
    pub state: RegistrationState,
    /// Ambiguous discovery candidates, awaiting the user's choice.
    pub auto_candidates: Vec<PathBuf>,
    /// Frame-id alignment warning against the run's `frames.json` (1-based vs
    /// 0-based numbering, low/no overlap).
    pub alignment_warning: Option<String>,
    /// Whether [`alignment_warning`](Self::alignment_warning) is up to date
    /// (reset by every import and unload).
    pub alignment_checked: bool,
    /// Discovery has run, or the user unloaded the data. Prevents re-imports on repaint.
    pub discovery_done: bool,
    /// The file picked in the "several tracking files found" chooser.
    pub candidate_choice: Option<PathBuf>,
    /// Text of the advanced path field while it is being edited.
    pub path_edit: Option<String>,
}

/// Persisted run choices, without source data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedRunTracking {
    pub mode: RegistrationMode,
    pub reference_frame: Option<FrameId>,
    pub reference_tracks: BTreeSet<TrackId>,
    pub min_confidence: f64,
    pub loading_axis_x: f64,
    pub loading_axis_y: f64,
    /// Persist the import path, not the sequence.
    pub import_path: Option<PathBuf>,
}

impl Default for PersistedRunTracking {
    fn default() -> Self {
        PersistedRunTracking {
            mode: RegistrationMode::Off,
            reference_frame: None,
            reference_tracks: BTreeSet::new(),
            min_confidence: 0.0,
            loading_axis_x: 1.0,
            loading_axis_y: 0.0,
            import_path: None,
        }
    }
}

/// Persisted run and overlay settings. Playback preferences stay here for compatibility;
/// playback state itself lives on the app. Imported data is not saved.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedTracking {
    /// Keyed by canonical run directory (as lossy string).
    pub runs: BTreeMap<String, PersistedRunTracking>,
    pub overlay: OverlaySettings,
    pub playback: PlaybackState,
    pub selected_track: Option<TrackId>,
    /// Stabilized replay is off by default.
    #[serde(default)]
    pub stabilize: StabilizeSettings,
}

/// The app-level tracking session.
#[derive(Debug)]
pub struct TrackingSession {
    pub runs: Vec<TrackedRunState>,
    /// The click-selected track (drives the inspector readout).
    pub selected_track: Option<TrackId>,
    /// Transient pointer-hover highlight in the image viewer overlay.
    pub hovered_track: Option<TrackId>,
    /// Ctrl/Shift-accumulated multi-selection (candidate reference set).
    pub multi_select: BTreeSet<TrackId>,
    pub overlay: OverlaySettings,
    /// The CoWTracker stabilized-replay switches (global, persisted).
    pub stabilize: StabilizeSettings,
    /// Persisted settings applied to runs the first time they are touched.
    persisted: PersistedTracking,
}

impl Default for TrackingSession {
    fn default() -> Self {
        TrackingSession::new(PersistedTracking::default())
    }
}

impl TrackingSession {
    pub fn new(persisted: PersistedTracking) -> Self {
        let mut session = TrackingSession {
            runs: Vec::new(),
            selected_track: persisted.selected_track.clone(),
            hovered_track: None,
            multi_select: BTreeSet::new(),
            overlay: persisted.overlay.clone(),
            stabilize: persisted.stabilize,
            persisted,
        };
        session.overlay = session.persisted.overlay.clone();
        session.stabilize = session.persisted.stabilize;
        session
    }

    pub fn run_state(&self, run_key: &Path) -> Option<&TrackedRunState> {
        self.runs.iter().find(|r| r.run_key == run_key)
    }

    pub fn run_state_mut(&mut self, run_key: &Path) -> Option<&mut TrackedRunState> {
        self.runs.iter_mut().find(|r| r.run_key == run_key)
    }

    /// Get-or-create a run state, applying any persisted per-run settings.
    pub fn ensure_run(&mut self, run_key: &Path) -> &mut TrackedRunState {
        let key = run_key.to_owned();
        if !self.runs.iter().any(|r| r.run_key == key) {
            let persisted = self
                .persisted
                .runs
                .get(&key.to_string_lossy().into_owned())
                .cloned()
                .unwrap_or_default();
            let mut state = TrackedRunState {
                run_key: key,
                import_path: persisted.import_path.clone(),
                ..Default::default()
            };
            state.state.mode = persisted.mode;
            state.state.reference_frame = persisted.reference_frame;
            state.state.reference_tracks = persisted.reference_tracks;
            state.state.min_confidence = persisted.min_confidence;
            state.state.loading_axis_x = persisted.loading_axis_x;
            state.state.loading_axis_y = persisted.loading_axis_y;
            self.runs.push(state);
        }
        self.runs
            .iter_mut()
            .find(|r| r.run_key == run_key)
            .expect("run state was just created")
    }

    /// Drop every run state not in `open` (used when projects close).
    pub fn retain_open_runs(&mut self, open: &BTreeSet<PathBuf>) {
        self.runs.retain(|r| open.contains(&r.run_key));
    }

    /// Import JSON or tabular tracks via [`fabric_data::tracking::importer_for_path`].
    pub fn import_file(&mut self, run_key: &Path, path: &Path) -> Result<(), String> {
        let run = self.ensure_run(run_key);
        let outcome = match fabric_data::tracking::import_tracking_path(path) {
            Ok(outcome) => outcome,
            Err(err) => {
                let message = format!("failed to load {}: {err}", path.display());
                run.error = Some(message.clone());
                return Err(message);
            }
        };
        if run.state.reference_frame.is_none() {
            // Default reference: the first tracking frame of the run.
            run.state.reference_frame = outcome.sequence.first_frame().map(|f| f.frame_id());
        }
        run.sequence = Some(outcome.sequence);
        run.import_path = Some(path.to_owned());
        run.import_warnings = outcome.warnings;
        run.error = None;
        run.auto_candidates.clear();
        run.candidate_choice = None;
        run.path_edit = None;
        run.alignment_warning = None;
        run.alignment_checked = false;
        run.discovery_done = true;
        Ok(())
    }

    /// Reload a saved import path if no sequence is loaded. Return whether an import happened.
    /// Keep missing paths so the user can fix them.
    fn auto_reload(&mut self, run_key: &Path) -> Result<bool, String> {
        let path = {
            let run = self.ensure_run(run_key);
            if run.sequence.is_some() {
                return Ok(false);
            }
            run.import_path.clone()
        };
        let Some(path) = path else {
            return Ok(false);
        };
        if !path.is_file() {
            let message = format!("tracking import path no longer exists: {}", path.display());
            if let Some(run) = self.run_state_mut(run_key) {
                run.error = Some(message.clone());
            }
            return Err(message);
        }
        self.import_file(run_key, &path)?;
        Ok(true)
    }

    /// Discover once per run, preferring its saved import path. Import a sole candidate;
    /// store multiple candidates in [`TrackedRunState::auto_candidates`] for user selection.
    /// Failed imports and explicit unloads are not retried on repaint.
    pub fn auto_discover(&mut self, run_key: &Path) -> Result<(), String> {
        let run = self.ensure_run(run_key);
        if run.discovery_done {
            return Ok(());
        }
        run.discovery_done = true;
        if run.import_path.is_some() {
            return self.auto_reload(run_key).map(|_| ());
        }
        let candidates = fabric_data::tracking::conventional_tracking_candidates(run_key);
        match candidates.len() {
            0 => Ok(()),
            1 => self.import_file(run_key, &candidates[0]),
            _ => {
                // Ask the user when discovery is ambiguous.
                if let Some(run) = self.run_state_mut(run_key) {
                    run.candidate_choice = candidates.first().cloned();
                    run.auto_candidates = candidates;
                }
                Ok(())
            }
        }
    }

    /// Unload tracks for the session without resetting registration choices.
    /// Discovery stays disabled until an explicit reload.
    pub fn clear_run(&mut self, run_key: &Path) {
        if let Some(run) = self.run_state_mut(run_key) {
            run.sequence = None;
            run.import_path = None;
            run.import_warnings.clear();
            run.error = None;
            run.auto_candidates.clear();
            run.candidate_choice = None;
            run.path_edit = None;
            run.alignment_warning = None;
            run.alignment_checked = false;
            run.discovery_done = true;
        }
    }

    /// Load the same tracking filename in other open, unloaded runs.
    /// Returns `(loaded, missing)` counts.
    pub fn load_similar_for_all_runs(&mut self, source_path: &Path) -> (usize, usize) {
        let Some(file_name) = source_path.file_name().map(PathBuf::from) else {
            return (0, 0);
        };
        let keys: Vec<PathBuf> = self.runs.iter().map(|r| r.run_key.clone()).collect();
        let mut loaded = 0;
        let mut missing = 0;
        for key in keys {
            if self
                .runs
                .iter()
                .any(|r| r.run_key == key && r.sequence.is_some())
            {
                continue;
            }
            let candidate = key.join(&file_name);
            if candidate.is_file() {
                if self.import_file(&key, &candidate).is_ok() {
                    loaded += 1;
                }
            } else {
                missing += 1;
            }
        }
        (loaded, missing)
    }

    /// Whether frame IDs still need checking after import.
    pub fn needs_alignment(&self, run_key: &Path) -> bool {
        self.run_state(run_key)
            .is_some_and(|r| r.sequence.is_some() && !r.alignment_checked)
    }

    /// Check imported frame IDs against `frames.json` once per import.
    /// Store no warning for an exact match or an unloaded sequence.
    pub fn refresh_alignment(&mut self, run_key: &Path, run_frame_ids: &[u32]) {
        let Some(run) = self.run_state(run_key) else {
            return;
        };
        let Some(sequence) = run.sequence.as_ref() else {
            return;
        };
        let alignment = fabric_data::tracking::align_tracking_frames(sequence, run_frame_ids);
        let warning = if alignment.status.is_warning() {
            Some(alignment.note)
        } else {
            None
        };
        if let Some(run) = self.run_state_mut(run_key) {
            run.alignment_warning = warning;
            run.alignment_checked = true;
        }
    }

    /// Export tracks using the run's registration settings.
    pub fn export_registered_csv(
        &mut self,
        run_key: &Path,
        run: &fabric_data::Run,
        target: &Path,
    ) -> Result<(), String> {
        let run_state = self
            .run_state(run_key)
            .ok_or_else(|| "no tracking loaded for this run".to_owned())?;
        let sequence = run_state
            .sequence
            .as_ref()
            .ok_or_else(|| "no tracking sequence loaded".to_owned())?;
        let frame_meta: Vec<(FrameId, fabric_data::tracking::RunFrameMeta)> = run
            .frames
            .frames
            .iter()
            .map(|f| {
                (
                    FrameId(f.frame_id),
                    fabric_data::tracking::RunFrameMeta {
                        displacement_mm: Some(f.displacement_mm),
                        force_n: run.force_n_for_frame(f),
                    },
                )
            })
            .collect();
        let csv =
            fabric_data::tracking::registered_tracks_csv(sequence, &run_state.state, &frame_meta);
        std::fs::write(target, csv)
            .map_err(|e| format!("failed to write {}: {e}", target.display()))?;
        Ok(())
    }

    /// Whether any run has tracking data imported.
    pub fn any_loaded(&self) -> bool {
        self.runs.iter().any(|r| r.sequence.is_some())
    }

    /// Reference positions filtered by track selection and confidence.
    pub fn reference_for(&self, run: &TrackedRunState) -> Option<ReferenceFrame> {
        let sequence = run.sequence.as_ref()?;
        let frame_id = run.state.reference_frame?;
        let frame = sequence.get_frame(&frame_id)?;
        Some(ReferenceFrame::from_frame(
            frame,
            &run.state.reference_tracks,
            run.state.min_confidence,
        ))
    }

    /// Compute registration for this frame. `None` means off or insufficient data;
    /// the viewer then uses the identity transform.
    pub fn result_for(
        &self,
        run: &TrackedRunState,
        current_frame_id: FrameId,
    ) -> Option<RegistrationResult> {
        if run.state.mode.is_off() {
            return None;
        }
        let sequence = run.sequence.as_ref()?;
        let reference = self.reference_for(run)?;
        let current = sequence.get_frame(&current_frame_id)?;
        Some(compute_registration(&reference, current, &run.state))
    }

    /// Use the chosen reference frame, falling back to the first tracking frame.
    pub fn stabilization_reference(&self, run: &TrackedRunState) -> Option<FrameId> {
        let sequence = run.sequence.as_ref()?;
        run.state
            .reference_frame
            .filter(|id| sequence.get_frame(id).is_some())
            .or_else(|| sequence.first_frame().map(|f| f.frame_id()))
    }

    /// Inverse anchor-band displacement for the displayed frame, or `None` for missing data.
    /// The caller decides whether stabilization is enabled.
    pub fn stabilization_for(
        &self,
        run: &TrackedRunState,
        current_frame_id: FrameId,
    ) -> Option<Stabilization> {
        let sequence = run.sequence.as_ref()?;
        let reference = self.stabilization_reference(run)?;
        stabilization_for_frame(sequence, reference, current_frame_id)
    }

    /// The point overlay takes priority; otherwise use the legacy diagnostic overlay.
    pub fn overlay_style(&self) -> OverlayStyle {
        if self.stabilize.show_points {
            OverlayStyle::Points
        } else if self.stabilize.enabled {
            OverlayStyle::Hidden
        } else {
            OverlayStyle::Full
        }
    }

    /// Rebuild the reference track set for the run (used after selection
    /// changes). Returns the new set for diagnostics.
    pub fn set_reference_tracks(&mut self, run_key: &Path, tracks: BTreeSet<TrackId>) {
        if let Some(run) = self.run_state_mut(run_key) {
            run.state.reference_tracks = tracks;
        }
    }

    /// `true` when a track is part of the run's reference set.
    pub fn is_reference_track(&self, run: &TrackedRunState, track: &TrackId) -> bool {
        run.state.reference_tracks.is_empty() || run.state.reference_tracks.contains(track)
    }

    /// Select a clear top band with at least three tracks in the first frame.
    pub fn select_top_row(&mut self, run_key: &Path) -> Option<BTreeSet<TrackId>> {
        let run = self.run_state(run_key)?;
        let sequence = run.sequence.as_ref()?;
        let first = sequence.first_frame()?;
        let mut ys: Vec<f64> = first
            .valid_points(run.state.min_confidence)
            .map(|(_, p)| p.y)
            .collect();
        if ys.len() < 3 {
            return None;
        }
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min_y = ys[0];
        let max_y = *ys.last().unwrap();
        let band = min_y + (max_y - min_y) * 0.25;
        let top: BTreeSet<TrackId> = first
            .valid_points(run.state.min_confidence)
            .filter(|(_, p)| p.y <= band)
            .map(|(id, _)| id.clone())
            .collect();
        if top.len() < 3 {
            return None;
        }
        let result = top.clone();
        self.selected_track = top.iter().next().cloned();
        self.multi_select = top;
        Some(result)
    }

    /// The current frame's valid points for a run, as `(track_id, point)`.
    pub fn points_in<'a>(
        &self,
        run: &'a TrackedRunState,
        frame_id: FrameId,
    ) -> Vec<(&'a TrackId, &'a TrackedPoint)> {
        run.sequence
            .as_ref()
            .and_then(|s| s.get_frame(&frame_id))
            .map(|f| f.valid_points(run.state.min_confidence).collect())
            .unwrap_or_default()
    }

    /// A track's trajectory across the run (ordered by frame).
    pub fn trajectory(
        &self,
        run: &TrackedRunState,
        track: &TrackId,
    ) -> Vec<(FrameId, TrackedPoint)> {
        run.sequence
            .as_ref()
            .map(|s| s.trajectory(track, run.state.min_confidence))
            .unwrap_or_default()
    }

    /// Frame ids of a run's tracking sequence.
    pub fn frame_ids(&self, run: &TrackedRunState) -> Vec<FrameId> {
        run.sequence
            .as_ref()
            .map(|s| s.frame_ids().collect())
            .unwrap_or_default()
    }

    pub fn to_persisted(&self) -> PersistedTracking {
        let default_state = RegistrationState::default();
        let runs = self
            .runs
            .iter()
            .filter(|r| r.sequence.is_some() || r.import_path.is_some() || r.state != default_state)
            .map(|r| {
                (
                    r.run_key.to_string_lossy().into_owned(),
                    PersistedRunTracking {
                        mode: r.state.mode,
                        reference_frame: r.state.reference_frame,
                        reference_tracks: r.state.reference_tracks.clone(),
                        min_confidence: r.state.min_confidence,
                        loading_axis_x: r.state.loading_axis_x,
                        loading_axis_y: r.state.loading_axis_y,
                        import_path: r.import_path.clone(),
                    },
                )
            })
            .collect();
        PersistedTracking {
            runs,
            overlay: self.overlay.clone(),
            playback: self.persisted.playback.clone(),
            selected_track: self.selected_track.clone(),
            stabilize: self.stabilize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_tracking_file(tag: &str, contents: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fabsim_tracked_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tracks.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(contents).unwrap();
        path
    }

    const SAMPLE_JSON: &[u8] = br#"{"frames":[
        {"frame_id":0,"points":[{"id":"a","x":10.0,"y":20.0},{"id":"b","x":30.0,"y":40.0}]},
        {"frame_id":1,"points":[{"id":"a","x":15.0,"y":17.0},{"id":"b","x":35.0,"y":37.0}]}
    ]}"#;

    fn sample_run_key() -> PathBuf {
        PathBuf::from("/project/sample_a/warp_loading")
    }

    #[test]
    fn import_sets_reference_to_first_tracking_frame() {
        let path = temp_tracking_file("ref", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        let run = session.run_state(&key).unwrap();
        assert_eq!(run.state.reference_frame, Some(FrameId(0)));
        assert_eq!(run.sequence.as_ref().unwrap().len(), 2);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn result_for_computes_registration_for_frame() {
        let path = temp_tracking_file("res", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        session.run_state_mut(&key).unwrap().state.mode = RegistrationMode::TranslationXY;
        let run = session.run_state(&key).unwrap();
        let result = session.result_for(run, FrameId(1)).unwrap();
        assert_eq!(result.mode_used, RegistrationMode::TranslationXY);
        let (tx, ty) = result.transform.translation();
        assert!((tx - 5.0).abs() < 1e-9, "tx={tx}");
        assert!((ty + 3.0).abs() < 1e-9, "ty={ty}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn persist_round_trip_preserves_choices_not_data() {
        let path = temp_tracking_file("persist", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        {
            let run = session.run_state_mut(&key).unwrap();
            run.state.mode = RegistrationMode::Rigid;
            run.state.reference_frame = Some(FrameId(1));
            run.state.reference_tracks.insert(TrackId::new("b"));
            run.state.min_confidence = 0.7;
        }
        let persisted = session.to_persisted();
        assert!(persisted
            .runs
            .contains_key(&key.to_string_lossy().into_owned()));

        // Persisted choices apply when the run is first accessed.
        let mut session2 = TrackingSession::new(persisted);
        session2.ensure_run(&key);
        let run = session2.run_state(&key).unwrap();
        assert_eq!(run.state.mode, RegistrationMode::Rigid);
        assert_eq!(run.state.reference_frame, Some(FrameId(1)));
        assert!(run.state.reference_tracks.contains(&TrackId::new("b")));
        assert_eq!(run.state.min_confidence, 0.7);
        // Imported sequences must not enter the saved settings.
        assert!(run.sequence.is_none());

        // Round trip through serde_json (the same code path eframe uses).
        let text = serde_json::to_string(&session2.to_persisted()).unwrap();
        let back: PersistedTracking = serde_json::from_str(&text).unwrap();
        let mut session3 = TrackingSession::new(back);
        session3.ensure_run(&key);
        assert_eq!(
            session3.run_state(&key).unwrap().state.reference_frame,
            Some(FrameId(1))
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn navigation_does_not_reset_registration_state() {
        // Navigation changes selection, not registration state.
        let path = temp_tracking_file("nav", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        session.run_state_mut(&key).unwrap().state.mode = RegistrationMode::TranslationXY;

        let before = session.to_persisted();
        // Selection is independent of the tracking session.
        let _ = session.result_for(session.run_state(&key).unwrap(), FrameId(0));
        let after = session.to_persisted();
        assert_eq!(before, after, "navigation must not mutate tracking state");
        assert_eq!(
            session.run_state(&key).unwrap().state.mode,
            RegistrationMode::TranslationXY
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn retain_open_runs_removes_only_closed_runs() {
        let path = temp_tracking_file("forget", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let a = PathBuf::from("/p/a");
        let b = PathBuf::from("/p/b");
        session.import_file(&a, &path).unwrap();
        session.import_file(&b, &path).unwrap();
        assert_eq!(session.runs.len(), 2);
        let open: BTreeSet<PathBuf> = BTreeSet::from([b.clone()]);
        session.retain_open_runs(&open);
        assert_eq!(session.runs.len(), 1);
        assert_eq!(session.runs[0].run_key, b);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn select_top_row_requires_a_defensible_band() {
        let json = br#"{"frames":[{"frame_id":0,"points":[
            {"id":"a","x":10.0,"y":10.0},
            {"id":"b","x":30.0,"y":12.0},
            {"id":"c","x":50.0,"y":11.0},
            {"id":"d","x":70.0,"y":500.0},
            {"id":"e","x":90.0,"y":510.0}
        ]}]}"#;
        let path = temp_tracking_file("toprow", json);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        let top = session.select_top_row(&key).unwrap();
        assert_eq!(top.len(), 3, "top band has three tracks");
        assert!(top.contains(&TrackId::new("a")));
        assert!(top.contains(&TrackId::new("c")));
        assert!(!top.contains(&TrackId::new("d")));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn select_top_row_is_none_when_band_is_thin() {
        let json = br#"{"frames":[{"frame_id":0,"points":[
            {"id":"a","x":10.0,"y":10.0},
            {"id":"b","x":30.0,"y":12.0}
        ]}]}"#;
        let path = temp_tracking_file("thin", json);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        assert_eq!(session.select_top_row(&key), None);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn trajectory_helper_is_analysis_ready() {
        let path = temp_tracking_file("traj", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        let run = session.run_state(&key).unwrap();
        let traj = session.trajectory(run, &TrackId::new("a"));
        let xs: Vec<f64> = traj.iter().map(|(_, p)| p.x).collect();
        assert_eq!(xs, vec![10.0, 15.0]);
        let pts = session.points_in(run, FrameId(0));
        assert_eq!(pts.len(), 2);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    fn run_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fabsim_runs_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn import_path_is_persisted_and_auto_reloaded() {
        let dir = run_dir("autoreload");
        let tracking_file = dir.join("tracking.json");
        std::fs::write(&tracking_file, SAMPLE_JSON).unwrap();

        let key = dir.join("warp_loading");
        let mut session = TrackingSession::default();
        session.import_file(&key, &tracking_file).unwrap();
        let persisted = session.to_persisted();
        let entry = persisted
            .runs
            .get(&key.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(entry.import_path.as_deref(), Some(tracking_file.as_path()));

        // A fresh session restores the path and re-imports it (file exists).
        let mut session2 = TrackingSession::new(persisted);
        assert!(session2.auto_reload(&key).unwrap(), "re-imported");
        assert!(session2.run_state(&key).unwrap().sequence.is_some());
        assert_eq!(
            session2.run_state(&key).unwrap().import_path.as_deref(),
            Some(tracking_file.as_path())
        );

        // A missing file reports an error and keeps the path.
        std::fs::remove_file(&tracking_file).unwrap();
        let mut session3 = TrackingSession::new(session.to_persisted());
        assert!(
            session3.auto_reload(&key).is_err(),
            "missing file is an error"
        );
        let run = session3.run_state(&key).unwrap();
        assert!(run.sequence.is_none());
        assert!(run
            .error
            .as_deref()
            .is_some_and(|e| e.contains("no longer exists")));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn auto_discover_imports_a_single_conventional_file() {
        let dir = run_dir("disc1");
        let run = dir.join("warp_loading");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(
            run.join("tracks.csv"),
            "frame_id,track_id,x,y\n0,a,1.0,2.0\n1,a,2.0,3.0\n",
        )
        .unwrap();

        let mut session = TrackingSession::default();
        session.auto_discover(&run).unwrap();
        let state = session.run_state(&run).unwrap();
        assert!(state.sequence.is_some(), "single candidate imported");
        assert_eq!(
            state.import_path.as_deref(),
            Some(run.join("tracks.csv").as_path())
        );

        // Idempotent: an already loaded run is left alone.
        session.auto_discover(&run).unwrap();
        assert!(session.run_state(&run).unwrap().sequence.is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unloaded_run_stays_unloaded_across_discovery_calls() {
        // Unload must survive the next repaint's discovery call.
        let dir = run_dir("unload_sticks");
        let run = dir.join("warp_loading");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("tracks.json"), SAMPLE_JSON).unwrap();

        let mut session = TrackingSession::default();
        session.auto_discover(&run).unwrap();
        assert!(session.run_state(&run).unwrap().sequence.is_some());
        session.clear_run(&run);
        for _ in 0..3 {
            session.auto_discover(&run).unwrap();
        }
        let state = session.run_state(&run).unwrap();
        assert!(state.sequence.is_none() && state.import_path.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_failing_import_is_reported_once_and_not_retried() {
        let dir = run_dir("failing");
        let run = dir.join("warp_loading");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("tracks.json"), "{ not json").unwrap();

        let mut session = TrackingSession::default();
        assert!(session.auto_discover(&run).is_err(), "first attempt fails");
        let error = session.run_state(&run).unwrap().error.clone();
        assert!(error.is_some());
        // A failed import requires an explicit retry, even after the file is fixed.
        std::fs::write(run.join("tracks.json"), SAMPLE_JSON).unwrap();
        assert!(session.auto_discover(&run).is_ok());
        assert!(session.run_state(&run).unwrap().sequence.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discovery_with_several_candidates_preselects_the_first() {
        let dir = run_dir("disc_choice");
        let run = dir.join("warp_loading");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(
            run.join("tracks.csv"),
            "frame_id,track_id,x,y\n0,a,1.0,2.0\n",
        )
        .unwrap();
        std::fs::write(run.join("tracking.json"), SAMPLE_JSON).unwrap();
        let mut session = TrackingSession::default();
        session.auto_discover(&run).unwrap();
        let state = session.run_state(&run).unwrap();
        assert_eq!(
            state.candidate_choice,
            state.auto_candidates.first().cloned()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn alignment_is_computed_once_per_import() {
        let path = temp_tracking_file("align_once", SAMPLE_JSON);
        let key = sample_run_key();
        let mut session = TrackingSession::default();
        assert!(!session.needs_alignment(&key), "nothing loaded");
        session.import_file(&key, &path).unwrap();
        assert!(session.needs_alignment(&key));
        session.refresh_alignment(&key, &[0, 1]);
        assert!(!session.needs_alignment(&key));
        session.import_file(&key, &path).unwrap();
        assert!(session.needs_alignment(&key), "a new import is rechecked");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn auto_discover_with_multiple_candidates_asks_for_a_choice() {
        let dir = run_dir("disc2");
        let run = dir.join("warp_loading");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(
            run.join("tracks.csv"),
            "frame_id,track_id,x,y\n0,a,1.0,2.0\n",
        )
        .unwrap();
        std::fs::write(run.join("tracking.json"), SAMPLE_JSON).unwrap();

        let mut session = TrackingSession::default();
        session.auto_discover(&run).unwrap();
        let state = session.run_state(&run).unwrap();
        assert!(
            state.sequence.is_none(),
            "never guess with several candidates"
        );
        assert_eq!(state.auto_candidates.len(), 2);
        assert!(state.import_path.is_none());

        // The user picks one -> it imports and clears the candidate list.
        session
            .import_file(&run, &state.auto_candidates[0].clone())
            .unwrap();
        let state = session.run_state(&run).unwrap();
        assert!(state.sequence.is_some());
        assert!(state.auto_candidates.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clear_run_unloads_data_but_keeps_choices() {
        let path = temp_tracking_file("clear", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        session.run_state_mut(&key).unwrap().state.mode = RegistrationMode::Rigid;

        session.clear_run(&key);
        let run = session.run_state(&key).unwrap();
        assert!(run.sequence.is_none());
        assert!(run.import_path.is_none());
        assert_eq!(run.state.mode, RegistrationMode::Rigid, "choice survives");
        // Keep registration choices after unload, but discard the source path.
        let persisted = session.to_persisted();
        let entry = persisted
            .runs
            .get(&key.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(entry.mode, RegistrationMode::Rigid);
        assert!(
            entry.import_path.is_none(),
            "unloaded path is not persisted"
        );

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn load_similar_for_all_runs_uses_the_file_name() {
        let dir = run_dir("similar");
        let a = dir.join("a");
        let b = dir.join("b");
        let c = dir.join("c");
        for d in [&a, &b, &c] {
            std::fs::create_dir_all(d).unwrap();
        }
        let csv = "frame_id,track_id,x,y\n0,a,1.0,2.0\n1,a,2.0,3.0\n";
        std::fs::write(a.join("tracks.csv"), csv).unwrap();
        std::fs::write(b.join("tracks.csv"), csv).unwrap();
        // c has no tracks.csv.

        let mut session = TrackingSession::default();
        session.import_file(&a, &a.join("tracks.csv")).unwrap();
        session.ensure_run(&b);
        session.ensure_run(&c);
        let (loaded, missing) = session.load_similar_for_all_runs(&a.join("tracks.csv"));
        assert_eq!(loaded, 1, "b got its same-named file");
        assert_eq!(missing, 1, "c had nothing to load");
        assert!(session.run_state(&b).unwrap().sequence.is_some());
        assert!(session.run_state(&c).unwrap().sequence.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refresh_alignment_flags_offset_numbering() {
        let json = br#"{"frames":[
            {"frame_id":1,"points":[{"id":"a","x":1.0,"y":2.0}]},
            {"frame_id":2,"points":[{"id":"a","x":3.0,"y":4.0}]}
        ]}"#;
        let path = temp_tracking_file("align", json);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        // Run frames are 0-based; tracking is 1-based.
        session.refresh_alignment(&key, &[0, 1]);
        let run = session.run_state(&key).unwrap();
        assert!(
            run.alignment_warning
                .as_deref()
                .is_some_and(|w| w.contains("1-based")),
            "{:?}",
            run.alignment_warning
        );
        // Matching numbering clears the warning.
        session.refresh_alignment(&key, &[1, 2]);
        assert_eq!(session.run_state(&key).unwrap().alignment_warning, None);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn export_registered_csv_writes_the_requested_file() {
        let path = temp_tracking_file("export", SAMPLE_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        session.run_state_mut(&key).unwrap().state.mode = RegistrationMode::TranslationXY;

        let run = fabric_data::Run {
            dir: PathBuf::from("/tmp"),
            session: fabric_data::Session::new(
                "s",
                "t",
                fabric_data::Direction::Warp,
                fabric_data::Branch::Loading,
                2,
            ),
            frames: fabric_data::FramesFile {
                frames: vec![
                    fabric_data::Frame::new(0, 0.0),
                    fabric_data::Frame::new(1, 1.0),
                ],
                ..Default::default()
            },
            force: fabric_data::ForceFile::default(),
            load_notes: vec![],
        };

        let out = path.parent().unwrap().join("registered.csv");
        session.export_registered_csv(&key, &run, &out).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.starts_with("frame_id,track_id,raw_x,raw_y,registered_x,registered_y"));
        assert!(text.contains("1,a,15.0000,17.0000,10.0000,20.0000"));
        assert!(text.contains("1.0000,"), "displacement attached");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// 3 × 3 grid: global shift (+20, +10), plus 30 px extension per row.
    const GRID_JSON: &[u8] = br#"{"frames":[
        {"frame_id":0,"points":[
            {"id":"r0c0","x":0.0,"y":0.0,"confidence":1.0},
            {"id":"r0c1","x":50.0,"y":0.0,"confidence":1.0},
            {"id":"r0c2","x":100.0,"y":0.0,"confidence":1.0},
            {"id":"r1c0","x":0.0,"y":100.0,"confidence":1.0},
            {"id":"r1c1","x":50.0,"y":100.0,"confidence":1.0},
            {"id":"r1c2","x":100.0,"y":100.0,"confidence":1.0},
            {"id":"r2c0","x":0.0,"y":200.0,"confidence":1.0},
            {"id":"r2c1","x":50.0,"y":200.0,"confidence":1.0},
            {"id":"r2c2","x":100.0,"y":200.0,"confidence":1.0}
        ]},
        {"frame_id":1,"points":[
            {"id":"r0c0","x":20.0,"y":10.0,"confidence":1.0},
            {"id":"r0c1","x":70.0,"y":10.0,"confidence":1.0},
            {"id":"r0c2","x":120.0,"y":10.0,"confidence":1.0},
            {"id":"r1c0","x":20.0,"y":140.0,"confidence":1.0},
            {"id":"r1c1","x":70.0,"y":140.0,"confidence":1.0},
            {"id":"r1c2","x":120.0,"y":140.0,"confidence":1.0},
            {"id":"r2c0","x":20.0,"y":270.0,"confidence":1.0},
            {"id":"r2c1","x":70.0,"y":270.0,"confidence":1.0},
            {"id":"r2c2","x":120.0,"y":270.0,"confidence":1.0}
        ]}
    ]}"#;

    #[test]
    fn stabilization_anchors_on_the_top_band_and_leaves_the_rest_moving() {
        let path = temp_tracking_file("stab", GRID_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        let run = session.run_state(&key).unwrap();

        // The reference frame needs no offset.
        assert_eq!(session.stabilization_reference(run), Some(FrameId(0)));
        let at_reference = session.stabilization_for(run, FrameId(0)).unwrap();
        assert!(at_reference.is_identity());

        let stab = session.stabilization_for(run, FrameId(1)).unwrap();
        assert_eq!(stab.anchor_total, 3, "only the top row anchors");
        assert_eq!(stab.anchor_pairs, 3);
        let (dx, dy) = stab.offset();
        assert!((dx + 20.0).abs() < 1e-9, "dx={dx}");
        assert!((dy + 10.0).abs() < 1e-9, "dy={dy}");

        // Top row pinned, bottom row still carries its extension downwards.
        let sequence = run.sequence.as_ref().unwrap();
        let top = sequence.point(&FrameId(1), &TrackId::new("r0c1")).unwrap();
        let (_, top_y) = stab.transform.apply(top.x, top.y);
        assert!(
            top_y.abs() < 1e-9,
            "top stays at its reference y, got {top_y}"
        );
        let bottom = sequence.point(&FrameId(1), &TrackId::new("r2c1")).unwrap();
        let (_, bottom_y) = stab.transform.apply(bottom.x, bottom.y);
        assert!(
            (bottom_y - 260.0).abs() < 1e-9,
            "bottom row keeps moving down, got {bottom_y}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn stabilization_follows_the_chosen_reference_frame() {
        let path = temp_tracking_file("stabref", GRID_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        // Moving the reference to frame 1 pins frame 1 and offsets frame 0.
        session.run_state_mut(&key).unwrap().state.reference_frame = Some(FrameId(1));
        let run = session.run_state(&key).unwrap();
        assert_eq!(session.stabilization_reference(run), Some(FrameId(1)));
        assert!(session
            .stabilization_for(run, FrameId(1))
            .unwrap()
            .is_identity());
        let (dx, dy) = session.stabilization_for(run, FrameId(0)).unwrap().offset();
        assert!((dx - 20.0).abs() < 1e-9, "dx={dx}");
        assert!((dy - 10.0).abs() < 1e-9, "dy={dy}");

        // Fall back to the first frame if the chosen reference is missing.
        session.run_state_mut(&key).unwrap().state.reference_frame = Some(FrameId(99));
        let run = session.run_state(&key).unwrap();
        assert_eq!(session.stabilization_reference(run), Some(FrameId(0)));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn stabilization_needs_tracking_data() {
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        let run = session.ensure_run(&key).run_key.clone();
        let run = session.run_state(&run).unwrap();
        assert!(session.stabilization_reference(run).is_none());
        assert!(session.stabilization_for(run, FrameId(0)).is_none());
    }

    #[test]
    fn untracked_frames_are_never_stabilized() {
        let path = temp_tracking_file("stabmiss", GRID_JSON);
        let mut session = TrackingSession::default();
        let key = sample_run_key();
        session.import_file(&key, &path).unwrap();
        let run = session.run_state(&key).unwrap();
        assert!(
            session.stabilization_for(run, FrameId(7)).is_none(),
            "a frame the tracker never saw is shown raw"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn overlay_style_follows_the_two_switches() {
        let mut session = TrackingSession::default();
        // Default: the legacy overlay, untouched.
        assert_eq!(session.stabilize, StabilizeSettings::default());
        assert_eq!(session.overlay_style(), OverlayStyle::Full);
        session.stabilize.enabled = true;
        assert_eq!(session.overlay_style(), OverlayStyle::Hidden);
        session.stabilize.show_points = true;
        assert_eq!(session.overlay_style(), OverlayStyle::Points);
        session.stabilize.enabled = false;
        assert_eq!(session.overlay_style(), OverlayStyle::Points);
        assert!(OverlayStyle::Points.is_pickable());
        assert!(OverlayStyle::Full.is_pickable());
        assert!(!OverlayStyle::Hidden.is_pickable());
    }

    #[test]
    fn stabilize_switches_survive_persistence() {
        let mut session = TrackingSession::default();
        session.stabilize.enabled = true;
        session.stabilize.show_points = true;
        let text = serde_json::to_string(&session.to_persisted()).unwrap();
        let back: PersistedTracking = serde_json::from_str(&text).unwrap();
        let session2 = TrackingSession::new(back);
        assert!(session2.stabilize.enabled);
        assert!(session2.stabilize.show_points);

        // Older persisted state without the field still loads (both off).
        let legacy: PersistedTracking = serde_json::from_str("{}").unwrap();
        let session3 = TrackingSession::new(legacy);
        assert!(!session3.stabilize.enabled);
        assert!(!session3.stabilize.show_points);
        assert_eq!(session3.overlay_style(), OverlayStyle::Full);
    }
}
