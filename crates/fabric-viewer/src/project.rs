//! Open projects and stable frame lookup across runs.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use fabric_data::{validate_run, Branch, Curve, RunValidation};

use crate::images::PsLayers;
use crate::selection::{canonical_key, FrameRef};

/// One run of an open project.
pub struct RunEntry {
    pub curve: Curve,
    pub validation: RunValidation,
    /// Plot visibility shared by the Curves and Runs tabs.
    pub visible: bool,
    pub legend_name: String,
    pub legend_id: egui::Id,
    /// Photometric-stereo albedo / normal files of the run, indexed once when
    /// the project opens.
    pub ps: PsLayers,
}

/// An open project folder with all its runs.
pub struct ProjectSlot {
    pub name: String,
    pub project: fabric_data::Project,
    pub entries: Vec<RunEntry>,
    pub enabled: bool,
    /// Canonical project root (selection identity).
    pub root_key: PathBuf,
    /// Canonical run directories, parallel to `entries`.
    pub run_keys: Vec<PathBuf>,
}

impl ProjectSlot {
    pub(crate) fn new(name: String, project: fabric_data::Project) -> Self {
        let mut used_names: HashSet<String> = HashSet::new();
        let entries = project
            .runs
            .iter()
            .map(|run| {
                let curve = Curve::from_run(run);
                let base = format!("{}: {}", name, curve.key);
                let mut legend_name = base.clone();
                let mut n = 2;
                while !used_names.insert(legend_name.clone()) {
                    legend_name = format!("{base} ({n})");
                    n += 1;
                }
                RunEntry {
                    validation: validate_run(run),
                    visible: true,
                    legend_id: egui::Id::new(&legend_name),
                    legend_name,
                    curve,
                    ps: PsLayers::scan(&run.dir),
                }
            })
            .collect();
        let root_key = canonical_key(&project.root);
        let run_keys = project
            .runs
            .iter()
            .map(|run| canonical_key(&run.dir))
            .collect();
        ProjectSlot {
            name,
            project,
            entries,
            enabled: true,
            root_key,
            run_keys,
        }
    }

    /// Displacement at which the release branch of run `idx` starts (the
    /// maximum of its loading counterpart), used to mirror release curves.
    pub fn release_start_mm(&self, idx: usize) -> f64 {
        let key = &self.entries[idx].curve.key;
        self.entries
            .iter()
            .find(|e| {
                e.curve.key.sample_id == key.sample_id
                    && e.curve.key.direction == key.direction
                    && e.curve.key.branch == Branch::Loading
            })
            .and_then(|e| e.curve.max_displacement_mm())
            .or_else(|| self.entries[idx].curve.max_displacement_mm())
            .unwrap_or(0.0)
    }

    /// Rescan runs without PS products to pick up maps generated while the viewer is open.
    pub fn rescan_missing_ps(&mut self) {
        for (entry, run) in self.entries.iter_mut().zip(&self.project.runs) {
            if entry.ps.is_empty() {
                entry.ps = PsLayers::scan(&run.dir);
            }
        }
    }

    /// Run path relative to the project root, with `/` separators. Includes sample directories
    /// to distinguish nested runs with the same name.
    pub fn run_name(&self, idx: usize) -> String {
        let dir = &self.project.runs[idx].dir;
        let rel = dir.strip_prefix(&self.project.root).unwrap_or(dir);
        let rel = if rel.as_os_str().is_empty() {
            dir.file_name().map(Path::new).unwrap_or(rel)
        } else {
            rel
        };
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
    }
}

/// Build a stable [`FrameRef`] for `(project, run, frame index)`.
pub fn frame_ref_for(
    slots: &[ProjectSlot],
    pidx: usize,
    ridx: usize,
    fidx: usize,
) -> Option<FrameRef> {
    let slot = slots.get(pidx)?;
    let run = slot.project.runs.get(ridx)?;
    let frame = run.frames.frames.get(fidx)?;
    Some(FrameRef::new(
        slot.root_key.clone(),
        slot.run_keys.get(ridx)?.clone(),
        frame.frame_id,
    ))
}

/// Resolve a [`FrameRef`] to `(project, run, frame index)`.
/// Returns `None` for closed, disabled or missing data.
pub fn resolve_frame_ref(slots: &[ProjectSlot], fr: &FrameRef) -> Option<(usize, usize, usize)> {
    let (pidx, slot) = slots
        .iter()
        .enumerate()
        .find(|(_, s)| s.enabled && s.root_key == fr.project_root)?;
    let ridx = slot.run_keys.iter().position(|k| *k == fr.run_dir)?;
    let fidx = slot.project.runs[ridx]
        .frames
        .frames
        .iter()
        .position(|f| f.frame_id == fr.frame_id)?;
    Some((pidx, ridx, fidx))
}
