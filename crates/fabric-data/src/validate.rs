//! Validate run completeness, force links, displacement and image paths.

use std::collections::HashSet;
use std::fmt;

use crate::discover::Run;
use crate::model::ForceStatus;

/// How bad the problems found in a run are; ordered so the worst of several
/// runs is their `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Nothing to report.
    Ok,
    /// Usable, but incomplete or partly unreliable (aborted run, missing
    /// images, bad load-cell readings).
    Warning,
    /// Not usable as is (no frames, broken force links, duplicate frame ids,
    /// non-monotonic displacement, unreadable files).
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Ok => write!(f, "OK"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Error => write!(f, "ERROR"),
        }
    }
}

/// Result of validating one run.
#[derive(Debug, Clone)]
pub struct RunValidation {
    pub frame_count: usize,
    /// `count` from `session.json`: the number of steps the run was
    /// configured for.
    pub expected_count: u32,
    /// `frames_written` from `session.json`: what the rig says it saved.
    pub frames_written: Option<u32>,
    pub abort_reason: Option<String>,

    /// Frames whose `frame_id` already occurred earlier in `frames.json`.
    pub duplicate_frame_ids: usize,

    pub force_records: usize,
    /// Records whose `index` already occurred earlier in `force.json`; the
    /// first one is used for frame links.
    pub duplicate_force_indices: usize,
    /// Records whose status is not `ok`.
    pub force_non_ok: usize,
    /// Records with `force_n == null`.
    pub force_null: usize,
    /// Frames whose `force_json_index` is absent or matches no force record.
    pub unresolved_force_links: usize,

    /// Whether `displacement_mm` is non-decreasing over frames (equal values
    /// are allowed - relaxation holds repeat a displacement).
    pub displacement_monotonic: bool,
    pub displacement_range: Option<(f64, f64)>,

    pub images_referenced: usize,
    /// Referenced image files that do not exist on disk.
    pub images_missing: Vec<String>,

    /// Recoverable problems from loading (missing/corrupt frames or force file).
    pub load_notes: Vec<String>,
}

impl RunValidation {
    pub fn severity(&self) -> Severity {
        if self.frame_count == 0
            || self.duplicate_frame_ids > 0
            || self.unresolved_force_links > 0
            || !self.displacement_monotonic
            || !self.load_notes.is_empty()
        {
            Severity::Error
        } else if self.abort_reason.is_some()
            || self.frame_count != self.expected_count as usize
            || self.frames_written_mismatch()
            || self.duplicate_force_indices > 0
            || self.force_non_ok > 0
            || self.force_null > 0
            || !self.images_missing.is_empty()
        {
            Severity::Warning
        } else {
            Severity::Ok
        }
    }

    /// The rig recorded a `frames_written` that differs from the number of
    /// frames actually present in `frames.json`.
    fn frames_written_mismatch(&self) -> bool {
        self.frames_written
            .is_some_and(|n| n as usize != self.frame_count)
    }

    /// Human-readable problem list, one item per issue; empty when clean.
    pub fn issues(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(reason) = &self.abort_reason {
            out.push(format!("run aborted: {reason}"));
        }
        if self.frame_count != self.expected_count as usize {
            out.push(format!(
                "frame count {} != configured count {}",
                self.frame_count, self.expected_count
            ));
        }
        if self.frames_written_mismatch() {
            out.push(format!(
                "frame count {} != frames_written {}",
                self.frame_count,
                self.frames_written.unwrap_or_default()
            ));
        }
        if self.frame_count == 0 {
            out.push("no frames".into());
        }
        if self.duplicate_frame_ids > 0 {
            out.push(format!(
                "{} duplicate frame_id(s) in frames.json",
                self.duplicate_frame_ids
            ));
        }
        if self.duplicate_force_indices > 0 {
            out.push(format!(
                "{} duplicate record index(es) in force.json",
                self.duplicate_force_indices
            ));
        }
        if self.unresolved_force_links > 0 {
            out.push(format!(
                "{} frame(s) with unresolved force link",
                self.unresolved_force_links
            ));
        }
        if self.force_non_ok > 0 {
            out.push(format!("{} non-ok force record(s)", self.force_non_ok));
        }
        if self.force_null > 0 {
            out.push(format!("{} null force_n record(s)", self.force_null));
        }
        if !self.displacement_monotonic {
            out.push("displacement_mm not monotonic".into());
        }
        if !self.images_missing.is_empty() {
            out.push(format!(
                "{}/{} referenced image(s) missing on disk",
                self.images_missing.len(),
                self.images_referenced
            ));
        }
        out.extend(self.load_notes.iter().cloned());
        out
    }
}

/// Check one loaded run and summarise the findings.
pub fn validate_run(run: &Run) -> RunValidation {
    let frames = &run.frames.frames;

    let unresolved_force_links = frames
        .iter()
        .filter(|f| run.force_for_frame(f).is_none())
        .count();
    let mut frame_ids = HashSet::new();
    let duplicate_frame_ids = frames
        .iter()
        .filter(|f| !frame_ids.insert(f.frame_id))
        .count();
    let mut force_indices = HashSet::new();
    let duplicate_force_indices = run
        .force
        .forces
        .iter()
        .filter(|r| !force_indices.insert(r.index))
        .count();

    let force_non_ok = run
        .force
        .forces
        .iter()
        .filter(|r| r.status != ForceStatus::Ok)
        .count();
    let force_null = run
        .force
        .forces
        .iter()
        .filter(|r| r.force_n.is_none())
        .count();

    let displacement_monotonic = frames
        .windows(2)
        .all(|w| w[1].displacement_mm >= w[0].displacement_mm);
    let displacement_range = match frames.len() {
        0 => None,
        _ => {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for f in frames {
                lo = lo.min(f.displacement_mm);
                hi = hi.max(f.displacement_mm);
            }
            Some((lo, hi))
        }
    };

    let images_dir = run.images_dir();
    let mut images_referenced = 0;
    let mut images_missing = Vec::new();
    for frame in frames {
        for (_role, filename) in frame.images() {
            images_referenced += 1;
            if !images_dir.join(filename).is_file() {
                images_missing.push(filename.to_owned());
            }
        }
    }

    RunValidation {
        frame_count: frames.len(),
        expected_count: run.session.count,
        frames_written: run.session.frames_written,
        abort_reason: run.session.abort_reason.clone(),
        duplicate_frame_ids,
        duplicate_force_indices,
        force_records: run.force.forces.len(),
        force_non_ok,
        force_null,
        unresolved_force_links,
        displacement_monotonic,
        displacement_range,
        images_referenced,
        images_missing,
        load_notes: run.load_notes.clone(),
    }
}

impl fmt::Display for RunValidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "frames: {}/{} (frames_written: {})",
            self.frame_count,
            self.expected_count,
            self.frames_written
                .map_or_else(|| "?".into(), |n| n.to_string()),
        )?;
        writeln!(
            f,
            "forces: {} record(s), {} non-ok, {} null, {} unresolved link(s), {} duplicate index(es)",
            self.force_records,
            self.force_non_ok,
            self.force_null,
            self.unresolved_force_links,
            self.duplicate_force_indices
        )?;
        match self.displacement_range {
            Some((lo, hi)) => writeln!(
                f,
                "displacement: {:.1} -> {:.1} mm, monotonic: {}",
                lo,
                hi,
                if self.displacement_monotonic {
                    "yes"
                } else {
                    "NO"
                }
            )?,
            None => writeln!(f, "displacement: no frames")?,
        }
        writeln!(
            f,
            "images: {}/{} present",
            self.images_referenced - self.images_missing.len(),
            self.images_referenced
        )?;
        if let Some(reason) = &self.abort_reason {
            writeln!(f, "aborted: {reason}")?;
        }
        for note in &self.load_notes {
            writeln!(f, "load note: {note}")?;
        }
        Ok(())
    }
}
