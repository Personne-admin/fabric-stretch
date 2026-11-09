//! Discover and group runs by `(sample_id, direction, branch)`.
//!
//! A `session.json` marks a run. Search at most three levels below the root, skipping hidden
//! directories and stopping at each run. The root may itself be a run; names need not match
//! the Python pipeline's stricter naming rules.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::{ForceFile, ForceRecord, Frame, FramesFile, RunKey, Session};

/// A failure to read or parse one of a run's files, or to list a directory.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, DataError> {
    let bytes = fs::read(path).map_err(|source| DataError::Io {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| DataError::Parse {
        path: path.to_owned(),
        source,
    })
}

/// One fully loaded run directory.
#[derive(Debug, Clone)]
pub struct Run {
    /// Directory the run was loaded from.
    pub dir: PathBuf,
    pub session: Session,
    pub frames: FramesFile,
    pub force: ForceFile,
    /// Recoverable problems hit while loading (e.g. missing force.json).
    pub load_notes: Vec<String>,
}

impl Run {
    /// The run's identity: sample, direction and branch.
    pub fn key(&self) -> RunKey {
        RunKey {
            sample_id: self.session.sample_id.clone(),
            direction: self.session.direction.clone(),
            branch: self.session.branch.clone(),
        }
    }

    /// Match force by [`ForceRecord::index`], not list position. First duplicate wins;
    /// validation reports duplicates.
    pub fn force_for_frame(&self, frame: &Frame) -> Option<&ForceRecord> {
        let idx = frame.force_json_index?;
        self.force.forces.iter().find(|r| r.index == idx)
    }

    /// Linked force in N, or `None` unless status is `ok` and the value is finite.
    pub fn force_n_for_frame(&self, frame: &Frame) -> Option<f64> {
        self.force_for_frame(frame)
            .and_then(ForceRecord::usable_force_n)
    }

    /// Directory where the frame's image files live.
    pub fn images_dir(&self) -> PathBuf {
        self.dir.join("images")
    }
}

/// A project directory with every run found under it.
#[derive(Debug, Default)]
pub struct Project {
    pub root: PathBuf,
    pub runs: Vec<Run>,
    /// Child-directory errors; discovery continues for the remaining runs.
    pub load_errors: Vec<(PathBuf, DataError)>,
}

impl Project {
    /// Find runs up to three levels below `root`. Only an unreadable root aborts discovery.
    pub fn discover(root: impl Into<PathBuf>) -> std::io::Result<Project> {
        let root = root.into();
        let mut project = Project {
            root: root.clone(),
            ..Default::default()
        };
        scan_dir(&root, 0, &mut project)?;
        project
            .runs
            .sort_by(|a, b| a.key().cmp(&b.key()).then(a.dir.cmp(&b.dir)));
        Ok(project)
    }

    /// Runs grouped by `(sample_id, direction, branch)`. Multiple runs can
    /// share a key (e.g. a re-measurement in a differently named directory).
    pub fn grouped(&self) -> BTreeMap<RunKey, Vec<&Run>> {
        let mut map: BTreeMap<RunKey, Vec<&Run>> = BTreeMap::new();
        for run in &self.runs {
            map.entry(run.key()).or_default().push(run);
        }
        map
    }
}

/// How far below the project root run directories are searched for.
const MAX_DEPTH: usize = 3;

/// Collect runs, recording child failures in `project.load_errors`.
fn scan_dir(dir: &Path, depth: usize, project: &mut Project) -> std::io::Result<()> {
    if dir.join("session.json").is_file() {
        match load_run(dir) {
            Ok(run) => project.runs.push(run),
            Err(err) => project.load_errors.push((dir.to_owned(), err)),
        }
        return Ok(());
    }
    if depth >= MAX_DEPTH {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = match entry {
            Ok(entry) => entry.path(),
            Err(source) => {
                let err = DataError::Io {
                    path: dir.to_owned(),
                    source,
                };
                project.load_errors.push((dir.to_owned(), err));
                continue;
            }
        };
        let hidden = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with('.'));
        if hidden || !path.is_dir() {
            continue;
        }
        if let Err(source) = scan_dir(&path, depth + 1, project) {
            let err = DataError::Io {
                path: path.clone(),
                source,
            };
            project.load_errors.push((path, err));
        }
    }
    Ok(())
}

fn load_run(dir: &Path) -> Result<Run, DataError> {
    let session: Session = read_json(&dir.join("session.json"))?;
    let mut load_notes = Vec::new();

    // Aborted runs may have missing/truncated JSON; keep the run with empty records.
    let frames = read_json::<FramesFile>(&dir.join("frames.json")).unwrap_or_else(|err| {
        load_notes.push(format!("frames.json unusable: {err}"));
        FramesFile::default()
    });
    let force = read_json::<ForceFile>(&dir.join("force.json")).unwrap_or_else(|err| {
        load_notes.push(format!("force.json unusable: {err}"));
        ForceFile::default()
    });

    Ok(Run {
        dir: dir.to_owned(),
        session,
        frames,
        force,
        load_notes,
    })
}
