//! Shared frame selection keyed by project path, run path and frame ID, rather than list indices.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Frame identity with canonical paths, so alternate spellings compare equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FrameRef {
    pub project_root: PathBuf,
    pub run_dir: PathBuf,
    pub frame_id: u32,
}

impl FrameRef {
    pub fn new(
        project_root: impl Into<PathBuf>,
        run_dir: impl Into<PathBuf>,
        frame_id: u32,
    ) -> Self {
        FrameRef {
            project_root: project_root.into(),
            run_dir: run_dir.into(),
            frame_id,
        }
    }
}

/// Shared selection and hover. Hover drives views unless pinned; Space toggles the pin.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSelection {
    pub selected: Option<FrameRef>,
    pub hovered: Option<FrameRef>,
    /// When true the hovered frame has been pinned into `selected` and
    /// hovering no longer overrides it.
    pub pinned: bool,
}

impl FrameSelection {
    /// The frame every consumer should render. Hovered wins unless pinned.
    pub fn active(&self) -> Option<&FrameRef> {
        if self.pinned {
            self.selected.as_ref()
        } else {
            self.hovered.as_ref().or(self.selected.as_ref())
        }
    }

    /// Clear the hovered frame (pointer left the plot, curves tab hidden, ...).
    pub fn clear_hover(&mut self) {
        self.hovered = None;
    }

    /// Space: pin the current hover into `selected`, or release a pin.
    pub fn toggle_pin(&mut self) {
        if self.pinned {
            self.pinned = false;
        } else if self.hovered.is_some() {
            self.selected = self.hovered.clone();
            self.pinned = true;
        }
    }

    /// Drop unresolved selections and hovers. Invalidating a pinned selection releases the pin.
    pub fn retain_valid(&mut self, resolve: &dyn Fn(&FrameRef) -> bool) {
        if self.selected.as_ref().is_some_and(|r| !resolve(r)) {
            self.selected = None;
            self.pinned = false;
        }
        if self.hovered.as_ref().is_some_and(|r| !resolve(r)) {
            self.hovered = None;
        }
    }
}

/// Canonical path for existing files; lexically normalized absolute path otherwise.
pub fn canonical_key(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_owned())
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs() -> (FrameRef, FrameRef) {
        let a = FrameRef::new("/p1", "/p1/r1", 3);
        let b = FrameRef::new("/p1", "/p1/r2", 7);
        (a, b)
    }

    #[test]
    fn hovered_prioritizes_over_selected() {
        let (a, b) = refs();
        let mut sel = FrameSelection {
            selected: Some(a.clone()),
            hovered: Some(b.clone()),
            pinned: false,
        };
        assert_eq!(sel.active(), Some(&b));
        sel.clear_hover();
        assert_eq!(sel.active(), Some(&a));
    }

    #[test]
    fn pin_uses_selected() {
        let (a, b) = refs();
        let mut sel = FrameSelection {
            selected: Some(a.clone()),
            hovered: Some(b.clone()),
            pinned: false,
        };
        sel.toggle_pin(); // hovered (b) gets pinned into selected
        assert_eq!(sel.selected, Some(b.clone()));
        assert_eq!(sel.active(), Some(&b));
        // Hover ending while pinned keeps the pinned selection.
        sel.clear_hover();
        assert_eq!(sel.active(), Some(&b));
        // Space again unpins; hover (now gone) no longer wins.
        sel.toggle_pin();
        assert_eq!(sel.active(), Some(&b));
    }

    #[test]
    fn toggle_pin_without_hover_is_a_no_op() {
        let (a, _) = refs();
        let mut sel = FrameSelection {
            selected: Some(a.clone()),
            hovered: None,
            pinned: false,
        };
        sel.toggle_pin();
        assert!(!sel.pinned);
        assert_eq!(sel.selected, Some(a.clone()));
    }

    #[test]
    fn invalid_refs_are_dropped_and_unpin() {
        let (a, b) = refs();
        let mut sel = FrameSelection {
            selected: Some(a.clone()),
            hovered: Some(b.clone()),
            pinned: true,
        };
        sel.retain_valid(&|r| r == &a);
        assert_eq!(sel.selected, Some(a.clone()));
        assert_eq!(sel.hovered, None);
        assert!(sel.pinned, "valid pinned selection survives");

        sel.retain_valid(&|_| false);
        assert_eq!(sel.selected, None);
        assert!(!sel.pinned);

        let mut sel2 = FrameSelection {
            selected: Some(b),
            hovered: None,
            pinned: false,
        };
        sel2.retain_valid(&|_| false);
        assert_eq!(sel2.selected, None);
    }

    #[test]
    fn active_prioritizes_hover_until_pinned() {
        let (a, b) = refs();
        let mut sel = FrameSelection {
            selected: Some(a.clone()),
            ..FrameSelection::default()
        };
        assert_eq!(sel.active(), Some(&a));
        sel.hovered = Some(b.clone());
        assert_eq!(sel.active(), Some(&b));
        sel.pinned = true;
        assert_eq!(sel.active(), Some(&a));
    }

    #[test]
    fn canonical_key_aliases_are_equivalent() {
        let dir = std::env::temp_dir().join(format!("fabsim_key_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("sub").join("img.png");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(&file, b"not really an image").unwrap();

        let via_parent = dir.join("sub").join("..").join("sub").join("img.png");
        let via_dot = dir.join("sub").join(".").join("img.png");
        assert_eq!(canonical_key(&file), canonical_key(&via_parent));
        assert_eq!(canonical_key(&file), canonical_key(&via_dot));
        assert_eq!(canonical_key(&file), canonical_key(&file));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn canonical_key_fallback_handles_missing_paths() {
        let base = canonical_key(Path::new("/tmp/nonexistent_fabsim_xyz/a.png"));
        let base2 = canonical_key(Path::new("/tmp/nonexistent_fabsim_xyz/./a.png"));
        assert_eq!(base, base2);
        assert!(!base.as_os_str().is_empty());
    }
}
