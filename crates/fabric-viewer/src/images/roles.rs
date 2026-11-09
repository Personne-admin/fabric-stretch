//! Declared image roles plus virtual PS albedo/normal roles.
//! Prefer the `ps/` manifest; fall back to legacy `ps_output/`. Scan per run, not per frame.
//! The image cache reduces 16-bit maps to 8-bit for display. Virtual paths are relative
//! to `images/`; declared roles take priority.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use fabric_data::Frame;
use serde::Deserialize;

/// Directory (next to `images/`) of the current PS layout.
const PS_DIR: &str = "ps";
/// Directory of the legacy PS layout.
const LEGACY_DIR: &str = "ps_output";
const MANIFEST: &str = "ps_manifest.json";

#[derive(Deserialize)]
struct Manifest {
    #[serde(default = "one")]
    albedo_scale: f32,
    #[serde(default)]
    frames: Vec<ManifestFrame>,
}

#[derive(Deserialize)]
struct ManifestFrame {
    frame_id: u32,
    albedo: Option<String>,
    normal: Option<String>,
}

fn one() -> f32 {
    1.0
}

/// The PS albedo / normal files of one run, indexed by `frame_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct PsLayers {
    /// Directory name below the run (`ps` or `ps_output`); empty when the run
    /// has no PS products.
    dir: &'static str,
    /// Multiplier that turns albedo bytes (`/255`) back into reflectance.
    scale: f32,
    albedo: HashMap<u32, String>,
    normal: HashMap<u32, String>,
}

impl Default for PsLayers {
    fn default() -> Self {
        PsLayers {
            dir: "",
            scale: 1.0,
            albedo: HashMap::new(),
            normal: HashMap::new(),
        }
    }
}

/// `true` for a bare file name (no directory components).
fn is_plain_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(std::path::Component::Normal(_))) && parts.next().is_none()
}

fn file_names(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Parse `img_<id>_<suffix>.png` into `id`.
fn frame_id_of(name: &str, suffix: &str) -> Option<u32> {
    name.strip_prefix("img_")?
        .strip_suffix(suffix)?
        .strip_suffix('_')?
        .parse()
        .ok()
}

impl PsLayers {
    /// Index `ps/` when readable, otherwise scan legacy `ps_output/` once.
    pub fn scan(run_dir: &Path) -> Self {
        Self::scan_manifest(&run_dir.join(PS_DIR))
            .unwrap_or_else(|| Self::scan_legacy(&run_dir.join(LEGACY_DIR)))
    }

    fn scan_manifest(dir: &Path) -> Option<Self> {
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(dir.join(MANIFEST)).ok()?).ok()?;
        let present = file_names(dir);
        let keep = |name: Option<String>| name.filter(|n| is_plain_name(n) && present.contains(n));
        let mut layers = PsLayers {
            dir: PS_DIR,
            scale: if manifest.albedo_scale.is_finite() && manifest.albedo_scale > 0.0 {
                manifest.albedo_scale
            } else {
                1.0
            },
            ..PsLayers::default()
        };
        for frame in manifest.frames {
            if let Some(name) = keep(frame.albedo) {
                layers.albedo.insert(frame.frame_id, name);
            }
            if let Some(name) = keep(frame.normal) {
                layers.normal.insert(frame.frame_id, name);
            }
        }
        Some(layers)
    }

    fn scan_legacy(dir: &Path) -> Self {
        let mut layers = PsLayers {
            dir: LEGACY_DIR,
            ..PsLayers::default()
        };
        // Sorted insertion makes `_normal.png` win over `_normals.png`.
        for name in file_names(dir) {
            if let Some(id) = frame_id_of(&name, "albedo.png") {
                layers.albedo.insert(id, name);
            } else if let Some(id) = frame_id_of(&name, "normal.png") {
                layers.normal.insert(id, name);
            } else if let Some(id) = frame_id_of(&name, "normals.png") {
                layers.normal.entry(id).or_insert(name);
            }
        }
        if layers.albedo.is_empty() && layers.normal.is_empty() {
            return PsLayers::default();
        }
        layers
    }

    /// Whether the run has any PS product.
    pub fn is_empty(&self) -> bool {
        self.albedo.is_empty() && self.normal.is_empty()
    }

    /// Reflectance scale for an albedo role file (a name from
    /// [`frame_roles`]): the recorded `albedo_scale` for PS albedo, 1 for
    /// anything else.
    pub fn albedo_scale_for(&self, filename: &str) -> f32 {
        if !self.dir.is_empty() && filename.starts_with(&format!("../{}/", self.dir)) {
            self.scale
        } else {
            1.0
        }
    }

    fn virtual_path(&self, name: &str) -> String {
        format!("../{}/{name}", self.dir)
    }
}

/// Declared roles plus PS roles, with declared entries taking priority. No file I/O.
pub fn frame_roles(frame: &Frame, ps: &PsLayers) -> Vec<(String, String)> {
    let mut roles: Vec<(String, String)> = frame
        .images()
        .map(|(r, f)| (r.to_string(), f.to_string()))
        .collect();
    for (role, table) in [("albedo", &ps.albedo), ("normal", &ps.normal)] {
        if roles.iter().any(|(r, _)| r == role) {
            continue;
        }
        if let Some(name) = table.get(&frame.frame_id) {
            roles.push((role.to_owned(), ps.virtual_path(name)));
        }
    }
    roles
}

#[cfg(test)]
mod tests {
    use fabric_data::Run;

    use super::*;

    /// A minimal on-disk run directory. `tag` must be unique per test: tests
    /// run in parallel threads and share one process.
    fn temp_run(tag: &str) -> (std::path::PathBuf, Run) {
        let dir = std::env::temp_dir().join(format!("fabsim_roles_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let run_dir = dir.join("run");
        std::fs::create_dir_all(&run_dir).unwrap();
        std::fs::write(
            run_dir.join("session.json"),
            r#"{"sample_id":"s","test_id":"t","direction":"warp","branch":"loading","count":0}"#,
        )
        .unwrap();
        std::fs::write(
            run_dir.join("frames.json"),
            r#"{"frames":[{"frame_id":0,"displacement_mm":0.0}]}"#,
        )
        .unwrap();
        std::fs::write(run_dir.join("force.json"), r#"{"forces":[]}"#).unwrap();
        let project = fabric_data::Project::discover(&dir).unwrap();
        let run = project.runs[0].clone();
        (dir, run)
    }

    fn touch(dir: &Path, names: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        for name in names {
            std::fs::write(dir.join(name), b"not a real png").unwrap();
        }
    }

    fn roles_of(run: &Run) -> Vec<(String, String)> {
        let ps = PsLayers::scan(&run.dir);
        frame_roles(&run.frames.frames[0], &ps)
    }

    fn named<'a>(roles: &'a [(String, String)], role: &str) -> Vec<&'a str> {
        roles
            .iter()
            .filter(|(r, _)| r == role)
            .map(|(_, f)| f.as_str())
            .collect()
    }

    #[test]
    fn no_ps_products_means_no_virtual_roles() {
        let (dir, run) = temp_run("none");
        assert!(roles_of(&run).is_empty());
        assert!(PsLayers::scan(&run.dir).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ps_manifest_layout_provides_albedo_normal_and_scale() {
        let (dir, run) = temp_run("manifest");
        let ps = run.dir.join("ps");
        touch(&ps, &["img_0000_albedo.png", "img_0000_normal.png"]);
        std::fs::write(
            ps.join("ps_manifest.json"),
            r#"{"format":"fabsim-ps-frames","version":1,"albedo_scale":1.75,
                "frames":[{"frame_id":0,"albedo":"img_0000_albedo.png","normal":"img_0000_normal.png"}]}"#,
        )
        .unwrap();
        let roles = roles_of(&run);
        assert_eq!(named(&roles, "albedo"), ["../ps/img_0000_albedo.png"]);
        assert_eq!(named(&roles, "normal"), ["../ps/img_0000_normal.png"]);
        let layers = PsLayers::scan(&run.dir);
        assert_eq!(layers.albedo_scale_for("../ps/img_0000_albedo.png"), 1.75);
        assert_eq!(layers.albedo_scale_for("declared.png"), 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_entries_without_a_file_or_with_paths_are_dropped() {
        let (dir, run) = temp_run("manifest_missing");
        let ps = run.dir.join("ps");
        touch(&ps, &["img_0000_albedo.png"]);
        std::fs::write(
            ps.join("ps_manifest.json"),
            r#"{"albedo_scale":2.0,"frames":[
                {"frame_id":0,"albedo":"img_0000_albedo.png","normal":"img_0000_normal.png"},
                {"frame_id":1,"albedo":"../escape.png"}]}"#,
        )
        .unwrap();
        let layers = PsLayers::scan(&run.dir);
        let roles = frame_roles(&run.frames.frames[0], &layers);
        assert_eq!(named(&roles, "albedo").len(), 1);
        assert!(
            named(&roles, "normal").is_empty(),
            "missing file is not offered"
        );
        assert!(!layers.albedo.contains_key(&1), "path entries are rejected");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_broken_manifest_falls_back_to_legacy() {
        let (dir, run) = temp_run("manifest_bad");
        touch(&run.dir.join("ps"), &["img_0000_albedo.png"]);
        std::fs::write(run.dir.join("ps").join("ps_manifest.json"), "{ nope").unwrap();
        assert!(roles_of(&run).is_empty(), "nothing usable anywhere");
        touch(&run.dir.join("ps_output"), &["img_0000_albedo.png"]);
        assert_eq!(
            named(&roles_of(&run), "albedo"),
            ["../ps_output/img_0000_albedo.png"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_ps_output_has_scale_one() {
        let (dir, run) = temp_run("legacy");
        touch(
            &run.dir.join("ps_output"),
            &["img_0000_albedo.png", "img_0000_normals.png"],
        );
        let roles = roles_of(&run);
        assert_eq!(
            named(&roles, "albedo"),
            ["../ps_output/img_0000_albedo.png"]
        );
        assert_eq!(
            named(&roles, "normal"),
            ["../ps_output/img_0000_normals.png"]
        );
        assert_eq!(roles.len(), 2, "no stray roles");
        let layers = PsLayers::scan(&run.dir);
        assert_eq!(
            layers.albedo_scale_for("../ps_output/img_0000_albedo.png"),
            1.0
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_singular_normal_wins_over_plural() {
        let (dir, run) = temp_run("legacy_both");
        touch(
            &run.dir.join("ps_output"),
            &["img_0000_normal.png", "img_0000_normals.png"],
        );
        let roles = roles_of(&run);
        assert_eq!(
            named(&roles, "normal"),
            ["../ps_output/img_0000_normal.png"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_ps_layout_takes_precedence_over_legacy() {
        let (dir, run) = temp_run("precedence");
        let ps = run.dir.join("ps");
        touch(&ps, &["img_0000_albedo.png"]);
        std::fs::write(
            ps.join("ps_manifest.json"),
            r#"{"albedo_scale":1.0,"frames":[{"frame_id":0,"albedo":"img_0000_albedo.png"}]}"#,
        )
        .unwrap();
        touch(
            &run.dir.join("ps_output"),
            &["img_0000_albedo.png", "img_0000_normal.png"],
        );
        let roles = roles_of(&run);
        assert_eq!(named(&roles, "albedo"), ["../ps/img_0000_albedo.png"]);
        assert!(named(&roles, "normal").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn declared_roles_are_never_shadowed_by_virtual_ones() {
        let (dir, mut run) = temp_run("shadow");
        for (key, value) in [("image_albedo", "alb.png"), ("image_normal", "decl_n.png")] {
            run.frames.frames[0]
                .extra
                .insert(key.to_owned(), serde_json::Value::String(value.to_owned()));
        }
        touch(
            &run.dir.join("ps_output"),
            &["img_0000_albedo.png", "img_0000_normal.png"],
        );
        let roles = roles_of(&run);
        assert_eq!(named(&roles, "albedo"), ["alb.png"]);
        assert_eq!(named(&roles, "normal"), ["decl_n.png"]);
        let layers = PsLayers::scan(&run.dir);
        assert_eq!(
            layers.albedo_scale_for("alb.png"),
            1.0,
            "declared albedo is unscaled"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frame_id_parsing_handles_large_ids() {
        assert_eq!(
            frame_id_of("img_12345_albedo.png", "albedo.png"),
            Some(12345)
        );
        assert_eq!(frame_id_of("img_0007_normals.png", "normals.png"), Some(7));
        assert_eq!(frame_id_of("img_0007_normals.png", "normal.png"), None);
        assert_eq!(frame_id_of("other_0007_albedo.png", "albedo.png"), None);
    }
}
