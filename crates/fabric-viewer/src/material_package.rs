//! Material package parsing, discovery and monotone force fitting.
//! Reads v2 and v3; ignores unknown fields for forward compatibility.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Value of the `format` key of every material package.
pub const FORMAT: &str = "fabsim-material-rgb";

/// Hover text of the Relight toggle for packages without photometric layers.
pub const NO_PHOTOMETRIC_NOTE: &str =
    "No photometric layers in this package \u{2014} rebuild with `fabsim package`";

/// Measured branch of the hysteresis loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PressBranch {
    #[default]
    Loading,
    Release,
}

impl PressBranch {
    pub fn name(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Release => "release",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Loading => "Loading",
            Self::Release => "Release",
        }
    }
    pub fn from_name(name: &str) -> Self {
        if name == "release" {
            Self::Release
        } else {
            Self::Loading
        }
    }
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    #[serde(default)]
    orientation: Option<String>,
    atlas: AtlasMeta,
    branches: BranchesMeta,
    #[serde(default)]
    tiling: Option<TilingMeta>,
    #[serde(default)]
    ps: Option<PsMeta>,
    #[serde(default)]
    force: Option<ForceMeta>,
    #[serde(default)]
    release_residual_stretch: Option<f32>,
}

#[derive(Deserialize)]
struct AtlasMeta {
    dimensions_px: [u32; 2],
    grid_cells: [u32; 4],
    #[serde(default)]
    origin: Option<String>,
    #[serde(default)]
    active_grid_axis: Option<String>,
}

#[derive(Deserialize)]
struct BranchesMeta {
    loading: Vec<StateMeta>,
    release: Vec<StateMeta>,
}

#[derive(Deserialize)]
struct StateMeta {
    #[serde(default)]
    frame_id: Option<i64>,
    stretch_ratio: f32,
    force_n: f32,
    #[serde(default)]
    force_raw_n: Option<f32>,
    /// Version 2 packages call this `machine_displacement_mm`.
    #[serde(default, alias = "machine_displacement_mm")]
    travel_mm: Option<f32>,
    texture: String,
    #[serde(default)]
    albedo: Option<String>,
    #[serde(default)]
    normal: Option<String>,
}

/// `tiling` block (absent or `enabled: false` in untiled packages).
#[derive(Deserialize)]
struct TilingMeta {
    enabled: bool,
    tile_size_px: [u32; 2],
    source_atlas_size_px: [u32; 2],
    period_px: [f32; 2],
    origin_uv: [f32; 2],
    period_uv: [f32; 2],
}

#[derive(Deserialize)]
struct PsMeta {
    albedo_scale: f32,
}

#[derive(Deserialize)]
struct ForceMeta {
    #[serde(default)]
    clamp_width_m: Option<f32>,
}

/// Tile mapping in source-atlas units: `(material_uv - origin_uv) / period_uv`.
/// Leave coordinates unwrapped until fragment sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tiling {
    pub origin_uv: [f32; 2],
    pub period_uv: [f32; 2],
}

impl Tiling {
    /// Keep coordinates unwrapped across the mesh; wrapping at vertices would break seams.
    pub fn texture_uv(&self, material_uv: [f32; 2]) -> [f32; 2] {
        [
            (material_uv[0] - self.origin_uv[0]) / self.period_uv[0],
            (material_uv[1] - self.origin_uv[1]) / self.period_uv[1],
        ]
    }
}

/// One measured state of a branch.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub frame_id: Option<i64>,
    pub stretch_ratio: f32,
    /// Strip tension in N, zero at the unloaded loading state (tared by the writer in version 3
    /// packages and by [`MaterialPackage::load`] in version 2 packages).
    pub force_n: f32,
    /// Tension before the tare (version 3 packages; for version 2 packages the viewer's own tare).
    pub force_raw_n: Option<f32>,
    /// Machine travel of the run (one branch) so far, in mm. It is a per-run displacement, not an
    /// absolute bar position: the release branch starts again at 0 at its maximum stretch.
    pub travel_mm: Option<f32>,
    pub texture: PathBuf,
    pub albedo: Option<PathBuf>,
    pub normal: Option<PathBuf>,
}

/// Photometric layers of a package.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Photometric {
    /// `albedo = value / 65535 * albedo_scale`.
    pub albedo_scale: f32,
}

/// Which two measured states bracket a stretch, and the blend weight.
#[derive(Debug, Clone, PartialEq)]
pub struct TexturePlan {
    pub branch: PressBranch,
    pub lo: usize,
    pub hi: usize,
    /// Weight of the `hi` state (0 = `lo`).
    pub t: f32,
    /// The requested stretch was outside the measured states and was clamped.
    pub clamped: bool,
    pub frames: [Option<i64>; 2],
    pub textures: [PathBuf; 2],
    pub albedo: Option<[PathBuf; 2]>,
    pub normal: Option<[PathBuf; 2]>,
}

/// A validated material package directory.
#[derive(Debug)]
pub struct MaterialPackage {
    root: PathBuf,
    version: u32,
    orientation: String,
    atlas_px: [u32; 2],
    pull_vertical: bool,
    tiling: Option<Tiling>,
    texture_px: [u32; 2],
    loading: Vec<State>,
    release: Vec<State>,
    photometric: Result<Photometric, String>,
    clamp_width_m: Option<f32>,
    residual_stretch: Option<f32>,
    legacy_tare_n: Option<f32>,
    loading_fit: Option<ForceFit>,
    release_fit: Option<ForceFit>,
}

#[derive(Default)]
pub struct LoadProgress {
    pub cancelled: AtomicBool,
    pub checked: AtomicUsize,
    pub total: AtomicUsize,
}

impl LoadProgress {
    fn check_cancelled(&self) -> Result<(), String> {
        if self.cancelled.load(Ordering::Relaxed) {
            Err("Loading cancelled".into())
        } else {
            Ok(())
        }
    }
}

impl MaterialPackage {
    /// Reads and validates the package in `root`.
    pub fn load(root: &Path) -> Result<Self, String> {
        Self::load_with_progress(root, &LoadProgress::default())
    }

    pub fn load_with_progress(root: &Path, progress: &LoadProgress) -> Result<Self, String> {
        progress.check_cancelled()?;
        let manifest_path = root.join("package.json");
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        let manifest: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        if manifest.format != FORMAT {
            return Err(format!(
                "{}: not a material package (format {:?})",
                manifest_path.display(),
                manifest.format
            ));
        }
        if !matches!(manifest.version, 2 | 3) {
            return Err(unsupported_version(manifest.version.into()));
        }
        progress.check_cancelled()?;
        let atlas = &manifest.atlas;
        if atlas.dimensions_px.contains(&0)
            || atlas.grid_cells[2..].contains(&0)
            || atlas.origin.as_deref() != Some("Loading reference grid top-left")
        {
            return Err("invalid shared material atlas".into());
        }
        let mut texture_px = atlas.dimensions_px;
        let mut tiling = None;
        if let Some(meta) = manifest.tiling.as_ref().filter(|t| t.enabled) {
            let consistent = (0..2).all(|a| {
                meta.tile_size_px[a] > 0
                    && meta.source_atlas_size_px[a] == atlas.dimensions_px[a]
                    && meta.period_px[a] == meta.tile_size_px[a] as f32
                    && meta.period_uv[a].is_finite()
                    && meta.period_uv[a] > 0.0
                    && meta.origin_uv[a].is_finite()
                    && (meta.period_uv[a] - meta.period_px[a] / atlas.dimensions_px[a] as f32).abs()
                        < 1.0e-4
            });
            if !consistent {
                return Err("inconsistent tiling metadata".into());
            }
            // Tiled textures cover one period, not the source atlas.
            texture_px = meta.tile_size_px;
            tiling = Some(Tiling {
                origin_uv: meta.origin_uv,
                period_uv: meta.period_uv,
            });
        }
        let convert = |branch: &str, states: Vec<StateMeta>| -> Result<Vec<State>, String> {
            let mut out = Vec::with_capacity(states.len());
            for (i, s) in states.into_iter().enumerate() {
                progress.check_cancelled()?;
                if !(s.stretch_ratio.is_finite() && s.stretch_ratio > 0.0 && s.force_n.is_finite())
                {
                    return Err(format!(
                        "{branch} state {i}: stretch_ratio {} and force_n {} must be finite numbers (stretch above 0)",
                        s.stretch_ratio, s.force_n
                    ));
                }
                out.push(State {
                    frame_id: s.frame_id,
                    stretch_ratio: s.stretch_ratio,
                    force_n: s.force_n,
                    force_raw_n: s.force_raw_n,
                    travel_mm: s.travel_mm,
                    texture: inside(root, &s.texture)?,
                    albedo: s.albedo.map(|p| inside(root, &p)).transpose()?,
                    normal: s.normal.map(|p| inside(root, &p)).transpose()?,
                });
            }
            out.sort_by(|a, b| a.stretch_ratio.total_cmp(&b.stretch_ratio));
            if let (Some(first), Some(last)) = (out.first(), out.last()) {
                if last.stretch_ratio - first.stretch_ratio < 1.0e-6 {
                    return Err(format!(
                        "the {branch} branch has no stretch range (every state is at \u{3bb} {:.3})",
                        first.stretch_ratio
                    ));
                }
            }
            Ok(out)
        };
        let mut loading = convert("loading", manifest.branches.loading)?;
        let mut release = convert("release", manifest.branches.release)?;
        // v2 stores raw force. Tare both branches to the first loading state;
        // it has no metadata to correct differing load-cell zeros.
        let legacy_tare_n = (manifest.version < 3)
            .then(|| loading.first().map(|s| s.force_n))
            .flatten();
        if let Some(offset) = legacy_tare_n {
            for state in loading.iter_mut().chain(release.iter_mut()) {
                state.force_raw_n.get_or_insert(state.force_n);
                state.force_n -= offset;
            }
        }
        check_sizes(
            loading
                .iter()
                .chain(&release)
                .map(|s| (s.texture.as_path(), texture_px)),
            progress,
        )?;
        let photometric = photometric_of(
            &manifest.ps,
            manifest.version,
            &loading,
            &release,
            texture_px,
            progress,
        );
        progress.check_cancelled()?;
        let pull_vertical = atlas.active_grid_axis.as_deref() != Some("horizontal");
        let loading_fit = ForceFit::from_states(&loading);
        let release_fit = ForceFit::from_states(&release);
        Ok(Self {
            root: root.into(),
            version: manifest.version,
            orientation: manifest.orientation.unwrap_or_default(),
            atlas_px: atlas.dimensions_px,
            pull_vertical,
            tiling,
            texture_px,
            loading,
            release,
            photometric,
            clamp_width_m: manifest.force.and_then(|f| f.clamp_width_m),
            residual_stretch: manifest.release_residual_stretch,
            legacy_tare_n,
            loading_fit,
            release_fit,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn orientation(&self) -> &str {
        &self.orientation
    }
    /// Size of the source atlas in pixels (width, height).
    pub fn atlas_px(&self) -> [u32; 2] {
        self.atlas_px
    }
    /// Size of one state texture: the atlas, or one tile for tiled packages.
    pub fn texture_px(&self) -> [u32; 2] {
        self.texture_px
    }
    /// The pull runs along the atlas rows (`active_grid_axis: vertical`, the default).
    pub fn pull_vertical(&self) -> bool {
        self.pull_vertical
    }
    /// Periodic mapping for tiled packages; `None` for clamped single-atlas packages.
    pub fn tiling(&self) -> Option<Tiling> {
        self.tiling
    }
    /// Albedo scale when every state has photometric layers; otherwise the reason it is unavailable.
    pub fn photometric(&self) -> Result<Photometric, &str> {
        self.photometric.as_ref().copied().map_err(String::as_str)
    }
    pub fn clamp_width_m(&self) -> Option<f32> {
        self.clamp_width_m
    }
    /// Permanent stretch left after the Release branch (version 3 packages).
    pub fn residual_stretch(&self) -> Option<f32> {
        self.residual_stretch
    }

    /// The offset (N) the viewer subtracted from the raw load-cell forces of a version 2 package;
    /// `None` for version 3 packages, which are tared by the writer.
    pub fn legacy_tare_n(&self) -> Option<f32> {
        self.legacy_tare_n
    }

    pub fn states(&self, branch: PressBranch) -> &[State] {
        match branch {
            PressBranch::Loading => &self.loading,
            PressBranch::Release => &self.release,
        }
    }

    /// Measured stretch range of a branch.
    pub fn stretch_range(&self, branch: PressBranch) -> Option<(f32, f32)> {
        let states = self.states(branch);
        Some((states.first()?.stretch_ratio, states.last()?.stretch_ratio))
    }

    /// Force fit of a branch, when its states allow one (see [`ForceFit::from_states`]).
    pub fn force_fit(&self, branch: PressBranch) -> Option<&ForceFit> {
        match branch {
            PressBranch::Loading => self.loading_fit.as_ref(),
            PressBranch::Release => self.release_fit.as_ref(),
        }
    }

    /// The two measured states that bracket `stretch` and the blend weight. A stretch outside the
    /// measured range uses the nearest state (`clamped`).
    pub fn texture_plan(&self, branch: PressBranch, stretch: f32) -> Result<TexturePlan, String> {
        let states = self.states(branch);
        if states.is_empty() || !stretch.is_finite() {
            return Err(format!("no measured {} states", branch.name()));
        }
        let first = states[0].stretch_ratio;
        let last = states[states.len() - 1].stretch_ratio;
        let clamped = stretch < first - 1.0e-5 || stretch > last + 1.0e-5;
        let stretch = stretch.clamp(first, last);
        let hi = states
            .partition_point(|s| s.stretch_ratio < stretch)
            .min(states.len() - 1);
        let lo = if states[hi].stretch_ratio > stretch {
            hi.saturating_sub(1)
        } else {
            hi
        };
        let t = if hi > lo && states[hi].stretch_ratio > states[lo].stretch_ratio {
            ((stretch - states[lo].stretch_ratio)
                / (states[hi].stretch_ratio - states[lo].stretch_ratio))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let pair = |f: &dyn Fn(&State) -> Option<&PathBuf>| -> Option<[PathBuf; 2]> {
            Some([f(&states[lo])?.clone(), f(&states[hi])?.clone()])
        };
        Ok(TexturePlan {
            branch,
            lo,
            hi,
            t,
            clamped,
            frames: [states[lo].frame_id, states[hi].frame_id],
            textures: [states[lo].texture.clone(), states[hi].texture.clone()],
            albedo: pair(&|s| s.albedo.as_ref()),
            normal: pair(&|s| s.normal.as_ref()),
        })
    }
}

fn check_size(path: &Path, expected: [u32; 2]) -> Result<(), String> {
    let size = image::image_dimensions(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if size != (expected[0], expected[1]) {
        return Err(format!(
            "{} is {}x{} but the package texture size is {}x{}",
            path.display(),
            size.0,
            size.1,
            expected[0],
            expected[1]
        ));
    }
    Ok(())
}

/// Check PNG headers in parallel, returning the first failure in input order.
fn check_sizes<'a>(
    files: impl Iterator<Item = (&'a Path, [u32; 2])>,
    progress: &LoadProgress,
) -> Result<(), String> {
    const THREADS: usize = 8;
    let files: Vec<(&Path, [u32; 2])> = files.collect();
    progress.total.fetch_add(files.len(), Ordering::Relaxed);
    let chunk = files.len().div_ceil(THREADS).max(1);
    std::thread::scope(|scope| {
        let workers: Vec<_> = files
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter().try_for_each(|(path, expected)| {
                        progress.check_cancelled()?;
                        check_size(path, *expected)?;
                        progress.checked.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    })
                })
            })
            .collect();
        workers
            .into_iter()
            .try_for_each(|w| w.join().unwrap_or_else(|_| Err("size check failed".into())))
    })
}

/// `root` joined with a path from the manifest. Absolute paths and `..` would leave the package
/// directory and are refused.
fn inside(root: &Path, relative: &str) -> Result<PathBuf, String> {
    use std::path::Component;
    let path = Path::new(relative);
    if relative.is_empty()
        || !path
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(format!(
            "{relative:?} in package.json points outside the package directory"
        ));
    }
    Ok(root.join(path))
}

fn unsupported_version(version: u64) -> String {
    format!(
        "unsupported material package version {version} (this viewer reads 2 and 3; rebuild with `fabsim package`)"
    )
}

/// Whether every state carries consistent photometric layers.
fn photometric_of(
    ps: &Option<PsMeta>,
    version: u32,
    loading: &[State],
    release: &[State],
    texture_px: [u32; 2],
    progress: &LoadProgress,
) -> Result<Photometric, String> {
    let Some(ps) = ps.as_ref().filter(|_| version >= 3) else {
        return Err(NO_PHOTOMETRIC_NOTE.into());
    };
    let all = loading.iter().chain(release);
    if all
        .clone()
        .any(|s| s.albedo.is_none() || s.normal.is_none())
    {
        return Err(NO_PHOTOMETRIC_NOTE.into());
    }
    if !(ps.albedo_scale.is_finite() && ps.albedo_scale > 0.0) {
        return Err("Photometric layers have an invalid albedo_scale".into());
    }
    check_sizes(
        all.flat_map(|state| {
            [&state.albedo, &state.normal]
                .into_iter()
                .flatten()
                .map(|path| (path.as_path(), texture_px))
        }),
        progress,
    )
    .map_err(|e| format!("Photometric layer: {e}"))?;
    Ok(Photometric {
        albedo_scale: ps.albedo_scale,
    })
}

/// Monotone tension-versus-stretch fit (pool-adjacent-violators), used only for interaction.
/// No fit with fewer than two distinct stretches. [`ForceFit::degenerate`] flags noisy fits
/// so the view can default to indentation.
#[derive(Debug, Clone)]
pub struct ForceFit {
    points: Vec<(f32, f32)>,
    max_correction_n: f32,
    rms_residual_n: f32,
}

/// Smallest tension range (N) of a usable fit.
const MIN_RANGE_N: f32 = 1.0;
/// A fit is a noise fit when its range is below this multiple of the residual rms.
const MIN_SIGNAL_TO_NOISE: f32 = 3.0;

impl ForceFit {
    /// `None` unless there are at least two states with strictly increasing, finite stretch.
    pub fn from_states(states: &[State]) -> Option<Self> {
        let measured: Vec<(f32, f32)> = states
            .iter()
            .filter(|s| s.stretch_ratio.is_finite() && s.force_n.is_finite())
            .map(|s| (s.stretch_ratio, s.force_n))
            .collect();
        if measured.len() < 2 || measured.windows(2).any(|p| p[1].0 <= p[0].0) {
            return None;
        }
        // Tension grows with stretch on both branches, so PAVA runs in stretch order.
        let mut blocks: Vec<(usize, usize, f32, usize)> = Vec::new();
        for (i, (_, force)) in measured.iter().enumerate() {
            blocks.push((i, i + 1, *force, 1));
            while blocks.len() >= 2 {
                let n = blocks.len();
                let (a, b) = (blocks[n - 2], blocks[n - 1]);
                if a.2 / a.3 as f32 <= b.2 / b.3 as f32 {
                    break;
                }
                blocks.pop();
                blocks.pop();
                blocks.push((a.0, b.1, a.2 + b.2, a.3 + b.3));
            }
        }
        let mut fitted = vec![0.0; measured.len()];
        for (start, end, sum, count) in blocks {
            for value in &mut fitted[start..end] {
                *value = sum / count as f32;
            }
        }
        let residuals: Vec<f32> = measured
            .iter()
            .zip(&fitted)
            .map(|((_, raw), fit)| raw - fit)
            .collect();
        let max_correction_n = residuals.iter().fold(0.0_f32, |m, r| m.max(r.abs()));
        let rms_residual_n =
            (residuals.iter().map(|r| r * r).sum::<f32>() / residuals.len() as f32).sqrt();
        Some(Self {
            points: measured
                .iter()
                .zip(fitted)
                .map(|((stretch, _), force)| (*stretch, force))
                .collect(),
            max_correction_n,
            rms_residual_n,
        })
    }

    pub fn force_at(&self, stretch: f32) -> f32 {
        if stretch <= self.points[0].0 {
            return self.points[0].1;
        }
        for pair in self.points.windows(2) {
            if stretch <= pair[1].0 {
                let t = (stretch - pair[0].0) / (pair[1].0 - pair[0].0);
                return pair[0].1 + t * (pair[1].1 - pair[0].1);
            }
        }
        self.points.last().map_or(0.0, |p| p.1)
    }

    /// Tension range the fit can drive: from the unloaded stretch to the highest measured one.
    pub fn range(&self) -> (f32, f32) {
        (self.force_at(1.0), self.points.last().map_or(0.0, |p| p.1))
    }

    /// Largest difference between a measured tension and the fit (N).
    pub fn max_correction_n(&self) -> f32 {
        self.max_correction_n
    }

    /// Why the fit is mostly noise, or `None` for a usable fit.
    pub fn degenerate(&self) -> Option<String> {
        let (lo, hi) = self.range();
        let range = hi - lo;
        let mut levels: Vec<f32> = self.points.iter().map(|p| (p.1 * 1.0e4).round()).collect();
        levels.dedup();
        if range < MIN_RANGE_N {
            Some(format!(
                "the measured tension changes by only {range:.2} N over the whole branch"
            ))
        } else if range < MIN_SIGNAL_TO_NOISE * self.rms_residual_n {
            Some(format!(
                "the tension signal ({range:.1} N) is not much larger than its noise ({:.1} N rms)",
                self.rms_residual_n
            ))
        } else if levels.len() < 4 {
            Some(format!(
                "the monotone fit has only {} distinct tension levels",
                levels.len()
            ))
        } else {
            None
        }
    }

    /// Stretch for a tension. A fit plateau has many valid stretches: keep `previous` there.
    pub fn stretch_for_force(&self, force: f32, previous: f32) -> f32 {
        let start = self.points[0].0.max(1.0);
        let end = self.points.last().map_or(start, |p| p.0);
        let (lo, hi) = self.range();
        if force <= lo {
            return start;
        }
        if force >= hi {
            return end;
        }
        let equal: Vec<f32> = self
            .points
            .iter()
            .filter(|(stretch, fitted)| *stretch >= 1.0 && (*fitted - force).abs() < 1.0e-5)
            .map(|(stretch, _)| *stretch)
            .collect();
        if let (Some(first), Some(last)) = (equal.first(), equal.last()) {
            return previous.clamp(*first, *last);
        }
        for pair in self.points.windows(2) {
            if pair[1].0 <= 1.0 {
                continue;
            }
            if pair[0].1 <= force && force <= pair[1].1 && pair[1].1 > pair[0].1 {
                let t = (force - pair[0].1) / (pair[1].1 - pair[0].1);
                return (pair[0].0 + t * (pair[1].0 - pair[0].0)).clamp(start, end);
            }
        }
        end
    }
}

/// A pair of package folders written by the same `fabsim package` generation.
struct SetDef {
    id: &'static str,
    label: &'static str,
    tiled: &'static str,
    untiled: &'static str,
}

/// Package sets in priority order: version 3 first, then the legacy version 2 folders.
const SET_DEFS: [SetDef; 2] = [
    SetDef {
        id: "v3",
        label: "v3",
        tiled: "material_package_v3",
        untiled: "material_package_v3_untiled",
    },
    SetDef {
        id: "v2",
        label: "legacy v2",
        tiled: "material_package",
        untiled: "material_package_untiled",
    },
];

/// The package directories of one orientation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OrientationDirs {
    pub name: String,
    pub tiled: Option<PathBuf>,
    pub untiled: Option<PathBuf>,
}

/// Which directory a Press View renders for its Texture switch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantChoice {
    pub dir: Option<PathBuf>,
    /// The chosen directory is the tiled package.
    pub tiled: bool,
    pub tiled_available: bool,
    pub untiled_available: bool,
}

impl OrientationDirs {
    /// The variant for the requested texture mode. When only the other variant exists it is used
    /// (the unavailable button is shown disabled with its reason, so this is never silent).
    pub fn choose(&self, tiled_requested: bool) -> VariantChoice {
        let pick_tiled = match (&self.tiled, &self.untiled) {
            (Some(_), Some(_)) => tiled_requested,
            (Some(_), None) => true,
            _ => false,
        };
        VariantChoice {
            dir: if pick_tiled {
                self.tiled.clone()
            } else {
                self.untiled.clone()
            },
            tiled: pick_tiled,
            tiled_available: self.tiled.is_some(),
            untiled_available: self.untiled.is_some(),
        }
    }
}

/// One generation of packages found in a project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageSet {
    pub id: &'static str,
    pub label: &'static str,
    /// Orientations with at least one valid package: `warp` and `weft` first, then by name.
    pub orientations: Vec<OrientationDirs>,
}

impl PackageSet {
    pub fn orientation(&self, name: &str) -> Option<&OrientationDirs> {
        self.orientations.iter().find(|o| o.name == name)
    }
}

/// What was found in a project: the usable package sets, and why anything else was skipped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Discovery {
    /// In priority order. Sets without a valid package are omitted.
    pub sets: Vec<PackageSet>,
    /// One line per package directory that has a `package.json` the viewer cannot use, for
    /// example `material_package_v3/weft: unsupported material package version 4 ...`.
    pub problems: Vec<String>,
}

/// Finds the package sets of a project and the packages that had to be skipped.
pub fn discover(project_root: &Path) -> Discovery {
    discover_with_progress(project_root, &LoadProgress::default())
}

pub fn discover_with_progress(project_root: &Path, progress: &LoadProgress) -> Discovery {
    let mut problems = Vec::new();
    let sets = SET_DEFS
        .iter()
        .filter_map(|def| {
            let mut names: Vec<String> = Vec::new();
            for folder in [def.tiled, def.untiled] {
                let Ok(entries) = std::fs::read_dir(project_root.join(folder)) else {
                    continue;
                };
                // Ignore report files beside orientation folders.
                for entry in entries.flatten().filter(|e| e.path().is_dir()) {
                    if progress.check_cancelled().is_err() {
                        return None;
                    }
                    let name = entry.file_name().to_string_lossy().into_owned();
                    match peek(&entry.path()) {
                        Peek::Package { .. } if !names.contains(&name) => names.push(name),
                        Peek::Broken(reason) => problems.push(format!("{folder}/{name}: {reason}")),
                        _ => {}
                    }
                }
            }
            names.sort_by_key(|n| {
                (
                    ["warp", "weft"].iter().position(|x| x == n).unwrap_or(2),
                    n.clone(),
                )
            });
            let orientations: Vec<OrientationDirs> = names
                .into_iter()
                .map(|name| {
                    let mut dirs = OrientationDirs {
                        name: name.clone(),
                        ..OrientationDirs::default()
                    };
                    // Classified by the package's own `tiling` block, not by the folder name.
                    for folder in [def.untiled, def.tiled] {
                        if progress.check_cancelled().is_err() {
                            break;
                        }
                        let dir = project_root.join(folder).join(&name);
                        match peek(&dir) {
                            Peek::Package { tiled: true } => {
                                dirs.tiled.get_or_insert(dir);
                            }
                            Peek::Package { tiled: false } => {
                                dirs.untiled.get_or_insert(dir);
                            }
                            _ => {}
                        }
                    }
                    dirs
                })
                .collect();
            (!orientations.is_empty()).then_some(PackageSet {
                id: def.id,
                label: def.label,
                orientations,
            })
        })
        .collect();
    problems.sort();
    problems.dedup();
    Discovery { sets, problems }
}

/// What a directory looks like from its `package.json`.
enum Peek {
    /// No `package.json`: not a package directory at all.
    Absent,
    Package {
        tiled: bool,
    },
    /// A `package.json` this viewer cannot use, with the reason.
    Broken(String),
}

fn peek(dir: &Path) -> Peek {
    let path = dir.join("package.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Peek::Absent,
        Err(e) => return Peek::Broken(format!("package.json cannot be read ({e})")),
    };
    let manifest: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(manifest) => manifest,
        Err(e) => return Peek::Broken(format!("package.json is not valid JSON ({e})")),
    };
    let format = manifest.get("format").and_then(|f| f.as_str());
    if format != Some(FORMAT) {
        return Peek::Broken(format!("not a material package (format {format:?})"));
    }
    match manifest.get("version").and_then(|v| v.as_u64()) {
        Some(2 | 3) => {}
        Some(other) => return Peek::Broken(unsupported_version(other)),
        None => return Peek::Broken("package.json has no version number".into()),
    }
    Peek::Package {
        tiled: manifest
            .get("tiling")
            .and_then(|t| t.get("enabled"))
            .and_then(|e| e.as_bool())
            .unwrap_or(false),
    }
}

/// Fixtures shared by the tests of the Press View modules.
#[cfg(test)]
pub mod test_support {
    use std::path::PathBuf;

    /// The committed demo project (`examples/demo`).
    pub fn demo_project() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/demo")
    }

    /// Directory of real measurement data (`FABSIM_DATA_DIR`) for the `#[ignore]`d tests.
    pub fn real_data_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("FABSIM_DATA_DIR").map(PathBuf::from)?;
        dir.is_dir().then_some(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    fn demo(dir: &str) -> MaterialPackage {
        MaterialPackage::load(&demo_project().join(dir).join("warp")).unwrap()
    }

    fn state(stretch: f32, force: f32) -> State {
        State {
            frame_id: None,
            stretch_ratio: stretch,
            force_n: force,
            force_raw_n: None,
            travel_mm: None,
            texture: PathBuf::new(),
            albedo: None,
            normal: None,
        }
    }

    fn states(values: &[(f32, f32)]) -> Vec<State> {
        values.iter().map(|(s, f)| state(*s, *f)).collect()
    }

    #[test]
    fn v3_demo_package_parses_with_all_blocks() {
        let tiled = demo("material_package_v3");
        assert_eq!(tiled.version(), 3);
        assert_eq!(tiled.orientation(), "warp");
        assert_eq!(tiled.atlas_px(), [164, 68]);
        assert_eq!(
            tiled.texture_px(),
            [152, 45],
            "tiled textures are one period"
        );
        assert!(tiled.pull_vertical());
        assert!(tiled.tiling().is_some());
        assert_eq!(tiled.clamp_width_m(), Some(0.202));
        let residual = tiled.residual_stretch().unwrap();
        assert!(residual > 1.0 && residual < 1.02, "residual {residual}");
        let ps = tiled.photometric().unwrap();
        assert!((ps.albedo_scale - 1.067).abs() < 0.01);
        // Tared force: the loading reference state carries zero tension.
        let first = &tiled.states(PressBranch::Loading)[0];
        assert_eq!((first.stretch_ratio, first.force_n), (1.0, 0.0));
        assert_eq!(
            (first.force_raw_n, first.travel_mm),
            (Some(-4.3), Some(0.0))
        );
        assert!(first.albedo.is_some() && first.normal.is_some());
        // Release is measured against the loading reference: it ends at the residual stretch > 1.
        let (lo, hi) = tiled.stretch_range(PressBranch::Release).unwrap();
        assert!((lo - residual).abs() < 1.0e-5 && hi > 1.2);

        let untiled = demo("material_package_v3_untiled");
        assert!(untiled.tiling().is_none());
        assert_eq!(untiled.texture_px(), untiled.atlas_px());
        assert!(untiled.photometric().is_ok());
    }

    /// Writes a package directory with solid-colour textures.
    fn synthetic(name: &str, manifest: impl FnOnce(&mut serde_json::Value)) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fabric_viewer_pkg_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut states = Vec::new();
        for (i, (stretch, force)) in [(1.0, 0.0), (1.1, 1.0), (1.2, 3.0)].iter().enumerate() {
            let file = format!("frame_{i}.png");
            image::RgbImage::from_pixel(8, 4, image::Rgb([10 * i as u8, 20, 30]))
                .save(dir.join(&file))
                .unwrap();
            states.push(serde_json::json!({
                "frame_id": i, "stretch_ratio": stretch, "force_n": force, "texture": file,
                "extra_future_key": [1, 2, 3],
            }));
        }
        let mut value = serde_json::json!({
            "format": FORMAT, "version": 2, "orientation": "warp",
            "atlas": {"dimensions_px": [8, 4], "grid_cells": [0, 0, 2, 1],
                       "origin": "Loading reference grid top-left", "active_grid_axis": "vertical"},
            "branches": {"loading": states, "release": states},
            "some_new_block": {"x": 1},
        });
        manifest(&mut value);
        std::fs::write(dir.join("package.json"), value.to_string()).unwrap();
        dir
    }

    #[test]
    fn v2_packages_are_still_readable_and_unknown_keys_are_ignored() {
        let dir = synthetic("v2", |_| {});
        let package = MaterialPackage::load(&dir).unwrap();
        assert_eq!(package.version(), 2);
        assert_eq!(package.states(PressBranch::Loading).len(), 3);
        assert_eq!(package.photometric(), Err(NO_PHOTOMETRIC_NOTE));
        assert!(package.force_fit(PressBranch::Loading).is_some());
        assert_eq!(package.residual_stretch(), None);
    }

    /// Adds `offset` to every force of a synthetic manifest (a raw load-cell reading).
    fn with_force_offset(m: &mut serde_json::Value, offset: f64) {
        for branch in ["loading", "release"] {
            for s in m["branches"][branch].as_array_mut().unwrap() {
                let force = s["force_n"].as_f64().unwrap() + offset;
                s["force_n"] = force.into();
            }
        }
    }

    #[test]
    fn version_two_forces_are_tared_at_the_first_loading_state() {
        let dir = synthetic("v2_tare", |m| with_force_offset(m, -4.3));
        let package = MaterialPackage::load(&dir).unwrap();
        assert!((package.legacy_tare_n().unwrap() + 4.3).abs() < 1.0e-5);
        let loading = package.states(PressBranch::Loading);
        assert!(
            loading[0].force_n.abs() < 1.0e-5,
            "unloaded state reads zero"
        );
        assert!((loading[0].force_raw_n.unwrap() + 4.3).abs() < 1.0e-5);
        assert!((loading[2].force_n - 3.0).abs() < 1.0e-5);
        let (lo, hi) = package.force_fit(PressBranch::Loading).unwrap().range();
        assert!(lo.abs() < 1.0e-5 && (hi - 3.0).abs() < 1.0e-5, "{lo}..{hi}");

        // Version 3 forces are tared by the writer and left alone.
        let dir = synthetic("v3_tare", |m| {
            m["version"] = 3.into();
            with_force_offset(m, 0.0);
        });
        let v3 = MaterialPackage::load(&dir).unwrap();
        assert_eq!(v3.legacy_tare_n(), None);
        assert_eq!(v3.states(PressBranch::Loading)[2].force_n, 3.0);
    }

    #[test]
    fn version_two_machine_displacement_is_read_as_travel() {
        let dir = synthetic("v2_travel", |m| {
            m["branches"]["loading"][1]["machine_displacement_mm"] = 12.5.into();
        });
        let package = MaterialPackage::load(&dir).unwrap();
        assert_eq!(
            package.states(PressBranch::Loading)[1].travel_mm,
            Some(12.5)
        );
    }

    #[test]
    fn numbers_that_would_make_the_view_silently_useless_are_rejected() {
        // 1e39 does not fit an f32 and parses as infinity.
        let inf = synthetic("inf", |m| {
            m["branches"]["loading"][2]["stretch_ratio"] = serde_json::json!(1.0e39);
        });
        let error = MaterialPackage::load(&inf).unwrap_err();
        assert!(
            error.contains("loading state 2") && error.contains("finite"),
            "{error}"
        );
        let flat = synthetic("flat", |m| {
            for s in m["branches"]["release"].as_array_mut().unwrap() {
                s["stretch_ratio"] = 1.0.into();
            }
        });
        let error = MaterialPackage::load(&flat).unwrap_err();
        assert!(
            error.contains("release branch has no stretch range"),
            "{error}"
        );
    }

    #[test]
    fn texture_paths_cannot_leave_the_package_directory() {
        for bad in ["/etc/hostname", "../frame_0.png", "sub/../../frame_0.png"] {
            let dir = synthetic("escape", |m| {
                m["branches"]["loading"][0]["texture"] = bad.into();
            });
            let error = MaterialPackage::load(&dir).unwrap_err();
            assert!(
                error.contains("outside the package directory"),
                "{bad}: {error}"
            );
        }
    }

    #[test]
    fn version_one_and_unknown_formats_are_rejected_with_a_message() {
        let v1 = synthetic("v1", |m| m["version"] = 1.into());
        let error = MaterialPackage::load(&v1).unwrap_err();
        assert!(
            error.contains("unsupported material package version 1"),
            "{error}"
        );
        let other = synthetic("fmt", |m| m["format"] = "something-else".into());
        assert!(MaterialPackage::load(&other)
            .unwrap_err()
            .contains("not a material package"));
        let missing = std::env::temp_dir().join("fabric_viewer_no_such_package");
        assert!(MaterialPackage::load(&missing)
            .unwrap_err()
            .contains("package.json"));
    }

    #[test]
    fn a_texture_of_the_wrong_size_names_the_file() {
        let dir = synthetic("size", |_| {});
        image::RgbImage::new(3, 3)
            .save(dir.join("frame_1.png"))
            .unwrap();
        let error = MaterialPackage::load(&dir).unwrap_err();
        assert!(
            error.contains("frame_1.png") && error.contains("3x3"),
            "{error}"
        );
    }

    #[test]
    fn inconsistent_tiling_metadata_is_rejected() {
        let dir = synthetic("tiling", |m| {
            m["tiling"] = serde_json::json!({
                "enabled": true, "tile_size_px": [4, 2], "source_atlas_size_px": [8, 4],
                "period_px": [4.0, 2.0], "origin_uv": [0.0, 0.0], "period_uv": [0.9, 0.5],
            });
        });
        assert_eq!(
            MaterialPackage::load(&dir).unwrap_err(),
            "inconsistent tiling metadata"
        );
    }

    #[test]
    fn broken_photometric_layers_disable_relight_but_not_the_package() {
        let dir = synthetic("ps", |m| {
            m["version"] = 3.into();
            m["ps"] = serde_json::json!({"albedo_scale": 0.5});
            for branch in ["loading", "release"] {
                for (i, s) in m["branches"][branch]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .enumerate()
                {
                    s["albedo"] = format!("a{i}.png").into();
                    s["normal"] = format!("n{i}.png").into();
                }
            }
        });
        let package = MaterialPackage::load(&dir).unwrap();
        assert!(package.photometric().unwrap_err().contains("a0.png"));
    }

    #[test]
    fn texture_plan_brackets_states_and_clamps_outside_the_range() {
        let package = demo("material_package_v3_untiled");
        let (lo, hi) = package.stretch_range(PressBranch::Loading).unwrap();
        let mid = package.states(PressBranch::Loading)[3].stretch_ratio;
        let between = (mid + package.states(PressBranch::Loading)[4].stretch_ratio) * 0.5;
        let plan = package.texture_plan(PressBranch::Loading, between).unwrap();
        assert_eq!((plan.lo, plan.hi), (3, 4));
        assert!((plan.t - 0.5).abs() < 1.0e-4 && !plan.clamped);
        assert!(plan.albedo.is_some() && plan.normal.is_some());
        let exact = package.texture_plan(PressBranch::Loading, mid).unwrap();
        assert_eq!((exact.lo, exact.t), (3, 0.0));
        let below = package
            .texture_plan(PressBranch::Loading, lo - 0.1)
            .unwrap();
        assert!(below.clamped && (below.lo, below.hi, below.t) == (0, 0, 0.0));
        let above = package
            .texture_plan(PressBranch::Loading, hi + 0.1)
            .unwrap();
        assert!(above.clamped && above.hi == package.states(PressBranch::Loading).len() - 1);
        assert!(package
            .texture_plan(PressBranch::Loading, f32::NAN)
            .is_err());
    }

    #[test]
    fn pava_preserves_monotone_data_and_inverts_intermediate_force() {
        let fit = ForceFit::from_states(&states(&[(1.0, 1.0), (1.1, 3.0), (1.2, 7.0)])).unwrap();
        assert_eq!(fit.points, vec![(1.0, 1.0), (1.1, 3.0), (1.2, 7.0)]);
        assert_eq!(fit.max_correction_n(), 0.0);
        assert_eq!(fit.stretch_for_force(1.0, 1.1), 1.0);
        assert_eq!(fit.stretch_for_force(7.0, 1.1), 1.2);
        assert!((fit.stretch_for_force(5.0, 1.0) - 1.15).abs() < 1.0e-6);
    }

    #[test]
    fn pava_pools_reversals_and_plateaus_keep_the_previous_stretch() {
        let fit = ForceFit::from_states(&states(&[
            (1.0, 0.0),
            (1.1, 4.0),
            (1.2, 2.0),
            (1.3, 3.0),
            (1.4, 9.0),
        ]))
        .unwrap();
        // 4, 2, 3 pool to their mean 3.
        assert_eq!(fit.points[1].1, 3.0);
        assert_eq!(fit.points[2].1, 3.0);
        assert_eq!(fit.points[3].1, 3.0);
        assert_eq!(fit.max_correction_n(), 1.0);
        assert_eq!(fit.stretch_for_force(3.0, 1.25), 1.25);
        assert_eq!(
            fit.stretch_for_force(3.0, 5.0),
            1.3,
            "clamped to the plateau"
        );
    }

    #[test]
    fn a_fit_needs_two_distinct_stretches() {
        assert!(ForceFit::from_states(&states(&[(1.0, 0.0)])).is_none());
        assert!(ForceFit::from_states(&states(&[(1.0, 0.0), (1.0, 2.0)])).is_none());
    }

    #[test]
    fn a_noise_fit_is_flagged_with_a_reason() {
        // Range 3 N hidden in 2 N noise (a fabric_002-like branch).
        let noisy: Vec<(f32, f32)> = (0..20)
            .map(|i| {
                (
                    1.0 + 0.01 * i as f32,
                    3.0 * i as f32 / 19.0 + if i % 2 == 0 { 1.5 } else { -1.5 },
                )
            })
            .collect();
        let fit = ForceFit::from_states(&states(&noisy)).unwrap();
        let reason = fit.degenerate().expect("noise fit");
        assert!(
            reason.contains("noise") || reason.contains("distinct"),
            "{reason}"
        );
        let flat = ForceFit::from_states(&states(&[(1.0, 0.0), (1.1, 0.2), (1.2, 0.4)])).unwrap();
        assert!(flat.degenerate().unwrap().contains("only 0.40 N"));
        let ramp: Vec<(f32, f32)> = (0..10)
            .map(|i| (1.0 + 0.02 * i as f32, 2.0 * i as f32))
            .collect();
        assert!(ForceFit::from_states(&states(&ramp))
            .unwrap()
            .degenerate()
            .is_none());
    }

    #[test]
    fn force_fit_quality_on_the_demo_package() {
        let package = demo("material_package_v3_untiled");
        for branch in [PressBranch::Loading, PressBranch::Release] {
            let fit = package.force_fit(branch).unwrap();
            let (lo, hi) = fit.range();
            assert!(hi > lo + 1.0, "{branch:?}: {lo} .. {hi}");
            assert!(fit.degenerate().is_none(), "{:?}", fit.degenerate());
            // The fit never leaves the measured envelope.
            let (s_lo, s_hi) = package.stretch_range(branch).unwrap();
            assert!(fit.stretch_for_force((lo + hi) * 0.5, 1.0) > s_lo);
            assert!(fit.stretch_for_force(hi, 1.0) <= s_hi + 1.0e-6);
        }
    }

    #[test]
    fn demo_sets_prefer_v3_and_list_the_discovered_orientations() {
        let sets = discover(&demo_project()).sets;
        assert_eq!(sets.len(), 1, "the demo only has v3 packages");
        assert_eq!(sets[0].id, "v3");
        let names: Vec<_> = sets[0]
            .orientations
            .iter()
            .map(|o| o.name.as_str())
            .collect();
        assert_eq!(names, ["warp"]);
        let warp = sets[0].orientation("warp").unwrap();
        assert!(warp
            .tiled
            .as_ref()
            .unwrap()
            .ends_with("material_package_v3/warp"));
        assert!(warp
            .untiled
            .as_ref()
            .unwrap()
            .ends_with("material_package_v3_untiled/warp"));
    }

    fn project_with_sets(name: &str, layout: &[(&str, &str, bool)]) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("fabric_viewer_sets_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (folder, orientation, tiled) in layout {
            let dir = root.join(folder).join(orientation);
            std::fs::create_dir_all(&dir).unwrap();
            let manifest = serde_json::json!({
                "format": FORMAT, "version": if folder.contains("v3") { 3 } else { 2 },
                "tiling": {"enabled": tiled},
            });
            std::fs::write(dir.join("package.json"), manifest.to_string()).unwrap();
        }
        root
    }

    #[test]
    fn v3_is_listed_before_legacy_v2_and_empty_directories_are_ignored() {
        let root = project_with_sets(
            "both",
            &[
                ("material_package", "warp", true),
                ("material_package_untiled", "warp", false),
                ("material_package_untiled", "weft", false),
                ("material_package_v3", "warp", true),
                ("material_package_v3_untiled", "warp", false),
            ],
        );
        std::fs::create_dir_all(root.join("material_package_v3/weft")).unwrap();
        let sets = discover(&root).sets;
        let ids: Vec<_> = sets.iter().map(|s| s.id).collect();
        assert_eq!(ids, ["v3", "v2"]);
        assert_eq!(
            sets[0].orientations.len(),
            1,
            "an empty weft directory is not an orientation"
        );
        let v2: Vec<_> = sets[1]
            .orientations
            .iter()
            .map(|o| o.name.as_str())
            .collect();
        assert_eq!(v2, ["warp", "weft"]);
        assert!(sets[1].orientation("weft").unwrap().tiled.is_none());
    }

    #[test]
    fn version_one_directories_are_not_packages() {
        let root = project_with_sets("v1", &[("material_package", "warp", false)]);
        let manifest = root.join("material_package/warp/package.json");
        std::fs::write(&manifest, r#"{"format":"fabsim-material-rgb","version":1}"#).unwrap();
        let found = discover(&root);
        assert!(found.sets.is_empty());
        assert!(
            found.problems[0].contains("version 1"),
            "{:?}",
            found.problems
        );
    }

    #[test]
    fn unusable_packages_are_reported_with_their_reason_instead_of_vanishing() {
        let root = project_with_sets(
            "problems",
            &[
                ("material_package_v3", "warp", true),
                ("material_package_v3", "weft", true),
                ("material_package_v3", "bad_json", true),
                ("material_package_v3", "future", true),
                ("material_package_v3", "other", true),
            ],
        );
        let at = |name: &str| {
            root.join("material_package_v3")
                .join(name)
                .join("package.json")
        };
        std::fs::write(at("bad_json"), "{bad").unwrap();
        std::fs::write(
            at("future"),
            r#"{"format":"fabsim-material-rgb","version":4}"#,
        )
        .unwrap();
        std::fs::write(at("other"), r#"{"format":"something-else","version":3}"#).unwrap();
        std::fs::write(root.join("material_package_v3/build_report.json"), "{}").unwrap();

        let found = discover(&root);
        let names: Vec<_> = found.sets[0]
            .orientations
            .iter()
            .map(|o| o.name.as_str())
            .collect();
        assert_eq!(names, ["warp", "weft"]);
        assert_eq!(found.problems.len(), 3, "{:?}", found.problems);
        let all = found.problems.join("\n");
        assert!(
            all.contains("material_package_v3/bad_json: package.json is not valid JSON"),
            "{all}"
        );
        assert!(
            all.contains("material_package_v3/future: unsupported material package version 4"),
            "{all}"
        );
        assert!(
            all.contains("material_package_v3/other: not a material package"),
            "{all}"
        );
    }

    #[test]
    fn texture_variant_choice_never_pretends() {
        let both = OrientationDirs {
            name: "warp".into(),
            tiled: Some("t".into()),
            untiled: Some("u".into()),
        };
        assert_eq!(both.choose(true).dir, Some("t".into()));
        assert_eq!(both.choose(false).dir, Some("u".into()));
        let only_untiled = OrientationDirs {
            tiled: None,
            ..both.clone()
        };
        let choice = only_untiled.choose(true);
        assert_eq!(choice.dir, Some("u".into()));
        assert!(!choice.tiled && !choice.tiled_available && choice.untiled_available);
        let only_tiled = OrientationDirs {
            untiled: None,
            ..both
        };
        let choice = only_tiled.choose(false);
        assert_eq!(choice.dir, Some("t".into()));
        assert!(choice.tiled && !choice.untiled_available);
    }

    #[test]
    fn tiling_maps_material_uv_to_continuous_tile_units() {
        let tiling = Tiling {
            origin_uv: [0.25, 0.5],
            period_uv: [0.5, 0.25],
        };
        assert_eq!(tiling.texture_uv([0.25, 0.5]), [0.0, 0.0]);
        assert_eq!(tiling.texture_uv([0.75, 0.75]), [1.0, 1.0]);
        assert_eq!(tiling.texture_uv([0.0, 0.5]), [-0.5, 0.0]);
    }

    /// Real packages (all projects in `FABSIM_DATA_DIR` with a v2 or v3 package set) load.
    #[test]
    #[ignore = "needs FABSIM_DATA_DIR pointing at real measurement projects"]
    fn real_projects_load_as_packages() {
        let Some(data) = real_data_dir() else {
            eprintln!("FABSIM_DATA_DIR is not set; skipping");
            return;
        };
        let mut loaded = 0;
        for project in std::fs::read_dir(&data).unwrap().flatten() {
            for set in discover(&project.path()).sets {
                for orientation in &set.orientations {
                    for dir in [&orientation.tiled, &orientation.untiled]
                        .into_iter()
                        .flatten()
                    {
                        let package = MaterialPackage::load(dir)
                            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
                        let (_, hi) = package.stretch_range(PressBranch::Loading).unwrap();
                        assert!(hi > 1.0, "{}", dir.display());
                        loaded += 1;
                    }
                }
            }
        }
        assert!(loaded > 0, "no package found under {}", data.display());
    }
}
