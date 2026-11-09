//! Shared package and GPU texture cache for Press Views.
//! Decode and build mipmaps on workers, then upload on the UI thread. Keep the last
//! complete pair while loading. In-use textures are protected from eviction; errors
//! are cached until rescan or a file change.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use eframe::egui_wgpu;

use crate::material_package::{discover, Discovery, MaterialPackage, PackageSet};
use crate::press_gpu::{GpuContext, GpuTexture};
use crate::press_render::{file_stamp, MipChain, SceneFiles, TexKind};

/// Video memory the cached state textures may take. Textures used by the current frame are never
/// evicted, so a scene that needs more than this still draws.
const GPU_BUDGET_BYTES: usize = 512 << 20;

/// Decoded textures uploaded per frame, so a burst of finished decodes cannot stall one frame.
const UPLOADS_PER_FRAME: usize = 2;

/// Worker threads decoding textures.
const DECODE_THREADS: usize = 2;

/// How often a cached texture's file is checked for changes.
const STAMP_RECHECK: Duration = Duration::from_secs(1);

/// Set `FABRIC_CPU_RENDER` to draw the Press View on the CPU even when a GPU is available.
const CPU_RENDER_ENV: &str = "FABRIC_CPU_RENDER";

struct PackageEntry {
    result: Result<Rc<MaterialPackage>, String>,
    /// Modification time of `package.json` when it was read.
    stamp: Option<SystemTime>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TexKey {
    path: PathBuf,
    kind: TexKind,
}

enum GpuState {
    Ready(Arc<GpuTexture>),
    /// Decoding or uploading failed; the file is not read again until it changes.
    Failed(String),
}

/// A cached texture (or its failure) and the modification time of the file it came from.
struct GpuEntry {
    key: TexKey,
    stamp: Option<SystemTime>,
    /// When the file's modification time was last compared.
    checked: Instant,
    /// Frame (see [`PackageStore::gpu_textures`]) that last used the entry.
    last_used: u64,
    state: GpuState,
}

struct Decoded {
    key: TexKey,
    chain: Result<MipChain, String>,
}

/// Decodes textures on background threads.
struct Decoder {
    jobs: mpsc::Sender<TexKey>,
    results: mpsc::Receiver<Decoded>,
}

impl Decoder {
    fn spawn(wake: egui::Context) -> Self {
        let (jobs, job_rx) = mpsc::channel::<TexKey>();
        let (result_tx, results) = mpsc::channel();
        let job_rx = Arc::new(Mutex::new(job_rx));
        for n in 0..DECODE_THREADS {
            let (job_rx, result_tx, wake) = (job_rx.clone(), result_tx.clone(), wake.clone());
            std::thread::Builder::new()
                .name(format!("press-decode-{n}"))
                .spawn(move || loop {
                    let job = match job_rx.lock() {
                        Ok(rx) => rx.recv(),
                        Err(_) => break,
                    };
                    let Ok(key) = job else { break };
                    let chain = std::panic::catch_unwind(|| MipChain::load(&key.path, key.kind))
                        .unwrap_or_else(|_| {
                            Err(format!(
                                "{}: decoding failed unexpectedly",
                                key.path.display()
                            ))
                        });
                    if result_tx.send(Decoded { key, chain }).is_err() {
                        break;
                    }
                    wake.request_repaint();
                })
                .expect("spawn a texture decode thread");
        }
        Self { jobs, results }
    }
}

/// What [`PackageStore::gpu_textures`] has for a scene.
pub enum Textures {
    /// Every texture of the scene, in the order of [`SceneFiles`].
    Ready(Vec<Arc<GpuTexture>>),
    /// Some textures are still being decoded or uploaded; ask again next frame.
    Loading,
}

pub struct PackageStore {
    discoveries: HashMap<PathBuf, Discovery>,
    packages: HashMap<PathBuf, PackageEntry>,
    gpu: Option<GpuContext>,
    gpu_textures: Vec<GpuEntry>,
    /// Textures handed to the decoder and not yet uploaded, with the file stamp at request time.
    pending: HashMap<TexKey, Option<SystemTime>>,
    decoder: Option<Decoder>,
    frame: u64,
    uploads_this_frame: usize,
    budget_bytes: usize,
    recheck: Duration,
}

impl Default for PackageStore {
    fn default() -> Self {
        Self {
            discoveries: HashMap::new(),
            packages: HashMap::new(),
            gpu: None,
            gpu_textures: Vec::new(),
            pending: HashMap::new(),
            decoder: None,
            frame: 0,
            uploads_this_frame: 0,
            budget_bytes: GPU_BUDGET_BYTES,
            recheck: STAMP_RECHECK,
        }
    }
}

fn manifest_stamp(dir: &Path) -> Option<SystemTime> {
    std::fs::metadata(dir.join("package.json"))
        .and_then(|m| m.modified())
        .ok()
}

impl PackageStore {
    /// Store that draws on the GPU when a wgpu render state is available; without one (or with
    /// `FABRIC_CPU_RENDER` set) the views fall back to the CPU worker.
    pub fn with_gpu(render_state: Option<&egui_wgpu::RenderState>) -> Self {
        let forced_cpu = std::env::var_os(CPU_RENDER_ENV).is_some();
        Self {
            gpu: render_state
                .filter(|_| !forced_cpu)
                .map(GpuContext::from_render_state),
            ..Self::default()
        }
    }

    /// Store drawing on an existing context (tests).
    #[cfg(test)]
    fn with_context(gpu: GpuContext) -> Self {
        Self {
            gpu: Some(gpu),
            ..Self::default()
        }
    }

    pub fn gpu(&self) -> Option<&GpuContext> {
        self.gpu.as_ref()
    }

    fn discovery(&mut self, project_root: &Path) -> &Discovery {
        self.discoveries
            .entry(project_root.to_path_buf())
            .or_insert_with(|| discover(project_root))
    }

    /// The package sets of a project (cached until [`PackageStore::rescan`]).
    pub fn sets(&mut self, project_root: &Path) -> &[PackageSet] {
        &self.discovery(project_root).sets
    }

    /// Package directories of the project that were skipped, each with the reason.
    pub fn problems(&mut self, project_root: &Path) -> &[String] {
        &self.discovery(project_root).problems
    }

    /// Forgets what was discovered and loaded below `project_root`, so new or rebuilt packages
    /// are picked up and failed loads (of packages and of textures) are retried.
    pub fn rescan(&mut self, project_root: &Path) {
        self.discoveries.remove(project_root);
        self.packages
            .retain(|dir, _| !dir.starts_with(project_root));
        self.gpu_textures.retain(|e| {
            !(e.key.path.starts_with(project_root) && matches!(e.state, GpuState::Failed(_)))
        });
    }

    /// Parsed package at `dir`. Failures are cached with their message; a changed `package.json`
    /// is read again.
    pub fn package(&mut self, dir: &Path) -> Result<Rc<MaterialPackage>, String> {
        let stamp = manifest_stamp(dir);
        let stale = self.packages.get(dir).is_some_and(|e| e.stamp != stamp);
        if stale {
            self.packages.remove(dir);
        }
        self.packages
            .entry(dir.to_path_buf())
            .or_insert_with(|| PackageEntry {
                result: MaterialPackage::load(dir).map(Rc::new),
                stamp,
            })
            .result
            .clone()
    }

    /// Drops everything cached below `root` (call when a project is closed).
    pub fn forget_project(&mut self, root: &Path) {
        self.discoveries.retain(|dir, _| !dir.starts_with(root));
        self.packages.retain(|dir, _| !dir.starts_with(root));
        self.gpu_textures.retain(|e| !e.key.path.starts_with(root));
        self.pending.retain(|key, _| !key.path.starts_with(root));
    }

    /// `(project discoveries, packages, GPU textures)` currently cached.
    #[cfg(test)]
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.discoveries.len(),
            self.packages.len(),
            self.gpu_textures.len(),
        )
    }

    /// Request scene textures without blocking on decode. Returns [`Textures::Loading`]
    /// until all are uploaded; decode errors are cached. Call each frame to protect
    /// the requested textures from eviction.
    pub fn gpu_textures(
        &mut self,
        ctx: &egui::Context,
        files: &SceneFiles,
    ) -> Result<Textures, String> {
        self.begin_frame(ctx);
        self.upload_decoded();
        let mut out = Vec::new();
        let mut loading = false;
        for key in scene_keys(files) {
            match self.lookup(&key) {
                Lookup::Ready(texture) => out.push(texture),
                Lookup::Failed(error) => return Err(error),
                Lookup::Missing => {
                    self.request(ctx, key);
                    loading = true;
                }
            }
        }
        self.evict();
        if loading {
            // Keep polling for uploads even if the decoder wake-up was already consumed.
            ctx.request_repaint_after(Duration::from_millis(16));
            Ok(Textures::Loading)
        } else {
            Ok(Textures::Ready(out))
        }
    }

    /// Starts decoding textures the view is likely to need next (the states beside the current
    /// pair), unless the cache is nearly full. Nothing is protected from eviction.
    pub fn prefetch(&mut self, ctx: &egui::Context, paths: &[PathBuf], kind: TexKind) {
        if self.gpu.is_none() || self.bytes() > self.budget_bytes / 4 * 3 {
            return;
        }
        for path in paths {
            let key = TexKey {
                path: path.clone(),
                kind,
            };
            let known = self.gpu_textures.iter().any(|e| e.key == key);
            if !known && !self.pending.contains_key(&key) {
                self.request(ctx, key);
            }
        }
    }

    fn begin_frame(&mut self, ctx: &egui::Context) {
        let frame = ctx.cumulative_frame_nr();
        if frame != self.frame {
            self.frame = frame;
            self.uploads_this_frame = 0;
        }
    }

    fn bytes(&self) -> usize {
        self.gpu_textures
            .iter()
            .map(|e| match &e.state {
                GpuState::Ready(texture) => texture.bytes(),
                GpuState::Failed(_) => 0,
            })
            .sum()
    }

    /// Uploads finished decodes, at most [`UPLOADS_PER_FRAME`] per frame.
    fn upload_decoded(&mut self) {
        let (Some(gpu), Some(decoder)) = (self.gpu.as_ref(), self.decoder.as_ref()) else {
            return;
        };
        while self.uploads_this_frame < UPLOADS_PER_FRAME {
            let Ok(done) = decoder.results.try_recv() else {
                break;
            };
            // Forgotten while it was decoding (the project was closed).
            let Some(stamp) = self.pending.remove(&done.key) else {
                continue;
            };
            let state = match done.chain {
                Ok(chain) => {
                    self.uploads_this_frame += 1;
                    match gpu.upload(&chain) {
                        Ok(texture) => GpuState::Ready(Arc::new(texture)),
                        Err(error) => GpuState::Failed(error),
                    }
                }
                Err(error) => GpuState::Failed(error),
            };
            self.gpu_textures.retain(|e| e.key != done.key);
            self.gpu_textures.push(GpuEntry {
                key: done.key,
                stamp,
                checked: Instant::now(),
                last_used: self.frame,
                state,
            });
        }
    }

    fn lookup(&mut self, key: &TexKey) -> Lookup {
        let Some(pos) = self.gpu_textures.iter().position(|e| e.key == *key) else {
            return Lookup::Missing;
        };
        let entry = &mut self.gpu_textures[pos];
        if entry.checked.elapsed() >= self.recheck {
            entry.checked = Instant::now();
            let stamp = file_stamp(&key.path);
            let changed = match entry.state {
                // A file that vanished keeps serving what was decoded from it.
                GpuState::Ready(_) => stamp.is_some() && stamp != entry.stamp,
                GpuState::Failed(_) => stamp != entry.stamp,
            };
            if changed {
                self.gpu_textures.remove(pos);
                return Lookup::Missing;
            }
        }
        entry.last_used = self.frame;
        match &entry.state {
            GpuState::Ready(texture) => Lookup::Ready(texture.clone()),
            GpuState::Failed(error) => Lookup::Failed(error.clone()),
        }
    }

    fn request(&mut self, ctx: &egui::Context, key: TexKey) {
        if self.gpu.is_none() || self.pending.contains_key(&key) {
            return;
        }
        let decoder = self
            .decoder
            .get_or_insert_with(|| Decoder::spawn(ctx.clone()));
        let stamp = file_stamp(&key.path);
        if decoder.jobs.send(key.clone()).is_ok() {
            self.pending.insert(key, stamp);
        }
    }

    /// Evicts least recently used textures until the cache fits its budget; textures used by the
    /// current frame stay.
    fn evict(&mut self) {
        let mut bytes = self.bytes();
        while bytes > self.budget_bytes {
            let oldest = self
                .gpu_textures
                .iter()
                .enumerate()
                .filter(|(_, e)| e.last_used != self.frame)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i);
            let Some(index) = oldest else { break };
            if let GpuState::Ready(texture) = &self.gpu_textures.remove(index).state {
                bytes -= texture.bytes();
            }
        }
    }
}

enum Lookup {
    Ready(Arc<GpuTexture>),
    Failed(String),
    Missing,
}

fn scene_keys(files: &SceneFiles) -> Vec<TexKey> {
    let key = |path: &PathBuf, kind| TexKey {
        path: path.clone(),
        kind,
    };
    match files {
        SceneFiles::Rgb(paths) => paths.iter().map(|p| key(p, TexKind::Srgb8)).collect(),
        SceneFiles::Relight { albedo, normal } => albedo
            .iter()
            .chain(normal)
            .map(|p| key(p, TexKind::Unorm16))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_package::test_support::demo_project;

    #[test]
    fn packages_and_sets_are_cached_and_forgotten_with_their_project() {
        let mut store = PackageStore::default();
        let project = demo_project();
        assert_eq!(store.sets(&project).len(), 1);
        assert!(store.problems(&project).is_empty());
        let dir = project.join("material_package_v3_untiled/warp");
        let a = store.package(&dir).unwrap();
        let b = store.package(&dir).unwrap();
        assert!(Rc::ptr_eq(&a, &b), "one parsed package per directory");
        assert_eq!(store.counts(), (1, 1, 0));
        store.forget_project(&project);
        assert_eq!(store.counts(), (0, 0, 0));
    }

    #[test]
    fn load_errors_are_cached_with_the_real_message_until_a_rescan() {
        let root = std::env::temp_dir().join(format!("press_store_err_{}", std::process::id()));
        let dir = root.join("material_package_v3/warp");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("package.json"),
            r#"{"format":"fabsim-material-rgb","version":3}"#,
        )
        .unwrap();
        let mut store = PackageStore::default();
        let error = store.package(&dir).err().unwrap();
        assert!(
            error.contains("package.json") && error.contains("atlas"),
            "{error}"
        );
        assert_eq!(
            store.package(&dir).err().unwrap(),
            error,
            "the failure is cached"
        );

        // Replace the package by a valid one: a rescan loads it.
        let src = demo_project().join("material_package_v3_untiled/warp");
        copy_dir(&src, &dir);
        store.rescan(&root);
        assert!(store.package(&dir).is_ok());
    }

    #[test]
    fn a_changed_manifest_is_noticed_without_a_rescan() {
        let root = std::env::temp_dir().join(format!("press_store_mtime_{}", std::process::id()));
        let dir = root.join("warp");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("package.json"), "{}").unwrap();
        let mut store = PackageStore::default();
        assert!(store.package(&dir).is_err());
        copy_dir(
            &demo_project().join("material_package_v3_untiled/warp"),
            &dir,
        );
        let later = SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(dir.join("package.json"))
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert!(
            store.package(&dir).is_ok(),
            "a newer package.json is read again"
        );
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap().flatten() {
            let target = to.join(entry.file_name());
            if entry.path().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    fn gpu_store() -> Option<PackageStore> {
        let Some(gpu) = GpuContext::headless() else {
            eprintln!("no GPU adapter available; skipping");
            return None;
        };
        Some(PackageStore::with_context(gpu))
    }

    /// Runs one egui frame so the store sees a new frame number.
    fn next_frame(ctx: &egui::Context) {
        let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
    }

    fn write_png(dir: &Path, name: &str, size: [u32; 2]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        image::RgbImage::from_pixel(size[0], size[1], image::Rgb([90, 120, 150]))
            .save(&path)
            .unwrap();
        path
    }

    fn rgb_files(a: &Path, b: &Path) -> SceneFiles {
        SceneFiles::Rgb([a.to_path_buf(), b.to_path_buf()])
    }

    /// Asks for the scene's textures every frame until they are not `Loading` any more.
    fn settle(
        store: &mut PackageStore,
        ctx: &egui::Context,
        files: &SceneFiles,
    ) -> Result<Vec<Arc<GpuTexture>>, String> {
        for _ in 0..500 {
            next_frame(ctx);
            match store.gpu_textures(ctx, files)? {
                Textures::Ready(textures) => return Ok(textures),
                Textures::Loading => std::thread::sleep(Duration::from_millis(10)),
            }
        }
        panic!("textures did not load");
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("press_store_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn textures_decode_off_the_calling_thread_and_are_reused() {
        let Some(mut store) = gpu_store() else { return };
        let dir = temp_dir("async");
        let (a, b) = (
            write_png(&dir, "a.png", [64, 32]),
            write_png(&dir, "b.png", [64, 32]),
        );
        let files = rgb_files(&a, &b);
        let ctx = egui::Context::default();
        next_frame(&ctx);
        assert!(
            matches!(store.gpu_textures(&ctx, &files), Ok(Textures::Loading)),
            "the first request must not block on the decode"
        );
        let first = settle(&mut store, &ctx, &files).unwrap();
        assert_eq!(first.len(), 2);
        assert!(first[0].bytes() > 64 * 32 * 4, "mip levels are counted");
        let again = settle(&mut store, &ctx, &files).unwrap();
        assert!(Arc::ptr_eq(&first[0], &again[0]), "uploaded once per file");
        assert!(store.pending.is_empty());
    }

    #[test]
    fn a_texture_that_cannot_be_decoded_is_an_error_that_is_not_retried() {
        let Some(mut store) = gpu_store() else { return };
        let dir = temp_dir("corrupt");
        let good = write_png(&dir, "good.png", [16, 16]);
        let bad = write_png(&dir, "bad.png", [16, 16]);
        let bytes = std::fs::read(&bad).unwrap();
        std::fs::write(&bad, &bytes[..bytes.len() / 2]).unwrap();
        let files = rgb_files(&good, &bad);
        let ctx = egui::Context::default();
        let error = settle(&mut store, &ctx, &files)
            .err()
            .expect("truncated PNG");
        assert!(error.contains("bad.png"), "{error}");
        // Cached failures must not queue another decode.
        next_frame(&ctx);
        assert_eq!(
            store.gpu_textures(&ctx, &files).err().as_deref(),
            Some(error.as_str())
        );
        assert!(store.pending.is_empty());

        // A repaired file (newer modification time) is read again.
        std::fs::write(&bad, &bytes).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&bad)
            .unwrap()
            .set_modified(SystemTime::now() + Duration::from_secs(5))
            .unwrap();
        store.recheck = Duration::ZERO;
        assert_eq!(settle(&mut store, &ctx, &files).unwrap().len(), 2);
    }

    #[test]
    fn a_vanished_file_keeps_serving_its_texture() {
        let Some(mut store) = gpu_store() else { return };
        let dir = temp_dir("vanish");
        let (a, b) = (
            write_png(&dir, "a.png", [16, 16]),
            write_png(&dir, "b.png", [16, 16]),
        );
        let files = rgb_files(&a, &b);
        let ctx = egui::Context::default();
        let first = settle(&mut store, &ctx, &files).unwrap();
        std::fs::remove_file(&a).unwrap();
        store.recheck = Duration::ZERO;
        next_frame(&ctx);
        let Ok(Textures::Ready(again)) = store.gpu_textures(&ctx, &files) else {
            panic!("the cached texture should be served");
        };
        assert!(Arc::ptr_eq(&first[0], &again[0]));
    }

    #[test]
    fn a_texture_larger_than_the_device_allows_is_an_error_not_a_crash() {
        let Some(mut store) = gpu_store() else { return };
        let limit = store.gpu().unwrap().max_texture_side();
        let dir = temp_dir("huge");
        let wide = write_png(&dir, "wide.png", [limit + 1, 2]);
        let ctx = egui::Context::default();
        let error = settle(&mut store, &ctx, &rgb_files(&wide, &wide))
            .err()
            .unwrap();
        assert!(error.contains("larger than this GPU supports"), "{error}");
    }

    #[test]
    fn the_cache_is_bounded_in_bytes_but_keeps_what_the_frame_uses() {
        let Some(mut store) = gpu_store() else { return };
        let dir = temp_dir("budget");
        let paths: Vec<PathBuf> = (0..4)
            .map(|i| write_png(&dir, &format!("t{i}.png"), [32, 32]))
            .collect();
        let ctx = egui::Context::default();
        let first = settle(&mut store, &ctx, &rgb_files(&paths[0], &paths[1])).unwrap();
        store.budget_bytes = first[0].bytes(); // room for one texture only
                                               // A scene of two textures still gets both: they belong to the current frame.
        let second = settle(&mut store, &ctx, &rgb_files(&paths[2], &paths[3])).unwrap();
        assert_eq!(second.len(), 2);
        let cached: Vec<_> = store
            .gpu_textures
            .iter()
            .map(|e| e.key.path.clone())
            .collect();
        assert!(
            !cached.contains(&paths[0]) && !cached.contains(&paths[1]),
            "{cached:?}"
        );
        assert!(cached.contains(&paths[2]) && cached.contains(&paths[3]));
    }

    #[test]
    fn prefetch_decodes_ahead_without_blocking() {
        let Some(mut store) = gpu_store() else { return };
        let dir = temp_dir("prefetch");
        let (a, b, next) = (
            write_png(&dir, "a.png", [16, 16]),
            write_png(&dir, "b.png", [16, 16]),
            write_png(&dir, "next.png", [16, 16]),
        );
        let ctx = egui::Context::default();
        settle(&mut store, &ctx, &rgb_files(&a, &b)).unwrap();
        store.prefetch(&ctx, std::slice::from_ref(&next), TexKind::Srgb8);
        // Prefetch should make the next pair ready without another decode.
        let files = rgb_files(&next, &next);
        for _ in 0..500 {
            next_frame(&ctx);
            let _ = store.gpu_textures(&ctx, &rgb_files(&a, &b));
            if store.pending.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        next_frame(&ctx);
        assert!(matches!(
            store.gpu_textures(&ctx, &files),
            Ok(Textures::Ready(_))
        ));
    }
}
