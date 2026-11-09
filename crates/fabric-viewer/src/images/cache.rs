//! Background image decoding with a byte-bounded LRU cache.
//! Poll once per UI frame. Requests after polling schedule their own repaint.
//!
//! Canonical paths deduplicate entries; decoded RGBA is retained for derived views.
//! The budget covers CPU buffers only (GPU textures add roughly another copy).
//! Zero means unbounded. Errors have a separate count limit.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use crate::selection::canonical_key;

/// Default cache budget: 512 MB of decoded image memory.
pub const DEFAULT_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Maximum number of cached *error* entries before the oldest is evicted.
pub const DEFAULT_MAX_ERRORS: usize = 2048;

/// Lifecycle state of one cached image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageStatus {
    /// Requested; the worker may still be reading/decoding it.
    Pending,
    /// Decoded (and, on the UI thread, textured).
    Ready,
    /// Reading or decoding failed; the error is cached so a broken file is
    /// not re-attempted every frame. [`ImageCache::invalidate`] resets it.
    Error(String),
}

struct Entry {
    status: ImageStatus,
    /// Estimated resident bytes of `decoded` (width * height * 4).
    bytes: usize,
    decoded: Option<Arc<image::RgbaImage>>,
    texture: Option<egui::TextureHandle>,
    texture_options: egui::TextureOptions,
    last_access: u64,
}

impl Entry {
    fn in_memory(&self) -> usize {
        self.bytes
    }
}

/// Bounded, lazily-populated image cache. Created on the UI thread; the
/// worker thread owns only the channel pair.
pub struct ImageCache {
    entries: HashMap<PathBuf, Entry>,
    /// Total bytes currently retained (decoded buffers only).
    total_bytes: usize,
    max_bytes: usize,
    /// Number of cached Error entries (bounded by `max_errors`).
    error_count: usize,
    max_errors: usize,
    clock: u64,
    /// In-flight path generations. Discard results from invalidated requests.
    pending: HashMap<PathBuf, u64>,
    /// Requests issued during the current frame (drains each poll).
    queued_this_frame: Vec<(PathBuf, u64)>,
    /// The texture filter requested for freshly created textures.
    wanted_options: egui::TextureOptions,
    /// Memoized canonical paths, filled by read-only lookups.
    keys: RefCell<HashMap<PathBuf, PathBuf>>,
    request_tx: Option<Sender<(PathBuf, u64)>>,
    result_rx: Receiver<Decoded>,
    worker: Option<std::thread::JoinHandle<()>>,
}

/// A finished decode: the key, the request generation and the pixels (or why
/// they could not be read).
type Decoded = (PathBuf, u64, Result<image::RgbaImage, String>);

fn spawn_worker() -> (
    Sender<(PathBuf, u64)>,
    Receiver<Decoded>,
    std::thread::JoinHandle<()>,
) {
    let (request_tx, request_rx) = channel::<(PathBuf, u64)>();
    let (result_tx, result_rx) = channel::<Decoded>();
    let handle = std::thread::Builder::new()
        .name("fabsim-image-loader".to_owned())
        .spawn(move || {
            while let Ok((path, generation)) = request_rx.recv() {
                let result = decode_rgba(&path);
                if result_tx.send((path, generation, result)).is_err() {
                    break; // cache dropped
                }
            }
        })
        .expect("failed to spawn image loader thread");
    (request_tx, result_rx, handle)
}

fn decode_rgba(path: &Path) -> Result<image::RgbaImage, String> {
    let reader = image::ImageReader::open(path).map_err(|e| e.to_string())?;
    let img = reader.decode().map_err(|e| e.to_string())?;
    Ok(img.to_rgba8())
}

impl ImageCache {
    pub fn new() -> Self {
        Self::with_budget(DEFAULT_MAX_BYTES)
    }

    /// `max_bytes == 0` means "unbounded" (no decoded-buffer eviction).
    pub fn with_budget(max_bytes: usize) -> Self {
        Self::with_limits(max_bytes, DEFAULT_MAX_ERRORS)
    }

    pub fn with_limits(max_bytes: usize, max_errors: usize) -> Self {
        let (request_tx, result_rx, worker) = spawn_worker();
        ImageCache {
            entries: HashMap::new(),
            total_bytes: 0,
            max_bytes,
            error_count: 0,
            max_errors,
            clock: 0,
            pending: HashMap::new(),
            queued_this_frame: Vec::new(),
            wanted_options: egui::TextureOptions::LINEAR,
            keys: RefCell::new(HashMap::new()),
            request_tx: Some(request_tx),
            result_rx,
            worker: Some(worker),
        }
    }

    /// Preferred sampling filter for freshly created textures.
    pub fn set_texture_options(&mut self, options: egui::TextureOptions) {
        if self.wanted_options != options {
            self.wanted_options = options;
            // Recreate textures from decoded pixels when the filter changes.
            for entry in self.entries.values_mut() {
                if entry.status == ImageStatus::Ready && entry.decoded.is_some() {
                    entry.texture_options = options;
                    entry.texture = None;
                }
            }
        }
    }

    /// The cache key of `path`: its canonical form, memoised per spelling for
    /// paths that exist.
    fn key_of(&self, path: &Path) -> PathBuf {
        if let Some(key) = self.keys.borrow().get(path) {
            return key.clone();
        }
        let key = canonical_key(path);
        // Do not memoize missing paths: their canonical form may change when created.
        if path.exists() {
            self.keys.borrow_mut().insert(path.to_owned(), key.clone());
        }
        key
    }

    /// Ensure `path` is being decoded or already cached; returns the current
    /// status (the UI uses [`ImageCache::request_and_repaint`]).
    #[cfg(test)]
    pub fn request(&mut self, path: impl Into<PathBuf>) -> ImageStatus {
        let key = self.key_of(&path.into());
        self.request_key(key).0
    }

    /// Request decode and repaint if newly queued, including requests made after this frame's poll.
    pub fn request_and_repaint(
        &mut self,
        ctx: &egui::Context,
        path: impl Into<PathBuf>,
    ) -> ImageStatus {
        let key = self.key_of(&path.into());
        let (status, newly_queued) = self.request_key(key);
        if newly_queued {
            ctx.request_repaint();
        }
        status
    }

    fn request_key(&mut self, key: PathBuf) -> (ImageStatus, bool) {
        self.clock += 1;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_access = self.clock;
            return (entry.status.clone(), false);
        }
        // Tag requests so invalidation can reject an older in-flight decode.
        let mut newly_queued = false;
        if !self.pending.contains_key(&key) {
            self.pending.insert(key.clone(), self.clock);
            self.queued_this_frame.push((key.clone(), self.clock));
            newly_queued = true;
        }
        let entry = Entry {
            status: ImageStatus::Pending,
            bytes: 0,
            decoded: None,
            texture: None,
            texture_options: self.wanted_options,
            last_access: self.clock,
        };
        self.entries.insert(key, entry);
        (ImageStatus::Pending, newly_queued)
    }

    /// Forget the entry so the next request retries decoding.
    pub fn invalidate(&mut self, path: &Path) {
        let key = self.key_of(path);
        self.keys.borrow_mut().remove(path);
        if let Some(entry) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(entry.in_memory());
            if matches!(entry.status, ImageStatus::Error(_)) {
                self.error_count = self.error_count.saturating_sub(1);
            }
        }
        // Discard in-flight results from the old generation.
        self.pending.remove(&key);
    }

    /// Current lifecycle status of `path` (without issuing a request).
    pub fn status(&self, path: &Path) -> Option<ImageStatus> {
        self.entries
            .get(&self.key_of(path))
            .map(|e| e.status.clone())
    }

    /// GPU texture for `path` if it is decoded and has been polled.
    pub fn texture(&self, path: &Path) -> Option<&egui::TextureHandle> {
        self.entries
            .get(&self.key_of(path))
            .and_then(|e| e.texture.as_ref())
    }

    /// Shared decoded RGBA for derived views; cloning the `Arc` does not copy pixels.
    pub fn decoded(&self, path: &Path) -> Option<Arc<image::RgbaImage>> {
        self.entries
            .get(&self.key_of(path))
            .and_then(|e| e.decoded.clone())
    }

    /// `(width, height)` of a decoded image, if available.
    pub fn size(&self, path: &Path) -> Option<(u32, u32)> {
        self.decoded(path).map(|img| (img.width(), img.height()))
    }

    /// Any entry still being decoded (used to keep repaints alive).
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
            || self
                .entries
                .values()
                .any(|e| e.status == ImageStatus::Pending)
    }

    /// Send queued requests to the worker. Called once per frame from the UI
    /// thread (usually via [`ImageCache::poll`]).
    pub fn flush_requests(&mut self) {
        let Some(tx) = &self.request_tx else {
            self.fail_all_pending("image loader unavailable");
            return;
        };
        let queued = std::mem::take(&mut self.queued_this_frame);
        for (key, generation) in queued {
            if tx.send((key, generation)).is_err() {
                // A dead worker cannot complete pending requests.
                self.fail_all_pending("image loader stopped");
                break;
            }
        }
    }

    /// The worker cannot accept requests anymore: convert every Pending entry
    /// to an Error so there is no permanent repaint loop.
    fn fail_all_pending(&mut self, reason: &str) {
        self.queued_this_frame.clear();
        for (key, _) in self.pending.drain() {
            if let Some(entry) = self.entries.get_mut(&key) {
                if entry.status == ImageStatus::Pending {
                    entry.status = ImageStatus::Error(reason.to_owned());
                    self.error_count += 1;
                }
            }
        }
        self.evict_errors_over_budget();
    }

    /// Collect decodes whose generation still matches `pending`. Does not touch the GPU.
    pub fn finish_pending(&mut self) {
        while let Ok((path, generation, result)) = self.result_rx.try_recv() {
            let is_current = self.pending.get(&path) == Some(&generation);
            if is_current {
                self.pending.remove(&path);
            }
            let Some(entry) = self.entries.get_mut(&path) else {
                continue; // evicted or invalidated without re-request
            };
            if !is_current {
                continue; // stale result for a superseded request
            }
            match result {
                Ok(img) => {
                    let size = (img.width() as usize)
                        .saturating_mul(img.height() as usize)
                        .saturating_mul(4);
                    self.total_bytes += size;
                    entry.bytes = size;
                    entry.decoded = Some(Arc::new(img));
                    entry.status = ImageStatus::Ready;
                    entry.texture_options = self.wanted_options;
                    entry.texture = None; // created by sync_textures on UI
                    self.evict_if_over_budget();
                }
                Err(err) => {
                    entry.bytes = 0;
                    entry.decoded = None;
                    entry.status = ImageStatus::Error(err);
                    self.error_count += 1;
                    self.evict_errors_over_budget();
                }
            }
        }
    }

    /// Create/recreate GPU textures for every ready entry, on the UI thread.
    pub fn sync_textures(&mut self, ctx: &egui::Context) {
        for entry in self.entries.values_mut() {
            if entry.status != ImageStatus::Ready {
                continue;
            }
            let Some(decoded) = &entry.decoded else {
                continue;
            };
            if entry.texture.is_none() || entry.texture_options != self.wanted_options {
                let size = [decoded.width() as usize, decoded.height() as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, decoded.as_raw());
                entry.texture = Some(ctx.load_texture("fabsim-image", color, self.wanted_options));
                entry.texture_options = self.wanted_options;
            }
        }
    }

    /// UI-thread poll: flush queued requests, absorb finished decodes,
    /// create textures, and keep repainting while anything is pending.
    pub fn poll(&mut self, ctx: &egui::Context) {
        self.flush_requests();
        self.finish_pending();
        self.sync_textures(ctx);
        if self.has_pending() {
            ctx.request_repaint();
        }
    }

    fn evict_if_over_budget(&mut self) {
        if self.max_bytes == 0 {
            return; // 0 = unbounded
        }
        while self.total_bytes > self.max_bytes {
            // Evict the least-used ready entry; a future request will decode again.
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| e.status == ImageStatus::Ready && e.bytes > 0)
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else {
                break; // nothing evictable
            };
            let bytes = self.entries.remove(&victim).map_or(0, |e| e.in_memory());
            self.total_bytes = self.total_bytes.saturating_sub(bytes);
        }
    }

    /// Keep cached error entries bounded so a flood of missing files cannot
    /// grow the map forever.
    fn evict_errors_over_budget(&mut self) {
        while self.error_count > self.max_errors {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| matches!(e.status, ImageStatus::Error(_)))
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else {
                break;
            };
            self.entries.remove(&victim);
            self.error_count = self.error_count.saturating_sub(1);
        }
    }
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ImageCache {
    fn drop(&mut self) {
        self.request_tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_png(path: &Path, width: u32, height: u32) {
        use image::ImageEncoder;
        let img = image::RgbaImage::from_pixel(width, height, image::Rgba([120, 60, 240, 255]));
        let mut file = std::fs::File::create(path).unwrap();
        let encoder = image::codecs::png::PngEncoder::new(&mut file);
        encoder
            .write_image(img.as_raw(), width, height, image::ExtendedColorType::Rgba8)
            .unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fabsim_cache_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Drain the worker until nothing is pending (bounded busy-wait).
    fn drain(cache: &mut ImageCache) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while cache.has_pending() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(2));
            cache.finish_pending();
        }
        assert!(!cache.has_pending(), "decode did not finish within timeout");
    }

    #[test]
    fn cache_key_dedupes_equivalent_paths() {
        let dir = temp_dir("key");
        write_png(&dir.join("a.png"), 4, 4);
        let canonical = canonical_key(&dir.join("a.png"));
        let alias = canonical_key(&dir.join(".").join("a.png"));
        assert_eq!(canonical, alias);

        let mut cache = ImageCache::new();
        let s1 = cache.request(dir.join("a.png"));
        let s2 = cache.request(dir.join(".").join("a.png"));
        assert_eq!(s1, ImageStatus::Pending);
        assert_eq!(s2, ImageStatus::Pending);
        assert_eq!(cache.entries.len(), 1, "aliases must share one entry");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_pending_to_ready() {
        let dir = temp_dir("decode");
        write_png(&dir.join("img.png"), 8, 8);
        let mut cache = ImageCache::new();
        assert_eq!(cache.request(dir.join("img.png")), ImageStatus::Pending);
        cache.flush_requests();
        drain(&mut cache);
        let status = cache.status(&dir.join("img.png"));
        assert_eq!(status, Some(ImageStatus::Ready));
        assert_eq!(cache.size(&dir.join("img.png")), Some((8, 8)));
        assert!(cache.decoded(&dir.join("img.png")).is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_caches_error_and_is_not_retried() {
        let dir = temp_dir("err");
        let mut cache = ImageCache::new();
        let missing = dir.join("nope.png");
        assert_eq!(cache.request(&missing), ImageStatus::Pending);
        cache.flush_requests();
        drain(&mut cache);
        let status = cache.status(&missing);
        assert!(matches!(status, Some(ImageStatus::Error(_))));

        // Re-requesting a failed file returns the cached error, no retry.
        assert!(matches!(cache.request(&missing), ImageStatus::Error(_)));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn invalidate_retries_a_failed_file() {
        let dir = temp_dir("retry");
        let missing = dir.join("later.png");
        let mut cache = ImageCache::new();
        cache.request(&missing);
        cache.flush_requests();
        drain(&mut cache);
        assert!(matches!(
            cache.status(&missing),
            Some(ImageStatus::Error(_))
        ));

        // The file appears; invalidate + re-request must decode it.
        write_png(&missing, 4, 4);
        cache.invalidate(&missing);
        assert!(cache.status(&missing).is_none());
        assert_eq!(cache.request(&missing), ImageStatus::Pending);
        cache.flush_requests();
        drain(&mut cache);
        assert_eq!(cache.status(&missing), Some(ImageStatus::Ready));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn invalidate_rerequest_discards_stale_decode() {
        // An invalidated decode may finish after its replacement. Ignore it without double-counting bytes.
        let dir = temp_dir("tomb");
        write_png(&dir.join("img.png"), 16, 16);
        let mut cache = ImageCache::new();
        cache.request(dir.join("img.png"));
        cache.flush_requests();
        cache.invalidate(&dir.join("img.png"));
        cache.request(dir.join("img.png"));
        cache.flush_requests();
        drain(&mut cache);
        assert_eq!(
            cache.total_bytes,
            16 * 16 * 4,
            "decoded bytes must be counted exactly once"
        );
        assert_eq!(cache.status(&dir.join("img.png")), Some(ImageStatus::Ready));
        assert!(!cache.has_pending());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn byte_budget_evicts_lru_and_rerequest_recovers() {
        let dir = temp_dir("budget");
        write_png(&dir.join("a.png"), 64, 64); // 16 KiB each
        write_png(&dir.join("b.png"), 64, 64);
        write_png(&dir.join("c.png"), 64, 64);

        let mut cache = ImageCache::with_budget(2 * 64 * 64 * 4 + 100);
        for name in ["a.png", "b.png", "c.png"] {
            cache.request(dir.join(name));
        }
        cache.flush_requests();
        drain(&mut cache);
        assert!(cache.total_bytes <= cache.max_bytes);
        // The oldest (a) must have been evicted; the two most recently used
        // (b, c) survive with decoded buffers.
        assert!(cache.status(&dir.join("a.png")).is_none(), "a evicted");
        assert!(matches!(
            cache.status(&dir.join("b.png")),
            Some(ImageStatus::Ready)
        ));
        assert!(matches!(
            cache.status(&dir.join("c.png")),
            Some(ImageStatus::Ready)
        ));

        // Re-requesting the evicted file decodes it again and the oldest
        // survivor (b) is now evicted to stay under budget.
        cache.request(dir.join("a.png"));
        cache.flush_requests();
        drain(&mut cache);
        assert!(cache.total_bytes <= cache.max_bytes);
        assert_eq!(cache.status(&dir.join("a.png")), Some(ImageStatus::Ready));
        assert!(
            cache.status(&dir.join("b.png")).is_none(),
            "b evicted after a re-request"
        );
        assert!(matches!(
            cache.status(&dir.join("c.png")),
            Some(ImageStatus::Ready)
        ));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn error_entries_are_bounded() {
        let dir = temp_dir("errors");
        let mut cache = ImageCache::with_limits(0, 5); // unbounded bytes, 5 errors
        let missing: Vec<PathBuf> = (0..20)
            .map(|i| dir.join(format!("missing_{i}.png")))
            .collect();
        for path in &missing {
            cache.request(path);
        }
        cache.flush_requests();
        drain(&mut cache);
        let error_count = cache
            .entries
            .values()
            .filter(|e| matches!(e.status, ImageStatus::Error(_)))
            .count();
        assert!(
            error_count <= 5,
            "error entries must be bounded, got {error_count}"
        );
        assert_eq!(cache.entries.len(), error_count);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn worker_shuts_down_cleanly() {
        let dir = temp_dir("shutdown");
        write_png(&dir.join("a.png"), 8, 8);
        let mut cache = ImageCache::new();
        cache.request(dir.join("a.png"));
        cache.flush_requests();
        // Dropping the cache must close the worker channel before joining.
        drop(cache);
        std::fs::remove_dir_all(&dir).ok();
    }
}
