//! Bilinear tracked-grid rectification and edge blending for the canvas preview.

use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Weak};

use fabric_data::tracking::{FrameId, TrackId, TrackingFrame};
use image::RgbaImage;

use crate::tracked::TrackedRunState;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Grid {
    cols: usize,
    size: [u32; 2],
    raster: [u32; 2],
    points: Vec<[f32; 2]>,
}

impl Grid {
    pub fn for_frame(run: &TrackedRunState, frame: FrameId) -> Option<Self> {
        let sequence = run.sequence.as_ref()?;
        let topology = sequence
            .source_reference()
            .or_else(|| sequence.first_frame())?;
        let first = ordered_points(topology)?;
        let cols = first
            .iter()
            .take_while(|p| (p[1] - first[0][1]).abs() < 1e-4)
            .count();
        if cols < 2 || first.len() % cols != 0 || first.len() / cols < 2 {
            return None;
        }
        let reference = run
            .state
            .reference_frame
            .and_then(|id| sequence.get_frame(&id))
            .unwrap_or(topology);
        let reference = ordered_points(reference)?;
        let points = ordered_points(sequence.get_frame(&frame)?)?;
        if points.len() != first.len() || reference.len() != first.len() {
            return None;
        }
        let rows = points.len() / cols;
        let distance = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        let width = reference
            .chunks_exact(cols)
            .map(|row| row.windows(2).map(|p| distance(p[0], p[1])).sum::<f32>())
            .sum::<f32>()
            / rows as f32;
        let height = (0..cols)
            .map(|col| {
                (0..rows - 1)
                    .map(|row| {
                        let i = row * cols + col;
                        distance(reference[i], reference[i + cols])
                    })
                    .sum::<f32>()
            })
            .sum::<f32>()
            / cols as f32;
        if !(width.is_finite() && height.is_finite() && width >= 1.0 && height >= 1.0) {
            return None;
        }
        // Keep comparison previews responsive while retaining the reference aspect ratio.
        let scale = (2048.0 / width.max(height)).min(1.0);
        let size = [width, height].map(|v| v.round().max(2.0) as u32);
        let raster = [width, height].map(|v| (v * scale).round().max(2.0) as u32);
        Some(Self {
            cols,
            size,
            raster,
            points,
        })
    }

    pub fn rectified_point(&self, id: &TrackId) -> Option<[f64; 2]> {
        let i = grid_index(id)?;
        if i >= self.points.len() {
            return None;
        }
        Some([
            (i % self.cols) as f64 * f64::from(self.size[0] - 1) / (self.cols - 1) as f64,
            (i / self.cols) as f64 * f64::from(self.size[1] - 1)
                / (self.points.len() / self.cols - 1) as f64,
        ])
    }

    fn source(&self, x: u32, y: u32) -> [f32; 2] {
        let rows = self.points.len() / self.cols;
        let gx = x as f32 * (self.cols - 1) as f32 / (self.raster[0] - 1) as f32;
        let gy = y as f32 * (rows - 1) as f32 / (self.raster[1] - 1) as f32;
        let col = (gx as usize).min(self.cols - 2);
        let row = (gy as usize).min(rows - 2);
        let i = row * self.cols + col;
        let u = gx - col as f32;
        let v = gy - row as f32;
        [0, 1].map(|a| {
            let top = mix(self.points[i][a], self.points[i + 1][a], u);
            let bottom = mix(
                self.points[i + self.cols][a],
                self.points[i + self.cols + 1][a],
                u,
            );
            mix(top, bottom, v)
        })
    }
}

fn grid_index(id: &TrackId) -> Option<usize> {
    id.as_str().strip_prefix("grid_")?.parse().ok()
}

fn ordered_points(frame: &TrackingFrame) -> Option<Vec<[f32; 2]>> {
    let mut ordered = frame
        .points()
        .iter()
        .map(|(id, p)| {
            let index = grid_index(id)?;
            let xy = [p.x as f32, p.y as f32];
            xy.iter().all(|v| v.is_finite()).then_some((index, xy))
        })
        .collect::<Option<Vec<_>>>()?;
    ordered.sort_unstable_by_key(|p| p.0);
    if ordered.len() < 4 || ordered.iter().enumerate().any(|(i, (id, _))| i != *id) {
        return None;
    }
    Some(ordered.into_iter().map(|(_, xy)| xy).collect())
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn sample(source: &RgbaImage, [x, y]: [f32; 2]) -> image::Rgba<u8> {
    let (w, h) = source.dimensions();
    if x < 0.0 || y < 0.0 || x > (w - 1) as f32 || y > (h - 1) as f32 {
        return image::Rgba([0; 4]);
    }
    let (i, j) = (x as u32, y as u32);
    let (u, v) = (x - i as f32, y - j as f32);
    let (right, bottom) = ((i + 1).min(w - 1), (j + 1).min(h - 1));
    let [tl, tr, bl, br] =
        [(i, j), (right, j), (i, bottom), (right, bottom)].map(|(x, y)| source.get_pixel(x, y).0);
    image::Rgba([0, 1, 2, 3].map(|c| {
        mix(
            mix(tl[c] as f32, tr[c] as f32, u),
            mix(bl[c] as f32, br[c] as f32, u),
            v,
        )
        .round() as u8
    }))
}

fn crossfade_edges(mut image: RgbaImage) -> RgbaImage {
    let (w, h) = image.dimensions();
    for (length, lines, horizontal) in [(w, h, true), (h, w, false)] {
        let band = (length / 10).clamp(1, 32).min(length / 2);
        for line in 0..lines {
            for d in 0..band {
                let t = d as f32 / band as f32;
                let weight = 0.5 * (1.0 - t * t * (3.0 - 2.0 * t));
                let coord = |x| if horizontal { (x, line) } else { (line, x) };
                let ((ax, ay), (bx, by)) = (coord(d), coord(length - 1 - d));
                let (a, b) = (image.get_pixel(ax, ay).0, image.get_pixel(bx, by).0);
                let blend = |from: [u8; 4], to: [u8; 4]| {
                    image::Rgba(
                        [0, 1, 2, 3]
                            .map(|c| mix(from[c] as f32, to[c] as f32, weight).round() as u8),
                    )
                };
                image.put_pixel(ax, ay, blend(a, b));
                image.put_pixel(bx, by, blend(b, a));
            }
        }
    }
    image
}

#[derive(Debug, Clone, PartialEq)]
struct Key {
    grid: Option<Arc<Grid>>,
    blend: bool,
}

struct Job {
    source: Arc<RgbaImage>,
    key: Key,
    ctx: egui::Context,
}

struct Entry {
    source: Weak<RgbaImage>,
    key: Key,
    texture: Option<egui::TextureHandle>,
    size: (u32, u32),
}

struct Finished {
    source: Weak<RgbaImage>,
    key: Key,
    image: RgbaImage,
    size: (u32, u32),
}

#[derive(Default)]
pub(super) struct PreviewCache {
    entries: VecDeque<Entry>,
    worker: Option<(mpsc::SyncSender<Job>, mpsc::Receiver<Finished>)>,
    active_count: usize,
    options: Option<egui::TextureOptions>,
}

impl std::fmt::Debug for PreviewCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreviewCache")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl PreviewCache {
    #[cfg(test)]
    pub fn ready_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.texture.is_some())
            .count()
    }

    pub fn set_active_count(&mut self, count: usize) {
        self.active_count = count;
    }

    fn capacity(&self) -> usize {
        self.active_count.saturating_mul(2).max(16)
    }

    pub fn poll(&mut self, ctx: &egui::Context, options: egui::TextureOptions) {
        if self.options.replace(options) != Some(options) {
            self.entries.retain(|e| e.texture.is_none());
        }
        if let Some((_, results)) = &self.worker {
            for result in results.try_iter() {
                if let Some(entry) = self
                    .entries
                    .iter_mut()
                    .find(|e| e.source.ptr_eq(&result.source) && e.key == result.key)
                {
                    let (w, h) = result.image.dimensions();
                    let color = egui::ColorImage::from_rgba_unmultiplied(
                        [w as usize, h as usize],
                        result.image.as_raw(),
                    );
                    entry.texture =
                        Some(ctx.load_texture("image-comparison-preview", color, options));
                    entry.size = result.size;
                }
            }
        }
        self.entries.retain(|e| e.source.strong_count() != 0);
        let capacity = self.capacity();
        while self.entries.len() > capacity {
            self.entries.pop_front();
        }
    }

    pub fn request(
        &mut self,
        source: Arc<RgbaImage>,
        grid: Option<Arc<Grid>>,
        blend: bool,
        ctx: &egui::Context,
    ) -> Option<(egui::TextureId, (u32, u32))> {
        let key = Key { grid, blend };
        let weak = Arc::downgrade(&source);
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.source.ptr_eq(&weak) && e.key == key)
        {
            let entry = self.entries.remove(i).expect("entry was found");
            let result = entry.texture.as_ref().map(|t| (t.id(), entry.size));
            self.entries.push_back(entry);
            return result;
        }
        let (jobs, _) = self.worker.get_or_insert_with(|| {
            let (send, jobs) = mpsc::sync_channel::<Job>(2);
            let (results, receive) = mpsc::channel();
            std::thread::spawn(move || {
                while let Ok(Job { source, key, ctx }) = jobs.recv() {
                    let size = key
                        .grid
                        .as_ref()
                        .map_or_else(|| source.dimensions(), |grid| (grid.size[0], grid.size[1]));
                    let image = if let Some(grid) = &key.grid {
                        RgbaImage::from_fn(grid.raster[0], grid.raster[1], |x, y| {
                            sample(&source, grid.source(x, y))
                        })
                    } else if source.width().max(source.height()) > 2048 {
                        let scale = 2048.0 / source.width().max(source.height()) as f32;
                        let [w, h] = [source.width(), source.height()]
                            .map(|v| (v as f32 * scale).round().max(1.0) as u32);
                        image::imageops::thumbnail(&*source, w, h)
                    } else {
                        (*source).clone()
                    };
                    let image = if key.blend {
                        crossfade_edges(image)
                    } else {
                        image
                    };
                    if results
                        .send(Finished {
                            source: Arc::downgrade(&source),
                            key,
                            image,
                            size,
                        })
                        .is_err()
                    {
                        break;
                    }
                    ctx.request_repaint();
                }
            });
            (send, receive)
        });
        if jobs
            .try_send(Job {
                source,
                key: key.clone(),
                ctx: ctx.clone(),
            })
            .is_ok()
        {
            // Pending work is bounded by the queue. Evict completed previews first.
            if self.entries.len() >= self.capacity() {
                if let Some(i) = self.entries.iter().position(|e| e.texture.is_some()) {
                    self.entries.remove(i);
                }
            }
            self.entries.push_back(Entry {
                source: weak,
                key,
                texture: None,
                size: (0, 0),
            });
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_samples_all_four_grid_vertices_and_bilinear_center() {
        let grid = Grid {
            cols: 2,
            size: [3, 3],
            raster: [3, 3],
            points: vec![[0.0, 1.0], [4.0, 0.0], [1.0, 5.0], [5.0, 6.0]],
        };
        assert_eq!(grid.source(0, 0), [0.0, 1.0]);
        assert_eq!(grid.source(2, 2), [5.0, 6.0]);
        assert_eq!(grid.source(1, 1), [2.5, 3.0]);
    }

    #[test]
    fn blending_matches_both_wrap_edges_and_preserves_the_center() {
        let source = RgbaImage::from_fn(30, 20, |x, y| image::Rgba([x as u8, y as u8, 0, 255]));
        let blended = crossfade_edges(source.clone());
        for x in 0..30 {
            assert_eq!(blended.get_pixel(x, 0), blended.get_pixel(x, 19));
        }
        for y in 0..20 {
            assert_eq!(blended.get_pixel(0, y), blended.get_pixel(29, y));
        }
        assert_eq!(blended.get_pixel(15, 10), source.get_pixel(15, 10));
    }
}
