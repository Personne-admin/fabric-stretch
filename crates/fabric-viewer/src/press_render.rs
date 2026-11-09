//! CPU reference for `press_gpu`, used by snapshots, tests and the fallback worker.
//! Colour mixes in linear light; relighting mixes albedo and normals before shading.
//! Both paths use trilinear sampling, repeating tiles and clamping untiled atlases.

use crate::lighting;
use crate::press_geometry::ClothQuad;
use egui::{Color32, Pos2};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};

/// How the 8- or 16-bit texel values are to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TexKind {
    /// 8-bit sRGB colour (decoded to linear light before filtering).
    Srgb8,
    /// 16-bit linear data: PS albedo and PS normals.
    Unorm16,
}

#[derive(Debug, Clone)]
pub enum Pixels {
    U8(Vec<[u8; 3]>),
    U16(Vec<[u16; 3]>),
}

#[derive(Debug, Clone)]
pub struct MipLevel {
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
}

/// A decoded texture with its full mip chain (level 0 is the image itself).
#[derive(Debug, Clone)]
pub struct MipChain {
    pub kind: TexKind,
    pub levels: Vec<MipLevel>,
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

impl MipChain {
    /// Decodes an image file. The file's bit depth is converted to the requested kind.
    pub fn load(path: &Path, kind: TexKind) -> Result<Self, String> {
        let image = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (width, height) = (image.width(), image.height());
        let pixels = match kind {
            TexKind::Srgb8 => Pixels::U8(image.to_rgb8().pixels().map(|p| p.0).collect()),
            TexKind::Unorm16 => Pixels::U16(image.to_rgb16().pixels().map(|p| p.0).collect()),
        };
        Ok(Self::from_level0(
            kind,
            MipLevel {
                width,
                height,
                pixels,
            },
        ))
    }

    /// Builds the chain below `level0` by 2x2 box filtering (sRGB colour is averaged in linear
    /// light; normals are averaged as encoded vectors and renormalised when sampled).
    pub fn from_level0(kind: TexKind, level0: MipLevel) -> Self {
        let mut levels = vec![level0];
        while let Some(last) = levels.last() {
            if last.width <= 1 && last.height <= 1 {
                break;
            }
            let next = downsample(last);
            levels.push(next);
        }
        Self { kind, levels }
    }

    pub fn size(&self) -> [u32; 2] {
        [self.levels[0].width, self.levels[0].height]
    }
}

/// Weights for one mip axis. Fractional coverage preserves the last row/column
/// when the source size is odd.
fn axis_taps(src: u32, dst: u32) -> Vec<Vec<(usize, f32)>> {
    let span = src as f64 / dst as f64;
    (0..dst)
        .map(|i| {
            let (from, to) = (i as f64 * span, (i + 1) as f64 * span);
            let first = from.floor() as u32;
            let last = (to.ceil() as u32).min(src);
            (first..last)
                .filter_map(|k| {
                    let overlap = to.min(k as f64 + 1.0) - from.max(k as f64);
                    (overlap > 1.0e-9).then_some((k as usize, (overlap / span) as f32))
                })
                .collect()
        })
        .collect()
}

fn downsample(src: &MipLevel) -> MipLevel {
    let (w, h) = ((src.width / 2).max(1), (src.height / 2).max(1));
    let (x_taps, y_taps) = (axis_taps(src.width, w), axis_taps(src.height, h));
    let gather = |x: usize, y: usize, value: &dyn Fn(usize) -> f32| -> f32 {
        let mut sum = 0.0;
        for &(sy, wy) in &y_taps[y] {
            for &(sx, wx) in &x_taps[x] {
                sum += wy * wx * value(sy * src.width as usize + sx);
            }
        }
        sum
    };
    let pixels = match &src.pixels {
        Pixels::U8(px) => {
            let table = srgb_table();
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    out.push(std::array::from_fn(|c| {
                        let mean = gather(x, y, &|i| table[px[i][c] as usize]);
                        (linear_to_srgb(mean) * 255.0 + 0.5) as u8
                    }));
                }
            }
            Pixels::U8(out)
        }
        Pixels::U16(px) => {
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    out.push(std::array::from_fn(|c| {
                        (gather(x, y, &|i| px[i][c] as f32) + 0.5) as u16
                    }));
                }
            }
            Pixels::U16(out)
        }
    };
    MipLevel {
        width: w,
        height: h,
        pixels,
    }
}

/// Edge handling: tiled packages repeat, untiled packages clamp to the edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrap {
    Clamp,
    Repeat,
}

fn texel(level: &MipLevel, x: i64, y: i64, wrap: Wrap) -> [f32; 3] {
    let (w, h) = (level.width as i64, level.height as i64);
    let (x, y) = match wrap {
        Wrap::Repeat => (x.rem_euclid(w), y.rem_euclid(h)),
        Wrap::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
    };
    let i = (y * w + x) as usize;
    match &level.pixels {
        Pixels::U8(px) => px[i].map(|v| srgb_table()[v as usize]),
        Pixels::U16(px) => px[i].map(|v| v as f32 / 65535.0),
    }
}

fn bilinear(level: &MipLevel, uv: [f32; 2], wrap: Wrap) -> [f32; 3] {
    let fx = uv[0] * level.width as f32 - 0.5;
    let fy = uv[1] * level.height as f32 - 0.5;
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let a = texel(level, x0, y0, wrap);
    let b = texel(level, x0 + 1, y0, wrap);
    let c = texel(level, x0, y0 + 1, wrap);
    let d = texel(level, x0 + 1, y0 + 1, wrap);
    std::array::from_fn(|k| {
        let top = a[k] + (b[k] - a[k]) * tx;
        let bottom = c[k] + (d[k] - c[k]) * tx;
        top + (bottom - top) * ty
    })
}

/// Trilinear lookup at normalised `uv` for the level of detail `lod` (0 = full resolution).
pub fn sample(chain: &MipChain, uv: [f32; 2], lod: f32, wrap: Wrap) -> [f32; 3] {
    let top = (chain.levels.len() - 1) as f32;
    let lod = lod.clamp(0.0, top);
    let lo = lod.floor();
    let frac = lod - lo;
    let a = bilinear(&chain.levels[lo as usize], uv, wrap);
    if frac < 1.0e-4 {
        return a;
    }
    let b = bilinear(
        &chain.levels[(lo as usize + 1).min(chain.levels.len() - 1)],
        uv,
        wrap,
    );
    std::array::from_fn(|k| a[k] + (b[k] - a[k]) * frac)
}

/// Modification time of a file, `None` when it cannot be read.
pub fn file_stamp(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Decoded textures by file, with least-recently-used eviction. A file that changed on disk
/// since it was decoded (a rebuilt package) is decoded again.
pub struct TextureCache {
    capacity: usize,
    entries: VecDeque<CacheEntry>,
}

struct CacheEntry {
    path: PathBuf,
    kind: TexKind,
    stamp: Option<std::time::SystemTime>,
    chain: Arc<MipChain>,
}

impl TextureCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
        }
    }

    pub fn get(&mut self, path: &Path, kind: TexKind) -> Result<Arc<MipChain>, String> {
        let stamp = file_stamp(path);
        if let Some(pos) = self
            .entries
            .iter()
            .position(|e| e.path == path && e.kind == kind)
        {
            let entry = self.entries.remove(pos).expect("position is valid");
            if entry.stamp == stamp {
                let chain = entry.chain.clone();
                self.entries.push_back(entry);
                return Ok(chain);
            }
        }
        let chain = Arc::new(MipChain::load(path, kind)?);
        self.entries.push_back(CacheEntry {
            path: path.to_path_buf(),
            kind,
            stamp,
            chain: chain.clone(),
        });
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
        Ok(chain)
    }
}

/// The files a scene samples, per bracketing state.
#[derive(Debug, Clone, PartialEq)]
pub enum SceneFiles {
    Rgb([PathBuf; 2]),
    Relight {
        albedo: [PathBuf; 2],
        normal: [PathBuf; 2],
    },
}

/// How the sampled texels are turned into colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shading {
    /// Mix the two states in linear light and encode as sRGB.
    Rgb { t: f32 },
    /// Mix albedo and normal, then shade with [`lighting::shade_pixel`].
    Relight {
        t: f32,
        intensity: f32,
        ambient: f32,
        albedo_scale: f32,
        /// Direction towards the light in the DATA frame, per cloth plane (tilt included).
        plane_lights: [[f32; 3]; 2],
    },
}

impl Shading {
    pub fn t(&self) -> f32 {
        match self {
            Self::Rgb { t } | Self::Relight { t, .. } => *t,
        }
    }
}

/// Everything needed to draw the cloth, independent of the renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct ClothScene {
    /// Quads in canvas coordinates (points).
    pub quads: [ClothQuad; 2],
    pub wrap: Wrap,
    pub shading: Shading,
    pub files: SceneFiles,
}

/// Decoded textures of a scene, in the order of [`SceneFiles`].
pub enum SceneTextures {
    Rgb([Arc<MipChain>; 2]),
    Relight {
        albedo: [Arc<MipChain>; 2],
        normal: [Arc<MipChain>; 2],
    },
}

impl SceneTextures {
    pub fn load(files: &SceneFiles, cache: &mut TextureCache) -> Result<Self, String> {
        let pair = |paths: &[PathBuf; 2], kind, cache: &mut TextureCache| {
            Ok::<_, String>([cache.get(&paths[0], kind)?, cache.get(&paths[1], kind)?])
        };
        match files {
            SceneFiles::Rgb(paths) => Ok(Self::Rgb(pair(paths, TexKind::Srgb8, cache)?)),
            SceneFiles::Relight { albedo, normal } => Ok(Self::Relight {
                albedo: pair(albedo, TexKind::Unorm16, cache)?,
                normal: pair(normal, TexKind::Unorm16, cache)?,
            }),
        }
    }

    fn size(&self) -> [u32; 2] {
        match self {
            Self::Rgb(c) => c[0].size(),
            Self::Relight { albedo, .. } => albedo[0].size(),
        }
    }
}

/// Shades one texel of an RGB scene: linear mix of the two states, sRGB encoded.
pub fn shade_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [u8; 3] {
    std::array::from_fn(|k| {
        let linear = a[k] + (b[k] - a[k]) * t;
        (linear_to_srgb(linear) * 255.0 + 0.5) as u8
    })
}

/// Shades one texel of a relit scene (see the module documentation).
#[allow(clippy::too_many_arguments)]
pub fn shade_relight(
    albedo_a: [f32; 3],
    albedo_b: [f32; 3],
    normal_a: [f32; 3],
    normal_b: [f32; 3],
    t: f32,
    light: [f32; 3],
    intensity: f32,
    ambient: f32,
    albedo_scale: f32,
) -> [u8; 3] {
    let mix = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
        std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
    };
    let albedo = mix(albedo_a, albedo_b).map(|v| v * albedo_scale);
    let n = mix(normal_a, normal_b).map(|v| v * 2.0 - 1.0);
    let n = crate::images::normal::normalize3(n);
    lighting::shade_pixel(albedo, n, light, intensity, ambient)
}

/// A rendered RGBA image (straight alpha: 0 outside the cloth, 255 inside).
#[derive(Debug)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height * 4) as usize],
        }
    }

    pub fn fill(&mut self, color: [u8; 4]) {
        for px in self.pixels.as_chunks_mut::<4>().0 {
            *px = color;
        }
    }

    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    /// Alpha-composites `top` (straight alpha) over this image.
    pub fn composite(&mut self, top: &RgbaImage) {
        let (dst_px, _) = self.pixels.as_chunks_mut::<4>();
        let (src_px, _) = top.pixels.as_chunks::<4>();
        for (dst, src) in dst_px.iter_mut().zip(src_px) {
            let alpha = src[3] as u32;
            if alpha == 0 {
                continue;
            }
            for k in 0..3 {
                dst[k] =
                    ((src[k] as u32 * alpha + dst[k] as u32 * (255 - alpha) + 127) / 255) as u8;
            }
            dst[3] = 255;
        }
    }

    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        image::save_buffer(
            path,
            &self.pixels,
            self.width,
            self.height,
            image::ColorType::Rgba8,
        )
        .map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Affine setup of one quad for rasterisation.
struct QuadSetup {
    origin: [f32; 2],
    /// Inverse of the (parameter -> pixel) basis: pixel offset -> `(t, across)`.
    inverse: [[f32; 2]; 2],
    uv0: [f32; 2],
    duv_dt: [f32; 2],
    duv_da: [f32; 2],
    lod: f32,
    bounds: [f32; 4],
    plane: usize,
}

impl QuadSetup {
    /// `None` for a degenerate (zero-area) quad.
    fn new(quad: &ClothQuad, to_px: impl Fn(Pos2) -> [f32; 2], tex: [u32; 2]) -> Option<Self> {
        let p: Vec<[f32; 2]> = quad.pos.iter().map(|&p| to_px(p)).collect();
        let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1]];
        let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1]];
        let det = e1[0] * e2[1] - e1[1] * e2[0];
        if det.abs() < 1.0e-6 {
            return None;
        }
        let inverse = [[e2[1] / det, -e2[0] / det], [-e1[1] / det, e1[0] / det]];
        let duv_dt = [quad.uv[1][0] - quad.uv[0][0], quad.uv[1][1] - quad.uv[0][1]];
        let duv_da = [quad.uv[2][0] - quad.uv[0][0], quad.uv[2][1] - quad.uv[0][1]];
        // Texel footprint of one pixel step in x and in y.
        let step = |axis: usize| {
            let dt = inverse[0][axis];
            let da = inverse[1][axis];
            let du = (duv_dt[0] * dt + duv_da[0] * da) * tex[0] as f32;
            let dv = (duv_dt[1] * dt + duv_da[1] * da) * tex[1] as f32;
            du.hypot(dv)
        };
        let rho = step(0).max(step(1));
        let xs = p.iter().map(|q| q[0]);
        let ys = p.iter().map(|q| q[1]);
        Some(Self {
            origin: p[0],
            inverse,
            uv0: quad.uv[0],
            duv_dt,
            duv_da,
            lod: if rho > 1.0 { rho.log2() } else { 0.0 },
            bounds: [
                xs.clone().fold(f32::MAX, f32::min),
                ys.clone().fold(f32::MAX, f32::min),
                xs.fold(f32::MIN, f32::max),
                ys.fold(f32::MIN, f32::max),
            ],
            plane: quad.plane,
        })
    }

    /// Texture coordinate at pixel centre `(x, y)`, or `None` outside the quad.
    fn uv_at(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        const EDGE: f32 = 1.0e-3;
        let (dx, dy) = (x - self.origin[0], y - self.origin[1]);
        let t = self.inverse[0][0] * dx + self.inverse[0][1] * dy;
        let a = self.inverse[1][0] * dx + self.inverse[1][1] * dy;
        if !(-EDGE..=1.0 + EDGE).contains(&t) || !(-EDGE..=1.0 + EDGE).contains(&a) {
            return None;
        }
        Some([
            self.uv0[0] + self.duv_dt[0] * t + self.duv_da[0] * a,
            self.uv0[1] + self.duv_dt[1] * t + self.duv_da[1] * a,
        ])
    }
}

/// Renders the cloth into a transparent image of `size` pixels. Scene positions are canvas
/// points; `origin` is the canvas corner and `scale` the pixels per point.
pub fn render_cloth(
    scene: &ClothScene,
    textures: &SceneTextures,
    size: [u32; 2],
    origin: Pos2,
    scale: f32,
) -> RgbaImage {
    let mut image = RgbaImage::new(size[0], size[1]);
    let to_px = |p: Pos2| [(p.x - origin.x) * scale, (p.y - origin.y) * scale];
    let tex = textures.size();
    let setups: Vec<QuadSetup> = scene
        .quads
        .iter()
        .filter_map(|q| QuadSetup::new(q, to_px, tex))
        .collect();
    let width = size[0] as usize;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 8);
    let rows_per_chunk = (size[1] as usize).div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in image
            .pixels
            .chunks_mut(rows_per_chunk * width * 4)
            .enumerate()
        {
            let setups = &setups;
            scope.spawn(move || {
                let first_row = chunk_index * rows_per_chunk;
                for (row, line) in chunk.chunks_exact_mut(width * 4).enumerate() {
                    let y = (first_row + row) as f32 + 0.5;
                    for setup in setups.iter() {
                        if y < setup.bounds[1] - 1.0 || y > setup.bounds[3] + 1.0 {
                            continue;
                        }
                        let x0 = (setup.bounds[0] - 1.0).floor().max(0.0) as usize;
                        let x1 = ((setup.bounds[2] + 1.0).ceil().max(0.0) as usize).min(width);
                        for x in x0..x1 {
                            let Some(uv) = setup.uv_at(x as f32 + 0.5, y) else {
                                continue;
                            };
                            let rgb = shade_texel(scene, textures, setup, uv);
                            line[x * 4..x * 4 + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                        }
                    }
                }
            });
        }
    });
    image
}

fn shade_texel(
    scene: &ClothScene,
    textures: &SceneTextures,
    setup: &QuadSetup,
    uv: [f32; 2],
) -> [u8; 3] {
    let (wrap, lod) = (scene.wrap, setup.lod);
    match (&scene.shading, textures) {
        (Shading::Rgb { t }, SceneTextures::Rgb(c)) => shade_rgb(
            sample(&c[0], uv, lod, wrap),
            sample(&c[1], uv, lod, wrap),
            *t,
        ),
        (
            Shading::Relight {
                t,
                intensity,
                ambient,
                albedo_scale,
                plane_lights,
            },
            SceneTextures::Relight { albedo, normal },
        ) => shade_relight(
            sample(&albedo[0], uv, lod, wrap),
            sample(&albedo[1], uv, lod, wrap),
            sample(&normal[0], uv, lod, wrap),
            sample(&normal[1], uv, lod, wrap),
            *t,
            plane_lights[setup.plane],
            *intensity,
            *ambient,
            *albedo_scale,
        ),
        _ => [255, 0, 255],
    }
}

/// A mark drawn over the cloth, in canvas points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlayItem {
    Disc {
        center: Pos2,
        radius: f32,
        color: Color32,
    },
    Line {
        a: Pos2,
        b: Pos2,
        width: f32,
        color: Color32,
    },
}

/// Draws overlay items onto an image (`origin` and `scale` as for [`render_cloth`]).
pub fn draw_overlay(image: &mut RgbaImage, items: &[OverlayItem], origin: Pos2, scale: f32) {
    let to_px = |p: Pos2| ((p.x - origin.x) * scale, (p.y - origin.y) * scale);
    for item in items {
        match *item {
            OverlayItem::Disc {
                center,
                radius,
                color,
            } => {
                let (cx, cy) = to_px(center);
                let r = radius * scale;
                blend_shape(image, [cx - r, cy - r, cx + r, cy + r], color, |x, y| {
                    0.5 - ((x - cx).hypot(y - cy) - r).clamp(-0.5, 0.5)
                });
            }
            OverlayItem::Line { a, b, width, color } => {
                let ((ax, ay), (bx, by)) = (to_px(a), to_px(b));
                let half = width * scale * 0.5;
                let (dx, dy) = (bx - ax, by - ay);
                let len2 = (dx * dx + dy * dy).max(1.0e-6);
                blend_shape(
                    image,
                    [
                        ax.min(bx) - half,
                        ay.min(by) - half,
                        ax.max(bx) + half,
                        ay.max(by) + half,
                    ],
                    color,
                    |x, y| {
                        let t = (((x - ax) * dx + (y - ay) * dy) / len2).clamp(0.0, 1.0);
                        let dist = (x - (ax + dx * t)).hypot(y - (ay + dy * t));
                        (half - dist).clamp(-0.5, 0.5) + 0.5
                    },
                );
            }
        }
    }
}

/// Blends `color` over the pixels in `bounds` with the coverage returned by `coverage(x, y)`.
fn blend_shape(
    image: &mut RgbaImage,
    bounds: [f32; 4],
    color: Color32,
    coverage: impl Fn(f32, f32) -> f32,
) {
    let x0 = (bounds[0].floor().max(0.0)) as u32;
    let y0 = (bounds[1].floor().max(0.0)) as u32;
    let x1 = (bounds[2].ceil().max(0.0) as u32 + 1).min(image.width);
    let y1 = (bounds[3].ceil().max(0.0) as u32 + 1).min(image.height);
    for y in y0..y1 {
        for x in x0..x1 {
            let cover =
                coverage(x as f32 + 0.5, y as f32 + 0.5).clamp(0.0, 1.0) * color.a() as f32 / 255.0;
            if cover <= 0.0 {
                continue;
            }
            let i = ((y * image.width + x) * 4) as usize;
            for (k, channel) in [color.r(), color.g(), color.b()].into_iter().enumerate() {
                let old = image.pixels[i + k] as f32;
                image.pixels[i + k] = (old + (channel as f32 - old) * cover + 0.5) as u8;
            }
            image.pixels[i + 3] = 255;
        }
    }
}

/// One request to the CPU worker.
pub struct CpuJob {
    pub id: u64,
    pub scene: ClothScene,
    /// Canvas corner and pixels per point.
    pub origin: Pos2,
    pub scale: f32,
    pub size: [u32; 2],
}

/// The answer to a [`CpuJob`].
pub struct CpuResult {
    pub id: u64,
    pub image: Result<RgbaImage, String>,
}

/// Renders scenes on a background thread. Only the newest queued job is rendered, so dragging a
/// slider never builds up a backlog; the UI keeps showing the previous image meanwhile.
pub struct CpuWorker {
    jobs: mpsc::Sender<CpuJob>,
    results: mpsc::Receiver<CpuResult>,
}

impl CpuWorker {
    /// Starts the worker. `wake` is called whenever a result is ready (for example to request a
    /// repaint).
    pub fn spawn(wake: impl Fn() + Send + 'static) -> Self {
        let (jobs, job_rx) = mpsc::channel::<CpuJob>();
        let (result_tx, results) = mpsc::channel();
        std::thread::Builder::new()
            .name("press-cpu-render".into())
            .spawn(move || {
                let mut cache = TextureCache::new(12);
                while let Ok(mut job) = job_rx.recv() {
                    while let Ok(newer) = job_rx.try_recv() {
                        job = newer;
                    }
                    // Surface worker panics instead of leaving the canvas waiting forever.
                    let image = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        SceneTextures::load(&job.scene.files, &mut cache).map(|textures| {
                            render_cloth(&job.scene, &textures, job.size, job.origin, job.scale)
                        })
                    }))
                    .unwrap_or_else(|_| Err("The CPU renderer failed unexpectedly".into()));
                    if result_tx.send(CpuResult { id: job.id, image }).is_err() {
                        break;
                    }
                    wake();
                }
            })
            .expect("spawn the CPU render thread");
        Self { jobs, results }
    }

    pub fn submit(&self, job: CpuJob) {
        let _ = self.jobs.send(job);
    }

    /// The newest finished result, if any arrived since the last call.
    pub fn poll(&self) -> Option<CpuResult> {
        let mut latest = None;
        while let Ok(result) = self.results.try_recv() {
            latest = Some(result);
        }
        latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::press_geometry::{cloth_quads, PackageGeometry, Projection, SheetMap, ViewMode};
    use egui::{Rect, Vec2};

    fn solid(kind: TexKind, w: u32, h: u32, value: [u16; 3]) -> MipChain {
        let pixels = match kind {
            TexKind::Srgb8 => Pixels::U8(vec![value.map(|v| v as u8); (w * h) as usize]),
            TexKind::Unorm16 => Pixels::U16(vec![value; (w * h) as usize]),
        };
        MipChain::from_level0(
            kind,
            MipLevel {
                width: w,
                height: h,
                pixels,
            },
        )
    }

    #[test]
    fn mip_chains_halve_down_to_one_texel() {
        let chain = solid(TexKind::Unorm16, 13, 5, [100, 200, 300]);
        let sizes: Vec<_> = chain.levels.iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, [(13, 5), (6, 2), (3, 1), (1, 1)]);
        // Averaging a flat texture keeps its value on every level.
        for level in &chain.levels {
            let Pixels::U16(px) = &level.pixels else {
                panic!()
            };
            assert!(px.iter().all(|p| *p == [100, 200, 300]));
        }
    }

    #[test]
    fn odd_sizes_keep_the_last_texel_in_the_next_level() {
        // Five texels become two: [0, 2.5) and [2.5, 5). The bright last texel must count.
        let level0 = MipLevel {
            width: 5,
            height: 1,
            pixels: Pixels::U16(vec![[0; 3], [0; 3], [0; 3], [0; 3], [65535; 3]]),
        };
        let chain = MipChain::from_level0(TexKind::Unorm16, level0);
        let Pixels::U16(px) = &chain.levels[1].pixels else {
            panic!()
        };
        assert_eq!(px[0], [0; 3]);
        assert!((px[1][0] as i32 - 26214).abs() <= 1, "{}", px[1][0]);
    }

    #[test]
    fn srgb_mips_average_in_linear_light() {
        // A checkerboard of black and white averages to linear 0.5 = sRGB 188, not 128.
        let mut px = Vec::new();
        for y in 0..2 {
            for x in 0..2 {
                px.push(if (x + y) % 2 == 0 { [255; 3] } else { [0; 3] });
            }
        }
        let chain = MipChain::from_level0(
            TexKind::Srgb8,
            MipLevel {
                width: 2,
                height: 2,
                pixels: Pixels::U8(px),
            },
        );
        let Pixels::U8(top) = &chain.levels[1].pixels else {
            panic!()
        };
        assert_eq!(top[0], [188; 3]);
    }

    #[test]
    fn sampling_is_bilinear_at_texel_centres_and_wraps_or_clamps() {
        let level = MipLevel {
            width: 2,
            height: 1,
            pixels: Pixels::U16(vec![[0; 3], [65535; 3]]),
        };
        let chain = MipChain {
            kind: TexKind::Unorm16,
            levels: vec![level],
        };
        let at = |u: f32, wrap| sample(&chain, [u, 0.5], 0.0, wrap)[0];
        assert!(
            (at(0.25, Wrap::Clamp) - 0.0).abs() < 1.0e-6,
            "centre of texel 0"
        );
        assert!((at(0.75, Wrap::Clamp) - 1.0).abs() < 1.0e-6);
        assert!(
            (at(0.5, Wrap::Clamp) - 0.5).abs() < 1.0e-6,
            "halfway between texels"
        );
        // Beyond the edge: clamp holds the border colour, repeat blends with the opposite side.
        assert!((at(1.0, Wrap::Clamp) - 1.0).abs() < 1.0e-6);
        assert!((at(1.0, Wrap::Repeat) - 0.5).abs() < 1.0e-6);
        assert!((at(1.25, Wrap::Repeat) - at(0.25, Wrap::Repeat)).abs() < 1.0e-6);
    }

    #[test]
    fn rgb_shading_mixes_in_linear_light() {
        assert_eq!(shade_rgb([0.0; 3], [1.0; 3], 0.0), [0; 3]);
        assert_eq!(shade_rgb([0.0; 3], [1.0; 3], 1.0), [255; 3]);
        // Half way between black and white in linear light is sRGB 188.
        assert_eq!(shade_rgb([0.0; 3], [1.0; 3], 0.5), [188; 3]);
        let v = srgb_to_linear(100.0 / 255.0);
        assert_eq!(shade_rgb([v; 3], [v; 3], 0.37), [100; 3]);
    }

    #[test]
    fn relight_shading_matches_the_lighting_model() {
        let up = [0.0, 0.0, 1.0];
        let encoded_up = [0.5, 0.5, 1.0];
        let rgb = shade_relight(
            [0.25; 3], [0.25; 3], encoded_up, encoded_up, 0.3, up, 1.0, 0.0, 2.0,
        );
        assert_eq!(rgb, lighting::shade_pixel([0.5; 3], up, up, 1.0, 0.0));
        // The mixed normal is renormalised: two opposite tilts average to flat.
        let left = [0.5 - 0.3, 0.5, 0.5 + 0.5 * 0.954];
        let right = [0.5 + 0.3, 0.5, 0.5 + 0.5 * 0.954];
        let flat = shade_relight([1.0; 3], [1.0; 3], left, right, 0.5, up, 1.0, 0.0, 1.0);
        assert_eq!(flat, [255; 3]);
    }

    #[test]
    fn the_texture_cache_reloads_a_file_that_changed_and_evicts_the_oldest() {
        let dir = std::env::temp_dir().join(format!("press_cache_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.png");
        image::RgbImage::from_pixel(2, 2, image::Rgb([10, 10, 10]))
            .save(&path)
            .unwrap();
        let mut cache = TextureCache::new(2);
        let first = cache.get(&path, TexKind::Srgb8).unwrap();
        assert!(
            Arc::ptr_eq(&first, &cache.get(&path, TexKind::Srgb8).unwrap()),
            "cached"
        );
        // The same file read as another kind is a separate entry.
        assert!(!Arc::ptr_eq(
            &first,
            &cache.get(&path, TexKind::Unorm16).unwrap()
        ));
        image::RgbImage::from_pixel(2, 2, image::Rgb([200, 200, 200]))
            .save(&path)
            .unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5))
            .unwrap();
        let again = cache.get(&path, TexKind::Srgb8).unwrap();
        let Pixels::U8(px) = &again.levels[0].pixels else {
            panic!()
        };
        assert_eq!(px[0], [200; 3], "a rebuilt file is decoded again");
        assert_eq!(cache.entries.len(), 2);
        let other = dir.join("b.png");
        image::RgbImage::new(1, 1).save(&other).unwrap();
        cache.get(&other, TexKind::Srgb8).unwrap();
        assert_eq!(cache.entries.len(), 2, "capacity holds");
    }

    #[test]
    fn alpha_composite_and_png_round_trip() {
        let mut base = RgbaImage::new(2, 1);
        base.fill([10, 20, 30, 255]);
        let mut top = RgbaImage::new(2, 1);
        top.pixels[..4].copy_from_slice(&[200, 100, 50, 255]);
        base.composite(&top);
        assert_eq!(base.get(0, 0), [200, 100, 50, 255]);
        assert_eq!(
            base.get(1, 0),
            [10, 20, 30, 255],
            "transparent pixels leave the base"
        );
        let path = std::env::temp_dir().join(format!("press_render_{}.png", std::process::id()));
        base.save_png(&path).unwrap();
        let back = image::open(&path).unwrap().to_rgba8();
        assert_eq!(back.get_pixel(0, 0).0, [200, 100, 50, 255]);
    }

    /// A four-corner-coloured atlas pulled vertically: the red corner is image top-left.
    #[test]
    fn an_asymmetric_texture_is_drawn_rotated_not_mirrored() {
        let colours = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        // Image corners (u, v): TL red, TR green, BL blue, BR yellow, as 2x2 sRGB texels.
        let chain = Arc::new(MipChain::from_level0(
            TexKind::Srgb8,
            MipLevel {
                width: 2,
                height: 2,
                pixels: Pixels::U8(colours.to_vec()),
            },
        ));
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(200.0, 200.0));
        let projection = Projection::fit(canvas, ViewMode::Top, 1.0, 0.0);
        let flat = PackageGeometry::new(0.5, 0.0).unwrap();
        let map = SheetMap {
            pull_vertical: true,
        };
        let scene = ClothScene {
            quads: cloth_quads(&flat, &projection, map, None),
            wrap: Wrap::Clamp,
            shading: Shading::Rgb { t: 0.0 },
            files: SceneFiles::Rgb([PathBuf::new(), PathBuf::new()]),
        };
        let image = render_cloth(
            &scene,
            &SceneTextures::Rgb([chain.clone(), chain]),
            [200, 200],
            Pos2::ZERO,
            1.0,
        );
        let q = &scene.quads;
        let (left, right) = (q[0].pos[0].x + 8.0, q[1].pos[1].x - 8.0);
        let (top, bottom) = (q[0].pos[0].y + 8.0, q[0].pos[2].y - 8.0);
        let at = |x: f32, y: f32| {
            let p = image.get(x as u32, y as u32);
            [p[0], p[1], p[2]]
        };
        // Expected corner order after a 90° counter-clockwise rotation.
        assert_eq!(at(left, bottom), colours[0], "image top-left (red)");
        assert_eq!(at(left, top), colours[1], "image top-right (green)");
        assert_eq!(at(right, bottom), colours[2], "image bottom-left (blue)");
        assert_eq!(at(right, top), colours[3], "image bottom-right (yellow)");
        assert_eq!(image.get(0, 0)[3], 0);
    }

    #[test]
    fn overlay_discs_and_lines_cover_their_pixels() {
        let mut image = RgbaImage::new(40, 20);
        image.fill([0, 0, 0, 255]);
        let items = [
            OverlayItem::Disc {
                center: Pos2::new(10.0, 10.0),
                radius: 4.0,
                color: Color32::WHITE,
            },
            OverlayItem::Line {
                a: Pos2::new(25.0, 2.0),
                b: Pos2::new(25.0, 18.0),
                width: 3.0,
                color: Color32::from_rgb(255, 195, 75),
            },
        ];
        draw_overlay(&mut image, &items, Pos2::ZERO, 1.0);
        assert_eq!(image.get(10, 10)[..3], [255, 255, 255]);
        assert_eq!(image.get(25, 10)[..3], [255, 195, 75]);
        assert_eq!(image.get(35, 10)[..3], [0, 0, 0]);
        assert_eq!(image.get(10, 2)[..3], [0, 0, 0]);
    }

    #[test]
    fn the_worker_renders_only_the_newest_job_and_reports_errors() {
        let worker = CpuWorker::spawn(|| {});
        let missing = ClothScene {
            quads: [ClothQuad {
                pos: [Pos2::ZERO; 4],
                uv: [[0.0; 2]; 4],
                plane: 0,
            }; 2],
            wrap: Wrap::Clamp,
            shading: Shading::Rgb { t: 0.0 },
            files: SceneFiles::Rgb([
                PathBuf::from("/nonexistent/a.png"),
                PathBuf::from("/nonexistent/b.png"),
            ]),
        };
        worker.submit(CpuJob {
            id: 7,
            scene: missing,
            origin: Pos2::ZERO,
            scale: 1.0,
            size: [4, 4],
        });
        let started = std::time::Instant::now();
        let result = loop {
            if let Some(result) = worker.poll() {
                break result;
            }
            assert!(started.elapsed().as_secs() < 10, "worker timed out");
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert_eq!(result.id, 7);
        assert!(result.image.err().unwrap().contains("a.png"));
    }
}
