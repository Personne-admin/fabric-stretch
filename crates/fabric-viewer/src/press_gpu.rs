//! Press View GPU renderer, with `press_render` as the CPU reference.
//!
//! Mixes the two state textures in the fragment shader to avoid re-uploading on slider changes.
//! Colour uses sRGB textures for linear interpolation; relighting uses raw albedo and normals.
//! Lights arrive in each plane's data frame. Textures are cached in `PackageStore`.

use std::sync::{Arc, Mutex};

use eframe::egui_wgpu::{self, wgpu, CallbackResources, CallbackTrait, ScreenDescriptor};
use egui::{PaintCallbackInfo, Pos2, Rect};
use wgpu::util::DeviceExt;

use crate::press_render::{
    ClothScene, MipChain, Pixels, RgbaImage, SceneFiles, Shading, TexKind, Wrap,
};

/// Shared by both shaders: the vertex stage, the colour helpers and the per-draw parameters.
const PRELUDE: &str = r#"
struct Params {
    // Light towards the viewer-side light in the data frame, plane 0 and plane 1 (w unused).
    light0: vec4<f32>,
    light1: vec4<f32>,
    // x = weight of the second state, y = intensity, z = ambient, w = albedo scale.
    shade: vec4<f32>,
}

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) plane: f32,
}
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) plane: f32,
}

@vertex
fn vs(v: VsIn) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(v.pos, 0.0, 1.0);
    out.uv = v.uv;
    out.plane = v.plane;
    return out;
}

fn srgb_from_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn linear_from_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// `display` holds the bytes to show (gamma space). A gamma surface takes them unchanged; an sRGB
// surface encodes on write, so decode first.
fn to_surface(display: vec3<f32>) -> vec3<f32> {
    if (TARGET_SRGB) {
        return linear_from_srgb(display);
    }
    return display;
}
"#;

const RGB_SHADER: &str = r#"
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> params: Params;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // The state textures are sRGB: these samples are linear light.
    let a = textureSample(tex_a, samp, in.uv).rgb;
    let b = textureSample(tex_b, samp, in.uv).rgb;
    let linear = clamp(mix(a, b, params.shade.x), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(to_surface(srgb_from_linear(linear)), 1.0);
}
"#;

const RELIGHT_SHADER: &str = r#"
@group(0) @binding(0) var albedo_a: texture_2d<f32>;
@group(0) @binding(1) var albedo_b: texture_2d<f32>;
@group(0) @binding(2) var normal_a: texture_2d<f32>;
@group(0) @binding(3) var normal_b: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var<uniform> params: Params;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let t = params.shade.x;
    let albedo = mix(textureSample(albedo_a, samp, in.uv).rgb,
                     textureSample(albedo_b, samp, in.uv).rgb, t) * params.shade.w;
    var n = mix(textureSample(normal_a, samp, in.uv).rgb,
                textureSample(normal_b, samp, in.uv).rgb, t) * 2.0 - vec3<f32>(1.0);
    // Renormalise the mixed normal; a degenerate one falls back to +z (as normal::normalize3).
    let len = length(n);
    if (len > 0.01) {
        n = n / len;
    } else {
        n = vec3<f32>(0.0, 0.0, 1.0);
    }
    var light = params.light0.xyz;
    if (in.plane > 0.5) {
        light = params.light1.xyz;
    }
    let gain = params.shade.z + params.shade.y * max(dot(n, light), 0.0);
    let display = clamp(albedo * gain, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(to_surface(display), 1.0);
}
"#;

/// Source of one shader module for a target colour space.
fn shader_source(body: &str, target_srgb: bool) -> String {
    format!("const TARGET_SRGB: bool = {target_srgb};\n{PRELUDE}\n{body}")
}

/// Both shader modules as WGSL text, for validation in tests.
#[cfg(test)]
pub fn shader_sources(target_srgb: bool) -> [String; 2] {
    [
        shader_source(RGB_SHADER, target_srgb),
        shader_source(RELIGHT_SHADER, target_srgb),
    ]
}

/// A texture uploaded to the GPU together with its mip chain.
pub struct GpuTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bytes: usize,
}

impl GpuTexture {
    /// Video memory taken by the texture, mip levels included.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

struct GpuPipelines {
    rgb: wgpu::RenderPipeline,
    rgb_layout: wgpu::BindGroupLayout,
    relight: wgpu::RenderPipeline,
    relight_layout: wgpu::BindGroupLayout,
    clamp: wgpu::Sampler,
    repeat: wgpu::Sampler,
}

/// Clone-cheap handle to the shared pipelines, held by every draw callback.
#[derive(Clone)]
pub struct GpuPipelinesHandle(Arc<GpuPipelines>);

/// Device, queue and pipelines for drawing and uploading Press View textures.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: GpuPipelinesHandle,
    wide_format: wgpu::TextureFormat,
}

impl GpuContext {
    /// Context for the egui render state (pipelines are built for its target format).
    pub fn from_render_state(render_state: &egui_wgpu::RenderState) -> Self {
        Self::new(
            render_state.device.clone(),
            render_state.queue.clone(),
            render_state.target_format,
        )
    }

    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let wide_format = if device
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            wgpu::TextureFormat::Rgba16Unorm
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let pipelines = GpuPipelinesHandle(Arc::new(GpuPipelines::new(&device, target_format)));
        Self {
            device,
            queue,
            pipelines,
            wide_format,
        }
    }

    /// A context without a window, on any available adapter (a software Vulkan driver works).
    /// `None` when no adapter is found. Renders into `Rgba8Unorm` targets.
    pub fn headless() -> Option<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        let mut required_features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            required_features |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("press view headless device"),
            required_features,
            ..Default::default()
        }))
        .ok()?;
        Some(Self::new(device, queue, wgpu::TextureFormat::Rgba8Unorm))
    }

    /// Largest texture width or height the device accepts.
    pub fn max_texture_side(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    /// Format of the 16-bit photometric textures on this device.
    pub fn wide_format(&self) -> wgpu::TextureFormat {
        self.wide_format
    }

    /// Uploads a decoded texture with all of its mip levels. Fails, instead of letting wgpu
    /// abort the process, when the texture is larger than the device supports.
    pub fn upload(&self, chain: &MipChain) -> Result<GpuTexture, String> {
        let format = match chain.kind {
            TexKind::Srgb8 => wgpu::TextureFormat::Rgba8UnormSrgb,
            TexKind::Unorm16 => self.wide_format,
        };
        let base = &chain.levels[0];
        let limit = self.max_texture_side();
        if base.width > limit || base.height > limit {
            return Err(format!(
                "A {}x{} texture is larger than this GPU supports ({limit} px per side). \
                 Rebuild the package with smaller textures or set FABRIC_CPU_RENDER=1.",
                base.width, base.height
            ));
        }
        let mut bytes = 0;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("press view state texture"),
            size: wgpu::Extent3d {
                width: base.width,
                height: base.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: chain.levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (mip, level) in chain.levels.iter().enumerate() {
            let (data, bytes_per_texel) = texel_bytes(&level.pixels, format);
            bytes += data.len();
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_texel * level.width),
                    rows_per_image: Some(level.height),
                },
                wgpu::Extent3d {
                    width: level.width,
                    height: level.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(GpuTexture {
            _texture: texture,
            view,
            bytes,
        })
    }

    /// The paint callback for one cloth draw. `textures` are in the order of [`SceneFiles`]
    /// (colour: two; relight: albedo pair then normal pair).
    pub fn draw(
        &self,
        canvas: Rect,
        scene: &ClothScene,
        textures: Vec<Arc<GpuTexture>>,
    ) -> egui::epaint::PaintCallback {
        egui_wgpu::Callback::new_paint_callback(canvas, self.quad_draw(canvas, scene, textures))
    }

    fn quad_draw(
        &self,
        canvas: Rect,
        scene: &ClothScene,
        textures: Vec<Arc<GpuTexture>>,
    ) -> QuadDraw {
        QuadDraw {
            pipelines: self.pipelines.clone(),
            textures,
            relight: matches!(scene.files, SceneFiles::Relight { .. }),
            wrap: scene.wrap,
            vertices: vertices(scene, canvas),
            params: params(&scene.shading),
            prepared: Mutex::new(None),
        }
    }

    /// Renders a scene into an offscreen `Rgba8Unorm` image (for tests and snapshots). Positions
    /// are canvas points; `origin` is the canvas corner and `scale` the pixels per point.
    pub fn render_offscreen(
        &self,
        scene: &ClothScene,
        textures: Vec<Arc<GpuTexture>>,
        size: [u32; 2],
        origin: Pos2,
        scale: f32,
    ) -> Result<RgbaImage, String> {
        let canvas = Rect::from_min_size(
            origin,
            egui::vec2(size[0] as f32 / scale, size[1] as f32 / scale),
        );
        let draw = self.quad_draw(canvas, scene, textures);
        draw.prepare_draw(&self.device);
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("press view offscreen target"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let row_bytes = size[0] * 4;
        let padded = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("press view readback"),
            size: padded as u64 * size[1] as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("press view offscreen pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            draw.draw_into(&mut pass);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("GPU readback failed: {e}"))?;
        let mapped = readback.slice(..).get_mapped_range();
        let mut image = RgbaImage::new(size[0], size[1]);
        for y in 0..size[1] as usize {
            let src = &mapped[y * padded as usize..y * padded as usize + row_bytes as usize];
            image.pixels[y * row_bytes as usize..(y + 1) * row_bytes as usize].copy_from_slice(src);
        }
        Ok(image)
    }
}

/// Texel bytes of one level in the layout of `format`, and the bytes per texel.
fn texel_bytes(pixels: &Pixels, format: wgpu::TextureFormat) -> (Vec<u8>, u32) {
    match (pixels, format) {
        (Pixels::U8(px), _) => (px.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect(), 4),
        (Pixels::U16(px), wgpu::TextureFormat::Rgba16Unorm) => (
            px.iter()
                .flat_map(|p| [p[0], p[1], p[2], u16::MAX])
                .flat_map(u16::to_le_bytes)
                .collect(),
            8,
        ),
        (Pixels::U16(px), _) => (
            px.iter()
                .flat_map(|p| [p[0], p[1], p[2], u16::MAX])
                .map(|v| ((v as u32 * 255 + 32767) / 65535) as u8)
                .collect(),
            4,
        ),
    }
}

impl GpuPipelines {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let sampler = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = |label: &str, textures: u32| {
            let mut entries: Vec<_> = (0..textures).map(|i| layout_entry(i, texture)).collect();
            entries.push(layout_entry(textures, sampler));
            entries.push(layout_entry(textures + 1, uniform));
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            })
        };
        let rgb_layout = layout("press rgb layout", 2);
        let relight_layout = layout("press relight layout", 4);
        let target_srgb = target_format.is_srgb();
        let pipeline = |label: &str, body: &str, layout: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source(body, target_srgb).into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            let attributes = [
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 16,
                    shader_location: 2,
                },
            ];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 20,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    }],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let rgb = pipeline("press rgb pipeline", RGB_SHADER, &rgb_layout);
        let relight = pipeline("press relight pipeline", RELIGHT_SHADER, &relight_layout);
        let make_sampler = |mode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("press view sampler"),
                address_mode_u: mode,
                address_mode_v: mode,
                address_mode_w: mode,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            })
        };
        Self {
            rgb,
            rgb_layout,
            relight,
            relight_layout,
            clamp: make_sampler(wgpu::AddressMode::ClampToEdge),
            repeat: make_sampler(wgpu::AddressMode::Repeat),
        }
    }
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

/// Vertices `[x, y, u, v, plane]` of the two quads: `x, y` in the normalised device coordinates of
/// `canvas`, the texture coordinate, and the plane index.
fn vertices(scene: &ClothScene, canvas: Rect) -> Vec<[f32; 5]> {
    let to_ndc = |p: Pos2| {
        [
            (p.x - canvas.min.x) / canvas.width() * 2.0 - 1.0,
            1.0 - (p.y - canvas.min.y) / canvas.height() * 2.0,
        ]
    };
    let mut out = Vec::with_capacity(12);
    for quad in &scene.quads {
        for triangle in crate::press_geometry::ClothQuad::TRIANGLES {
            for i in triangle {
                let i = i as usize;
                let [x, y] = to_ndc(quad.pos[i]);
                out.push([x, y, quad.uv[i][0], quad.uv[i][1], quad.plane as f32]);
            }
        }
    }
    out
}

/// The uniform block of a draw (12 floats, see `Params` in the shaders).
fn params(shading: &Shading) -> [f32; 12] {
    match *shading {
        Shading::Rgb { t } => {
            let mut p = [0.0; 12];
            p[8] = t;
            p
        }
        Shading::Relight {
            t,
            intensity,
            ambient,
            albedo_scale,
            plane_lights,
        } => {
            let [l0, l1] = plane_lights;
            [
                l0[0],
                l0[1],
                l0[2],
                0.0,
                l1[0],
                l1[1],
                l1[2],
                0.0,
                t,
                intensity,
                ambient,
                albedo_scale,
            ]
        }
    }
}

/// Draws the textured triangles of one Press View canvas inside the egui render pass.
struct QuadDraw {
    pipelines: GpuPipelinesHandle,
    textures: Vec<Arc<GpuTexture>>,
    relight: bool,
    wrap: Wrap,
    vertices: Vec<[f32; 5]>,
    params: [f32; 12],
    prepared: Mutex<Option<(wgpu::Buffer, wgpu::BindGroup)>>,
}

impl QuadDraw {
    fn prepare_draw(&self, device: &wgpu::Device) {
        let p = &self.pipelines.0;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("press view vertices"),
            contents: bytes_of_f32(self.vertices.as_flattened()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("press view params"),
            contents: bytes_of_f32(&self.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sampler = match self.wrap {
            Wrap::Clamp => &p.clamp,
            Wrap::Repeat => &p.repeat,
        };
        let mut entries: Vec<wgpu::BindGroupEntry> = self
            .textures
            .iter()
            .enumerate()
            .map(|(i, t)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(&t.view),
            })
            .collect();
        let n = self.textures.len() as u32;
        entries.push(wgpu::BindGroupEntry {
            binding: n,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: n + 1,
            resource: uniform.as_entire_binding(),
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("press view bind group"),
            layout: if self.relight {
                &p.relight_layout
            } else {
                &p.rgb_layout
            },
            entries: &entries,
        });
        *self.prepared.lock().expect("quad draw lock") = Some((vertex_buffer, bind));
    }

    fn draw_into(&self, pass: &mut wgpu::RenderPass<'static>) {
        let guard = self.prepared.lock().expect("quad draw lock");
        let Some((buffer, bind)) = guard.as_ref() else {
            return;
        };
        let p = &self.pipelines.0;
        pass.set_pipeline(if self.relight { &p.relight } else { &p.rgb });
        pass.set_bind_group(0, bind, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}

impl CallbackTrait for QuadDraw {
    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.prepare_draw(device);
        Vec::new()
    }

    fn paint(
        &self,
        _info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &CallbackResources,
    ) {
        self.draw_into(render_pass);
    }
}

fn bytes_of_f32(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and every bit pattern is a valid byte sequence.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// Minimal executor for the wgpu setup futures, which complete without waiting on anything.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Waker};
    let mut context = Context::from_waker(Waker::noop());
    let mut future = Box::pin(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::Light;
    use crate::material_package::test_support::demo_project;
    use crate::material_package::{MaterialPackage, PressBranch};
    use crate::press_geometry::ViewMode;
    use crate::press_render::{render_cloth, SceneTextures, TextureCache};
    use crate::press_sim::{build_scene, Control, SceneRequest};

    /// Parses and validates both shaders with naga (no GPU needed), for both target colour spaces.
    #[test]
    fn the_wgsl_shaders_validate() {
        for srgb in [false, true] {
            for (i, source) in shader_sources(srgb).iter().enumerate() {
                let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
                    panic!(
                        "shader {i} (srgb {srgb}) does not parse:\n{}",
                        e.emit_to_string(source)
                    )
                });
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("shader {i} (srgb {srgb}) is invalid: {e:?}"));
            }
        }
    }

    #[test]
    fn uniform_blocks_carry_the_blend_weight_and_the_per_plane_lights() {
        assert_eq!(params(&Shading::Rgb { t: 0.25 })[8], 0.25);
        let p = params(&Shading::Relight {
            t: 0.5,
            intensity: 1.5,
            ambient: 0.1,
            albedo_scale: 0.6,
            plane_lights: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        });
        assert_eq!(
            p,
            [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.5, 1.5, 0.1, 0.6]
        );
    }

    #[test]
    fn wide_textures_fall_back_to_eight_bit_without_the_16_bit_feature() {
        let px = Pixels::U16(vec![[0, 32768, 65535]]);
        let (bytes, per_texel) = texel_bytes(&px, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(
            (bytes.as_slice(), per_texel),
            (&[0u8, 128, 255, 255][..], 4)
        );
        let (bytes, per_texel) = texel_bytes(&px, wgpu::TextureFormat::Rgba16Unorm);
        assert_eq!(per_texel, 8);
        assert_eq!(&bytes[2..4], &32768u16.to_le_bytes());
    }

    fn pair(chains: &[Arc<MipChain>], ctx: &GpuContext) -> Vec<Arc<GpuTexture>> {
        chains
            .iter()
            .map(|c| Arc::new(ctx.upload(c).unwrap()))
            .collect()
    }

    /// Mean absolute difference over pixels covered in both images (and the covered fraction that differs in coverage).
    fn compare(a: &RgbaImage, b: &RgbaImage) -> (f32, f32) {
        let (mut sum, mut n, mut coverage_diff, mut covered) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let ((a_px, _), (b_px, _)) = (a.pixels.as_chunks::<4>(), b.pixels.as_chunks::<4>());
        for (pa, pb) in a_px.iter().zip(b_px) {
            let (ca, cb) = (pa[3] > 0, pb[3] > 0);
            covered += (ca || cb) as u8 as f32;
            if ca != cb {
                coverage_diff += 1.0;
            } else if ca {
                for k in 0..3 {
                    sum += (pa[k] as f32 - pb[k] as f32).abs();
                }
                n += 3.0;
            }
        }
        (sum / n.max(1.0), coverage_diff / covered.max(1.0))
    }

    /// Compare with the CPU renderer. Skips without a Vulkan adapter; SwiftShader also works.
    #[test]
    fn gpu_rendering_matches_the_cpu_reference() {
        let Some(ctx) = GpuContext::headless() else {
            eprintln!("no GPU adapter available; skipping the GPU/CPU comparison");
            return;
        };
        let size = [640u32, 400u32];
        let canvas = Rect::from_min_size(Pos2::ZERO, egui::vec2(size[0] as f32, size[1] as f32));
        let mut cache = TextureCache::new(16);
        for (dir, relight) in [
            ("material_package_v3_untiled", None),
            ("material_package_v3", None),
            (
                "material_package_v3_untiled",
                Some(Light {
                    azimuth_deg: 200.0,
                    elevation_deg: 40.0,
                    intensity: 1.2,
                    ambient: 0.05,
                }),
            ),
            ("material_package_v3", Some(Light::default())),
        ] {
            let package = MaterialPackage::load(&demo_project().join(dir).join("warp")).unwrap();
            let request = SceneRequest {
                branch: PressBranch::Loading,
                control: Control::Indentation(0.18),
                beta: 0.42,
                previous_lambda: 1.0,
                view: ViewMode::Oblique,
                relight,
            };
            let scene = build_scene(&package, &request, canvas).unwrap();
            let cpu_textures = SceneTextures::load(&scene.cloth.files, &mut cache).unwrap();
            let cpu = render_cloth(&scene.cloth, &cpu_textures, size, Pos2::ZERO, 1.0);
            let gpu_textures = match &cpu_textures {
                SceneTextures::Rgb(c) => pair(c, &ctx),
                SceneTextures::Relight { albedo, normal } => pair(
                    &[
                        albedo[0].clone(),
                        albedo[1].clone(),
                        normal[0].clone(),
                        normal[1].clone(),
                    ],
                    &ctx,
                ),
            };
            let gpu = ctx
                .render_offscreen(&scene.cloth, gpu_textures, size, Pos2::ZERO, 1.0)
                .unwrap();
            let (mean_diff, coverage_diff) = compare(&cpu, &gpu);
            eprintln!("{dir} relight={}: mean abs diff {mean_diff:.2}/255, coverage diff {coverage_diff:.4}", relight.is_some());
            assert!(
                mean_diff < 3.0,
                "{dir}: GPU and CPU differ by {mean_diff}/255"
            );
            assert!(
                coverage_diff < 0.01,
                "{dir}: coverage differs by {coverage_diff}"
            );
        }
    }
}
