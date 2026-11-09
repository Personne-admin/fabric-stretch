//! Fabric Stretch Viewer executable.

// No console window for the release build on Windows.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use eframe::egui_wgpu::{wgpu, WgpuConfiguration, WgpuSetup};
use std::sync::Arc;

/// Window size: the default, or full HD for the scripted demo recording.
fn window_size() -> [f32; 2] {
    #[cfg(feature = "demo")]
    if fabric_viewer::demo::requested() {
        return [1920.0, 1080.0];
    }
    [1400.0, 850.0]
}

/// The default wgpu setup, plus 16-bit normalised textures where the adapter has them: the Press
/// View stores photometric normals and albedo with 16 bits per channel.
fn wgpu_options() -> WgpuConfiguration {
    let mut options = WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(create) = &mut options.wgpu_setup {
        let base = create.device_descriptor.clone();
        create.device_descriptor = Arc::new(move |adapter| {
            let mut descriptor = base(adapter);
            if adapter
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
            {
                descriptor.required_features |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
            }
            descriptor
        });
    }
    options
}

fn main() -> eframe::Result {
    env_logger::init();

    let paths: Vec<std::path::PathBuf> = std::env::args()
        .skip(1)
        .map(std::path::PathBuf::from)
        .collect();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(window_size())
            .with_title("Fabric Stretch Viewer"),
        wgpu_options: wgpu_options(),
        ..Default::default()
    };
    eframe::run_native(
        "fabric-viewer",
        options,
        Box::new(move |cc| {
            Ok(Box::new(fabric_viewer::app::ViewerApp::new(
                paths,
                cc.storage,
                cc.wgpu_render_state.as_ref(),
            )))
        }),
    )
}
