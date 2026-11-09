//! Scripted recorder behind the `demo` feature.
//! `FABRIC_VIEWER_DEMO_DIR` enables a 1920 × 1080 recording, one PPM per UI frame.
//! The viewer runs the script, then exits; assemble the frames externally.

use std::path::PathBuf;

use crate::app::ViewerApp;
use crate::material_package::PressBranch;
use crate::press_tab::PressViews;
use crate::ui::workspace::WorkspaceTab;

/// Environment variable naming the output directory.
pub const ENV_DIR: &str = "FABRIC_VIEWER_DEMO_DIR";

/// Frames rendered before the script starts (layout and textures settle).
const WARMUP: usize = 6;
/// Length of the scripted animation in frames.
const TOTAL_FRAMES: usize = 105;
/// Extra frames to wait for outstanding screenshots before giving up.
const GRACE: usize = 20;

/// Whether the recorder was requested.
pub fn requested() -> bool {
    std::env::var_os(ENV_DIR).is_some()
}

/// State of one recording.
pub struct DemoRecorder {
    dir: PathBuf,
    frame: usize,
    saved: usize,
}

impl DemoRecorder {
    pub fn from_env() -> Option<Self> {
        std::env::var_os(ENV_DIR).map(|dir| DemoRecorder {
            dir: PathBuf::from(dir),
            frame: 0,
            saved: 0,
        })
    }

    /// Curves on the left, a Press View on the first project on the right.
    pub fn install_layout(app: &mut ViewerApp) {
        let Some(slot) = app.slots.first() else {
            return;
        };
        let id = app.press.add(Some(slot.root_key.clone()));
        let mut state = egui_dock::DockState::new(vec![WorkspaceTab::Curves]);
        state.main_surface_mut().split_right(
            egui_dock::NodeIndex::root(),
            0.38,
            vec![WorkspaceTab::Press(id)],
        );
        app.workspace.dock_state = state;
    }

    /// Advance the script by one UI frame: store arrived screenshots, drive
    /// the Press Views, request the next screenshot, and close when done.
    pub fn tick(&mut self, ctx: &egui::Context, press: &mut PressViews) {
        let frame = self.frame;
        self.frame += 1;
        for event in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = event {
                self.save_ppm(&image);
            }
        }

        if frame < WARMUP {
            ctx.request_repaint();
        } else if frame < WARMUP + TOTAL_FRAMES {
            let (force, beta, branch) = script(frame - WARMUP);
            for view in press.views_mut() {
                view.set_demo_state(force, beta, branch);
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            ctx.request_repaint();
        } else if self.saved >= TOTAL_FRAMES || frame >= WARMUP + TOTAL_FRAMES + GRACE {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else {
            ctx.request_repaint();
        }
    }

    fn save_ppm(&mut self, image: &egui::ColorImage) {
        let (w, h) = (image.size[0], image.size[1]);
        let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
        data.reserve(w * h * 3);
        for p in &image.pixels {
            data.extend_from_slice(&[p.r(), p.g(), p.b()]);
        }
        let path = self.dir.join(format!("frame_{:05}.ppm", self.saved));
        self.saved += 1;
        if let Err(err) = std::fs::write(&path, data) {
            log::warn!("demo: could not write {}: {err}", path.display());
        }
    }
}

/// `(tension as a fraction of the force range, bar position as a fraction of the span, branch)` at
/// script frame `f`: ramp the tension, sweep the bar left and right, then release.
fn script(f: usize) -> (f32, f32, PressBranch) {
    use PressBranch::{Loading, Release};
    let ease = |t: f32| (1.0 - (t * std::f32::consts::PI).cos()) * 0.5;
    match f {
        0..=24 => (0.05 + 0.9 * f as f32 / 25.0, 0.50, Loading),
        25..=49 => (0.95, 0.50 - 0.20 * ease((f - 25) as f32 / 25.0), Loading),
        50..=74 => (0.95, 0.30 + 0.40 * ease((f - 50) as f32 / 25.0), Loading),
        75..=89 => (0.95, 0.70 - 0.20 * ease((f - 75) as f32 / 15.0), Loading),
        _ => (0.95 - 0.9 * (f - 90) as f32 / 15.0, 0.50, Release),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_script_ramps_sweeps_and_releases() {
        let (f0, b0, branch0) = script(0);
        assert!(f0 < 0.2 && b0 == 0.5 && branch0 == PressBranch::Loading);
        assert!(script(49).1 < 0.31, "bar swept left");
        assert!(script(74).1 > 0.69, "and right");
        let (f_end, _, branch_end) = script(TOTAL_FRAMES - 1);
        assert!(f_end < 0.2 && branch_end == PressBranch::Release);
    }
}
