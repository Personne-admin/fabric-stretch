//! Image Viewer playback controls, driving the shared [`FrameSelection`].

use crate::playback::PlaybackState;
use crate::project::{frame_ref_for, ProjectSlot};
use crate::selection::FrameSelection;

/// The frame being shown: `(project, run, frame)` indices plus the run length.
#[derive(Debug, Clone, Copy)]
pub(super) struct ReplayTarget {
    pub project: usize,
    pub run: usize,
    pub frame: usize,
    pub n_frames: usize,
}

pub(super) fn replay_row(
    ui: &mut egui::Ui,
    slots: &[ProjectSlot],
    selection: &mut FrameSelection,
    playback: &mut PlaybackState,
    at: ReplayTarget,
) {
    let ReplayTarget {
        project,
        run,
        frame: fidx,
        n_frames,
    } = at;
    if n_frames == 0 {
        return;
    }
    let last = n_frames - 1;
    let mut target: Option<usize> = None;
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Frame {} / {}", fidx + 1, n_frames));
        if ui
            .add_enabled(fidx > 0, egui::Button::new("<"))
            .on_hover_text("previous frame (Left arrow)")
            .clicked()
        {
            target = Some(fidx - 1);
        }
        let play_label = if playback.playing { "Pause" } else { "Play" };
        if ui
            .button(play_label)
            .on_hover_text("replay this run frame by frame")
            .clicked()
        {
            playback.playing = !playback.playing;
        }
        if ui
            .add_enabled(fidx < last, egui::Button::new(">"))
            .on_hover_text("next frame (Right arrow)")
            .clicked()
        {
            target = Some(fidx + 1);
        }
        ui.label("FPS");
        ui.add(
            egui::DragValue::new(&mut playback.fps)
                .speed(0.25)
                .range(0.5..=60.0),
        )
        .on_hover_text("playback rate in frames per second");
        ui.checkbox(&mut playback.loop_enabled, "Loop")
            .on_hover_text("restart at the first frame instead of stopping at the end");
        if last > 0 {
            let mut scrub = fidx;
            if ui
                .add(
                    egui::Slider::new(&mut scrub, 0..=last)
                        .show_value(false)
                        .text("timeline"),
                )
                .changed()
            {
                target = Some(scrub);
            }
        }
    });
    if let Some(index) = target {
        if let Some(fr) = frame_ref_for(slots, project, run, index.min(last)) {
            selection.selected = Some(fr);
            selection.hovered = None;
            selection.pinned = false;
        }
    }
}
