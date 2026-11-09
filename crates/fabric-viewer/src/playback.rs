//! Replay of a run: the playback rate / loop preferences (persisted) and the
//! timer that advances the central frame selection while playing.

use serde::{Deserialize, Serialize};

use crate::project::{frame_ref_for, resolve_frame_ref, ProjectSlot};
use crate::selection::FrameSelection;

/// Default playback rate (frames per second).
pub const DEFAULT_FPS: f32 = 5.0;

/// Playback rate / loop preference (persisted) plus the live play state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlaybackState {
    #[serde(skip)]
    pub playing: bool,
    #[serde(skip)]
    timer: f64,
    pub fps: f32,
    pub loop_enabled: bool,
}

impl Default for PlaybackState {
    fn default() -> Self {
        PlaybackState {
            playing: false,
            timer: 0.0,
            fps: DEFAULT_FPS,
            loop_enabled: true,
        }
    }
}

impl PlaybackState {
    /// Advance the selected frame using the loop setting. Clear transient hover
    /// so plot scrubbing does not override playback.
    pub fn step(&mut self, slots: &[ProjectSlot], selection: &mut FrameSelection) {
        let Some(active) = selection.active().cloned() else {
            self.playing = false;
            return;
        };
        let Some((pidx, ridx, fidx)) = resolve_frame_ref(slots, &active) else {
            self.playing = false;
            return;
        };
        let n = slots[pidx].project.runs[ridx].frames.frames.len();
        if n == 0 {
            self.playing = false;
            return;
        }
        let next = fidx + 1;
        if next >= n {
            if self.loop_enabled {
                if let Some(fr) = frame_ref_for(slots, pidx, ridx, 0) {
                    selection.selected = Some(fr);
                }
            } else {
                if let Some(fr) = frame_ref_for(slots, pidx, ridx, n - 1) {
                    selection.selected = Some(fr);
                }
                self.playing = false;
            }
        } else if let Some(fr) = frame_ref_for(slots, pidx, ridx, next) {
            selection.selected = Some(fr);
        }
        selection.hovered = None;
    }

    /// Timer-driven playback: advances `selection` at the configured rate and
    /// keeps repainting while playing.
    pub fn advance(
        &mut self,
        ctx: &egui::Context,
        slots: &[ProjectSlot],
        selection: &mut FrameSelection,
    ) {
        if !self.playing {
            return;
        }
        self.timer += ctx.input(|i| i.stable_dt) as f64;
        let interval = 1.0 / self.fps.max(0.1) as f64;
        let mut steps = 0;
        while self.timer >= interval && self.playing && steps < 1000 {
            self.timer -= interval;
            self.step(slots, selection);
            steps += 1;
        }
        if self.playing {
            ctx.request_repaint();
        }
    }
}
