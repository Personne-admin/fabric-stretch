//! Small colour / badge helpers shared by several tabs.

use egui::{Color32, RichText};
use fabric_data::{Direction, Severity};

/// Direction colour shared by the plot and the runs tab.
pub fn direction_color(direction: &Direction) -> Color32 {
    match direction {
        Direction::Warp => Color32::from_rgb(231, 97, 84),
        Direction::Weft => Color32::from_rgb(87, 144, 231),
        Direction::Bias => Color32::from_rgb(96, 189, 115),
        Direction::Other(_) => Color32::from_rgb(160, 160, 160),
    }
}

/// Validation severity badge.
pub fn severity_badge(severity: Severity) -> RichText {
    let (text, color) = match severity {
        Severity::Ok => ("OK", Color32::from_rgb(96, 189, 115)),
        Severity::Warning => ("WARN", Color32::from_rgb(235, 178, 66)),
        Severity::Error => ("ERR", Color32::from_rgb(231, 84, 84)),
    };
    RichText::new(text).color(color).strong().small()
}

/// Amber used for warnings and fallbacks.
pub const WARN: Color32 = Color32::from_rgb(235, 178, 66);
/// Red used for errors.
pub const ERROR: Color32 = Color32::from_rgb(231, 84, 84);
