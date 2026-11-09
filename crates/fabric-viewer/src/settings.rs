//! Persisted plot, image and UI settings. `serde(default)` keeps older RON files readable.

use serde::{Deserialize, Serialize};

use crate::lighting::Light;

/// Which curve quantity is plotted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlotQuantity {
    ForceN,
    LineLoad,
}

impl PlotQuantity {
    pub fn label(self) -> &'static str {
        match self {
            PlotQuantity::ForceN => "Force [N]",
            PlotQuantity::LineLoad => "Line load [N/m]",
        }
    }

    pub fn of(self, p: &fabric_data::CurvePoint) -> Option<f64> {
        match self {
            PlotQuantity::ForceN => p.force_n,
            PlotQuantity::LineLoad => p.line_load_n_per_m,
        }
    }
}

/// How the image viewer fits the image into its canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitMode {
    /// Scale to fit the canvas, preserving aspect ratio.
    Fit,
    /// One image pixel = one point.
    OneToOne,
}

/// Texture sampling filter for displayed images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Nearest,
    Linear,
}

impl Interpolation {
    pub fn texture_options(self) -> egui::TextureOptions {
        match self {
            Interpolation::Nearest => egui::TextureOptions::NEAREST,
            Interpolation::Linear => egui::TextureOptions::LINEAR,
        }
    }
}

/// Comparison layout for the image viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonMode {
    Single,
    SideBySide,
    SplitWipe,
}

impl ComparisonMode {
    pub const ALL: [ComparisonMode; 3] = [
        ComparisonMode::Single,
        ComparisonMode::SideBySide,
        ComparisonMode::SplitWipe,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ComparisonMode::Single => "Single",
            ComparisonMode::SideBySide => "Side-by-side",
            ComparisonMode::SplitWipe => "Split / wipe",
        }
    }
}

/// Persisted Image Viewer light. Legacy flat light fields are ignored and use defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RelightSettings {
    pub enabled: bool,
    pub light: Light,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PlotSettings {
    pub quantity: PlotQuantity,
    pub mirror_release: bool,
    pub show_relax: bool,
    pub smooth: bool,
    pub show_legend: bool,
    pub show_grid: bool,
    pub line_width: f32,
    pub selected_line_width: f32,
    pub marker_radius: f32,
    /// Hovering the plot pops up a small image preview at the pointer.
    pub hover_preview: bool,
    pub hover_preview_max_px: f32,
}

impl Default for PlotSettings {
    fn default() -> Self {
        PlotSettings {
            quantity: PlotQuantity::ForceN,
            mirror_release: true,
            show_relax: true,
            smooth: false,
            show_legend: true,
            show_grid: true,
            line_width: 1.5,
            selected_line_width: 3.0,
            marker_radius: 5.0,
            hover_preview: true,
            hover_preview_max_px: 280.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ImageSettings {
    /// Preferred image role, with diffuse → rgb → albedo → first sorted role as fallback.
    /// Relighting always uses albedo.
    pub preferred_role: Option<String>,
    pub fit: FitMode,
    pub interpolation: Interpolation,
    /// Follow the hovered frame while scrubbing the curve; off means the
    /// viewer only follows the clicked/scrubbed selection.
    pub hover_follow: bool,
    pub comparison: ComparisonMode,
    pub relight: RelightSettings,
}

impl Default for ImageSettings {
    fn default() -> Self {
        ImageSettings {
            preferred_role: Some("diffuse".into()),
            fit: FitMode::Fit,
            interpolation: Interpolation::Linear,
            hover_follow: true,
            comparison: ComparisonMode::Single,
            relight: RelightSettings::default(),
        }
    }
}

/// User theme preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Theme {
    #[default]
    System,
    Light,
    Dark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiSettings {
    pub theme: Theme,
    /// Restore the dock layout (tab positions/surfaces) on next launch.
    pub restore_workspace: bool,
    /// Show the legacy tracking workbench. Hidden by default; toggling it preserves its state.
    pub advanced_tracking: bool,
}

impl Default for UiSettings {
    fn default() -> Self {
        UiSettings {
            theme: Theme::System,
            restore_workspace: true,
            advanced_tracking: false,
        }
    }
}

/// Top-level settings envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[derive(Default)]
pub struct ViewerSettings {
    pub plot: PlotSettings,
    pub images: ImageSettings,
    pub ui: UiSettings,
}

impl ViewerSettings {
    /// Apply the configured theme preference.
    pub fn apply_theme(&self, ctx: &egui::Context) {
        let pref = match self.ui.theme {
            Theme::System => egui::ThemePreference::System,
            Theme::Light => egui::ThemePreference::Light,
            Theme::Dark => egui::ThemePreference::Dark,
        };
        ctx.set_theme(pref);
    }

    /// Treat a blank preferred role as no preference.
    pub fn normalize(&mut self) {
        if self
            .images
            .preferred_role
            .as_deref()
            .is_some_and(|s| s.trim().is_empty())
        {
            self.images.preferred_role = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_serialize_round_trip() {
        let settings = ViewerSettings::default();
        let text = serde_json::to_string(&settings).unwrap();
        let back: ViewerSettings = serde_json::from_str(&text).unwrap();
        assert_eq!(settings, back);
    }

    #[test]
    fn settings_deserialize_partial_defaults() {
        // Older configs fill missing groups from defaults.
        let text = serde_json::to_string(&PlotSettings::default()).unwrap();
        let outer = format!(r#"{{"plot":{text}}}"#);
        let settings: ViewerSettings = serde_json::from_str(&outer).unwrap();
        assert_eq!(settings.images, ImageSettings::default());
        assert_eq!(settings.ui, UiSettings::default());
    }

    #[test]
    fn settings_ron_round_trip_matches_eframe() {
        // Round-trip the RON format used by eframe.
        let settings = ViewerSettings::default();
        let text = ron::ser::to_string(&settings).unwrap();
        let back: ViewerSettings = ron::de::from_str(&text).unwrap();
        assert_eq!(settings, back);
    }

    #[test]
    fn settings_ron_partial_group_defaults() {
        // Partial RON configs use defaults for missing groups.
        let plot_text = ron::ser::to_string(&PlotSettings::default()).unwrap();
        let outer = format!("(plot:{plot_text},)");
        let settings: ViewerSettings = ron::de::from_str(&outer).unwrap();
        assert_eq!(settings.images, ImageSettings::default());
        assert_eq!(settings.ui, UiSettings::default());
    }

    #[test]
    fn empty_preferred_role_normalizes_to_none() {
        let mut settings = ViewerSettings::default();
        settings.images.preferred_role = Some("   ".into());
        settings.normalize();
        assert_eq!(settings.images.preferred_role, None);

        settings.images.preferred_role = Some("diffuse".into());
        settings.normalize();
        assert_eq!(settings.images.preferred_role.as_deref(), Some("diffuse"));
    }

    #[test]
    fn defaults_preserve_previous_behavior() {
        let p = PlotSettings::default();
        assert_eq!(p.quantity, PlotQuantity::ForceN);
        assert!(p.mirror_release);
        assert!(p.show_relax);
        assert!(p.show_legend);
        assert!(p.show_grid);
        assert_eq!(p.line_width, 1.5);
        assert_eq!(p.selected_line_width, 3.0);
        let i = ImageSettings::default();
        assert!(i.hover_follow);
        assert_eq!(i.fit, FitMode::Fit);
        assert_eq!(i.interpolation, Interpolation::Linear);
        assert_eq!(i.preferred_role.as_deref(), Some("diffuse"));
    }

    #[test]
    fn advanced_tracking_defaults_off() {
        assert!(!UiSettings::default().advanced_tracking);
        assert!(!ViewerSettings::default().ui.advanced_tracking);
    }

    #[test]
    fn settings_written_before_the_advanced_flag_still_load() {
        // Older configs default advanced tracking to off.
        let legacy_json = r#"{"ui":{"theme":"Dark","restore_workspace":false}}"#;
        let settings: ViewerSettings = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(settings.ui.theme, Theme::Dark);
        assert!(!settings.ui.restore_workspace);
        assert!(!settings.ui.advanced_tracking);

        let legacy_ron = "(ui:(theme:Dark,restore_workspace:false),)";
        let settings: ViewerSettings = ron::de::from_str(legacy_ron).unwrap();
        assert_eq!(settings.ui.theme, Theme::Dark);
        assert!(!settings.ui.advanced_tracking);
    }

    #[test]
    fn advanced_tracking_round_trips_when_enabled() {
        let mut settings = ViewerSettings::default();
        settings.ui.advanced_tracking = true;
        let ron_text = ron::ser::to_string(&settings).unwrap();
        let back: ViewerSettings = ron::de::from_str(&ron_text).unwrap();
        assert!(back.ui.advanced_tracking);
        let json_text = serde_json::to_string(&settings).unwrap();
        let back: ViewerSettings = serde_json::from_str(&json_text).unwrap();
        assert!(back.ui.advanced_tracking);
    }

    #[test]
    fn raw_measurements_are_plotted_by_default() {
        assert!(!PlotSettings::default().smooth);
    }

    #[test]
    fn relight_settings_from_older_builds_still_load() {
        // Flat light fields and the removed `advanced_visualization` flag are
        // ignored; everything else is kept.
        let legacy = "(ui:(theme:Dark,advanced_visualization:true),\
            images:(relight:(enabled:true,azimuth_deg:10.0,elevation_deg:20.0,\
            intensity:2.0,ambient:0.3)),)";
        let settings: ViewerSettings = ron::de::from_str(legacy).unwrap();
        assert_eq!(settings.ui.theme, Theme::Dark);
        assert!(settings.images.relight.enabled);
        assert_eq!(settings.images.relight.light, Light::default());
    }

    #[test]
    fn relight_light_round_trips() {
        let mut settings = ViewerSettings::default();
        settings.images.relight.light.azimuth_deg = 200.0;
        settings.images.relight.enabled = true;
        let text = ron::ser::to_string(&settings).unwrap();
        let back: ViewerSettings = ron::de::from_str(&text).unwrap();
        assert_eq!(back.images.relight.light.azimuth_deg, 200.0);
        assert!(back.images.relight.enabled);
    }

    #[test]
    fn the_removed_manual_indentation_preference_in_old_settings_is_ignored() {
        let older: ViewerSettings =
            ron::de::from_str("(ui:(theme:Dark,manual_indentation:true),)").unwrap();
        assert_eq!(older.ui.theme, Theme::Dark);
    }

    #[test]
    fn quantity_extraction_matches_curve_points() {
        let p = fabric_data::CurvePoint {
            frame_id: 0,
            displacement_mm: 1.0,
            force_n: Some(2.0),
            line_load_n_per_m: Some(10.0),
            is_relaxation_hold: false,
        };
        assert_eq!(PlotQuantity::ForceN.of(&p), Some(2.0));
        assert_eq!(PlotQuantity::LineLoad.of(&p), Some(10.0));
    }
}
