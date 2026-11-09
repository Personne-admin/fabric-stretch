//! Press View controls and persistence. Scene building lives in `press_sim`;
//! geometry and rendering live in `press_geometry`, `press_gpu` and `press_render`.

use crate::lighting::{self, Light};
use crate::material_package::{
    ForceFit, MaterialPackage, PackageSet, PressBranch, TexturePlan, VariantChoice,
};
use crate::press_geometry::{near_segment, PackageGeometry, ViewMode};
use crate::press_gpu::GpuTexture;
use crate::press_render::{
    ClothScene, CpuJob, CpuWorker, OverlayItem, RgbaImage, SceneFiles, TexKind,
};
use crate::press_sim::{build_scene, Control, PressScene, SceneRequest, BETA_RANGE};
use crate::press_store::{PackageStore, Textures};
use crate::project::ProjectSlot;
use egui::{Color32, CursorIcon, Pos2, Rect, Sense, Stroke, Vec2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

/// What the slider of the second row drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMode {
    /// Strip tension in N, mapped to a stretch by the force fit.
    Force,
    /// Bar indentation as a percentage of the clamp span.
    Indentation,
}

/// Largest side, in pixels, of an image rendered on the CPU fallback path.
const MAX_CPU_SIDE: f32 = 1600.0;

/// Span (mm) in which old viewer files stored the bar position and depth.
const LEGACY_SPAN_MM: f32 = 200.0;

/// Press View settings saved in the dock layout and `.fvproj` files.
/// Legacy `position_mm` / `depth_mm` assume a 200 mm span; `manual_indentation`
/// selects Indentation control.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PersistedPressView {
    /// Source path of the project this view shows (matched against the open projects on restore).
    pub project: Option<PathBuf>,
    /// Package set id (`v3`, `v2`); `None` follows the default (newest).
    pub package_set: Option<String>,
    pub orientation: String,
    pub branch: String,
    /// The Tiled texture is requested (Untiled otherwise).
    pub tiling: bool,
    pub relight: bool,
    pub light: Light,
    pub top_view: bool,
    /// `None` chooses Force when the force fit is reliable, else Indentation.
    pub control: Option<ControlMode>,
    /// Strip tension in N. Not clamped to zero: tared tension can be negative on old packages.
    pub force_n: Option<f32>,
    /// Indentation as a fraction of the clamp span.
    pub indentation: Option<f32>,
    /// Bar position as a fraction of the clamp span.
    pub beta: Option<f32>,
    #[serde(skip_serializing)]
    pub manual_indentation: bool,
    #[serde(skip_serializing)]
    pub position_mm: Option<[f32; 2]>,
    #[serde(skip_serializing)]
    pub depth_mm: Option<f32>,
}

impl Default for PersistedPressView {
    fn default() -> Self {
        Self {
            project: None,
            package_set: None,
            orientation: "warp".into(),
            branch: "loading".into(),
            tiling: true,
            relight: false,
            light: Light::default(),
            top_view: false,
            control: None,
            force_n: None,
            indentation: None,
            beta: None,
            manual_indentation: false,
            position_mm: None,
            depth_mm: None,
        }
    }
}

/// Read-only summary of one view's state, for tests and diagnostics.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct PressSnapshot {
    pub project: Option<PathBuf>,
    pub orientation: String,
    pub branch: PressBranch,
    pub force_n: Option<f32>,
    pub beta: f32,
    pub lambda: Option<f32>,
    pub indentation: Option<f32>,
    pub tiled: bool,
    pub control: Option<ControlMode>,
}

/// The CPU fallback: a background worker and the last image it delivered.
#[derive(Default)]
struct CpuRender {
    worker: Option<CpuWorker>,
    submitted: Option<(ClothScene, Rect, f32)>,
    next_id: u64,
    texture: Option<egui::TextureHandle>,
    error: Option<String>,
}

/// Independent view state; packages and textures are shared through [`PackageStore`].
pub struct PressTab {
    id: u32,
    project_root: Option<PathBuf>,
    package_set: Option<String>,
    orientation: String,
    branch: PressBranch,
    tiled: bool,
    relight: bool,
    light: Light,
    view: ViewMode,
    /// `None` = automatic (Force when the fit is reliable).
    control: Option<ControlMode>,
    force_n: Option<f32>,
    indentation: Option<f32>,
    beta: f32,
    /// Stretch of the previous frame, to stay put on flat parts of the force fit.
    lambda: Option<f32>,
    /// Mode the last frame was drawn in, to hand the state over when the mode changes.
    last_mode: Option<ControlMode>,
    dragging_bar: bool,
    cpu: CpuRender,
    /// The textures of the last complete GPU draw, shown while the next ones load.
    gpu_last: Option<Vec<Arc<GpuTexture>>>,
    /// Fraction of the force range scripted by the screenshot recorder.
    #[cfg(feature = "demo")]
    demo_force: Option<f32>,
}

impl Default for PressTab {
    fn default() -> Self {
        Self::new()
    }
}

/// Requests a view makes of its host while drawing.
#[derive(Debug, Default, Clone, Copy)]
pub struct PressUiOutput {
    pub add_view: bool,
}

/// Default bar position.
const DEFAULT_BETA: f32 = 0.5;
/// Default indentation (fraction of the span) of a fresh view.
const DEFAULT_INDENTATION: f32 = 0.12;

impl PressTab {
    pub fn new() -> Self {
        Self {
            id: 0,
            project_root: None,
            package_set: None,
            orientation: "warp".into(),
            branch: PressBranch::Loading,
            tiled: true,
            relight: false,
            light: Light::default(),
            view: ViewMode::Oblique,
            control: None,
            force_n: None,
            indentation: None,
            beta: DEFAULT_BETA,
            lambda: None,
            last_mode: None,
            dragging_bar: false,
            cpu: CpuRender::default(),
            gpu_last: None,
            #[cfg(feature = "demo")]
            demo_force: None,
        }
    }

    #[cfg(test)]
    pub fn project_root(&self) -> Option<&Path> {
        self.project_root.as_deref()
    }

    /// Scripts the view for the screenshot recorder: tension as a fraction of the fit range.
    #[cfg(feature = "demo")]
    pub fn set_demo_state(&mut self, force_fraction: f32, beta: f32, branch: PressBranch) {
        self.branch = branch;
        self.control = Some(ControlMode::Force);
        self.demo_force = Some(force_fraction.clamp(0.0, 1.0));
        self.beta = beta.clamp(BETA_RANGE.0, BETA_RANGE.1);
    }

    #[cfg(test)]
    pub fn snapshot(&self) -> PressSnapshot {
        PressSnapshot {
            project: self.project_root.clone(),
            orientation: self.orientation.clone(),
            branch: self.branch,
            force_n: self.force_n,
            beta: self.beta,
            lambda: self.lambda,
            indentation: self.indentation,
            tiled: self.tiled,
            control: self.control,
        }
    }

    /// Switches this view to another project and restores its default press state.
    pub fn select_project(&mut self, root: PathBuf) {
        self.project_root = Some(root);
        self.package_set = None;
        self.reset_press_state();
    }

    /// Default bar, tension, indentation and branch; the project and the display choices stay.
    fn reset_press_state(&mut self) {
        self.branch = PressBranch::Loading;
        self.control = None;
        self.force_n = None;
        self.indentation = None;
        self.beta = DEFAULT_BETA;
        self.lambda = None;
        self.last_mode = None;
        self.dragging_bar = false;
    }

    pub fn persisted(&self, slots: &[ProjectSlot]) -> PersistedPressView {
        PersistedPressView {
            project: self
                .project_root
                .as_ref()
                .and_then(|root| slots.iter().find(|s| &s.root_key == root))
                .map(|s| s.project.root.clone()),
            package_set: self.package_set.clone(),
            orientation: self.orientation.clone(),
            branch: self.branch.name().into(),
            tiling: self.tiled,
            relight: self.relight,
            light: self.light,
            top_view: self.view == ViewMode::Top,
            control: self.control,
            force_n: self.force_n,
            indentation: self.indentation,
            beta: Some(self.beta),
            ..PersistedPressView::default()
        }
    }

    pub fn restore(&mut self, s: PersistedPressView, slots: &[ProjectSlot]) {
        self.reset_press_state();
        self.project_root = s.project.as_ref().and_then(|source| {
            let key = crate::selection::canonical_key(source);
            slots
                .iter()
                .find(|slot| slot.root_key == key)
                .map(|slot| slot.root_key.clone())
        });
        self.package_set = s.package_set;
        let axis = usize::from(s.orientation == "weft");
        self.orientation = s.orientation;
        self.branch = PressBranch::from_name(&s.branch);
        self.tiled = s.tiling;
        self.relight = s.relight;
        self.light = s.light;
        self.view = if s.top_view {
            ViewMode::Top
        } else {
            ViewMode::Oblique
        };
        self.control = s
            .control
            .or(s.manual_indentation.then_some(ControlMode::Indentation));
        self.force_n = s.force_n.filter(|f| f.is_finite());
        self.indentation = s
            .indentation
            .or(s.depth_mm.map(|d| d / LEGACY_SPAN_MM))
            .filter(|d| d.is_finite())
            .map(|d| d.max(0.0));
        let beta = s
            .beta
            .or(s.position_mm.map(|p| p[axis] / LEGACY_SPAN_MM))
            .filter(|b| b.is_finite())
            .unwrap_or(DEFAULT_BETA);
        self.beta = beta.clamp(BETA_RANGE.0, BETA_RANGE.1);
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        slots: &[ProjectSlot],
        default_project: Option<&Path>,
        store: &mut PackageStore,
    ) -> PressUiOutput {
        let mut out = PressUiOutput::default();
        if slots.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("Load a project to use Press View.");
            });
            return out;
        }
        if self.project_root.is_none() {
            let fallback = default_project
                .and_then(|p| slots.iter().find(|s| s.root_key == p))
                .or_else(|| slots.first())
                .map(|s| s.root_key.clone());
            if let Some(root) = fallback {
                self.select_project(root);
            }
        }
        let slot = slots
            .iter()
            .find(|s| Some(&s.root_key) == self.project_root.as_ref());

        let sets: Vec<PackageSet> = slot
            .map(|s| store.sets(&s.root_key).to_vec())
            .unwrap_or_default();
        let set = self
            .package_set
            .as_deref()
            .and_then(|id| sets.iter().find(|s| s.id == id))
            .or_else(|| sets.first());
        let orientation = set.and_then(|set| {
            set.orientation(&self.orientation)
                .or_else(|| set.orientations.first())
        });
        let variant = orientation.map(|o| o.choose(self.tiled));
        let package: Option<Result<Rc<MaterialPackage>, String>> = variant
            .as_ref()
            .and_then(|v| v.dir.as_ref())
            .map(|dir| store.package(dir));

        let actions = self.controls_row(
            ui,
            slots,
            slot,
            &sets,
            set,
            orientation.map(|o| o.name.as_str()),
            variant.as_ref(),
            package.as_ref(),
            &mut out,
        );
        let problems: Vec<String> = slot
            .map(|s| store.problems(&s.root_key).to_vec())
            .unwrap_or_default();
        if let Some(root) = actions.project {
            if self.project_root.as_ref() != Some(&root) {
                self.select_project(root);
            }
        }
        if let (Some(root), true) = (&self.project_root, actions.rescan) {
            store.rescan(root);
        }
        if actions.reset {
            self.reset_press_state();
        }
        if actions.changed {
            ui.ctx().request_repaint();
        }

        let Some(slot) = slot else {
            ui.label("The project for this view was closed. Choose another project.");
            return out;
        };
        let (Some(set), Some(orientation), Some(variant)) = (set, orientation, variant.as_ref())
        else {
            ui.label(
                "No material package for this project (looked for material_package_v3 and material_package). \
                 Build one with `fabsim package`, then press Rescan.",
            );
            show_problems(ui, &problems);
            if ui.button("Rescan").clicked() {
                store.rescan(&slot.root_key);
            }
            return out;
        };
        show_problems(ui, &problems);
        match package {
            Some(Ok(package)) => {
                self.package_body(ui, store, &package, set, orientation.name.as_str(), variant)
            }
            Some(Err(error)) => {
                ui.colored_label(
                    Color32::YELLOW,
                    format!("Could not load the package: {error}"),
                );
                if ui.button("Rescan").clicked() {
                    store.rescan(&slot.root_key);
                }
            }
            None => {
                ui.label("This orientation has no package.");
            }
        }
        out
    }

    /// Apply local controls and return actions that need the package store.
    #[allow(clippy::too_many_arguments)]
    fn controls_row(
        &mut self,
        ui: &mut egui::Ui,
        slots: &[ProjectSlot],
        slot: Option<&ProjectSlot>,
        sets: &[PackageSet],
        set: Option<&PackageSet>,
        orientation: Option<&str>,
        variant: Option<&VariantChoice>,
        package: Option<&Result<Rc<MaterialPackage>, String>>,
        out: &mut PressUiOutput,
    ) -> RowActions {
        let mut actions = RowActions::default();
        ui.horizontal_wrapped(|ui| {
            let selected = slot.map_or("Choose project...", |s| s.name.as_str());
            egui::ComboBox::from_id_salt(("press-project", self.id))
                .selected_text(selected)
                .width(130.0)
                .show_ui(ui, |ui| {
                    for candidate in slots {
                        let is_current = Some(&candidate.root_key) == self.project_root.as_ref();
                        if ui.selectable_label(is_current, &candidate.name).clicked() {
                            actions.project = Some(candidate.root_key.clone());
                        }
                    }
                });
            if sets.len() > 1 {
                egui::ComboBox::from_id_salt(("press-set", self.id))
                    .selected_text(format!("Package {}", set.map_or("", |s| s.label)))
                    .show_ui(ui, |ui| {
                        for candidate in sets {
                            let current = set.is_some_and(|s| s.id == candidate.id);
                            if ui.selectable_label(current, candidate.label).clicked() {
                                self.package_set = Some(candidate.id.to_owned());
                                actions.changed = true;
                            }
                        }
                    })
                    .response
                    .on_hover_text("Package generation: v3 is the current format, v2 is read for older projects");
            }
            ui.separator();
            if let Some(set) = set {
                for dirs in &set.orientations {
                    let active = orientation == Some(dirs.name.as_str());
                    if ui
                        .add(egui::Button::selectable(active, capitalised(&dirs.name)))
                        .clicked()
                    {
                        self.orientation = dirs.name.clone();
                        actions.changed = true;
                    }
                }
            }
            ui.separator();
            for branch in [PressBranch::Loading, PressBranch::Release] {
                if ui
                    .add(egui::Button::selectable(self.branch == branch, branch.label()))
                    .clicked()
                {
                    self.branch = branch;
                }
            }
            ui.separator();
            ui.label("Texture");
            let tiled_now = variant.is_some_and(|v| v.tiled);
            let (tiled_ok, untiled_ok) = variant.map_or((false, false), |v| {
                (v.tiled_available, v.untiled_available)
            });
            let tiled_button = ui.add_enabled(
                tiled_ok,
                egui::Button::selectable(tiled_ok && tiled_now, "Tiled"),
            );
            let tiled_button = if tiled_ok {
                tiled_button.on_hover_text(
                    "Seamless periodic texture (findTile), repeated over the sheet",
                )
            } else {
                tiled_button.on_disabled_hover_text(
                    "No tiled package for this orientation (build one without --no-tiling)",
                )
            };
            if tiled_button.clicked() {
                self.tiled = true;
            }
            let untiled_button = ui.add_enabled(
                untiled_ok,
                egui::Button::selectable(untiled_ok && !tiled_now, "Untiled"),
            );
            let untiled_button = if untiled_ok {
                untiled_button.on_hover_text("The registered atlas as measured, drawn once")
            } else {
                untiled_button.on_disabled_hover_text(
                    "No untiled package for this orientation (build one with --untiled-out)",
                )
            };
            if untiled_button.clicked() {
                self.tiled = false;
            }
            ui.separator();
            let photometric = relight_availability(package);
            let relight_ok = matches!(photometric, Some(Ok(_)));
            let relight_button = ui.add_enabled(
                relight_ok,
                egui::Button::selectable(relight_ok && self.relight, "Relight"),
            );
            let relight_button = match photometric {
                Some(Ok(_)) => relight_button.on_hover_text(
                    "Shade the photometric albedo and normals with a movable light",
                ),
                Some(Err(reason)) => relight_button.on_disabled_hover_text(reason),
                None => relight_button.on_disabled_hover_text("No package for this orientation"),
            };
            if relight_button.clicked() {
                self.relight = !self.relight;
            }
            ui.separator();
            for view in [ViewMode::Oblique, ViewMode::Top] {
                if ui
                    .add(egui::Button::selectable(self.view == view, view.label()))
                    .clicked()
                {
                    self.view = view;
                }
            }
            ui.separator();
            if ui
                .button("Reset")
                .on_hover_text("Default press state; also looks for new or rebuilt packages")
                .clicked()
            {
                actions.reset = true;
                actions.rescan = true;
            }
            if ui
                .button("+ View")
                .on_hover_text("Open another Press View for side-by-side comparison")
                .clicked()
            {
                out.add_view = true;
            }
        });
        actions
    }

    /// Everything below the first row for a loaded package.
    fn package_body(
        &mut self,
        ui: &mut egui::Ui,
        store: &mut PackageStore,
        package: &Rc<MaterialPackage>,
        set: &PackageSet,
        orientation: &str,
        variant: &VariantChoice,
    ) {
        let branch = self.branch;
        let Some((_, lambda_max)) = package.stretch_range(branch) else {
            ui.label(format!(
                "No measured {} states in this package.",
                branch.name()
            ));
            return;
        };
        let fit = package.force_fit(branch).filter(|f| {
            let (lo, hi) = f.range();
            hi > lo + 1.0e-4
        });
        let fit_problem = fit.and_then(ForceFit::degenerate);
        let mode = self.effective_mode(fit, fit_problem.is_some());
        self.hand_over_mode(mode, fit);

        // Clamp state after switching branches or packages.
        self.beta = self.beta.clamp(BETA_RANGE.0, BETA_RANGE.1);
        let max_delta = PackageGeometry::max_delta(self.beta, lambda_max);
        if let Some(fit) = fit {
            let (lo, hi) = fit.range();
            #[cfg(feature = "demo")]
            if let Some(fraction) = self.demo_force {
                self.force_n = Some(lo + fraction * (hi - lo));
            }
            let default = lo + 0.4 * (hi - lo);
            self.force_n = Some(
                self.force_n
                    .unwrap_or(default)
                    .clamp(lo.min(hi), hi.max(lo)),
            );
        }
        self.indentation = Some(
            self.indentation
                .unwrap_or(DEFAULT_INDENTATION)
                .clamp(0.0, max_delta),
        );

        let relight_light = (self.relight && package.photometric().is_ok()).then_some(self.light);
        self.mode_row(
            ui,
            package,
            fit,
            fit_problem.as_deref(),
            mode,
            max_delta,
            relight_light.is_some(),
        );

        let status_height = ui.text_style_height(&egui::TextStyle::Body) + 8.0;
        let free = ui.available_rect_before_wrap();
        if free.width() < 120.0 || free.height() < status_height + 80.0 {
            ui.label("Press view needs more space; enlarge the tab.");
            return;
        }
        let canvas = Rect::from_min_max(free.min, free.max - Vec2::new(0.0, status_height));
        let response = ui.allocate_rect(canvas, Sense::click_and_drag());

        let control = match mode {
            ControlMode::Force => Control::Force(self.force_n.unwrap_or(0.0)),
            ControlMode::Indentation => Control::Indentation(self.indentation.unwrap_or(0.0)),
        };
        let request = SceneRequest {
            branch,
            control,
            beta: self.beta,
            previous_lambda: self.lambda.unwrap_or(1.0),
            view: self.view,
            relight: relight_light,
        };
        let scene = match build_scene(package, &request, canvas) {
            Ok(scene) => scene,
            Err(error) => {
                ui.colored_label(Color32::YELLOW, error);
                return;
            }
        };
        match scene.solved.control {
            Control::Force(f) => self.force_n = Some(f),
            Control::Indentation(d) => self.indentation = Some(d),
        }
        if mode == ControlMode::Force {
            self.indentation = Some(scene.solved.geometry.delta);
        }
        self.lambda = Some(scene.solved.geometry.lambda);

        self.interact_with_bar(ui, &response, &scene);
        self.draw_cloth(ui, store, canvas, package, &scene);
        let painter = ui.painter_at(canvas);
        for item in &scene.overlay {
            match *item {
                OverlayItem::Disc {
                    center,
                    radius,
                    color,
                } => {
                    painter.circle_filled(center, radius, color);
                }
                OverlayItem::Line { a, b, width, color } => {
                    painter.line_segment([a, b], Stroke::new(width, color));
                }
            }
        }
        ui.label(status_line(
            &scene,
            package,
            set.label,
            orientation,
            variant.tiled,
            relight_light.is_some(),
            mode,
        ));
    }

    /// The control mode in effect: the user's choice, or Force when the fit is reliable.
    fn effective_mode(&self, fit: Option<&ForceFit>, fit_degenerate: bool) -> ControlMode {
        match (self.control, fit) {
            (Some(ControlMode::Force), Some(_)) => ControlMode::Force,
            (Some(_), _) => ControlMode::Indentation,
            (None, Some(_)) if !fit_degenerate => ControlMode::Force,
            (None, _) => ControlMode::Indentation,
        }
    }

    /// When the mode changes, the new control starts at the state the old one produced.
    fn hand_over_mode(&mut self, mode: ControlMode, fit: Option<&ForceFit>) {
        if self.last_mode.is_some_and(|last| last != mode) {
            match mode {
                ControlMode::Indentation => {
                    if let Some(lambda) = self.lambda {
                        self.indentation = Some(delta_for_lambda(self.beta, lambda));
                    }
                }
                ControlMode::Force => {
                    if let (Some(fit), Some(lambda)) = (fit, self.lambda) {
                        self.force_n = Some(fit.force_at(lambda));
                    }
                }
            }
        }
        self.last_mode = Some(mode);
    }

    /// Second row: control mode, its slider, and the light pad when relighting.
    #[allow(clippy::too_many_arguments)]
    fn mode_row(
        &mut self,
        ui: &mut egui::Ui,
        package: &MaterialPackage,
        fit: Option<&ForceFit>,
        fit_problem: Option<&str>,
        mode: ControlMode,
        max_delta: f32,
        relight: bool,
    ) {
        ui.horizontal_wrapped(|ui| {
            let force_button = ui.add_enabled(
                fit.is_some(),
                egui::Button::selectable(mode == ControlMode::Force, "Force"),
            );
            let force_button = if fit.is_some() {
                force_button.on_hover_text("Drive the press by the strip tension, through the monotone fit of the measured curve")
            } else {
                force_button.on_disabled_hover_text(
                    "This branch has no usable force curve (fewer than two distinct stretches or no tension change)",
                )
            };
            if force_button.clicked() {
                self.control = Some(ControlMode::Force);
            }
            if ui
                .add(egui::Button::selectable(
                    mode == ControlMode::Indentation,
                    "Indentation",
                ))
                .on_hover_text("Drive the press by how deep the bar pushes the sheet in")
                .clicked()
            {
                self.control = Some(ControlMode::Indentation);
            }
            match mode {
                ControlMode::Force => {
                    if let (Some(fit), Some(force)) = (fit, self.force_n.as_mut()) {
                        let (lo, hi) = fit.range();
                        let hover = tension_hover(package);
                        ui.add(
                            egui::Slider::new(force, lo.min(hi)..=hi.max(lo))
                                .suffix(" N")
                                .text("sheet tension (strip test)"),
                        )
                        .on_hover_text(hover);
                    }
                }
                ControlMode::Indentation => {
                    let mut percent = self.indentation.unwrap_or(0.0) * 100.0;
                    let response = ui.add(
                        egui::Slider::new(&mut percent, 0.0..=(max_delta * 100.0).max(0.01))
                            .suffix(" % of span")
                            .text("indentation"),
                    );
                    if response.changed() {
                        self.indentation = Some(percent / 100.0);
                    }
                }
            }
            if let Some(problem) = fit_problem {
                let driven = if mode == ControlMode::Indentation {
                    " Indentation drives the press."
                } else {
                    ""
                };
                ui.colored_label(
                    Color32::YELLOW,
                    format!("\u{26a0} Force curve unreliable: {problem}.{driven}"),
                );
            }
            if relight {
                lighting::light_pad(ui, &mut self.light);
            }
        });
    }

    /// Bar dragging: grab the bar, follow the pointer, keep the bar inside the sheet.
    fn interact_with_bar(&mut self, ui: &egui::Ui, response: &egui::Response, scene: &PressScene) {
        let bar = scene.projection.bar(&scene.solved.geometry);
        let pointer = response.hover_pos();
        if response.drag_started() {
            let start = ui.input(|i| i.pointer.press_origin());
            self.dragging_bar = start.is_some_and(|p| near_segment(p, bar));
        }
        if self.dragging_bar && response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let beta = scene.projection.pull_at_x(pos.x);
                self.beta = beta.clamp(BETA_RANGE.0, BETA_RANGE.1);
                ui.ctx().request_repaint();
            }
        }
        if response.drag_stopped() {
            self.dragging_bar = false;
        }
        if self.dragging_bar {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        } else if pointer.is_some_and(|p| near_segment(p, bar)) {
            ui.ctx().set_cursor_icon(CursorIcon::Grab);
        }
    }

    /// Draws the cloth with the GPU when there is a render state, else from the CPU worker.
    fn draw_cloth(
        &mut self,
        ui: &mut egui::Ui,
        store: &mut PackageStore,
        canvas: Rect,
        package: &MaterialPackage,
        scene: &PressScene,
    ) {
        let painter = ui.painter_at(canvas);
        if store.gpu().is_some() {
            let expected = match scene.cloth.files {
                SceneFiles::Rgb(_) => 2,
                SceneFiles::Relight { .. } => 4,
            };
            let message = |text: &str| {
                painter.text(
                    canvas.center(),
                    egui::Align2::CENTER_CENTER,
                    text,
                    egui::FontId::proportional(14.0),
                    Color32::YELLOW,
                );
            };
            let textures = match store.gpu_textures(ui.ctx(), &scene.cloth.files) {
                Ok(Textures::Ready(textures)) => {
                    self.gpu_last = Some(textures.clone());
                    prefetch_neighbours(ui.ctx(), store, package, &scene.plan, expected == 4);
                    Some(textures)
                }
                // Keep showing the last complete pair (blended with the new weight) meanwhile.
                Ok(Textures::Loading) => {
                    self.gpu_last.clone().filter(|last| last.len() == expected)
                }
                Err(error) => {
                    self.gpu_last = None;
                    message(&error);
                    return;
                }
            };
            match (textures, store.gpu()) {
                (Some(textures), Some(gpu)) => {
                    painter.add(gpu.draw(canvas, &scene.cloth, textures));
                }
                _ => message("Loading textures..."),
            }
            return;
        }
        let ppp = ui.ctx().pixels_per_point();
        let scale = ppp.min(MAX_CPU_SIDE / canvas.width().max(canvas.height()).max(1.0));
        let size = [
            (canvas.width() * scale).ceil().max(1.0) as u32,
            (canvas.height() * scale).ceil().max(1.0) as u32,
        ];
        let ctx = ui.ctx().clone();
        let cpu = &mut self.cpu;
        let worker = cpu.worker.get_or_insert_with(|| {
            let wake = ctx.clone();
            CpuWorker::spawn(move || wake.request_repaint())
        });
        let key = (scene.cloth.clone(), canvas, scale);
        if cpu.submitted.as_ref() != Some(&key) {
            cpu.next_id += 1;
            worker.submit(CpuJob {
                id: cpu.next_id,
                scene: scene.cloth.clone(),
                origin: canvas.min,
                scale,
                size,
            });
            cpu.submitted = Some(key);
        }
        if let Some(result) = worker.poll() {
            match result.image {
                Ok(image) => {
                    cpu.error = None;
                    let color = color_image(&image);
                    match cpu.texture.as_mut() {
                        Some(texture) => texture.set(color, egui::TextureOptions::LINEAR),
                        None => {
                            cpu.texture = Some(ctx.load_texture(
                                "press-cpu",
                                color,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }
                }
                Err(error) => cpu.error = Some(error),
            }
        }
        if let Some(error) = &cpu.error {
            painter.text(
                canvas.center(),
                egui::Align2::CENTER_CENTER,
                error,
                egui::FontId::proportional(14.0),
                Color32::YELLOW,
            );
        } else if let Some(texture) = &cpu.texture {
            painter.image(
                texture.id(),
                canvas,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }
}

/// Actions collected from the first row.
#[derive(Default)]
struct RowActions {
    project: Option<PathBuf>,
    reset: bool,
    rescan: bool,
    changed: bool,
}

/// Starts decoding the states next to the bracketing pair, so slow scrubbing finds them ready.
fn prefetch_neighbours(
    ctx: &egui::Context,
    store: &mut PackageStore,
    package: &MaterialPackage,
    plan: &TexturePlan,
    relit: bool,
) {
    let states = package.states(plan.branch);
    let below = plan.lo.checked_sub(1);
    let above = Some(plan.hi + 1).filter(|i| *i < states.len());
    let mut paths: Vec<PathBuf> = Vec::new();
    for state in [below, above].into_iter().flatten().map(|i| &states[i]) {
        if relit {
            paths.extend(state.albedo.iter().chain(&state.normal).cloned());
        } else {
            paths.push(state.texture.clone());
        }
    }
    let kind = if relit {
        TexKind::Unorm16
    } else {
        TexKind::Srgb8
    };
    store.prefetch(ctx, &paths, kind);
}

/// Relight availability and failure reason; `None` without a package.
fn relight_availability(
    package: Option<&Result<Rc<MaterialPackage>, String>>,
) -> Option<Result<(), String>> {
    package.map(|p| match p {
        Ok(p) => p.photometric().map(|_| ()).map_err(str::to_owned),
        Err(_) => Err("The package could not be loaded".to_owned()),
    })
}

/// Lists the package directories discovery had to skip, with the reason.
fn show_problems(ui: &mut egui::Ui, problems: &[String]) {
    for problem in problems {
        ui.colored_label(Color32::YELLOW, format!("Skipped {problem}"));
    }
}

fn color_image(image: &RgbaImage) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied(
        [image.width as usize, image.height as usize],
        &image.pixels,
    )
}

fn capitalised(name: &str) -> String {
    let mut chars = name.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
        .unwrap_or_default()
}

/// Indentation (fraction of the span) that produces stretch `lambda` with the bar at `beta`.
fn delta_for_lambda(beta: f32, lambda: f32) -> f32 {
    PackageGeometry::max_delta(beta, lambda)
}

fn tension_hover(package: &MaterialPackage) -> String {
    let mut text = String::from(
        "Tension of the strip test: the load-cell reading of the whole sheet, zero at the \
         unloaded state. It is the sheet tension that stretches the fabric to the shown state, \
         not the force of the bar.",
    );
    if let Some(offset) = package.legacy_tare_n() {
        text.push_str(&format!(
            " This is a version {} package, which stores the raw reading; the viewer set the \
             zero to the reading at the first loading state ({offset:.2} N, the release branch \
             uses the same zero).",
            package.version()
        ));
    }
    if let Some(w) = package.clamp_width_m().filter(|w| *w > 0.0) {
        text.push_str(&format!(
            " Over the {:.0} mm clamp width, 1 N is {:.1} N/m.",
            w * 1000.0,
            1.0 / w
        ));
    }
    text
}

/// The status line: stretch, tension, indentation, bar, bracketing states and the package.
pub fn status_line(
    scene: &PressScene,
    package: &MaterialPackage,
    set_label: &str,
    orientation: &str,
    tiled: bool,
    relit: bool,
    mode: ControlMode,
) -> String {
    let g = &scene.solved.geometry;
    let force = scene.solved.fit_force.map_or_else(
        || "F n/a".to_owned(),
        |f| {
            format!(
                "F {f:.2} N{}",
                if mode == ControlMode::Force {
                    ""
                } else {
                    " (fit)"
                }
            )
        },
    );
    let frame = |i: Option<i64>| i.map_or_else(|| "?".to_owned(), |f| f.to_string());
    let plan = &scene.plan;
    let clamped = if plan.clamped {
        ", clamped to the nearest state"
    } else {
        ""
    };
    format!(
        "\u{3bb} {:.3} \u{b7} {force} \u{b7} indentation {:.1} % \u{b7} bar {:.0} % \u{b7} states {} / {} @ t={:.2}{clamped} \u{b7} {} {orientation}, {}{} (v{})",
        g.lambda,
        g.delta * 100.0,
        g.beta * 100.0,
        frame(plan.frames[0]),
        frame(plan.frames[1]),
        plan.t,
        set_label,
        if tiled { "tiled" } else { "untiled" },
        if relit { ", relit" } else { "" },
        package.version(),
    )
}

/// All open Press Views, keyed by the id carried by their dock tab.
#[derive(Default)]
pub struct PressViews {
    views: BTreeMap<u32, PressTab>,
    next_id: u32,
    /// A view asked for another one; the host (which owns the dock) acts on it.
    pub add_requested: bool,
}

impl PressViews {
    pub fn new() -> Self {
        Self::default()
    }
    #[cfg(feature = "demo")]
    pub fn views_mut(&mut self) -> impl Iterator<Item = &mut PressTab> {
        self.views.values_mut()
    }
    #[cfg(test)]
    pub fn ids(&self) -> Vec<u32> {
        self.views.keys().copied().collect()
    }
    #[cfg(test)]
    pub fn get(&self, id: u32) -> Option<&PressTab> {
        self.views.get(&id)
    }
    #[cfg(test)]
    pub fn get_mut(&mut self, id: u32) -> Option<&mut PressTab> {
        self.views.get_mut(&id)
    }
    /// A new independent view; it adopts `project` (the host's current project) when given.
    pub fn add(&mut self, project: Option<PathBuf>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.insert(id, project)
    }
    /// The view for a dock tab, created on demand (e.g. when a saved layout names an unknown id).
    pub fn get_or_create(&mut self, id: u32) -> &mut PressTab {
        self.next_id = self.next_id.max(id + 1);
        if !self.views.contains_key(&id) {
            self.insert(id, None);
        }
        self.views.get_mut(&id).expect("view was just inserted")
    }
    fn insert(&mut self, id: u32, project: Option<PathBuf>) -> u32 {
        let mut view = PressTab::new();
        view.id = id;
        if let Some(root) = project {
            view.select_project(root);
        }
        self.views.insert(id, view);
        id
    }
    pub fn remove(&mut self, id: u32) {
        self.views.remove(&id);
    }
    pub fn clear(&mut self) {
        self.views.clear();
        self.next_id = 0;
    }
    pub fn persisted(&self, slots: &[ProjectSlot]) -> Vec<PersistedPressView> {
        self.views.values().map(|v| v.persisted(slots)).collect()
    }
    /// Rebuilds the views from saved state; returns the new ids in saved order.
    pub fn restore(&mut self, saved: Vec<PersistedPressView>, slots: &[ProjectSlot]) -> Vec<u32> {
        self.clear();
        saved
            .into_iter()
            .map(|state| {
                let id = self.add(None);
                if let Some(view) = self.views.get_mut(&id) {
                    view.restore(state, slots);
                }
                id
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_package::test_support::demo_project;

    fn demo_slot() -> ProjectSlot {
        let root = demo_project();
        let project = fabric_data::Project::discover(&root).expect("demo project");
        ProjectSlot::new("demo".into(), project)
    }

    fn frame(
        view: &mut PressTab,
        slots: &[ProjectSlot],
        store: &mut PackageStore,
    ) -> egui::Context {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1000., 700.))),
            ..Default::default()
        };
        let _ = ctx.run_ui(input, |ui| {
            view.ui(ui, slots, None, store);
        });
        ctx
    }

    /// Runs a few frames so state that depends on the previous frame settles.
    fn settle(view: &mut PressTab, slots: &[ProjectSlot], store: &mut PackageStore) {
        for _ in 0..3 {
            frame(view, slots, store);
        }
    }

    #[test]
    fn views_keep_independent_state() {
        let mut views = PressViews::new();
        let (a, b) = (views.add(None), views.add(None));
        assert_ne!(a, b);
        let before = views.get(b).unwrap().snapshot();
        {
            let va = views.get_mut(a).unwrap();
            va.project_root = Some(PathBuf::from("/projects/one"));
            va.orientation = "weft".into();
            va.branch = PressBranch::Release;
            va.force_n = Some(7.5);
            va.beta = 0.15;
            va.tiled = false;
            va.control = Some(ControlMode::Indentation);
        }
        assert_eq!(
            views.get(b).unwrap().snapshot(),
            before,
            "view B changed with view A"
        );
        let sa = views.get(a).unwrap().snapshot();
        assert_eq!(sa.project, Some(PathBuf::from("/projects/one")));
        assert_eq!(
            (sa.orientation.as_str(), sa.branch),
            ("weft", PressBranch::Release)
        );
        assert_eq!((sa.force_n, sa.beta, sa.tiled), (Some(7.5), 0.15, false));
        assert!(views.get(b).unwrap().snapshot().tiled);
    }

    #[test]
    fn selecting_a_project_in_one_view_does_not_touch_another() {
        let mut views = PressViews::new();
        let (a, b) = (
            views.add(Some(PathBuf::from("/p/x"))),
            views.add(Some(PathBuf::from("/p/x"))),
        );
        views
            .get_mut(a)
            .unwrap()
            .select_project(PathBuf::from("/p/y"));
        assert_eq!(
            views.get(a).unwrap().project_root(),
            Some(Path::new("/p/y"))
        );
        assert_eq!(
            views.get(b).unwrap().project_root(),
            Some(Path::new("/p/x"))
        );
    }

    #[test]
    fn closing_a_view_leaves_the_others() {
        let mut views = PressViews::new();
        let (a, b, c) = (views.add(None), views.add(None), views.add(None));
        views.remove(b);
        assert_eq!(views.ids(), vec![a, c]);
        assert_eq!(views.add(None), c + 1, "ids are never reused");
    }

    #[test]
    fn persisted_views_round_trip_and_keep_their_own_settings() {
        let slots = vec![demo_slot()];
        let mut views = PressViews::new();
        let (a, b) = (views.add(Some(slots[0].root_key.clone())), views.add(None));
        {
            let va = views.get_mut(a).unwrap();
            va.orientation = "warp".into();
            va.branch = PressBranch::Release;
            va.tiled = false;
            va.relight = true;
            va.light = Light {
                azimuth_deg: 10.0,
                elevation_deg: 20.0,
                intensity: 1.5,
                ambient: 0.3,
            };
            va.view = ViewMode::Top;
            va.control = Some(ControlMode::Indentation);
            va.force_n = Some(-3.0);
            va.indentation = Some(0.2);
            va.beta = 0.3;
            va.package_set = Some("v3".into());
        }
        let saved = views.persisted(&slots);
        let json = serde_json::to_string(&saved).unwrap();
        for legacy in ["manual_indentation", "position_mm", "depth_mm", "scene"] {
            assert!(!json.contains(legacy), "{legacy} is no longer written");
        }
        let loaded: Vec<PersistedPressView> = serde_json::from_str(&json).unwrap();
        let mut restored = PressViews::new();
        let ids = restored.restore(loaded, &slots);
        assert_eq!(ids.len(), 2);
        let (ra, rb) = (restored.get(ids[0]).unwrap(), restored.get(ids[1]).unwrap());
        assert_eq!(
            ra.snapshot(),
            views.get(a).unwrap().snapshot(),
            "negative tension is kept"
        );
        assert_eq!(ra.package_set.as_deref(), Some("v3"));
        assert_eq!(
            (ra.relight, ra.light, ra.view),
            (true, views.get(a).unwrap().light, ViewMode::Top)
        );
        assert_eq!(rb.snapshot().project, None);
        assert_eq!(rb.snapshot(), PressTab::new().snapshot());
        let _ = b;
    }

    #[test]
    fn old_payloads_are_migrated() {
        // Format of earlier viewers: fictitious millimetres, a manual-indentation switch, scene.
        let old: PersistedPressView = serde_json::from_str(
            r#"{"force_n":3.0,"depth_mm":40.0,"width_mm":40.0,"position_mm":[60.0,150.0],
                "orientation":"weft","branch":"release","tiling":false,"tile_repeat":2.0,
                "manual_indentation":true,"relight":true,
                "light":{"azimuth_deg":45.0,"elevation_deg":45.0,"intensity":1.0,"ambient":0.15},
                "scene":{"surface":"Cylinder"}}"#,
        )
        .unwrap();
        let mut view = PressTab::new();
        view.restore(old, &[]);
        let s = view.snapshot();
        assert_eq!(s.orientation, "weft");
        assert_eq!(s.branch, PressBranch::Release);
        assert_eq!(s.control, Some(ControlMode::Indentation));
        assert_eq!(s.force_n, Some(3.0));
        assert!(
            (s.indentation.unwrap() - 0.2).abs() < 1.0e-6,
            "40 mm of 200 mm"
        );
        assert!(
            (s.beta - 0.75).abs() < 1.0e-6,
            "weft uses the second position: 150 of 200 mm"
        );
        assert!(!s.tiled && view.relight);
        let warp: PersistedPressView =
            serde_json::from_str(r#"{"position_mm":[60.0,150.0]}"#).unwrap();
        let mut view = PressTab::new();
        view.restore(warp, &[]);
        assert!((view.snapshot().beta - 0.3).abs() < 1.0e-6);
        let mut view = PressTab::new();
        view.restore(serde_json::from_str("{}").unwrap(), &[]);
        assert_eq!(view.snapshot(), PressTab::new().snapshot());
    }

    #[test]
    fn a_fresh_view_opens_in_force_mode_when_the_fit_is_reliable() {
        let slots = vec![demo_slot()];
        let mut store = PackageStore::default();
        let mut view = PressTab::new();
        settle(&mut view, &slots, &mut store);
        let s = view.snapshot();
        assert_eq!(s.control, None, "automatic");
        assert!(s.force_n.is_some() && s.lambda.unwrap() > 1.0, "{s:?}");
        let package = store
            .package(&demo_project().join("material_package_v3/warp"))
            .unwrap();
        let (lo, hi) = package.force_fit(PressBranch::Loading).unwrap().range();
        let force = s.force_n.unwrap();
        assert!(force > lo && force < hi);
        // The sheet is visibly pressed at the default tension.
        assert!(view.indentation.unwrap() > 0.02);
    }

    #[test]
    fn an_unreliable_force_fit_defaults_to_indentation_and_says_why() {
        let tab = PressTab::new();
        let noisy: Vec<crate::material_package::State> = (0..20)
            .map(|i| crate::material_package::State {
                frame_id: None,
                stretch_ratio: 1.0 + 0.01 * i as f32,
                force_n: 2.0 * i as f32 / 19.0 + if i % 2 == 0 { 1.5 } else { -1.5 },
                force_raw_n: None,
                travel_mm: None,
                texture: PathBuf::new(),
                albedo: None,
                normal: None,
            })
            .collect();
        let fit = ForceFit::from_states(&noisy).unwrap();
        let problem = fit.degenerate();
        assert!(problem.is_some());
        assert_eq!(
            tab.effective_mode(Some(&fit), true),
            ControlMode::Indentation
        );
        // The user can still pick Force explicitly.
        let mut forced = PressTab::new();
        forced.control = Some(ControlMode::Force);
        assert_eq!(forced.effective_mode(Some(&fit), true), ControlMode::Force);
        // Without any fit even an explicit Force request falls back.
        assert_eq!(forced.effective_mode(None, false), ControlMode::Indentation);
        assert_eq!(
            PressTab::new().effective_mode(None, false),
            ControlMode::Indentation
        );
    }

    #[test]
    fn switching_mode_hands_the_state_over() {
        let slots = vec![demo_slot()];
        let mut store = PackageStore::default();
        let mut view = PressTab::new();
        view.control = Some(ControlMode::Force);
        settle(&mut view, &slots, &mut store);
        let (lambda, delta) = (view.lambda.unwrap(), view.indentation.unwrap());
        view.control = Some(ControlMode::Indentation);
        settle(&mut view, &slots, &mut store);
        assert!(
            (view.indentation.unwrap() - delta).abs() < 2.0e-3,
            "indentation starts where force left it"
        );
        assert!((view.lambda.unwrap() - lambda).abs() < 2.0e-3);
        let force_before = view.force_n.unwrap();
        view.indentation = Some(delta * 0.5);
        settle(&mut view, &slots, &mut store);
        view.control = Some(ControlMode::Force);
        settle(&mut view, &slots, &mut store);
        assert!(
            view.force_n.unwrap() < force_before,
            "half the indentation needs less tension"
        );
    }

    #[test]
    fn branch_switches_keep_force_and_indentation_inside_the_new_range() {
        let slots = vec![demo_slot()];
        let mut store = PackageStore::default();
        let mut view = PressTab::new();
        view.control = Some(ControlMode::Force);
        settle(&mut view, &slots, &mut store);
        let package = store
            .package(&demo_project().join("material_package_v3/warp"))
            .unwrap();
        let loading = package.force_fit(PressBranch::Loading).unwrap().range();
        view.force_n = Some(loading.1 + 100.0);
        view.indentation = Some(10.0);
        view.branch = PressBranch::Release;
        settle(&mut view, &slots, &mut store);
        let release = package.force_fit(PressBranch::Release).unwrap().range();
        let force = view.force_n.unwrap();
        assert!(
            force <= release.1 + 1.0e-4 && force >= release.0 - 1.0e-4,
            "{force} not in {release:?}"
        );
        assert!(
            view.indentation.unwrap() < 1.0,
            "indentation is bounded by the stretch limit"
        );
        // A restored negative tension is not clamped to zero.
        let mut negative = PressTab::new();
        negative.restore(
            PersistedPressView {
                force_n: Some(-0.4),
                control: Some(ControlMode::Force),
                ..Default::default()
            },
            &[],
        );
        assert_eq!(negative.force_n, Some(-0.4));
    }

    #[test]
    fn toggling_the_texture_changes_only_the_texture_source() {
        let slots = vec![demo_slot()];
        let mut store = PackageStore::default();
        let mut view = PressTab::new();
        view.control = Some(ControlMode::Indentation);
        view.indentation = Some(0.15);
        settle(&mut view, &slots, &mut store);
        let tiled = view.snapshot();
        assert!(tiled.tiled && tiled.lambda.is_some());
        assert_eq!(store.counts().1, 1, "the tiled package is loaded");

        view.tiled = false;
        settle(&mut view, &slots, &mut store);
        let mut untiled = view.snapshot();
        assert_eq!(store.counts().1, 2, "the untiled package is loaded as well");
        untiled.tiled = true;
        assert_eq!(
            untiled, tiled,
            "lambda, bar and indentation survive the switch"
        );
    }

    #[test]
    fn relight_is_independent_of_the_texture_variant() {
        let slots = vec![demo_slot()];
        let mut store = PackageStore::default();
        for tiled in [true, false] {
            let mut view = PressTab::new();
            view.tiled = tiled;
            view.relight = true;
            settle(&mut view, &slots, &mut store);
            assert!(
                view.relight && view.snapshot().tiled == tiled,
                "no forced variant switch"
            );
        }
    }

    #[test]
    fn two_views_share_packages_and_parse_once() {
        let slots = vec![demo_slot()];
        let mut store = PackageStore::default();
        let mut views = PressViews::new();
        let (a, b) = (
            views.add(Some(slots[0].root_key.clone())),
            views.add(Some(slots[0].root_key.clone())),
        );
        for id in [a, b] {
            settle(views.get_mut(id).unwrap(), &slots, &mut store);
        }
        assert_eq!(store.counts().0, 1, "one discovery");
        assert_eq!(store.counts().1, 1, "one parsed package for both views");
        assert_eq!(
            views.get(a).unwrap().snapshot(),
            views.get(b).unwrap().snapshot()
        );
    }

    #[test]
    fn relight_explains_why_it_is_unavailable() {
        assert_eq!(relight_availability(None), None);
        let error = Err("boom".to_owned());
        assert_eq!(
            relight_availability(Some(&error)),
            Some(Err("The package could not be loaded".to_owned()))
        );
        let package = Rc::new(
            MaterialPackage::load(&demo_project().join("material_package_v3/warp")).unwrap(),
        );
        assert_eq!(relight_availability(Some(&Ok(package))), Some(Ok(())));
    }

    #[test]
    fn the_tension_hover_only_claims_a_tare_the_package_carries() {
        let v3 = MaterialPackage::load(&demo_project().join("material_package_v3/warp")).unwrap();
        let hover = tension_hover(&v3);
        assert!(
            hover.contains("zero at the unloaded state") && !hover.contains("raw"),
            "{hover}"
        );
        assert!(hover.contains("202 mm clamp width"), "{hover}");

        let dir = std::env::temp_dir().join(format!("press_tab_v2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(demo_project().join("material_package_v3_untiled/warp/package.json"))
                .unwrap(),
        )
        .unwrap();
        manifest["version"] = 2.into();
        manifest["ps"] = serde_json::Value::Null;
        // Point the textures at the demo files and add a raw offset.
        let src = demo_project().join("material_package_v3_untiled/warp");
        for branch in ["loading", "release"] {
            for s in manifest["branches"][branch].as_array_mut().unwrap() {
                let texture = src.join(s["texture"].as_str().unwrap());
                let name = texture.file_name().unwrap().to_owned();
                std::fs::copy(&texture, dir.join(&name)).unwrap();
                s["texture"] = name.to_string_lossy().into_owned().into();
                s["force_n"] = (s["force_n"].as_f64().unwrap() - 4.3).into();
            }
        }
        std::fs::write(dir.join("package.json"), manifest.to_string()).unwrap();
        let v2 = MaterialPackage::load(&dir).unwrap();
        let hover = tension_hover(&v2);
        assert!(
            hover.contains("version 2") && hover.contains("(-4.30 N"),
            "{hover}"
        );
    }

    #[test]
    fn a_project_without_packages_shows_the_unavailable_state() {
        let root = std::env::temp_dir().join(format!("press_tab_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Reuse a run of the demo as a project without any package.
        let run = root.join("warp_loading");
        std::fs::create_dir_all(&run).unwrap();
        for file in ["frames.json", "force.json", "session.json"] {
            std::fs::copy(
                demo_project().join("warp_loading").join(file),
                run.join(file),
            )
            .unwrap();
        }
        let Ok(project) = fabric_data::Project::discover(&root) else {
            return;
        };
        let slots = vec![ProjectSlot::new("empty".into(), project)];
        let mut store = PackageStore::default();
        let mut view = PressTab::new();
        settle(&mut view, &slots, &mut store);
        assert_eq!(view.snapshot().lambda, None, "no package, no simulation");
        assert_eq!(store.counts().1, 0);
    }

    #[test]
    fn the_status_line_reports_the_state() {
        let package =
            MaterialPackage::load(&demo_project().join("material_package_v3_untiled/warp"))
                .unwrap();
        let request = SceneRequest {
            branch: PressBranch::Loading,
            control: Control::Indentation(0.1),
            beta: 0.5,
            previous_lambda: 1.0,
            view: ViewMode::Oblique,
            relight: None,
        };
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 500.0));
        let scene = build_scene(&package, &request, canvas).unwrap();
        let text = status_line(
            &scene,
            &package,
            "v3",
            "warp",
            false,
            false,
            ControlMode::Indentation,
        );
        for part in [
            "\u{3bb} 1.",
            "(fit)",
            "indentation 10.0 %",
            "bar 50 %",
            "states ",
            "@ t=",
            "v3 warp, untiled",
            "(v3)",
        ] {
            assert!(text.contains(part), "{part:?} missing from {text:?}");
        }
        let forced = status_line(
            &scene,
            &package,
            "v3",
            "warp",
            true,
            true,
            ControlMode::Force,
        );
        assert!(forced.contains("tiled") && forced.contains("relit") && !forced.contains("(fit)"));
    }

    #[test]
    fn dragging_one_view_never_moves_the_other_bar() {
        let mut views = PressViews::new();
        let (a, b) = (views.add(None), views.add(None));
        views.get_mut(a).unwrap().dragging_bar = true;
        views.get_mut(a).unwrap().beta = 0.2;
        assert_eq!(views.get(b).unwrap().snapshot().beta, DEFAULT_BETA);
        assert!(!views.get(b).unwrap().dragging_bar);
    }

    #[test]
    fn the_cpu_fallback_delivers_an_image() {
        let slots = vec![demo_slot()];
        let mut store = PackageStore::default();
        let mut view = PressTab::new();
        let started = std::time::Instant::now();
        loop {
            let ctx = egui::Context::default();
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1000., 700.))),
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                view.ui(ui, &slots, None, &mut store);
            });
            if view.cpu.texture.is_some() || started.elapsed().as_secs() > 20 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(view.cpu.error.is_none(), "{:?}", view.cpu.error);
        assert!(
            view.cpu.texture.is_some(),
            "the worker did not deliver an image"
        );
    }

    /// Real packages of every project in `FABSIM_DATA_DIR` open in the view's solver.
    #[test]
    #[ignore = "needs FABSIM_DATA_DIR pointing at real measurement projects"]
    fn real_packages_drive_the_press() {
        let Some(data) = crate::material_package::test_support::real_data_dir() else {
            eprintln!("FABSIM_DATA_DIR is not set; skipping");
            return;
        };
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 500.0));
        let mut checked = 0;
        for project in std::fs::read_dir(data).unwrap().flatten() {
            for set in crate::material_package::discover(&project.path()).sets {
                for o in &set.orientations {
                    for dir in [&o.tiled, &o.untiled].into_iter().flatten() {
                        let package = MaterialPackage::load(dir).unwrap();
                        for branch in [PressBranch::Loading, PressBranch::Release] {
                            let request = SceneRequest {
                                branch,
                                control: Control::Indentation(0.2),
                                beta: 0.5,
                                previous_lambda: 1.0,
                                view: ViewMode::Oblique,
                                relight: None,
                            };
                            build_scene(&package, &request, canvas).unwrap();
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn color_images_keep_straight_alpha() {
        let mut image = RgbaImage::new(2, 1);
        image.pixels.copy_from_slice(&[10, 20, 30, 255, 0, 0, 0, 0]);
        let color = color_image(&image);
        assert_eq!(color.pixels[0], Color32::from_rgb(10, 20, 30));
        assert_eq!(color.pixels[1], Color32::TRANSPARENT);
        assert_eq!(capitalised("warp"), "Warp");
        assert_eq!(capitalised(""), "");
    }
}
