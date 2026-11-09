//! Dock layout and persistence. Tabs borrow app state through [`WorkspaceTabViewer`].

use egui_dock::{DockArea, DockState, NodeIndex, TabViewer};
use serde::{Deserialize, Serialize};

use crate::images::ImageCache;
use crate::playback::PlaybackState;
use crate::press_store::PackageStore;
use crate::press_tab::PressViews;
use crate::project::ProjectSlot;
use crate::selection::FrameSelection;
use crate::settings::ViewerSettings;
use crate::tracked::TrackingSession;
use crate::ui::curves::{self, PlotState};
use crate::ui::image_viewer::{self, ImageViewerCtx, ImageViewerState};
use crate::ui::inspector::{self, InspectorState};
use crate::ui::press;
use crate::ui::runs_panel::{self, RunsFilter};

/// The set of dockable tabs. `Press(id)` is one independent Press View instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "TabRepr", into = "TabRepr")]
pub enum WorkspaceTab {
    Curves,
    ImageViewer,
    Inspector,
    Runs,
    Press(u32),
}

/// On-disk form. Layouts saved before multiple Press Views stored the single
/// view as the unit variant `Press`, which reads back as view 0.
#[derive(Serialize, Deserialize)]
enum TabRepr {
    Curves,
    ImageViewer,
    Inspector,
    Runs,
    Press,
    PressView(u32),
}

impl From<TabRepr> for WorkspaceTab {
    fn from(repr: TabRepr) -> Self {
        match repr {
            TabRepr::Curves => WorkspaceTab::Curves,
            TabRepr::ImageViewer => WorkspaceTab::ImageViewer,
            TabRepr::Inspector => WorkspaceTab::Inspector,
            TabRepr::Runs => WorkspaceTab::Runs,
            TabRepr::Press => WorkspaceTab::Press(0),
            TabRepr::PressView(id) => WorkspaceTab::Press(id),
        }
    }
}

/// A tab as read from a saved layout: `None` for the removed 3D View tab,
/// which [`restore_dock`] prunes so old layouts still load.
#[derive(Deserialize)]
#[serde(from = "LegacyRepr")]
struct LegacyTab(Option<WorkspaceTab>);

#[derive(Deserialize)]
enum LegacyRepr {
    Curves,
    ImageViewer,
    Inspector,
    Runs,
    Press,
    PressView(u32),
    ThreeD,
}

impl From<LegacyRepr> for LegacyTab {
    fn from(repr: LegacyRepr) -> Self {
        LegacyTab(match repr {
            LegacyRepr::Curves => Some(WorkspaceTab::Curves),
            LegacyRepr::ImageViewer => Some(WorkspaceTab::ImageViewer),
            LegacyRepr::Inspector => Some(WorkspaceTab::Inspector),
            LegacyRepr::Runs => Some(WorkspaceTab::Runs),
            LegacyRepr::Press => Some(WorkspaceTab::Press(0)),
            LegacyRepr::PressView(id) => Some(WorkspaceTab::Press(id)),
            LegacyRepr::ThreeD => None,
        })
    }
}

/// Rebuild a saved dock layout, dropping tabs that no longer exist (the 3D
/// View). `None` when the layout is unreadable or empty after pruning.
pub fn restore_dock(storage: &dyn eframe::Storage) -> Option<DockState<WorkspaceTab>> {
    let mut dock: DockState<LegacyTab> = eframe::get_value(storage, DOCK_KEY)?;
    dock.retain_tabs(|tab| tab.0.is_some());
    let dock = dock.map_tabs(|tab| tab.0.expect("pruned above"));
    (!dock.main_surface().is_empty()).then_some(dock)
}

/// Storage key of the dock layout.
pub const DOCK_KEY: &str = "workspace_dock";

impl From<WorkspaceTab> for TabRepr {
    fn from(tab: WorkspaceTab) -> Self {
        match tab {
            WorkspaceTab::Curves => TabRepr::Curves,
            WorkspaceTab::ImageViewer => TabRepr::ImageViewer,
            WorkspaceTab::Inspector => TabRepr::Inspector,
            WorkspaceTab::Runs => TabRepr::Runs,
            WorkspaceTab::Press(id) => TabRepr::PressView(id),
        }
    }
}

impl WorkspaceTab {
    /// Singleton tabs listed in the View menu; Press Views are added individually.
    pub const SINGLETONS: [WorkspaceTab; 4] = [
        WorkspaceTab::Curves,
        WorkspaceTab::ImageViewer,
        WorkspaceTab::Inspector,
        WorkspaceTab::Runs,
    ];

    pub fn title(self) -> String {
        match self {
            WorkspaceTab::Curves => "Curves".into(),
            WorkspaceTab::ImageViewer => "Image Viewer".into(),
            WorkspaceTab::Inspector => "Inspector".into(),
            WorkspaceTab::Runs => "Runs".into(),
            WorkspaceTab::Press(0) => "Press View".into(),
            WorkspaceTab::Press(id) => format!("Press View {}", id + 1),
        }
    }

    pub fn shortcut_hint(self) -> &'static str {
        match self {
            WorkspaceTab::Curves => "click/hover a curve point; right-click for run actions",
            WorkspaceTab::ImageViewer => "F fit, 1 one-to-one, wheel zoom, drag pan",
            WorkspaceTab::Inspector => "pick a run and scrub frames; click a curve to select",
            WorkspaceTab::Runs => "toggle run visibility; context menu for solo",
            WorkspaceTab::Press(_) => {
                "drag the bar along its axis; each view has its own project and force"
            }
        }
    }

    fn press_id(self) -> Option<u32> {
        match self {
            WorkspaceTab::Press(id) => Some(id),
            _ => None,
        }
    }
}

/// The docking state plus layout helpers.
pub struct Workspace {
    pub dock_state: DockState<WorkspaceTab>,
}

impl Workspace {
    pub fn new() -> Self {
        Workspace {
            dock_state: Self::default_layout(),
        }
    }

    /// Curves left, Image Viewer above Inspector on the right, Runs below.
    /// Split fractions belong to the old child: `split_right(0.62)` leaves 62% on the left.
    pub fn default_layout() -> DockState<WorkspaceTab> {
        let mut state = DockState::new(vec![WorkspaceTab::Curves]);
        {
            let main = state.main_surface_mut();
            let [_, image_viewer] =
                main.split_right(NodeIndex::root(), 0.62, vec![WorkspaceTab::ImageViewer]);
            let [_, _inspector] =
                main.split_below(image_viewer, 0.55, vec![WorkspaceTab::Inspector]);
            let [_, _runs] = main.split_below(NodeIndex::root(), 0.72, vec![WorkspaceTab::Runs]);
        }
        state
    }

    pub fn reset_layout(&mut self) {
        self.dock_state = Self::default_layout();
    }

    pub fn has_tab(&self, tab: &WorkspaceTab) -> bool {
        self.dock_state.find_tab(tab).is_some()
    }

    /// Reopen tabs near their default region. Restore the default layout first if empty.
    pub fn open_tab(&mut self, tab: WorkspaceTab) {
        if self.has_tab(&tab) {
            return;
        }
        if self.dock_state.main_surface().num_tabs() == 0 {
            self.dock_state = Self::default_layout();
            if self.has_tab(&tab) {
                return;
            }
        }
        let sibling = match tab {
            WorkspaceTab::Press(_) => self.last_press_tab(),
            _ => None,
        };
        let main = self.dock_state.main_surface_mut();
        // Anchor on Curves, then the Image/Inspector column, then the first leaf.
        let anchor = main
            .find_tab(&WorkspaceTab::Curves)
            .map(|(n, _)| n)
            .or_else(|| main.find_tab(&WorkspaceTab::ImageViewer).map(|(n, _)| n))
            .or_else(|| main.find_tab(&WorkspaceTab::Inspector).map(|(n, _)| n))
            .or_else(|| main.iter().position(|n| n.is_leaf()).map(NodeIndex))
            .unwrap_or(NodeIndex::root());
        match tab {
            WorkspaceTab::Curves => {
                // `split_left` keeps the old share; 0.38 leaves 62% for reopened Curves.
                let [_, _new] = main.split_left(anchor, 0.38, vec![tab]);
            }
            WorkspaceTab::Press(_) if sibling.is_some() => {
                // Put Press Views side by side, moving a stacked first view into its own pane.
                let sibling = sibling.unwrap();
                let (node, _) = main.find_tab(&sibling).unwrap();
                let shares_leaf = main
                    .leaf_mut(node)
                    .map(|leaf| leaf.tabs.len() > 1)
                    .unwrap_or(false);
                let mut target = node;
                if shares_leaf {
                    if let Some(path) = self.dock_state.find_tab(&sibling) {
                        self.dock_state.remove_tab(path);
                    }
                    let main = self.dock_state.main_surface_mut();
                    let [_, pane] = main.split_right(node, 0.4, vec![sibling]);
                    target = pane;
                }
                let main = self.dock_state.main_surface_mut();
                let [_, _new] = main.split_right(target, 0.5, vec![tab]);
            }
            WorkspaceTab::Press(_) => {
                // Stack with Curves, or use the main content anchor.
                if let Some((curves_node, _)) = main.find_tab(&WorkspaceTab::Curves) {
                    if let Ok(leaf) = main.leaf_mut(curves_node) {
                        leaf.append_tab(tab);
                    }
                } else {
                    let [_, _new] = main.split_left(anchor, 0.38, vec![tab]);
                }
            }
            WorkspaceTab::ImageViewer => {
                let [_, _new] = main.split_right(anchor, 0.62, vec![tab]);
            }
            WorkspaceTab::Inspector => {
                let below = main
                    .find_tab(&WorkspaceTab::ImageViewer)
                    .map(|(n, _)| n)
                    .unwrap_or(anchor);
                let [_, _new] = main.split_below(below, 0.55, vec![tab]);
            }
            WorkspaceTab::Runs => {
                let [_, _new] = main.split_below(NodeIndex::root(), 0.72, vec![tab]);
            }
        }
    }

    /// The Press View tab with the highest id that is currently open.
    fn last_press_tab(&self) -> Option<WorkspaceTab> {
        self.dock_state
            .iter_all_tabs()
            .filter_map(|(_, tab)| tab.press_id().map(|_| *tab))
            .max_by_key(|tab| tab.press_id())
    }

    pub fn press_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .dock_state
            .iter_all_tabs()
            .filter_map(|(_, tab)| tab.press_id())
            .collect();
        ids.sort_unstable();
        ids
    }

    pub fn close_tab(&mut self, tab: WorkspaceTab) {
        if let Some(path) = self.dock_state.find_tab(&tab) {
            self.dock_state.remove_tab(path);
        }
    }

    pub fn toggle_tab(&mut self, tab: WorkspaceTab) {
        if self.has_tab(&tab) {
            self.close_tab(tab);
        } else {
            self.open_tab(tab);
        }
    }

    /// Whether the tab has focus; used to clear stale plot hover.
    pub fn is_tab_active(&mut self, tab: &WorkspaceTab) -> bool {
        matches!(self.dock_state.find_active_focused(), Some((_, active)) if active == tab)
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrows the disjoint app fields every tab needs. The `DockState` itself is
/// borrowed separately, so no unsafe/self-referential tricks are required.
pub struct WorkspaceTabViewer<'a> {
    pub slots: &'a mut Vec<ProjectSlot>,
    pub selection: &'a mut FrameSelection,
    pub settings: &'a mut ViewerSettings,
    pub image_cache: &'a mut ImageCache,
    pub press: &'a mut PressViews,
    pub packages: &'a mut PackageStore,
    pub plot: &'a mut PlotState,
    pub image_viewer: &'a mut ImageViewerState,
    pub inspector: &'a mut InspectorState,
    pub runs_filter: &'a mut RunsFilter,
    pub tracking: &'a mut TrackingSession,
    pub playback: &'a mut PlaybackState,
}

impl TabViewer for WorkspaceTabViewer<'_> {
    type Tab = WorkspaceTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            WorkspaceTab::Curves => curves::curves_tab(
                ui,
                self.slots,
                self.selection,
                self.settings,
                self.image_cache,
                self.plot,
            ),
            WorkspaceTab::ImageViewer => image_viewer::image_viewer_tab(
                ui,
                self.image_viewer,
                &mut ImageViewerCtx {
                    slots: self.slots,
                    selection: self.selection,
                    settings: self.settings,
                    image_cache: self.image_cache,
                    tracking: self.tracking,
                    playback: self.playback,
                },
            ),
            WorkspaceTab::Inspector => {
                inspector::inspector_tab(ui, self.slots, self.selection, self.inspector)
            }
            WorkspaceTab::Runs => runs_panel::runs_tab(ui, self.slots, self.runs_filter),
            WorkspaceTab::Press(id) => press::press_tab(
                ui,
                self.slots,
                self.selection,
                self.press,
                self.packages,
                *id,
            ),
        }
    }

    fn on_close(&mut self, tab: &mut Self::Tab) -> egui_dock::tab_viewer::OnCloseResponse {
        if let WorkspaceTab::Press(id) = tab {
            self.press.remove(*id);
        }
        egui_dock::tab_viewer::OnCloseResponse::Close
    }

    /// Tabs own their scroll areas, so the dock must not add another.
    fn scroll_bars(&self, _tab: &Self::Tab) -> [bool; 2] {
        [false, false]
    }
}

/// Show the whole docking area inside `ui`.
pub fn show_dock_area(
    ui: &mut egui::Ui,
    workspace: &mut Workspace,
    viewer: &mut WorkspaceTabViewer<'_>,
) {
    DockArea::new(&mut workspace.dock_state)
        .style(egui_dock::Style::from_egui(ui.style()))
        .show_inside(ui, viewer);
}

/// A preferences window shared by all tabs.
pub fn preferences_window(
    ctx: &egui::Context,
    settings: &mut ViewerSettings,
    image_viewer: &mut ImageViewerState,
    open: &mut bool,
) {
    let mut is_open = *open;
    let mut changed = false;
    egui::Window::new("Preferences")
        .open(&mut is_open)
        .resizable(true)
        .default_size([400.0, 480.0])
        .show(ctx, |ui| {
            ui.heading("Images");
            changed |= image_preferences(ui, settings, image_viewer);
            ui.separator();
            ui.heading("Plots");
            changed |= plot_preferences(ui, settings);
            ui.separator();
            ui.heading("Interface");
            changed |= ui_preferences(ui, settings);
        });
    *open = is_open;
    if changed {
        settings.apply_theme(ctx);
        ctx.request_repaint();
    }
}

fn plot_preferences(ui: &mut egui::Ui, settings: &mut ViewerSettings) -> bool {
    let mut changed = false;
    let p = &mut settings.plot;
    changed |= ui
        .radio_value(
            &mut p.quantity,
            crate::settings::PlotQuantity::ForceN,
            "Force [N]",
        )
        .changed()
        | ui.radio_value(
            &mut p.quantity,
            crate::settings::PlotQuantity::LineLoad,
            "Line load [N/m]",
        )
        .changed();
    ui.horizontal(|ui| {
        changed |= ui
            .checkbox(&mut p.mirror_release, "Mirror release")
            .changed();
        changed |= ui.checkbox(&mut p.show_relax, "Relaxation holds").changed();
        changed |= ui.checkbox(&mut p.smooth, "Smooth").changed();
    });
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut p.show_legend, "Legend").changed();
        changed |= ui.checkbox(&mut p.show_grid, "Grid").changed();
    });
    changed |= ui
        .checkbox(&mut p.hover_preview, "Hover preview image")
        .on_hover_text("Show the hovered frame's image near the pointer")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut p.hover_preview_max_px, 80.0..=600.0).text("hover preview size"),
        )
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut p.line_width, 0.5..=6.0).text("line width"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut p.selected_line_width, 1.0..=8.0).text("selected line width"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut p.marker_radius, 1.0..=12.0).text("relax marker radius"))
        .changed();
    changed
}

fn image_preferences(
    ui: &mut egui::Ui,
    settings: &mut ViewerSettings,
    image_viewer: &mut ImageViewerState,
) -> bool {
    let mut changed = false;
    let mut role_text = settings.images.preferred_role.clone().unwrap_or_default();
    if ui
        .add(
            egui::TextEdit::singleline(&mut role_text)
                .hint_text("preferred role (diffuse, ps_1, ...)"),
        )
        .on_hover_text(
            "Role shown by default; falls back to diffuse -> rgb -> albedo -> first. \
             Relight always uses the albedo.",
        )
        .changed()
    {
        settings.images.preferred_role = Some(role_text);
        settings.normalize();
        changed = true;
    }
    let i = &mut settings.images;
    ui.horizontal(|ui| {
        let fit = ui.radio_value(&mut i.fit, crate::settings::FitMode::Fit, "Fit");
        if fit.clicked() {
            // Clicking "Fit" while already Fit must still reapply it.
            image_viewer.fit_force = true;
        }
        changed |= fit.changed();
        let one_to_one = ui.radio_value(&mut i.fit, crate::settings::FitMode::OneToOne, "1:1");
        if one_to_one.clicked() {
            image_viewer.fit_force = true;
        }
        changed |= one_to_one.changed();
    });
    ui.horizontal(|ui| {
        changed |= ui
            .radio_value(
                &mut i.interpolation,
                crate::settings::Interpolation::Nearest,
                "Nearest",
            )
            .changed();
        changed |= ui
            .radio_value(
                &mut i.interpolation,
                crate::settings::Interpolation::Linear,
                "Linear",
            )
            .changed();
    });
    changed |= ui
        .checkbox(&mut i.hover_follow, "Follow hovered frame")
        .on_hover_text("While scrubbing the curve, the viewer follows the hovered frame")
        .changed();
    ui.label("Comparison");
    for mode in crate::settings::ComparisonMode::ALL {
        changed |= ui
            .radio_value(&mut i.comparison, mode, mode.label())
            .changed();
    }
    changed
}

fn ui_preferences(ui: &mut egui::Ui, settings: &mut ViewerSettings) -> bool {
    let mut changed = false;
    let u = &mut settings.ui;
    ui.horizontal(|ui| {
        changed |= ui
            .radio_value(&mut u.theme, crate::settings::Theme::System, "System")
            .changed();
        changed |= ui
            .radio_value(&mut u.theme, crate::settings::Theme::Light, "Light")
            .changed();
        changed |= ui
            .radio_value(&mut u.theme, crate::settings::Theme::Dark, "Dark")
            .changed();
    });
    changed |= ui
        .checkbox(&mut u.restore_workspace, "Restore dock layout on launch")
        .changed();
    changed |= ui
        .checkbox(
            &mut u.advanced_tracking,
            "Show advanced tracking & registration controls",
        )
        .on_hover_text(
            "Advanced/debug/research mode: adds the legacy registration modes, manual \
             reference-track selection, overlay/canvas configuration and diagnostics to \
             the Image Viewer's Tracking section (collapsed). The normal tracking \
             controls and the stabilized replay are unaffected.",
        )
        .changed();
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_contains_all_builtin_tabs() {
        let ws = Workspace::new();
        assert!(ws.has_tab(&WorkspaceTab::Curves));
        assert!(ws.has_tab(&WorkspaceTab::ImageViewer));
        assert!(ws.has_tab(&WorkspaceTab::Inspector));
        assert!(ws.has_tab(&WorkspaceTab::Runs));
        assert!(
            !ws.has_tab(&WorkspaceTab::Press(0)),
            "Press opens via the menu"
        );
    }

    #[test]
    fn reopen_every_tab_from_empty_restores_a_sane_workspace() {
        let mut ws = Workspace::new();
        // Closing all tabs must still allow reopening a usable layout.
        let all: Vec<WorkspaceTab> = WorkspaceTab::SINGLETONS
            .into_iter()
            .chain([WorkspaceTab::Press(0)])
            .collect();
        for tab in all.iter().copied() {
            ws.close_tab(tab);
        }
        for tab in all {
            ws.open_tab(tab);
            assert!(ws.has_tab(&tab), "{tab:?} should be reopenable");
        }
    }

    #[test]
    fn reopen_single_tab_after_close_is_a_noop_respectfully() {
        let mut ws = Workspace::new();
        ws.close_tab(WorkspaceTab::ImageViewer);
        assert!(!ws.has_tab(&WorkspaceTab::ImageViewer));
        ws.open_tab(WorkspaceTab::ImageViewer);
        assert!(ws.has_tab(&WorkspaceTab::ImageViewer));
        ws.open_tab(WorkspaceTab::ImageViewer);
        assert!(ws.has_tab(&WorkspaceTab::ImageViewer));
    }

    fn leaf_tabs(dock: &DockState<WorkspaceTab>, idx: NodeIndex) -> Vec<WorkspaceTab> {
        match &dock.main_surface()[idx] {
            egui_dock::Node::Leaf(leaf) => leaf.tabs.clone(),
            _ => vec![],
        }
    }

    #[test]
    fn reopen_curves_lands_in_its_own_dominant_pane() {
        let mut ws = Workspace::new();
        ws.close_tab(WorkspaceTab::Curves);
        assert!(!ws.has_tab(&WorkspaceTab::Curves));
        ws.open_tab(WorkspaceTab::Curves);
        assert!(ws.has_tab(&WorkspaceTab::Curves));

        let main = &ws.dock_state.main_surface();
        let curves_node = main.find_tab(&WorkspaceTab::Curves).unwrap().0;
        let image_node = main.find_tab(&WorkspaceTab::ImageViewer).unwrap().0;
        assert_ne!(
            curves_node, image_node,
            "Curves must reopen as its own pane, not stacked into the side column"
        );
        assert_eq!(
            leaf_tabs(&ws.dock_state, curves_node),
            vec![WorkspaceTab::Curves]
        );
        // The root must be a split again (Curves back as a separate region).
        assert!(matches!(
            main[NodeIndex::root()],
            egui_dock::Node::Horizontal(_) | egui_dock::Node::Vertical(_)
        ));
    }

    #[test]
    fn reopen_press_stacks_with_curves() {
        let mut ws = Workspace::new();
        ws.open_tab(WorkspaceTab::Press(0));
        let main = &ws.dock_state.main_surface();
        let curves_node = main.find_tab(&WorkspaceTab::Curves).unwrap().0;
        let press_node = main.find_tab(&WorkspaceTab::Press(0)).unwrap().0;
        assert_eq!(curves_node, press_node, "Press stacks into the Curves leaf");
        assert_eq!(
            leaf_tabs(&ws.dock_state, curves_node),
            vec![WorkspaceTab::Curves, WorkspaceTab::Press(0)]
        );
    }

    #[test]
    fn reopen_press_without_curves_uses_the_main_anchor() {
        let mut ws = Workspace::new();
        ws.close_tab(WorkspaceTab::Curves);
        ws.open_tab(WorkspaceTab::Press(0));
        assert!(ws.has_tab(&WorkspaceTab::Press(0)));
        let main = &ws.dock_state.main_surface();
        let press_node = main.find_tab(&WorkspaceTab::Press(0)).unwrap().0;
        let image_node = main.find_tab(&WorkspaceTab::ImageViewer).unwrap().0;
        assert_ne!(
            press_node, image_node,
            "Press without Curves must be its own main pane, not the side column"
        );
    }

    #[test]
    fn reset_layout_restores_the_exact_default() {
        let mut ws = Workspace::new();
        let before = serde_json::to_string(&ws.dock_state).unwrap();
        ws.close_tab(WorkspaceTab::Runs);
        ws.open_tab(WorkspaceTab::Press(0));
        ws.reset_layout();
        let after = serde_json::to_string(&ws.dock_state).unwrap();
        assert_eq!(before, after, "Reset Layout must restore the exact default");
    }

    #[test]
    fn legacy_press_tab_in_a_saved_layout_loads_as_view_zero() {
        let legacy: WorkspaceTab = serde_json::from_str("\"Press\"").unwrap();
        assert_eq!(legacy, WorkspaceTab::Press(0));
        let modern = serde_json::to_string(&WorkspaceTab::Press(3)).unwrap();
        assert_eq!(
            serde_json::from_str::<WorkspaceTab>(&modern).unwrap(),
            WorkspaceTab::Press(3)
        );
        for tab in WorkspaceTab::SINGLETONS {
            let json = serde_json::to_string(&tab).unwrap();
            assert_eq!(serde_json::from_str::<WorkspaceTab>(&json).unwrap(), tab);
        }
    }

    #[test]
    fn dock_layout_with_several_press_views_round_trips() {
        let mut ws = Workspace::new();
        for id in 0..3 {
            ws.open_tab(WorkspaceTab::Press(id));
        }
        assert_eq!(ws.press_ids(), vec![0, 1, 2]);
        let saved = ron::to_string(&ws.dock_state).unwrap();
        let restored: DockState<WorkspaceTab> = ron::from_str(&saved).unwrap();
        assert_eq!(
            restored.iter_all_tabs().count(),
            ws.dock_state.iter_all_tabs().count()
        );
        assert!(restored.find_tab(&WorkspaceTab::Press(2)).is_some());
    }

    #[test]
    fn two_press_views_open_side_by_side_in_their_own_panes() {
        let mut ws = Workspace::new();
        ws.open_tab(WorkspaceTab::Press(0));
        ws.open_tab(WorkspaceTab::Press(1));
        let main = ws.dock_state.main_surface();
        let first = main.find_tab(&WorkspaceTab::Press(0)).unwrap().0;
        let second = main.find_tab(&WorkspaceTab::Press(1)).unwrap().0;
        let curves = main.find_tab(&WorkspaceTab::Curves).unwrap().0;
        assert_ne!(first, second, "views must not be stacked as tabs");
        assert_ne!(
            first, curves,
            "the first view leaves the Curves pane so both are visible"
        );
        assert_eq!(
            leaf_tabs(&ws.dock_state, first),
            vec![WorkspaceTab::Press(0)]
        );
        assert_eq!(
            leaf_tabs(&ws.dock_state, second),
            vec![WorkspaceTab::Press(1)]
        );
    }

    #[test]
    fn closing_one_press_view_keeps_the_other() {
        let mut ws = Workspace::new();
        ws.open_tab(WorkspaceTab::Press(0));
        ws.open_tab(WorkspaceTab::Press(1));
        ws.close_tab(WorkspaceTab::Press(0));
        assert_eq!(ws.press_ids(), vec![1]);
    }
}
