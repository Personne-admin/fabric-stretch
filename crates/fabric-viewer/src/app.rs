//! App state, project lifecycle and eframe integration. Tab logic lives in `crate::ui`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use egui::RichText;

use crate::images::ImageCache;
use crate::playback::PlaybackState;
use crate::press_store::PackageStore;
use crate::press_tab::{PersistedPressView, PressViews};
use crate::project::{frame_ref_for, resolve_frame_ref, ProjectSlot};
use crate::selection::{canonical_key, FrameSelection};
use crate::settings::ViewerSettings;
use crate::tracked::{PersistedTracking, TrackingSession};
use crate::ui::curves::PlotState;
use crate::ui::help::help_menu;
use crate::ui::image_viewer::ImageViewerState;
use crate::ui::inspector::InspectorState;
use crate::ui::runs_panel::RunsFilter;
use crate::ui::style;
use crate::ui::workspace::{
    preferences_window, restore_dock, show_dock_area, Workspace, WorkspaceTab, WorkspaceTabViewer,
    DOCK_KEY,
};
use crate::viewer_project::{DatasetState, ViewerProjectState};

/// eframe storage keys.
const SETTINGS_KEY: &str = "settings";
const TRACKING_KEY: &str = "tracking";
/// Press View states by dock-tab id, stored next to the dock layout.
const PRESS_KEY: &str = "press_views";

pub struct ViewerApp {
    pub(crate) slots: Vec<ProjectSlot>,
    /// Status / error line shown under the top bar.
    pub(crate) error: Option<String>,
    pub(crate) selection: FrameSelection,
    pub(crate) settings: ViewerSettings,
    pub(crate) image_cache: ImageCache,
    pub(crate) workspace: Workspace,
    pub(crate) press: PressViews,
    /// Restored Press View states waiting for their project to be opened
    /// again (source data is not persisted), keyed by dock-tab id.
    pending_press: Vec<(u32, PersistedPressView)>,
    /// Package metadata, decoded images and GPU textures shared by every Press View.
    packages: PackageStore,
    plot: PlotState,
    image_viewer: ImageViewerState,
    inspector: InspectorState,
    runs_filter: RunsFilter,
    pub(crate) tracking: TrackingSession,
    pub(crate) playback: PlaybackState,
    show_preferences: bool,
    /// Whether the window had focus on the previous frame; regaining it
    /// re-indexes photometric-stereo products written meanwhile.
    was_focused: bool,
    /// Workspace path for Save, separate from source data paths.
    viewer_project_path: Option<PathBuf>,
    #[cfg(feature = "demo")]
    demo: Option<crate::demo::DemoRecorder>,
}

impl ViewerApp {
    pub fn new(
        paths: Vec<PathBuf>,
        storage: Option<&dyn eframe::Storage>,
        render_state: Option<&eframe::egui_wgpu::RenderState>,
    ) -> Self {
        let mut settings: ViewerSettings = storage
            .and_then(|s| eframe::get_value(s, SETTINGS_KEY))
            .unwrap_or_default();
        settings.normalize();

        let mut workspace = Workspace::new();
        if settings.ui.restore_workspace {
            if let Some(dock) = storage.and_then(restore_dock) {
                workspace.dock_state = dock;
            }
        }
        // Reserve restored tab IDs before allowing new views.
        let mut press = PressViews::new();
        for id in workspace.press_ids() {
            press.get_or_create(id);
        }
        let pending_press: Vec<(u32, PersistedPressView)> = if settings.ui.restore_workspace {
            storage
                .and_then(|s| eframe::get_value(s, PRESS_KEY))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Persist tracking choices and display preferences, not imported data.
        let persisted: PersistedTracking = storage
            .and_then(|s| eframe::get_value(s, TRACKING_KEY))
            .unwrap_or_default();
        let playback = persisted.playback.clone();

        let mut app = ViewerApp {
            slots: Vec::new(),
            error: None,
            selection: FrameSelection::default(),
            settings,
            image_cache: ImageCache::new(),
            workspace,
            press,
            pending_press,
            packages: PackageStore::with_gpu(render_state),
            plot: PlotState::default(),
            image_viewer: ImageViewerState::default(),
            inspector: InspectorState::default(),
            runs_filter: RunsFilter::default(),
            tracking: TrackingSession::new(persisted),
            playback,
            show_preferences: false,
            was_focused: true,
            viewer_project_path: None,
            #[cfg(feature = "demo")]
            demo: crate::demo::DemoRecorder::from_env(),
        };
        app.open_paths(paths);
        #[cfg(feature = "demo")]
        if app.demo.is_some() {
            crate::demo::DemoRecorder::install_layout(&mut app);
        }
        app
    }

    /// Open one project or each child project in a data-root folder. Collect failures in the status line.
    fn open_path(&mut self, path: PathBuf) {
        self.open_paths([path]);
    }

    /// [`Self::open_path`] for several folders; the status line reports the
    /// failures of all of them.
    fn open_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        let mut errors = Vec::new();
        for path in paths {
            for target in project_dirs(&path) {
                if let Err(err) = self.open_project(target) {
                    errors.push(err);
                }
            }
        }
        self.error = (!errors.is_empty()).then(|| errors.join("\n"));
    }

    /// Append a project and return its slot. Reject an already-open canonical root.
    fn open_project(&mut self, path: PathBuf) -> Result<usize, String> {
        // Resolve the root before duplicate checks so both use the same path key.
        let project = fabric_data::Project::discover(&path)
            .map_err(|err| format!("Failed to scan {}: {err}", path.display()))?;
        let root_key = canonical_key(&path);
        if let Some(dup) = self.slots.iter().find(|s| s.root_key == root_key) {
            return Err(format!(
                "Already open: {} ({})",
                dup.name,
                project.root.display()
            ));
        }
        if project.runs.is_empty() && project.load_errors.is_empty() {
            return Err(format!(
                "No runs found under {} (no session.json in any subdirectory)",
                path.display()
            ));
        }

        let base = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let mut name = base.clone();
        let mut n = 2;
        while self.slots.iter().any(|s| s.name == name) {
            name = format!("{base} ({n})");
            n += 1;
        }
        self.slots.push(ProjectSlot::new(name, project));
        self.restore_pending_press_views();
        // Loading data explicitly makes that dataset the inspection context.
        let index = self.slots.len() - 1;
        self.select_project(index);
        Ok(index)
    }

    fn close_project(&mut self, idx: usize) {
        let removed = self.slots.remove(idx);
        self.packages.forget_project(&removed.root_key);
        self.after_projects_closed();
    }

    /// Close every project, with the same cleanup as closing them one by one.
    fn close_all_projects(&mut self) {
        for slot in std::mem::take(&mut self.slots) {
            self.packages.forget_project(&slot.root_key);
        }
        self.after_projects_closed();
    }

    /// Forget everything that referred to the closed projects.
    fn after_projects_closed(&mut self) {
        self.ensure_selection_usable();
        // Discard registration state for closed runs.
        let open: BTreeSet<PathBuf> = self
            .slots
            .iter()
            .flat_map(|s| s.run_keys.iter().cloned())
            .collect();
        self.tracking.retain_open_runs(&open);
        // Discard the context-menu target when its project closes.
        self.plot.context_target = None;
    }

    /// Clear unresolved selections without switching to another dataset.
    fn ensure_selection_usable(&mut self) {
        let resolve = |r: &crate::selection::FrameRef| resolve_frame_ref(&self.slots, r).is_some();
        self.selection.retain_valid(&resolve);
    }

    /// Select the dataset's first available frame.
    fn select_project(&mut self, project_index: usize) {
        self.selection.clear_hover();
        self.selection.selected = frame_ref_for(&self.slots, project_index, 0, 0);
    }

    fn viewer_project_state(&self) -> ViewerProjectState {
        let views = self.press.persisted(&self.slots);
        ViewerProjectState {
            datasets: self
                .slots
                .iter()
                .map(|slot| DatasetState {
                    source_path: slot.project.root.clone(),
                    display_name: slot.name.clone(),
                    enabled: slot.enabled,
                    hidden_runs: slot
                        .entries
                        .iter()
                        .enumerate()
                        .filter(|(_, entry)| !entry.visible)
                        .map(|(i, _)| slot.run_name(i))
                        .collect(),
                    run_visible: Vec::new(),
                })
                .collect(),
            press_view: views.first().cloned().unwrap_or_default(),
            press_views: views,
            selection: self.selection.clone(),
            ..Default::default()
        }
    }

    fn save_viewer_project(&mut self, path: PathBuf) {
        match serde_json::to_string_pretty(&self.viewer_project_state()) {
            Ok(text) => match fs::write(&path, text) {
                Ok(()) => {
                    self.viewer_project_path = Some(path);
                    self.error = None;
                }
                Err(e) => self.error = Some(format!("Could not save {}: {e}", path.display())),
            },
            Err(e) => self.error = Some(format!("Could not serialize viewer project: {e}")),
        }
    }

    /// Replace the open projects by the ones a `.fvproj` names. Datasets that
    /// fail to open are reported together, and the rest still load.
    fn load_viewer_project(&mut self, path: PathBuf) {
        let state = match fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|s| ViewerProjectState::from_json(&s))
        {
            Ok(s) => s,
            Err(e) => {
                self.error = Some(format!("Could not load {}: {e}", path.display()));
                return;
            }
        };
        self.close_all_projects();
        self.selection = FrameSelection::default();
        let mut errors = Vec::new();
        for saved in state.datasets {
            match self.open_project(saved.source_path.clone()) {
                Ok(index) => {
                    let slot = &mut self.slots[index];
                    slot.name = saved.display_name.clone();
                    slot.enabled = saved.enabled;
                    for i in 0..slot.entries.len() {
                        let name = slot.run_name(i);
                        slot.entries[i].visible = saved.is_run_visible(&name, i);
                    }
                }
                Err(err) => errors.push(format!("{}: {err}", saved.display_name)),
            }
        }
        self.restore_press_views(state.press_views);
        self.selection = state.selection;
        self.viewer_project_path = Some(path);
        self.ensure_selection_usable();
        self.error = (!errors.is_empty()).then(|| errors.join("\n"));
    }

    /// Restore views whose projects are open; retain the others for a later reopen.
    fn restore_pending_press_views(&mut self) {
        let open_sources: Vec<PathBuf> = self
            .slots
            .iter()
            .map(|s| canonical_key(&s.project.root))
            .collect();
        let live = self.workspace.press_ids();
        let (ready, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending_press)
            .into_iter()
            .filter(|(id, state)| live.contains(id) && state.project.is_some())
            .partition(|(_, state)| {
                state
                    .project
                    .as_ref()
                    .is_some_and(|p| open_sources.contains(&canonical_key(p)))
            });
        self.pending_press = waiting;
        for (id, state) in ready {
            self.press.get_or_create(id).restore(state, &self.slots);
        }
    }

    /// Opens another independent Press View, starting on the currently selected project.
    fn add_press_view(&mut self) {
        let project = self
            .selection
            .selected
            .as_ref()
            .map(|f| f.project_root.clone());
        let id = self.press.add(project);
        self.workspace.open_tab(WorkspaceTab::Press(id));
    }

    /// Replaces all Press Views (and their dock tabs) with the saved ones.
    fn restore_press_views(&mut self, saved: Vec<PersistedPressView>) {
        self.pending_press.clear();
        for id in self.workspace.press_ids() {
            self.workspace.close_tab(WorkspaceTab::Press(id));
        }
        for id in self.press.restore(saved, &self.slots) {
            self.workspace.open_tab(WorkspaceTab::Press(id));
        }
    }

    /// State of every Press View in the dock, by tab id: the live state of
    /// views that are set up, the still-waiting restored state otherwise.
    fn press_states(&mut self) -> Vec<(u32, PersistedPressView)> {
        let mut states = Vec::new();
        for id in self.workspace.press_ids() {
            if let Some(waiting) = self.pending_press.iter().find(|(p, _)| *p == id) {
                states.push(waiting.clone());
            } else {
                states.push((id, self.press.get_or_create(id).persisted(&self.slots)));
            }
        }
        states
    }

    /// Move the selection forward/backward by one frame (or ten with Shift).
    fn step_selection(&mut self, delta: isize, jump: bool) {
        let Some(active) = self.selection.active().cloned() else {
            return;
        };
        let Some((pidx, ridx, fidx)) = resolve_frame_ref(&self.slots, &active) else {
            return;
        };
        let n = self.slots[pidx].project.runs[ridx].frames.frames.len();
        if n == 0 {
            return;
        }
        let step = if jump { 10 } else { 1 };
        let new_fidx = (fidx as isize + delta * step).clamp(0, n as isize - 1) as usize;
        if new_fidx != fidx {
            if let Some(fr) = frame_ref_for(&self.slots, pidx, ridx, new_fidx) {
                self.selection.selected = Some(fr);
                self.selection.hovered = None;
                self.selection.pinned = false;
            }
        }
    }

    fn handle_shortcuts(&mut self, ui: &egui::Ui) {
        let ctx = ui.ctx();
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (left, right, space) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Space),
            )
        });
        if left || right {
            let jump = ctx.input(|i| i.modifiers.shift);
            self.step_selection(if left { -1 } else { 1 }, jump);
        }
        if space {
            self.selection.toggle_pin();
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                self.project_menu(ui);
                self.projects_menu(ui);
                ui.separator();
                self.workspace_menu(ui);
                if ui.button("Preferences").clicked() {
                    self.show_preferences = true;
                }
                help_menu(ui);

                ui.separator();
                if self.tracking.any_loaded() {
                    let loaded = self
                        .tracking
                        .runs
                        .iter()
                        .filter(|r| r.sequence.is_some())
                        .count();
                    ui.label(
                        RichText::new(format!("tracking: {loaded} run(s) loaded"))
                            .small()
                            .weak(),
                    );
                }
            });
            if let Some(err) = &self.error {
                ui.colored_label(style::ERROR, err);
            }
        });

        if self.show_preferences {
            preferences_window(
                ui.ctx(),
                &mut self.settings,
                &mut self.image_viewer,
                &mut self.show_preferences,
            );
        }
    }

    fn project_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Project", |ui| {
            if ui.button("Load Data...").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.open_path(dir);
                }
                ui.close();
            }
            ui.separator();
            let can_save = self.viewer_project_path.is_some();
            if ui
                .add_enabled(can_save, egui::Button::new("Save Viewer Project"))
                .clicked()
            {
                if let Some(path) = self.viewer_project_path.clone() {
                    self.save_viewer_project(path);
                }
                ui.close();
            }
            if ui.button("Save Viewer Project As...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name("viewer.fvproj")
                    .add_filter("Fabric Viewer project", &["fvproj"])
                    .save_file()
                {
                    self.save_viewer_project(path);
                }
                ui.close();
            }
            if ui.button("Load Viewer Project...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Fabric Viewer project", &["fvproj", "json"])
                    .pick_file()
                {
                    self.load_viewer_project(path);
                }
                ui.close();
            }
        });
    }

    fn projects_menu(&mut self, ui: &mut egui::Ui) {
        let label = format!("Projects ({})", self.slots.len());
        ui.menu_button(label, |ui| {
            if self.slots.is_empty() {
                ui.label("none open");
            }
            let mut close: Option<usize> = None;
            let mut select: Option<usize> = None;
            let mut enabled_changed = false;
            for (i, slot) in self.slots.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    if ui
                        .checkbox(&mut slot.enabled, "")
                        .on_hover_text(
                            "Enabled: plot this project's curves and list its runs; \
                             unchecked keeps it open but out of the way",
                        )
                        .changed()
                    {
                        enabled_changed = true;
                    }
                    if ui.small_button("close").clicked() {
                        close = Some(i);
                    }
                    let selected = self
                        .selection
                        .selected
                        .as_ref()
                        .is_some_and(|frame| frame.project_root == slot.root_key);
                    if ui
                        .selectable_label(selected, &slot.name)
                        .on_hover_text(slot.project.root.display().to_string())
                        .clicked()
                    {
                        select = Some(i);
                    }
                });
            }
            if let Some(i) = close {
                self.close_project(i);
            } else if let Some(i) = select {
                self.select_project(i);
            } else if enabled_changed {
                self.ensure_selection_usable();
            }
        });
    }

    fn workspace_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("View", |ui| {
            for tab in WorkspaceTab::SINGLETONS {
                let mut on = self.workspace.has_tab(&tab);
                if ui
                    .toggle_value(&mut on, tab.title())
                    .on_hover_text(tab.shortcut_hint())
                    .clicked()
                {
                    self.workspace.toggle_tab(tab);
                }
            }
            ui.separator();
            for id in self.workspace.press_ids() {
                let tab = WorkspaceTab::Press(id);
                let mut on = true;
                if ui
                    .toggle_value(&mut on, tab.title())
                    .on_hover_text("Click to close this Press View")
                    .clicked()
                {
                    self.workspace.close_tab(tab);
                    self.press.remove(id);
                }
            }
            if ui
                .button("Press View")
                .on_hover_text("Open another independent Press View for side-by-side comparison")
                .clicked()
            {
                self.add_press_view();
                ui.close();
            }
            ui.separator();
            if ui
                .button("Reset Layout")
                .on_hover_text("Restore the default tab arrangement")
                .clicked()
            {
                self.workspace.reset_layout();
                self.press.clear();
                self.pending_press.clear();
            }
        });
    }
}

/// Use the folder itself if it contains runs, otherwise its child projects.
fn project_dirs(path: &Path) -> Vec<PathBuf> {
    if path.join("session.json").is_file() {
        return vec![path.to_owned()];
    }
    let mut children: Vec<PathBuf> = fs::read_dir(path)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && !p
                            .file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with('.'))
                })
                .collect()
        })
        .unwrap_or_default();
    children.sort();
    let has_direct_runs = children.iter().any(|c| c.join("session.json").is_file());
    if !has_direct_runs {
        let sub_projects: Vec<PathBuf> = children
            .into_iter()
            .filter(|c| contains_run(c, 2))
            .collect();
        if !sub_projects.is_empty() {
            return sub_projects;
        }
    }
    vec![path.to_owned()]
}

/// Whether `dir` contains `session.json`, directly or within `depth` levels.
fn contains_run(dir: &Path, depth: usize) -> bool {
    if dir.join("session.json").is_file() {
        return true;
    }
    if depth == 0 {
        return false;
    }
    fs::read_dir(dir).is_ok_and(|rd| {
        rd.filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .any(|p| contains_run(&p, depth - 1))
    })
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.settings.apply_theme(ui.ctx());
        self.image_cache.poll(ui.ctx());
        let focused = ui.ctx().input(|i| i.viewport().focused.unwrap_or(true));
        if focused && !self.was_focused {
            for slot in &mut self.slots {
                slot.rescan_missing_ps();
            }
        }
        self.was_focused = focused;
        self.top_bar(ui);
        self.handle_shortcuts(ui);
        self.playback
            .advance(ui.ctx(), &self.slots, &mut self.selection);

        // Clear plot hover when Curves loses focus. egui_dock reports focus from the previous pass.
        if !self.workspace.is_tab_active(&WorkspaceTab::Curves) {
            self.selection.clear_hover();
        }

        #[cfg(feature = "demo")]
        if let Some(demo) = &mut self.demo {
            demo.tick(ui.ctx(), &mut self.press);
        }

        egui::CentralPanel::default().show(ui, |ui| {
            let mut viewer = WorkspaceTabViewer {
                slots: &mut self.slots,
                selection: &mut self.selection,
                settings: &mut self.settings,
                image_cache: &mut self.image_cache,
                press: &mut self.press,
                packages: &mut self.packages,
                plot: &mut self.plot,
                image_viewer: &mut self.image_viewer,
                inspector: &mut self.inspector,
                runs_filter: &mut self.runs_filter,
                tracking: &mut self.tracking,
                playback: &mut self.playback,
            };
            show_dock_area(ui, &mut self.workspace, &mut viewer);
        });
        if std::mem::take(&mut self.press.add_requested) {
            self.add_press_view();
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, SETTINGS_KEY, &self.settings);
        // The playback preferences are stored with the tracking envelope, as
        // before the state moved out of the tracking session.
        let mut tracking = self.tracking.to_persisted();
        tracking.playback = self.playback.clone();
        eframe::set_value(storage, TRACKING_KEY, &tracking);
        if self.settings.ui.restore_workspace {
            eframe::set_value(storage, DOCK_KEY, &self.workspace.dock_state);
            eframe::set_value(storage, PRESS_KEY, &self.press_states());
        }
    }
}

#[cfg(test)]
mod tests;
