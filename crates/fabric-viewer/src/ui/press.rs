//! Hosts one Press View instance inside the docking workspace.

use crate::press_store::PackageStore;
use crate::press_tab::PressViews;
use crate::project::ProjectSlot;
use crate::selection::FrameSelection;

pub fn press_tab(
    ui: &mut egui::Ui,
    slots: &mut [ProjectSlot],
    selection: &mut FrameSelection,
    views: &mut PressViews,
    packages: &mut PackageStore,
    id: u32,
) {
    if ui.available_height() < 120.0 {
        ui.label("Press view needs more vertical space; drag the splitter.");
        return;
    }
    let default_project = selection.selected.as_ref().map(|f| f.project_root.clone());
    let out = views
        .get_or_create(id)
        .ui(ui, slots, default_project.as_deref(), packages);
    if out.add_view {
        views.add_requested = true;
    }
}
