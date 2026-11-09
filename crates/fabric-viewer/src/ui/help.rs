//! The Help menu: keyboard and mouse shortcuts of the viewer.

/// `(keys, what they do)`, listed in the Help menu.
pub const SHORTCUTS: &[(&str, &str)] = &[
    ("Left / Right", "previous / next frame of the selected run"),
    ("Shift + Left / Right", "jump ten frames"),
    (
        "Space",
        "pin the hovered curve point as the selection (again to release)",
    ),
    ("F", "Image Viewer: fit the image to the canvas"),
    ("1", "Image Viewer: one image pixel per point"),
    (
        "Mouse wheel / drag",
        "Image Viewer: zoom at the pointer / pan",
    ),
    (
        "Click / right-click",
        "Curves: select a frame / run actions",
    ),
    ("Drag the bar", "Press View: move the bar along the sheet"),
    (
        "Ctrl or Shift + click",
        "Image Viewer: add a tracked point to the reference set",
    ),
];

pub fn help_menu(ui: &mut egui::Ui) {
    ui.menu_button("Help", |ui| {
        ui.label(egui::RichText::new("Shortcuts").strong());
        egui::Grid::new("help_shortcuts").show(ui, |ui| {
            for (keys, what) in SHORTCUTS {
                ui.label(egui::RichText::new(*keys).monospace());
                ui.label(*what);
                ui.end_row();
            }
        });
        ui.separator();
        ui.label("Hover a control for its description.");
    });
}
