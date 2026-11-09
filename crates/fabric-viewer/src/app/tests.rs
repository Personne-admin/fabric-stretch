//! Tests of project open / close / save / load and the app-level state.

use std::fs;
use std::path::{Path, PathBuf};

use eframe::App as _;

use super::*;
use crate::selection::FrameRef;
use crate::ui::curves::legend_hidden_ids;

fn write_min_run(dir: &Path, sample: &str) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("session.json"),
        format!(
            r#"{{"sample_id":"{sample}","test_id":"t","direction":"warp","branch":"loading","count":0}}"#
        ),
    )
    .unwrap();
    fs::write(dir.join("frames.json"), r#"{"frames":[]}"#).unwrap();
    fs::write(dir.join("force.json"), r#"{"forces":[]}"#).unwrap();
}

fn write_single_frame(dir: &Path) {
    fs::write(
        dir.join("frames.json"),
        r#"{"frames":[{"frame_id":1,"step_index":0,"displacement_mm":0.0}]}"#,
    )
    .unwrap();
}

#[test]
fn opening_a_parent_folder_imports_each_child_as_its_own_project() {
    let root = std::env::temp_dir().join(format!("fabsim_parent_import_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    write_min_run(&root.join("sample_a").join("warp_loading"), "a");
    write_min_run(&root.join("sample_b").join("warp_loading"), "b");

    let parent = ViewerApp::new(vec![root.clone()], None, None);
    assert_eq!(parent.slots.len(), 2);
    assert!(parent.slots.iter().any(|s| s.name == "sample_a"));
    assert!(parent.slots.iter().any(|s| s.name == "sample_b"));

    // A folder that itself contains run dirs still opens as one project.
    let single = ViewerApp::new(vec![root.join("sample_a")], None, None);
    assert_eq!(single.slots.len(), 1);

    fs::remove_dir_all(&root).ok();
}

#[test]
fn selection_is_invalidated_when_project_is_closed() {
    let root = std::env::temp_dir().join(format!("fabsim_selection_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    write_min_run(&root.join("sample_a").join("warp_loading"), "a");
    write_single_frame(&root.join("sample_a").join("warp_loading"));

    let mut app = ViewerApp::new(vec![root.clone()], None, None);
    assert!(app.selection.selected.is_some(), "first run auto-selected");
    app.close_project(0);
    assert_eq!(app.selection.selected, None);
    assert_eq!(app.slots.len(), 0);

    fs::remove_dir_all(&root).ok();
}

#[test]
fn disabling_the_selected_project_invalidates_selection() {
    let root = std::env::temp_dir().join(format!("fabsim_disable_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    write_min_run(&root.join("sample_a").join("warp_loading"), "a");
    write_single_frame(&root.join("sample_a").join("warp_loading"));

    let mut app = ViewerApp::new(vec![root.clone()], None, None);
    assert!(app.selection.selected.is_some());
    app.slots[0].enabled = false;
    app.ensure_selection_usable();
    assert_eq!(app.selection.selected, None);
    // Re-enabling keeps the existing selection; explicit selection changes context.
    app.slots[0].enabled = true;
    app.ensure_selection_usable();
    assert!(app.selection.selected.is_none());
    app.select_project(0);
    assert!(app.selection.selected.is_some());

    fs::remove_dir_all(&root).ok();
}

#[test]
fn frame_refs_resolve_round_trip() {
    let root = std::env::temp_dir().join(format!("fabsim_refs_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    write_min_run(&root.join("sample_a").join("warp_loading"), "a");
    write_single_frame(&root.join("sample_a").join("warp_loading"));

    let app = ViewerApp::new(vec![root.clone()], None, None);
    let fr = frame_ref_for(&app.slots, 0, 0, 0).expect("run has a frame 0");
    assert_eq!(resolve_frame_ref(&app.slots, &fr), Some((0, 0, 0)));

    // A frame id that does not exist does not resolve.
    let missing = FrameRef {
        project_root: fr.project_root.clone(),
        run_dir: fr.run_dir.clone(),
        frame_id: 999,
    };
    assert_eq!(resolve_frame_ref(&app.slots, &missing), None);

    fs::remove_dir_all(&root).ok();
}

#[test]
fn visibility_source_of_truth_round_trips() {
    let root = std::env::temp_dir().join(format!("fabsim_vis_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    write_min_run(&root.join("sample_a").join("warp_loading"), "a");
    write_min_run(&root.join("sample_b").join("warp_loading"), "b");
    write_single_frame(&root.join("sample_a").join("warp_loading"));

    let mut app = ViewerApp::new(vec![root.clone()], None, None);
    assert_eq!(app.slots.len(), 2);
    assert!(legend_hidden_ids(&app.slots).is_empty());

    // Hide one run via the Runs-tab flags: the hidden set reflects it.
    app.slots[0].entries[0].visible = false;
    let hidden = legend_hidden_ids(&app.slots);
    assert_eq!(hidden.len(), 1);
    assert!(hidden.contains(&app.slots[0].entries[0].legend_id));

    // Disabled projects are excluded from the hidden set entirely.
    app.slots[1].enabled = false;
    let hidden = legend_hidden_ids(&app.slots);
    assert_eq!(hidden.len(), 1, "disabled project contributes nothing");

    // Hidden runs must survive repeated legend read-back without becoming visible again.
    for _ in 0..3 {
        for slot in app.slots.iter().filter(|s| s.enabled) {
            for e in &slot.entries {
                let read_back = !hidden.contains(&e.legend_id);
                assert_eq!(
                    e.visible, read_back,
                    "read-back must agree with the hidden set (no bounce-back)"
                );
            }
        }
    }
    assert!(
        app.slots[1].entries[0].visible,
        "flags untouched by disabled sync"
    );

    fs::remove_dir_all(&root).ok();
}

#[test]
fn opening_an_already_open_project_is_deduplicated() {
    let root = std::env::temp_dir().join(format!("fabsim_dedup_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let project_dir = root.join("sample_a");
    write_min_run(&project_dir.join("warp_loading"), "a");

    // Open the project directory itself (a parent with direct runs opens
    // as one project, not as child imports).
    let mut app = ViewerApp::new(vec![project_dir.clone()], None, None);
    assert_eq!(app.slots.len(), 1);
    let err = app.open_project(project_dir.clone()).unwrap_err();
    assert_eq!(
        app.slots.len(),
        1,
        "reopening the same canonical root must not add a slot"
    );
    assert!(
        err.contains("Already open"),
        "the error explains the no-op: {err}"
    );
    app.open_path(project_dir.clone());
    assert!(
        app.error
            .as_deref()
            .is_some_and(|e| e.contains("Already open")),
        "and is shown in the status line: {:?}",
        app.error
    );

    fs::remove_dir_all(&root).ok();
}

/// A run with several frames, for navigation tests.
fn write_multi_frame_run(dir: &Path) {
    fs::write(
        dir.join("frames.json"),
        r#"{"frames":[
            {"frame_id":0,"step_index":0,"displacement_mm":0.0},
            {"frame_id":1,"step_index":1,"displacement_mm":1.0},
            {"frame_id":2,"step_index":2,"displacement_mm":2.0}
        ]}"#,
    )
    .unwrap();
}

fn write_tracking(path: &Path) {
    fs::write(
        path,
        r#"{"frames":[
            {"frame_id":0,"points":[{"id":"a","x":10.0,"y":20.0},{"id":"b","x":30.0,"y":40.0}]},
            {"frame_id":1,"points":[{"id":"a","x":15.0,"y":17.0},{"id":"b","x":35.0,"y":37.0}]},
            {"frame_id":2,"points":[{"id":"a","x":20.0,"y":14.0},{"id":"b","x":40.0,"y":34.0}]}
        ]}"#,
    )
    .unwrap();
}

#[test]
fn sequential_navigation_never_resets_registration_state() {
    // Navigation must leave per-run registration state unchanged.
    let root = std::env::temp_dir().join(format!("fabsim_nav_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let run_dir = root.join("sample_a").join("warp_loading");
    write_min_run(&run_dir, "a");
    write_multi_frame_run(&run_dir);
    fs::create_dir_all(&root).unwrap();

    let mut app = ViewerApp::new(vec![root.join("sample_a")], None, None);
    let run_key = app.slots[0].run_keys[0].clone();
    let tracking_path = root.join("tracks.json");
    write_tracking(&tracking_path);
    app.tracking.import_file(&run_key, &tracking_path).unwrap();
    app.tracking.run_state_mut(&run_key).unwrap().state.mode =
        fabric_data::tracking::registration::RegistrationMode::TranslationXY;
    app.tracking
        .run_state_mut(&run_key)
        .unwrap()
        .state
        .reference_frame = Some(fabric_data::tracking::FrameId(0));

    let before = app.tracking.to_persisted();

    app.step_selection(1, false);
    app.step_selection(1, false);
    app.step_selection(-1, false);
    app.playback.step(&app.slots, &mut app.selection);
    app.playback.step(&app.slots, &mut app.selection);

    assert_eq!(
        before,
        app.tracking.to_persisted(),
        "navigation must not mutate tracking state"
    );
    assert_eq!(
        app.tracking.run_state(&run_key).unwrap().state.mode,
        fabric_data::tracking::registration::RegistrationMode::TranslationXY,
        "registration mode survives navigation"
    );

    fs::remove_dir_all(&root).ok();
}

#[test]
fn playback_loops_and_stops_at_the_ends() {
    let root = std::env::temp_dir().join(format!("fabsim_play_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let run_dir = root.join("sample_a").join("warp_loading");
    write_min_run(&run_dir, "a");
    write_multi_frame_run(&run_dir);

    let mut app = ViewerApp::new(vec![root.join("sample_a")], None, None);
    app.selection.selected = frame_ref_for(&app.slots, 0, 0, 0);
    app.selection.hovered = None;
    app.selection.pinned = false;

    // Playback without loop stops on the last frame.
    app.playback.loop_enabled = false;
    app.playback.playing = true;
    for _ in 0..5 {
        app.playback.step(&app.slots, &mut app.selection);
    }
    assert!(!app.playback.playing, "playback stops at the end");
    let resolved = app
        .selection
        .active()
        .and_then(|fr| resolve_frame_ref(&app.slots, fr));
    assert_eq!(resolved, Some((0, 0, 2)), "ended on the last frame");

    // Playback with loop wraps back to the first frame.
    app.selection.selected = frame_ref_for(&app.slots, 0, 0, 2);
    app.playback.loop_enabled = true;
    app.playback.playing = true;
    app.playback.step(&app.slots, &mut app.selection);
    let resolved = app
        .selection
        .active()
        .and_then(|fr| resolve_frame_ref(&app.slots, fr));
    assert_eq!(resolved, Some((0, 0, 0)), "wrapped to the first frame");

    // Clear hover when playback takes over selection.
    app.selection.hovered = frame_ref_for(&app.slots, 0, 0, 1);
    app.playback.step(&app.slots, &mut app.selection);
    assert!(app.selection.hovered.is_none());

    fs::remove_dir_all(&root).ok();
}

#[test]
fn closing_a_project_forgets_its_tracking_state() {
    let root = std::env::temp_dir().join(format!("fabsim_close_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let run_dir = root.join("sample_a").join("warp_loading");
    write_min_run(&run_dir, "a");
    write_multi_frame_run(&run_dir);
    fs::create_dir_all(&root).unwrap();

    let mut app = ViewerApp::new(vec![root.join("sample_a")], None, None);
    let run_key = app.slots[0].run_keys[0].clone();
    let tracking_path = root.join("tracks.json");
    write_tracking(&tracking_path);
    app.tracking.import_file(&run_key, &tracking_path).unwrap();
    assert!(app.tracking.any_loaded());

    app.close_project(0);
    assert!(
        !app.tracking.any_loaded(),
        "tracking state dropped with the project"
    );

    fs::remove_dir_all(&root).ok();
}

#[test]
fn tracking_settings_persist_through_eframe_style_storage() {
    // Round-trip tracking choices through eframe storage.
    use crate::tracked::PersistedTracking;
    #[derive(Default)]
    struct MemoryStorage(std::collections::BTreeMap<String, String>);
    impl eframe::Storage for MemoryStorage {
        fn get_string(&self, key: &str) -> Option<String> {
            self.0.get(key).cloned()
        }
        fn set_string(&mut self, key: &str, value: String) {
            self.0.insert(key.to_owned(), value);
        }
        fn remove_string(&mut self, key: &str) {
            self.0.remove(key);
        }
        fn flush(&mut self) {}
    }

    let root = std::env::temp_dir().join(format!("fabsim_persist_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let run_dir = root.join("sample_a").join("warp_loading");
    write_min_run(&run_dir, "a");
    write_multi_frame_run(&run_dir);
    fs::create_dir_all(&root).unwrap();
    let tracking_path = root.join("tracks.json");
    write_tracking(&tracking_path);

    let mut storage = MemoryStorage::default();
    {
        let mut app = ViewerApp::new(vec![root.join("sample_a")], Some(&storage), None);
        let run_key = app.slots[0].run_keys[0].clone();
        app.tracking.import_file(&run_key, &tracking_path).unwrap();
        {
            let run = app.tracking.run_state_mut(&run_key).unwrap();
            run.state.mode = fabric_data::tracking::registration::RegistrationMode::Rigid;
            run.state.reference_frame = Some(fabric_data::tracking::FrameId(2));
        }
        app.save(&mut storage);
    }
    // Persist only the import path, not source data.
    let stored = &storage.0["tracking"];
    assert!(
        !stored.contains("frames"),
        "source points must not be embedded in storage"
    );
    assert!(stored.contains("tracks.json"), "the import path is stored");

    // Restore choices now; reload source data when the run is viewed.
    let mut app = ViewerApp::new(vec![root.join("sample_a")], Some(&storage), None);
    assert!(!app.tracking.any_loaded(), "nothing is parsed at startup");
    let run_key = app.slots[0].run_keys[0].clone();
    app.tracking.auto_discover(&run_key).unwrap();
    assert!(
        app.tracking.any_loaded(),
        "discovery re-imports the persisted path while it exists"
    );
    let run = app.tracking.run_state(&run_key).unwrap();
    assert_eq!(
        run.state.mode,
        fabric_data::tracking::registration::RegistrationMode::Rigid,
        "persisted mode restored"
    );
    assert_eq!(
        run.state.reference_frame,
        Some(fabric_data::tracking::FrameId(2))
    );
    assert_eq!(
        run.import_path.as_deref(),
        Some(tracking_path.as_path()),
        "the source path is remembered"
    );

    let _: PersistedTracking =
        ron::de::from_str(&ron::ser::to_string(&app.tracking.to_persisted()).unwrap()).unwrap();

    fs::remove_dir_all(&root).ok();
}

/// A minimal in-memory eframe storage.
#[derive(Default)]
struct MemoryStorage(std::collections::BTreeMap<String, String>);

impl eframe::Storage for MemoryStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.0.get(key).cloned()
    }
    fn set_string(&mut self, key: &str, value: String) {
        self.0.insert(key.to_owned(), value);
    }
    fn remove_string(&mut self, key: &str) {
        self.0.remove(key);
    }
    fn flush(&mut self) {}
}

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("fabsim_app_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    root
}

#[test]
fn the_inspector_follows_the_pinned_selection_not_hover() {
    let root = temp_root("insp");
    write_min_run(&root.join("sample_a").join("warp_loading"), "a");
    write_min_run(&root.join("sample_a").join("warp_release"), "a");
    write_single_frame(&root.join("sample_a").join("warp_loading"));
    write_single_frame(&root.join("sample_a").join("warp_release"));

    let app = ViewerApp::new(vec![root.join("sample_a")], None, None);
    let slots = &app.slots;
    let r1 = frame_ref_for(slots, 0, 0, 0).expect("frame 0 of warp_loading");
    let r2 = frame_ref_for(slots, 0, 1, 0).expect("frame 0 of warp_release");

    let mut state = InspectorState::default();
    state.follow(Some(&r1), slots);
    assert_eq!(state.resolve(slots), Some((0, 0)));
    state.follow(Some(&r1), slots);
    assert_eq!(state.resolve(slots), Some((0, 0)));
    state.follow(Some(&r2), slots);
    assert_eq!(state.resolve(slots), Some((0, 1)));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn closing_an_earlier_project_keeps_the_inspector_on_its_run() {
    // Closing an earlier project must not retarget the inspector.
    let root = temp_root("insp_close");
    for sample in ["sample_a", "sample_b"] {
        let run = root.join(sample).join("warp_loading");
        write_min_run(&run, sample);
        write_single_frame(&run);
    }
    let mut app = ViewerApp::new(vec![root.clone()], None, None);
    assert_eq!(app.slots.len(), 2);
    let b_root = app.slots[1].root_key.clone();
    let b_ref = frame_ref_for(&app.slots, 1, 0, 0).unwrap();
    app.inspector.follow(Some(&b_ref), &app.slots);
    assert_eq!(app.inspector.resolve(&app.slots), Some((1, 0)));

    app.close_project(0);
    assert_eq!(app.slots[0].root_key, b_root);
    assert_eq!(
        app.inspector.resolve(&app.slots),
        Some((0, 0)),
        "still the run of the project that stayed open"
    );
    app.close_project(0);
    assert_eq!(app.inspector.resolve(&app.slots), None, "closed run");
    fs::remove_dir_all(&root).ok();
}

#[test]
fn closing_a_project_drops_the_latched_context_menu_target() {
    let root = temp_root("ctx");
    let run = root.join("sample_a").join("warp_loading");
    write_min_run(&run, "a");
    write_single_frame(&run);
    let mut app = ViewerApp::new(vec![root.join("sample_a")], None, None);
    app.plot.context_target = frame_ref_for(&app.slots, 0, 0, 0);
    app.close_project(0);
    assert!(app.plot.context_target.is_none());
    fs::remove_dir_all(&root).ok();
}

#[test]
fn a_viewer_project_round_trips_visibility_by_run_name() {
    let root = temp_root("fvproj");
    let sample = root.join("sample_a");
    for run in ["warp_loading", "warp_release"] {
        write_min_run(&sample.join(run), "a");
        write_single_frame(&sample.join(run));
    }
    let mut app = ViewerApp::new(vec![sample.clone()], None, None);
    app.slots[0].name = "renamed".into();
    let hidden = app.slots[0].run_name(1);
    app.slots[0].entries[1].visible = false;
    let file = root.join("viewer.fvproj");
    app.save_viewer_project(file.clone());
    assert!(app.error.is_none(), "{:?}", app.error);
    let text = fs::read_to_string(&file).unwrap();
    assert!(
        text.contains(&hidden) && !text.contains("view_3d"),
        "{text}"
    );

    // Visibility must survive changes to the discovered run list.
    write_min_run(&sample.join("a_new_run"), "a");
    let mut other = ViewerApp::new(Vec::new(), None, None);
    other.load_viewer_project(file);
    assert!(other.error.is_none(), "{:?}", other.error);
    assert_eq!(other.slots[0].name, "renamed");
    for (i, entry) in other.slots[0].entries.iter().enumerate() {
        assert_eq!(entry.visible, other.slots[0].run_name(i) != hidden);
    }
    fs::remove_dir_all(&root).ok();
}

#[test]
fn loading_a_viewer_project_reports_every_failing_dataset() {
    let root = temp_root("fvproj_err");
    let good = root.join("good");
    write_min_run(&good.join("warp_loading"), "g");
    let project = ViewerProjectState {
        datasets: vec![
            DatasetState {
                source_path: root.join("missing_one"),
                display_name: "first".into(),
                ..DatasetState::default()
            },
            DatasetState {
                source_path: good.clone(),
                display_name: "good".into(),
                ..DatasetState::default()
            },
            DatasetState {
                source_path: root.join("missing_two"),
                display_name: "last".into(),
                ..DatasetState::default()
            },
        ],
        ..ViewerProjectState::default()
    };
    let file = root.join("p.fvproj");
    fs::write(&file, serde_json::to_string(&project).unwrap()).unwrap();

    let mut app = ViewerApp::new(Vec::new(), None, None);
    app.load_viewer_project(file);
    assert_eq!(app.slots.len(), 1, "the readable dataset still loads");
    let error = app.error.expect("failures are surfaced");
    assert!(
        error.contains("first") && error.contains("last"),
        "all failures are listed, not just the last: {error}"
    );
    fs::remove_dir_all(&root).ok();
}

#[test]
fn loading_a_viewer_project_cleans_up_the_previous_projects() {
    let root = temp_root("fvproj_clean");
    let run = root.join("sample_a").join("warp_loading");
    write_min_run(&run, "a");
    write_multi_frame_run(&run);
    let mut app = ViewerApp::new(vec![root.join("sample_a")], None, None);
    let run_key = app.slots[0].run_keys[0].clone();
    let tracking_path = root.join("tracks.json");
    write_tracking(&tracking_path);
    app.tracking.import_file(&run_key, &tracking_path).unwrap();
    app.plot.context_target = frame_ref_for(&app.slots, 0, 0, 0);

    let file = root.join("empty.fvproj");
    fs::write(
        &file,
        serde_json::to_string(&ViewerProjectState::default()).unwrap(),
    )
    .unwrap();
    app.load_viewer_project(file);
    assert!(app.slots.is_empty());
    assert!(
        !app.tracking.any_loaded(),
        "tracking of closed runs is dropped"
    );
    assert!(app.plot.context_target.is_none());
    fs::remove_dir_all(&root).ok();
}

#[test]
fn a_new_press_view_never_reuses_an_id_of_the_restored_layout() {
    // Restored dock IDs must be reserved before creating another view.
    let mut layout = Workspace::new();
    layout.open_tab(WorkspaceTab::Press(0));
    layout.open_tab(WorkspaceTab::Press(1));
    let mut storage = MemoryStorage::default();
    eframe::set_value(&mut storage, DOCK_KEY, &layout.dock_state);

    let mut app = ViewerApp::new(Vec::new(), Some(&storage), None);
    assert_eq!(app.workspace.press_ids(), vec![0, 1]);
    app.add_press_view();
    assert_eq!(app.workspace.press_ids(), vec![0, 1, 2]);
}

#[test]
fn a_saved_layout_with_the_removed_3d_view_still_restores() {
    let mut storage = MemoryStorage::default();
    let mut dock = ron::to_string(&Workspace::new().dock_state).unwrap();
    assert!(dock.contains("Runs"));
    dock = dock.replace("Runs", "ThreeD");
    eframe::Storage::set_string(&mut storage, DOCK_KEY, dock);

    let app = ViewerApp::new(Vec::new(), Some(&storage), None);
    for tab in [
        WorkspaceTab::Curves,
        WorkspaceTab::ImageViewer,
        WorkspaceTab::Inspector,
    ] {
        assert!(app.workspace.has_tab(&tab), "{tab:?} survives");
    }
    assert_eq!(
        app.workspace.dock_state.iter_all_tabs().count(),
        3,
        "the 3D tab is dropped"
    );
}

#[test]
fn old_settings_with_the_removed_options_still_load_through_storage() {
    let mut storage = MemoryStorage::default();
    eframe::Storage::set_string(
        &mut storage,
        SETTINGS_KEY,
        "(ui:(theme:Dark,advanced_visualization:true),images:(relight:(enabled:true,azimuth_deg:12.0)),)"
            .into(),
    );
    let app = ViewerApp::new(Vec::new(), Some(&storage), None);
    assert!(app.settings.images.relight.enabled);
    assert_eq!(app.settings.ui.theme, crate::settings::Theme::Dark);
}

#[test]
fn playback_preferences_survive_a_restart() {
    let mut storage = MemoryStorage::default();
    let mut app = ViewerApp::new(Vec::new(), None, None);
    app.playback.fps = 12.0;
    app.playback.loop_enabled = false;
    app.save(&mut storage);
    let restored = ViewerApp::new(Vec::new(), Some(&storage), None);
    assert_eq!(restored.playback.fps, 12.0);
    assert!(!restored.playback.loop_enabled);
}

#[test]
fn press_views_survive_a_restart_together_with_the_dock_layout() {
    let root = std::env::temp_dir().join(format!("fabsim_press_restart_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let sample = root.join("sample_a");
    write_min_run(&sample.join("warp_loading"), "a");
    write_single_frame(&sample.join("warp_loading"));

    let mut first = ViewerApp::new(vec![sample.clone()], None, None);
    first.add_press_view();
    let id = *first.workspace.press_ids().last().unwrap();
    {
        let mut state = first.press.get_or_create(id).persisted(&first.slots);
        state.force_n = Some(7.5);
        state.orientation = "weft".into();
        first.press.get_or_create(id).restore(state, &first.slots);
    }
    let mut storage = MemoryStorage::default();
    first.save(&mut storage);

    let again = ViewerApp::new(vec![sample.clone()], Some(&storage), None);
    let restored = again.press.get(id).unwrap().persisted(&again.slots);
    assert_eq!(restored.force_n, Some(7.5));
    assert_eq!(restored.orientation, "weft");

    // Retain pending view state across restarts without its project open.
    let mut empty = ViewerApp::new(Vec::new(), Some(&storage), None);
    let mut storage2 = MemoryStorage::default();
    empty.save(&mut storage2);
    let mut later = ViewerApp::new(Vec::new(), Some(&storage2), None);
    later.open_path(sample);
    assert_eq!(
        later.press.get(id).unwrap().persisted(&later.slots).force_n,
        Some(7.5)
    );
    fs::remove_dir_all(&root).ok();
}
