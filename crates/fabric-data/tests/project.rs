//! Fixture runs: clean warp loading, truncated warp release, and weft loading
//! with missing images, a timed-out force reading and a relaxation hold.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use fabric_data::{
    validate_run, Branch, Curve, DataError, Direction, ForceFile, Frame, FramesFile, Project, Run,
    RunKey, Session, Severity,
};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/project")
}

fn run_named<'p>(project: &'p Project, dir: &str) -> &'p Run {
    project
        .runs
        .iter()
        .find(|r| r.dir.file_name().is_some_and(|n| n == dir))
        .unwrap_or_else(|| panic!("run {dir} not discovered"))
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn discovery_finds_runs_and_skips_hidden_and_package_dirs() {
    let project = Project::discover(fixture()).unwrap();
    assert!(project.load_errors.is_empty(), "{:?}", project.load_errors);
    let dirs: Vec<String> = project
        .runs
        .iter()
        .map(|r| r.dir.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    // Sorted by (sample, direction, branch): warp < weft, loading < release.
    assert_eq!(dirs, ["warp_loading", "warp_release", "weft_loading"]);
}

#[test]
fn runs_are_grouped_by_sample_direction_and_branch() {
    let project = Project::discover(fixture()).unwrap();
    let groups = project.grouped();
    assert_eq!(groups.len(), 3);
    let key = RunKey {
        sample_id: "fixture".into(),
        direction: Direction::Weft,
        branch: Branch::Loading,
    };
    assert_eq!(groups[&key].len(), 1);
}

#[test]
fn a_run_directory_can_be_the_root() {
    let project = Project::discover(fixture().join("warp_loading")).unwrap();
    assert_eq!(project.runs.len(), 1);
}

#[test]
fn nested_layouts_load_down_to_three_levels() {
    let tmp = tempfile::tempdir().unwrap();
    // Runs three levels below the root are found, a fourth level is not.
    copy_dir(
        &fixture().join("warp_loading"),
        &tmp.path().join("a/b/warp_loading"),
    );
    copy_dir(
        &fixture().join("weft_loading"),
        &tmp.path().join("a/weft_loading"),
    );
    copy_dir(
        &fixture().join("warp_release"),
        &tmp.path().join("a/b/c/warp_release"),
    );
    let project = Project::discover(tmp.path()).unwrap();
    let names: Vec<_> = project
        .runs
        .iter()
        .map(|r| r.dir.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["warp_loading", "weft_loading"]);
}

#[test]
fn unparseable_session_is_a_load_error_and_other_runs_still_load() {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(&fixture().join("warp_loading"), &tmp.path().join("good"));
    fs::create_dir_all(tmp.path().join("bad")).unwrap();
    fs::write(tmp.path().join("bad/session.json"), "{ not json").unwrap();
    let project = Project::discover(tmp.path()).unwrap();
    assert_eq!(project.runs.len(), 1);
    assert_eq!(project.load_errors.len(), 1);
    assert!(matches!(project.load_errors[0].1, DataError::Parse { .. }));
}

#[test]
fn a_missing_root_is_an_error() {
    assert!(Project::discover(fixture().join("does_not_exist")).is_err());
}

#[cfg(unix)]
#[test]
fn an_unreadable_subdirectory_does_not_abort_discovery() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    copy_dir(
        &fixture().join("warp_loading"),
        &tmp.path().join("warp_loading"),
    );
    let locked = tmp.path().join("locked");
    fs::create_dir(&locked).unwrap();
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    // Permission bits do not bind root; nothing to test then.
    let enforced = fs::read_dir(&locked).is_err();

    let project = Project::discover(tmp.path());
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();
    let project = project.expect("discovery must survive an unreadable subdirectory");
    assert_eq!(project.runs.len(), 1);
    if enforced {
        assert_eq!(project.load_errors.len(), 1);
        assert!(matches!(project.load_errors[0].1, DataError::Io { .. }));
    }
}

#[test]
fn damaged_frames_or_force_files_degrade_to_notes() {
    let project = Project::discover(fixture()).unwrap();
    let release = run_named(&project, "warp_release");
    assert!(release.force.forces.is_empty());
    assert_eq!(release.load_notes.len(), 1);
    assert!(release.load_notes[0].starts_with("force.json unusable"));
    assert_eq!(release.frames.frames.len(), 2, "frames.json still loads");
}

#[test]
fn current_rig_output_parses_from_disk() {
    let project = Project::discover(fixture()).unwrap();
    let run = run_named(&project, "warp_loading");
    let s = &run.session;
    assert_eq!(s.led_mask, Some(85), "float-typed counter");
    assert_eq!(s.mechanics.gear_ratio, Some(49.0));
    assert_eq!(s.frames_written, Some(2));
    assert_eq!(run.force.forces[0].raw_counts, Some(40));
    assert_eq!(run.frames.frames[1].force_csv_row, Some(1));
}

#[test]
fn severities_follow_the_problems() {
    let project = Project::discover(fixture()).unwrap();

    let clean = validate_run(run_named(&project, "warp_loading"));
    assert_eq!(clean.severity(), Severity::Ok, "{:?}", clean.issues());
    assert!(clean.issues().is_empty());
    assert_eq!(clean.images_referenced, 2);
    assert_eq!(clean.displacement_range, Some((0.0, 1.0)));

    // Truncated force.json: loaded with a note, so the links are unresolved.
    let aborted = validate_run(run_named(&project, "warp_release"));
    assert_eq!(aborted.severity(), Severity::Error);
    let issues = aborted.issues().join("\n");
    assert!(issues.contains("run aborted: user_stop"), "{issues}");
    assert!(
        issues.contains("frame count 2 != configured count 4"),
        "{issues}"
    );
    assert!(
        issues.contains("2 frame(s) with unresolved force link"),
        "{issues}"
    );
    assert!(issues.contains("force.json unusable"), "{issues}");

    // Missing images and a timed-out reading are warnings, not errors.
    let weft = validate_run(run_named(&project, "weft_loading"));
    assert_eq!(weft.severity(), Severity::Warning);
    assert_eq!(weft.images_missing.len(), 4);
    assert_eq!(weft.force_non_ok, 1);
    assert_eq!(weft.force_null, 1);
    assert_eq!(weft.unresolved_force_links, 0);
    assert!(weft.displacement_monotonic, "holds repeat a displacement");
}

#[test]
fn frames_written_must_match_the_frame_count() {
    let project = Project::discover(fixture()).unwrap();
    let mut run = run_named(&project, "warp_loading").clone();
    run.session.frames_written = Some(1);
    let report = validate_run(&run);
    assert_eq!(report.severity(), Severity::Warning);
    assert!(
        report
            .issues()
            .iter()
            .any(|i| i == "frame count 2 != frames_written 1"),
        "{:?}",
        report.issues()
    );
}

#[test]
fn duplicate_frame_ids_are_an_error() {
    let project = Project::discover(fixture()).unwrap();
    let mut run = run_named(&project, "warp_loading").clone();
    run.frames.frames[1].frame_id = 0;
    let report = validate_run(&run);
    assert_eq!(report.duplicate_frame_ids, 1);
    assert_eq!(report.severity(), Severity::Error);
    assert!(report
        .issues()
        .iter()
        .any(|i| i.contains("duplicate frame_id")));
}

#[test]
fn duplicate_force_indices_are_reported() {
    let project = Project::discover(fixture()).unwrap();
    let mut run = run_named(&project, "warp_loading").clone();
    run.force.forces[1].index = 0;
    let report = validate_run(&run);
    assert_eq!(report.duplicate_force_indices, 1);
    assert_eq!(report.unresolved_force_links, 1, "index 1 no longer exists");
    assert!(report
        .issues()
        .iter()
        .any(|i| i.contains("duplicate record index")));
}

#[test]
fn non_monotonic_displacement_is_an_error() {
    let project = Project::discover(fixture()).unwrap();
    let mut run = run_named(&project, "warp_loading").clone();
    run.frames.frames[1].displacement_mm = -1.0;
    let report = validate_run(&run);
    assert!(!report.displacement_monotonic);
    assert_eq!(report.severity(), Severity::Error);
}

#[test]
fn an_empty_run_is_an_error() {
    let run = Run {
        dir: PathBuf::new(),
        session: Session::new("s", "t", Direction::Warp, Branch::Loading, 3),
        frames: FramesFile::default(),
        force: ForceFile::default(),
        load_notes: vec![],
    };
    let report = validate_run(&run);
    assert_eq!(report.severity(), Severity::Error);
    assert!(report.issues().iter().any(|i| i == "no frames"));
    assert_eq!(report.displacement_range, None);
}

#[test]
fn curve_series_use_the_linked_force_and_line_load() {
    let project = Project::discover(fixture()).unwrap();
    let curve = Curve::from_run(run_named(&project, "warp_loading"));
    assert_eq!(curve.test_id, "warp_stretch_20260101_100000");
    assert_eq!(curve.force_series(false), [(0.0, 0.5), (1.0, 1.5)]);
    assert_eq!(curve.line_load_series(false), [(0.0, 2.5), (1.0, 7.5)]);
    assert_eq!(curve.max_displacement_mm(), Some(1.0));
}

#[test]
fn curve_skips_unusable_forces_and_separates_relaxation_holds() {
    let project = Project::discover(fixture()).unwrap();
    let curve = Curve::from_run(run_named(&project, "weft_loading"));
    assert_eq!(curve.points.len(), 4);
    // Frame 2 is a relaxation hold whose reading timed out.
    assert_eq!(curve.points[2].force_n, None);
    assert!(curve.points[2].is_relaxation_hold);
    assert_eq!(
        curve.force_series(true),
        [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    );
    assert_eq!(
        curve.force_series(false),
        [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    );
    // The hold still carries a line load.
    assert_eq!(curve.line_load_series(false).len(), 3);
    assert_eq!(curve.line_load_series(true).len(), 4);
}

#[test]
fn curve_of_a_run_without_force_file_has_no_forces() {
    let project = Project::discover(fixture()).unwrap();
    let curve = Curve::from_run(run_named(&project, "warp_release"));
    assert_eq!(curve.points.len(), 2);
    assert!(curve.force_series(true).is_empty());
    assert_eq!(curve.line_load_series(true).len(), 2);
}

#[test]
fn curve_force_follows_the_frame_link_not_the_list_position() {
    let project = Project::discover(fixture()).unwrap();
    let mut run = run_named(&project, "warp_loading").clone();
    run.force.forces.reverse();
    let curve = Curve::from_run(&run);
    assert_eq!(curve.force_series(false), [(0.0, 0.5), (1.0, 1.5)]);
    // A frame pointing at a missing record has no force.
    run.frames.frames.push(Frame {
        force_json_index: Some(42),
        ..Frame::new(2, 2.0)
    });
    assert_eq!(Curve::from_run(&run).points[2].force_n, None);
}

#[test]
fn release_travel_is_mirrored_against_the_loading_maximum() {
    let project = Project::discover(fixture()).unwrap();
    let release = Curve::from_run(run_named(&project, "warp_release"));
    let loading = Curve::from_run(run_named(&project, "warp_loading"));
    let start = loading.max_displacement_mm().unwrap();
    assert_eq!(release.mirrored_displacement(start, 0.25), 0.75);
    assert_eq!(loading.mirrored_displacement(start, 0.25), 0.25);
}

fn validate_cli(args: &[&Path]) -> (Option<i32>, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_validate"))
        .args(args)
        .output()
        .expect("run the validate binary");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn cli_exits_1_when_a_run_has_errors() {
    let (code, stdout, _) = validate_cli(&[&fixture()]);
    assert_eq!(code, Some(1));
    assert!(stdout.contains("[ERROR]"), "{stdout}");
    assert!(stdout.contains("[OK]"), "{stdout}");
    assert!(stdout.contains("[WARN]"), "{stdout}");
    assert!(
        stdout.contains("3 run(s) in 3 group(s), 0 load error(s); worst severity: ERROR"),
        "{stdout}"
    );
}

#[test]
fn cli_exits_0_for_a_clean_project() {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(
        &fixture().join("warp_loading"),
        &tmp.path().join("warp_loading"),
    );
    let (code, stdout, _) = validate_cli(&[tmp.path()]);
    assert_eq!(code, Some(0), "{stdout}");
    assert!(stdout.contains("worst severity: OK"), "{stdout}");
}

#[test]
fn cli_exits_0_for_warnings_only() {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(
        &fixture().join("weft_loading"),
        &tmp.path().join("weft_loading"),
    );
    let (code, stdout, _) = validate_cli(&[tmp.path()]);
    assert_eq!(code, Some(0), "{stdout}");
    assert!(stdout.contains("worst severity: WARN"), "{stdout}");
}

#[test]
fn cli_exits_2_on_usage_and_empty_projects() {
    let (code, _, stderr) = validate_cli(&[]);
    assert_eq!(code, Some(2));
    assert!(stderr.contains("usage: validate"), "{stderr}");

    let tmp = tempfile::tempdir().unwrap();
    let (code, _, stderr) = validate_cli(&[tmp.path()]);
    assert_eq!(code, Some(2));
    assert!(stderr.contains("no runs found"), "{stderr}");

    let (code, _, stderr) = validate_cli(&[&tmp.path().join("missing")]);
    assert_eq!(code, Some(2));
    assert!(stderr.contains("failed to scan"), "{stderr}");
}

#[test]
fn cli_reports_load_errors_and_fails() {
    let tmp = tempfile::tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("bad")).unwrap();
    fs::write(tmp.path().join("bad/session.json"), "nope").unwrap();
    let (code, stdout, _) = validate_cli(&[tmp.path()]);
    assert_eq!(code, Some(1));
    assert!(stdout.contains("failed to load"), "{stdout}");
}
