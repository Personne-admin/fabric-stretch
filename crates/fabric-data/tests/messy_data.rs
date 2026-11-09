//! Coverage for incomplete captures and older metadata formats.

use fabric_data::model::{Branch, Direction, ForceFile, ForceStatus, FramesFile, Session};
use fabric_data::Run;
use std::path::PathBuf;

#[test]
fn frame_images_are_collected_dynamically() {
    // fabric_001 frames have no image_diffuse, only ps_1..ps_3.
    let frames: FramesFile = serde_json::from_str(
        r#"{
            "sample_id": "s", "test_id": "t", "direction": "warp", "branch": "loading",
            "frames": [{
                "frame_id": 0, "step_index": 0, "displacement_mm": 0.0,
                "force_json_index": 0, "line_load_N_per_m": 1.5,
                "is_relaxation_hold": false,
                "image_ps_1": "a.png", "image_ps_2": "b.png", "image_ps_3": "c.png"
            }]
        }"#,
    )
    .unwrap();
    let roles: Vec<&str> = frames.frames[0].images().map(|(r, _)| r).collect();
    assert_eq!(roles, ["ps_1", "ps_2", "ps_3"]);
    assert_eq!(frames.frames[0].line_load_n_per_m, Some(1.5));
    assert!(frames.frames[0].extra.contains_key("image_ps_1"));
}

#[test]
fn null_and_non_ok_force_records_parse_and_are_unusable() {
    let force: ForceFile = serde_json::from_str(
        r#"{"forces": [
            {"index": 0, "label": "0", "timestamp": null, "raw": null, "value": null,
             "force_n": null, "status": "timeout"},
            {"index": 1, "label": "1", "timestamp": "x", "raw": "tension: 2.0",
             "value": "2.0", "force_n": 2.0, "status": "invalid"},
            {"index": 2, "label": "2", "timestamp": "x", "raw": "tension: 3.0",
             "value": "3.0", "force_n": 3.0, "status": "ok"}
        ]}"#,
    )
    .unwrap();
    assert_eq!(force.forces[0].status, ForceStatus::Timeout);
    assert_eq!(force.forces[0].usable_force_n(), None);
    // Non-ok status makes the value untrustworthy even if present.
    assert_eq!(force.forces[1].usable_force_n(), None);
    assert_eq!(force.forces[2].usable_force_n(), Some(3.0));
}

#[test]
fn frame_force_follows_the_index_not_the_position() {
    let session = Session::new("s", "t", Direction::Warp, Branch::Loading, 3);
    let frames: FramesFile = serde_json::from_str(
        r#"{
        "frames":[
          {"frame_id":0,"displacement_mm":0.0,"force_json_index":7,"line_load_N_per_m":248.0},
          {"frame_id":1,"displacement_mm":1.0,"force_json_index":99,"line_load_N_per_m":null},
          {"frame_id":2,"displacement_mm":2.0,"force_json_index":8},
          {"frame_id":3,"displacement_mm":3.0}
        ]
    }"#,
    )
    .unwrap();
    // Record 8 sits *before* record 7 in the list and is not ok.
    let force: ForceFile = serde_json::from_str(
        r#"{"forces":[
        {"index":8,"force_n":9.0,"status":"invalid"},
        {"index":7,"force_n":12.4,"status":"ok"}
    ]}"#,
    )
    .unwrap();
    let run = Run {
        dir: PathBuf::new(),
        session,
        frames,
        force,
        load_notes: vec![],
    };
    let f = &run.frames.frames;
    assert_eq!(run.force_n_for_frame(&f[0]), Some(12.4));
    assert_eq!(run.force_for_frame(&f[0]).map(|r| r.index), Some(7));
    assert_eq!(run.force_n_for_frame(&f[1]), None, "dangling link");
    assert_eq!(run.force_n_for_frame(&f[2]), None, "status is not ok");
    assert!(
        run.force_for_frame(&f[2]).is_some(),
        "record still resolves"
    );
    assert_eq!(run.force_n_for_frame(&f[3]), None, "no link at all");
}

#[test]
fn duplicate_force_indices_resolve_to_the_first_record() {
    let force: ForceFile = serde_json::from_str(
        r#"{"forces":[
        {"index":3,"force_n":1.0,"status":"ok"},
        {"index":3,"force_n":2.0,"status":"ok"}
    ]}"#,
    )
    .unwrap();
    let run = Run {
        dir: PathBuf::new(),
        session: Session::new("s", "t", Direction::Warp, Branch::Loading, 1),
        frames: FramesFile::default(),
        force,
        load_notes: vec![],
    };
    let mut frame = fabric_data::Frame::new(0, 0.0);
    frame.force_json_index = Some(3);
    assert_eq!(run.force_n_for_frame(&frame), Some(1.0));
}

#[test]
fn unknown_direction_and_missing_optionals_still_parse() {
    let session: Session = serde_json::from_str(
        r#"{
            "sample_id": "s", "test_id": "t", "direction": "diagonal",
            "branch": "loading", "count": 5,
            "abort_reason": "user_stop"
        }"#,
    )
    .unwrap();
    assert_eq!(session.direction, Direction::Other("diagonal".into()));
    assert!(session.abort_reason.is_some());
    assert_eq!(session.frames_written, None);
}

#[test]
fn explicit_motion_and_motor_mechanics_session_is_readable() {
    let session: Session = serde_json::from_str(
        r#"{
            "sample_id": "s", "test_id": "t", "direction": "weft",
            "branch": "release", "motion_mode": "release", "count": 80,
            "step_size": -10000, "commanded_motor_steps_per_increment": 10000,
            "screw_pitch_um_per_output_rev": 1500, "gear_ratio": 49,
            "motor_full_steps_per_rev": 200, "microsteps": 1,
            "motor_steps_per_output_rev": 9800,
            "linear_um_per_motor_step": 0.1530612245
        }"#,
    )
    .unwrap();
    assert_eq!(session.branch.to_string(), "release");
    assert_eq!(session.mechanics.motion_mode.as_deref(), Some("release"));
    assert_eq!(session.mechanics.motor_steps_per_output_rev, Some(9800.0));
    assert_eq!(
        session.mechanics.commanded_motor_steps_per_increment,
        Some(10000.0)
    );
}

#[test]
fn empty_frames_list_is_fine() {
    let frames: FramesFile = serde_json::from_str(r#"{"sample_id": "s", "frames": []}"#).unwrap();
    assert!(frames.frames.is_empty());
    let frames: FramesFile = serde_json::from_str(r#"{"sample_id": "s"}"#).unwrap();
    assert!(frames.frames.is_empty());
}

#[test]
fn current_rig_output_is_fully_typed() {
    let session: Session = serde_json::from_str(
        r#"{
            "sample_id": "s", "test_id": "t", "direction": "warp", "branch": "loading",
            "count": 72, "motion_mode": "loading",
            "commanded_motor_steps_per_increment": 8000,
            "requested_displacement_um_per_increment": 1224.49,
            "screw_pitch_um_per_output_rev": 1500.0, "gear_ratio": 49.0,
            "motor_full_steps_per_rev": 200, "microsteps": 1,
            "motor_steps_per_output_rev": 9800,
            "linear_um_per_motor_step": 0.15306122448979592,
            "led_mask": 85.0, "led_power": 128, "settle_ms": 50.0,
            "relax_duration_ms": 4999.0, "frames_written": 72.0,
            "color_calibration": {"applied": false}
        }"#,
    )
    .unwrap();
    let m = &session.mechanics;
    assert_eq!(m.motion_mode.as_deref(), Some("loading"));
    assert_eq!(m.gear_ratio, Some(49.0));
    assert_eq!(m.linear_um_per_motor_step, Some(0.15306122448979592));
    assert_eq!(m.motor_steps_per_output_rev, Some(9800.0));
    // Integral floats are accepted for integer metadata.
    assert_eq!(session.led_mask, Some(85));
    assert_eq!(session.settle_ms, Some(50));
    assert_eq!(session.relax_duration_ms, Some(4999));
    assert_eq!(session.frames_written, Some(72));
    // Typed keys leave `extra`; unknown ones stay.
    assert!(!session.extra.contains_key("gear_ratio"));
    assert!(session.extra.contains_key("color_calibration"));
}

#[test]
fn non_integral_counter_is_rejected() {
    let err = serde_json::from_str::<Session>(
        r#"{"sample_id":"s","test_id":"t","direction":"warp","branch":"loading",
            "count":1,"settle_ms":50.5}"#,
    );
    assert!(err.is_err());
}

#[test]
fn force_records_keep_calibration_counts() {
    let force: ForceFile = serde_json::from_str(
        r#"{"forces":[{"index":0,"label":"0","timestamp":"x","raw":"TENSION",
            "value":-4.3,"force_n":-4.3,"raw_counts":-284,"zero_counts":-628.0,
            "calibration_n_per_count":0.0125,"status":"ok","note":"kept"}]}"#,
    )
    .unwrap();
    let r = &force.forces[0];
    assert_eq!((r.raw_counts, r.zero_counts), (Some(-284), Some(-628)));
    assert_eq!(r.calibration_n_per_count, Some(0.0125));
    assert_eq!(
        r.usable_force_n(),
        Some(-4.3),
        "negative tare offsets are fine"
    );
    assert!(r.extra.contains_key("note"));
}

#[test]
fn frame_level_force_row_and_frames_file_extras_are_kept() {
    let frames: FramesFile = serde_json::from_str(
        r#"{"sample_id":"s","gear_ratio":49.0,"motion_mode":"loading",
            "frames":[{"frame_id":0,"displacement_mm":0.0,"force_csv_row":4,"settle_ms":50.0}]}"#,
    )
    .unwrap();
    assert_eq!(frames.frames[0].force_csv_row, Some(4));
    assert_eq!(frames.frames[0].settle_ms, Some(50));
    assert_eq!(frames.extra["gear_ratio"], 49.0);
}
