//! Run JSON types. Optional fields allow older or aborted captures to load;
//! flattened `extra` maps preserve unmodelled keys.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// Stretch direction of a run. `Other` preserves unknown spellings instead of
/// failing deserialization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    Warp,
    Weft,
    Bias,
    Other(String),
}

/// Loading or release branch of the hysteresis loop.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Branch {
    Loading,
    Release,
    Other(String),
}

/// Status of a single load-cell reading.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ForceStatus {
    Ok,
    Timeout,
    Invalid,
    Other(String),
}

macro_rules! string_enum_deserialize {
    ($ty:ident { $($lit:literal => $variant:ident),+ $(,)? }) => {
        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let s = String::deserialize(d)?;
                Ok(match s.to_ascii_lowercase().as_str() {
                    $($lit => $ty::$variant,)+
                    _ => $ty::Other(s),
                })
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $($ty::$variant => f.write_str($lit),)+
                    $ty::Other(s) => f.write_str(s),
                }
            }
        }
    };
}

string_enum_deserialize!(Direction { "warp" => Warp, "weft" => Weft, "bias" => Bias });
string_enum_deserialize!(Branch { "loading" => Loading, "release" => Release });
string_enum_deserialize!(ForceStatus { "ok" => Ok, "timeout" => Timeout, "invalid" => Invalid });

/// Motor-step conversion metadata from session/frames JSON. Fields are optional for
/// older captures that only recorded `motor_step_um`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct MotorMechanics {
    /// `"loading"` or `"release"`: the direction the motor was driven in.
    pub motion_mode: Option<String>,
    pub commanded_motor_steps_per_increment: Option<f64>,
    pub requested_displacement_um_per_increment: Option<f64>,
    pub screw_pitch_um_per_output_rev: Option<f64>,
    pub gear_ratio: Option<f64>,
    pub motor_full_steps_per_rev: Option<f64>,
    pub microsteps: Option<f64>,
    pub motor_steps_per_output_rev: Option<f64>,
    /// Linear travel per motor step in micrometres.
    pub linear_um_per_motor_step: Option<f64>,
}

/// `session.json` - metadata for one acquisition run.
#[derive(Debug, Clone, Deserialize)]
pub struct Session {
    pub sample_id: String,
    pub test_id: String,
    pub direction: Direction,
    pub branch: Branch,
    /// Number of steps the run was configured for.
    pub count: u32,
    pub timestamp: Option<String>,
    /// Motor microsteps per increment.
    pub step_size: Option<i64>,
    /// Legacy: linear travel per motor step in micrometres, as written by
    /// early acquisitions. Current runs carry
    /// [`MotorMechanics::linear_um_per_motor_step`] instead.
    pub motor_step_um: Option<f64>,
    pub clamp_width_m: Option<f64>,
    pub max_stretch_mm: Option<f64>,
    pub max_force_n: Option<f64>,
    #[serde(default, deserialize_with = "lenient_int")]
    pub led_mask: Option<u32>,
    #[serde(default, deserialize_with = "lenient_int")]
    pub led_power: Option<u32>,
    #[serde(default, deserialize_with = "lenient_int")]
    pub settle_ms: Option<u32>,
    pub cam_exposure_ms: Option<f64>,
    pub speed_mult: Option<f64>,
    pub relax_enabled: Option<bool>,
    #[serde(default)]
    pub relax_indices: Vec<u32>,
    #[serde(default, deserialize_with = "lenient_int")]
    pub relax_duration_ms: Option<u64>,
    /// Non-null when the run stopped early, e.g. `"user_stop"` or
    /// `"firmware_force_limit (54.125 N)"`.
    pub abort_reason: Option<String>,
    /// Frames actually written before the run ended; may be < `count`.
    #[serde(default, deserialize_with = "lenient_int")]
    pub frames_written: Option<u32>,
    #[serde(flatten)]
    pub mechanics: MotorMechanics,
    /// Any session fields this struct doesn't model explicitly (for example
    /// the `color_calibration` provenance block).
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Session {
    /// A session with only the required fields set; every optional field is
    /// empty. Mainly useful to build runs in tests.
    pub fn new(
        sample_id: impl Into<String>,
        test_id: impl Into<String>,
        direction: Direction,
        branch: Branch,
        count: u32,
    ) -> Self {
        Session {
            sample_id: sample_id.into(),
            test_id: test_id.into(),
            direction,
            branch,
            count,
            timestamp: None,
            step_size: None,
            motor_step_um: None,
            clamp_width_m: None,
            max_stretch_mm: None,
            max_force_n: None,
            led_mask: None,
            led_power: None,
            settle_ms: None,
            cam_exposure_ms: None,
            speed_mult: None,
            relax_enabled: None,
            relax_indices: Vec::new(),
            relax_duration_ms: None,
            abort_reason: None,
            frames_written: None,
            mechanics: MotorMechanics::default(),
            extra: BTreeMap::new(),
        }
    }
}

/// `frames.json` - per-step records linking displacement, force and images.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct FramesFile {
    pub sample_id: Option<String>,
    pub test_id: Option<String>,
    pub direction: Option<Direction>,
    pub branch: Option<Branch>,
    pub clamp_width_m: Option<f64>,
    #[serde(default)]
    pub frames: Vec<Frame>,
    /// Remaining top-level keys (the drive-train description repeated from
    /// `session.json`, see [`MotorMechanics`]).
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Captured step. Collect variable `image_<role>` fields into [`Frame::images`].
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Frame {
    pub frame_id: u32,
    pub step_index: Option<u32>,
    pub displacement_mm: f64,
    /// Link into `force.json`: matched against [`ForceRecord::index`], never
    /// used as a list position.
    pub force_json_index: Option<i64>,
    /// Row of the rig's CSV force log this frame was read from.
    pub force_csv_row: Option<i64>,
    #[serde(rename = "line_load_N_per_m")]
    pub line_load_n_per_m: Option<f64>,
    #[serde(default)]
    pub is_relaxation_hold: bool,
    #[serde(default, deserialize_with = "lenient_int")]
    pub settle_ms: Option<u32>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Frame {
    /// A frame with only the required fields set.
    pub fn new(frame_id: u32, displacement_mm: f64) -> Self {
        Frame {
            frame_id,
            displacement_mm,
            ..Frame::default()
        }
    }

    /// `(role, filename)` pairs from the frame's `image_<role>` keys.
    pub fn images(&self) -> impl Iterator<Item = (&str, &str)> {
        self.extra.iter().filter_map(|(k, v)| {
            let role = k.strip_prefix("image_")?;
            Some((role, v.as_str()?))
        })
    }
}

/// `force.json` - the load-cell record log.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ForceFile {
    #[serde(default)]
    pub forces: Vec<ForceRecord>,
}

/// One load-cell reading. `force_n` is null on timeout/invalid readings.
#[derive(Debug, Clone, Deserialize)]
pub struct ForceRecord {
    /// Identity of the record; frames link to it via
    /// [`Frame::force_json_index`].
    pub index: i64,
    pub label: Option<String>,
    pub timestamp: Option<String>,
    pub raw: Option<String>,
    /// A string in older acquisition logs, a JSON number in newer ones.
    #[serde(default, deserialize_with = "string_or_number")]
    pub value: Option<String>,
    pub force_n: Option<f64>,
    /// ADC counts of the reading (current logs only).
    #[serde(default, deserialize_with = "lenient_int")]
    pub raw_counts: Option<i64>,
    /// ADC counts at the unloaded zero the reading was taken against.
    #[serde(default, deserialize_with = "lenient_int")]
    pub zero_counts: Option<i64>,
    /// Calibration slope: newtons per ADC count.
    pub calibration_n_per_count: Option<f64>,
    pub status: ForceStatus,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Accepts a JSON number or string for the textual `value` field.
fn string_or_number<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(Value::Number(n)) => Ok(Some(n.to_string())),
        Some(other) => Err(D::Error::custom(format!("unexpected force value {other}"))),
    }
}

/// Accept integer-valued floats as well as integers for optional counters.
fn lenient_int<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: TryFrom<i64>,
{
    use serde::de::Error;
    let Some(v) = Option::<f64>::deserialize(deserializer)? else {
        return Ok(None);
    };
    // 2^53 is the largest range in which every integer is exact in an f64.
    if v.fract() != 0.0 || v.abs() > 9_007_199_254_740_992.0 {
        return Err(D::Error::custom(format!("expected an integer, got {v}")));
    }
    T::try_from(v as i64)
        .map(Some)
        .map_err(|_| D::Error::custom(format!("integer {v} out of range")))
}

impl ForceRecord {
    /// The reading, but only when it is trustworthy: status `ok` and a finite
    /// value.
    pub fn usable_force_n(&self) -> Option<f64> {
        if self.status == ForceStatus::Ok {
            self.force_n.filter(|v| v.is_finite())
        } else {
            None
        }
    }
}

/// Identity of a run within a project: one sample stretched in one direction
/// along one branch.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RunKey {
    pub sample_id: String,
    pub direction: Direction,
    pub branch: Branch,
}

impl fmt::Display for RunKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} / {} {}", self.sample_id, self.direction, self.branch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_value_accepts_string_number_and_null() {
        let parse = |value: &str| -> ForceRecord {
            serde_json::from_str(&format!(
                r#"{{"index":0,"force_n":0.5875,"status":"ok","value":{value}}}"#
            ))
            .unwrap()
        };
        assert_eq!(parse("\"0.5875\"").value.as_deref(), Some("0.5875"));
        assert_eq!(parse("0.5875").value.as_deref(), Some("0.5875"));
        assert_eq!(parse("null").value, None);
        let missing: ForceRecord =
            serde_json::from_str(r#"{"index":1,"status":"ok","force_n":1.0}"#).unwrap();
        assert_eq!(missing.value, None);
        assert_eq!(missing.usable_force_n(), Some(1.0));
    }
}
