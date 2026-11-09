//! Per-run force/displacement curves, with relaxation holds flagged separately.
//! Keep raw run travel; [`Curve::mirrored_displacement`] mirrors release travel
//! against the paired loading endpoint for hysteresis plots.

use crate::discover::Run;
use crate::model::{Branch, RunKey};

/// One frame's contribution to a curve.
#[derive(Debug, Clone, Copy)]
pub struct CurvePoint {
    pub frame_id: u32,
    /// Travel within the run, as recorded by the rig.
    pub displacement_mm: f64,
    /// Load-cell reading; `None` when the linked force record is missing,
    /// non-ok or null.
    pub force_n: Option<f64>,
    pub line_load_n_per_m: Option<f64>,
    pub is_relaxation_hold: bool,
}

/// The extracted series for one run.
#[derive(Debug, Clone)]
pub struct Curve {
    pub key: RunKey,
    pub test_id: String,
    pub points: Vec<CurvePoint>,
}

impl Curve {
    /// Extract readings through [`Run::force_n_for_frame`]; invalid force links become missing values.
    pub fn from_run(run: &Run) -> Curve {
        let points = run
            .frames
            .frames
            .iter()
            .map(|frame| CurvePoint {
                frame_id: frame.frame_id,
                displacement_mm: frame.displacement_mm,
                force_n: run.force_n_for_frame(frame),
                line_load_n_per_m: frame.line_load_n_per_m.filter(|v| v.is_finite()),
                is_relaxation_hold: frame.is_relaxation_hold,
            })
            .collect();
        Curve {
            key: run.key(),
            test_id: run.session.test_id.clone(),
            points,
        }
    }

    /// Largest recorded travel, `None` for a curve without points.
    pub fn max_displacement_mm(&self) -> Option<f64> {
        self.points
            .iter()
            .map(|p| p.displacement_mm)
            .fold(None, |acc, d| Some(acc.map_or(d, |a: f64| a.max(d))))
    }

    /// `(displacement_mm, force_n)` for points with a usable force reading.
    /// `include_relax` controls whether relaxation-hold frames are included.
    pub fn force_series(&self, include_relax: bool) -> Vec<(f64, f64)> {
        self.series(include_relax, |p| p.force_n)
    }

    /// `(displacement_mm, line_load_N_per_m)` for points carrying a line load.
    pub fn line_load_series(&self, include_relax: bool) -> Vec<(f64, f64)> {
        self.series(include_relax, |p| p.line_load_n_per_m)
    }

    fn series(
        &self,
        include_relax: bool,
        value: impl Fn(&CurvePoint) -> Option<f64>,
    ) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .filter(|p| include_relax || !p.is_relaxation_hold)
            .filter_map(|p| Some((p.displacement_mm, value(p)?)))
            .collect()
    }

    /// Mirror release travel against `start_stretch_mm` for hysteresis plots. Loading is unchanged.
    pub fn mirrored_displacement(&self, start_stretch_mm: f64, displacement_mm: f64) -> f64 {
        match self.key.branch {
            Branch::Release => start_stretch_mm - displacement_mm,
            _ => displacement_mm,
        }
    }
}
