//! Loading, validation, curves and tracking for stretch-rig measurements.
//!
//! ```no_run
//! use fabric_data::{Project, Curve, validate_run};
//!
//! let project = Project::discover("/path/to/project").unwrap();
//! for run in &project.runs {
//!     let report = validate_run(run);
//!     println!("{} [{}]", run.key(), report.severity());
//!     let curve = Curve::from_run(run);
//!     let _series = curve.force_series(false);
//! }
//! ```

pub mod curves;
pub mod discover;
pub mod model;
pub mod tracking;
pub mod validate;

pub use curves::{Curve, CurvePoint};
pub use discover::{DataError, Project, Run};
pub use model::{
    Branch, Direction, ForceFile, ForceRecord, ForceStatus, Frame, FramesFile, MotorMechanics,
    RunKey, Session,
};
pub use validate::{validate_run, RunValidation, Severity};
