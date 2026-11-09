//! Rigid/translation alignment from selected reference tracks, with scale fixed at 1.
//!
//! Translations use coordinate-wise medians; rigid fitting rejects residual outliers with MAD.
//! Small point sets may still give unreliable estimates. Insufficient data falls back through
//! `Rigid → TranslationXY → Off`.
//!
//! [`RegistrationResult::transform`] maps reference to current; use
//! [`RegistrationResult::to_reference`] to align a displayed frame. The loading axis
//! is configurable in image coordinates.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{FrameId, TrackId, TrackedPoint, TrackingFrame};

/// How tracked points are used to align the current frame onto the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegistrationMode {
    /// No alignment; the display uses the raw image.
    Off,
    /// Robust 2-D translation (coordinate-wise median displacement).
    TranslationXY,
    /// Translation along the calibrated loading axis only.
    LoadingAxisOnly,
    /// Translation perpendicular to the calibrated loading axis only.
    CrossAxisOnly,
    /// Robust rotation/translation with scale fixed at 1.
    Rigid,
}

impl RegistrationMode {
    pub const ALL: [RegistrationMode; 5] = [
        RegistrationMode::Off,
        RegistrationMode::TranslationXY,
        RegistrationMode::LoadingAxisOnly,
        RegistrationMode::CrossAxisOnly,
        RegistrationMode::Rigid,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RegistrationMode::Off => "Off",
            RegistrationMode::TranslationXY => "Translation (X,Y)",
            RegistrationMode::LoadingAxisOnly => "Loading axis only",
            RegistrationMode::CrossAxisOnly => "Cross axis only",
            RegistrationMode::Rigid => "Rigid (no scale)",
        }
    }

    /// Stable ASCII identifier for exports, independent of UI labels.
    pub fn code(self) -> &'static str {
        match self {
            RegistrationMode::Off => "off",
            RegistrationMode::TranslationXY => "translation_xy",
            RegistrationMode::LoadingAxisOnly => "loading_axis",
            RegistrationMode::CrossAxisOnly => "cross_axis",
            RegistrationMode::Rigid => "rigid",
        }
    }

    pub fn is_off(self) -> bool {
        self == RegistrationMode::Off
    }
}

/// Registration choices only; computed transforms are stored separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RegistrationState {
    pub mode: RegistrationMode,
    /// The frame whose tracked positions are the registration targets.
    pub reference_frame: Option<FrameId>,
    /// Tracks eligible to estimate the transform. Empty set = every valid
    /// track.
    #[serde(default)]
    pub reference_tracks: BTreeSet<TrackId>,
    /// Points with `confidence < min_confidence` are rejected.
    pub min_confidence: f64,
    /// Calibrated loading axis in image pixel space (unnormalized is fine).
    pub loading_axis_x: f64,
    pub loading_axis_y: f64,
}

impl Default for RegistrationState {
    fn default() -> Self {
        RegistrationState {
            mode: RegistrationMode::Off,
            reference_frame: None,
            reference_tracks: BTreeSet::new(),
            min_confidence: 0.0,
            loading_axis_x: 1.0,
            loading_axis_y: 0.0,
        }
    }
}

impl RegistrationState {
    /// The normalized image-space loading axis; a zero vector falls back to
    /// `+x` (no direction information).
    pub fn loading_axis(&self) -> (f64, f64) {
        let len = (self.loading_axis_x * self.loading_axis_x
            + self.loading_axis_y * self.loading_axis_y)
            .sqrt();
        if len <= f64::EPSILON {
            (1.0, 0.0)
        } else {
            (self.loading_axis_x / len, self.loading_axis_y / len)
        }
    }

    /// Loading-axis angle: 0 = image-right, positive clockwise (y down).
    pub fn loading_axis_angle_deg(&self) -> f64 {
        let (x, y) = self.loading_axis();
        // atan2 with the image's y-down convention: +x is 0°, +y is +90°.
        y.atan2(x).to_degrees().rem_euclid(360.0)
    }

    pub fn set_loading_axis_angle_deg(&mut self, deg: f64) {
        let rad = deg.to_radians();
        self.loading_axis_x = rad.cos();
        self.loading_axis_y = rad.sin();
    }
}

/// Selected reference-frame positions passing the confidence threshold.
#[derive(Debug, Clone)]
pub struct ReferenceFrame {
    pub frame_id: FrameId,
    points: Vec<(TrackId, TrackedPoint)>,
}

impl ReferenceFrame {
    /// All points of `frame`, optionally restricted to `only` (empty = all)
    /// and `min_confidence`.
    pub fn from_frame(
        frame: &TrackingFrame,
        only: &BTreeSet<TrackId>,
        min_confidence: f64,
    ) -> Self {
        let points = frame
            .valid_points(min_confidence)
            .filter(|(id, _)| only.is_empty() || only.contains(*id))
            .map(|(id, p)| (id.clone(), *p))
            .collect();
        ReferenceFrame {
            frame_id: frame.frame_id(),
            points,
        }
    }

    pub fn points(&self) -> &[(TrackId, TrackedPoint)] {
        &self.points
    }

    pub fn get(&self, id: &TrackId) -> Option<&TrackedPoint> {
        self.points.iter().find(|(t, _)| t == id).map(|(_, p)| p)
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Image-space rotation and translation with scale fixed at 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegistrationTransform {
    Identity,
    Translation { tx: f64, ty: f64 },
    Rigid { theta: f64, tx: f64, ty: f64 },
}

impl RegistrationTransform {
    pub const IDENTITY: RegistrationTransform = RegistrationTransform::Identity;

    pub fn is_identity(&self) -> bool {
        matches!(self, RegistrationTransform::Identity)
    }

    /// Apply the transform to an image-space point.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        match *self {
            RegistrationTransform::Identity => (x, y),
            RegistrationTransform::Translation { tx, ty } => (x + tx, y + ty),
            RegistrationTransform::Rigid { theta, tx, ty } => {
                let (c, s) = (theta.cos(), theta.sin());
                (c * x - s * y + tx, s * x + c * y + ty)
            }
        }
    }

    /// The inverse transform (maps registered space back to raw image space).
    pub fn inverse(&self) -> RegistrationTransform {
        match *self {
            RegistrationTransform::Identity => RegistrationTransform::Identity,
            RegistrationTransform::Translation { tx, ty } => {
                RegistrationTransform::Translation { tx: -tx, ty: -ty }
            }
            RegistrationTransform::Rigid { theta, tx, ty } => {
                // Inverse of R(theta) is R(-theta); t' = -R(-theta) * t.
                let (c, s) = (theta.cos(), theta.sin());
                let nx = -tx * c - ty * s;
                let ny = tx * s - ty * c;
                RegistrationTransform::Rigid {
                    theta: -theta,
                    tx: nx,
                    ty: ny,
                }
            }
        }
    }

    /// The translation part, in image pixels.
    pub fn translation(&self) -> (f64, f64) {
        match *self {
            RegistrationTransform::Identity => (0.0, 0.0),
            RegistrationTransform::Translation { tx, ty } => (tx, ty),
            RegistrationTransform::Rigid { tx, ty, .. } => (tx, ty),
        }
    }
}

/// The computed alignment for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct RegistrationResult {
    /// Which mode actually produced the transform (may be simpler than the
    /// requested mode after a fallback).
    pub mode_used: RegistrationMode,
    /// Estimated reference-to-current motion. Invert with [`RegistrationResult::to_reference`].
    pub transform: RegistrationTransform,
    /// Number of `(reference, current)` pairs that entered the estimate
    /// (after outlier rejection in rigid mode).
    pub valid_count: usize,
    /// Pairs dropped as outliers by the rigid estimator (always 0 for the
    /// median-based translation modes).
    pub outliers_rejected: usize,
    /// Number of reference points available on the reference frame (before
    /// pairing against the current frame).
    pub reference_count: usize,
    /// RMS residual in pixels for accepted pairs, or `None` without an estimate.
    pub rms_residual_px: Option<f64>,
    /// Set when the requested mode could not be estimated and a simpler one
    /// was used instead.
    pub fallback: Option<String>,
    /// Recoverable problems, e.g. skipped low-confidence/missing points.
    pub warnings: Vec<String>,
}

impl RegistrationResult {
    /// Current-to-reference transform, the inverse of [`RegistrationResult::transform`].
    /// Matches the direction used by replay stabilization.
    pub fn to_reference(&self) -> RegistrationTransform {
        self.transform.inverse()
    }

    /// The "nothing registered" state.
    pub fn off(reference_count: usize, warnings: Vec<String>) -> Self {
        RegistrationResult {
            mode_used: RegistrationMode::Off,
            transform: RegistrationTransform::IDENTITY,
            valid_count: 0,
            outliers_rejected: 0,
            reference_count,
            rms_residual_px: None,
            fallback: None,
            warnings,
        }
    }
}

/// Estimate alignment from `current` to `reference` using `state`.
pub fn compute_registration(
    reference: &ReferenceFrame,
    current: &TrackingFrame,
    state: &RegistrationState,
) -> RegistrationResult {
    if state.mode.is_off() {
        return RegistrationResult::off(reference.len(), Vec::new());
    }

    // Pair matching track IDs, dropping missing or invalid observations.
    let mut pairs: Vec<Pair> = Vec::new();
    for (id, ref_pt) in reference.points() {
        let Some(cur_pt) = current.get(id) else {
            continue; // track absent/lost in the current frame
        };
        if cur_pt.is_valid(state.min_confidence) {
            pairs.push((id.clone(), *ref_pt, *cur_pt));
        }
    }
    let reference_count = reference.len();

    if pairs.is_empty() {
        return RegistrationResult::off(
            reference_count,
            vec!["no valid reference/current point pairs for this frame".to_owned()],
        );
    }

    // Use robust estimates so lost or jumping tracks do not dominate alignment.
    let mut outliers_rejected = 0;
    let (mode_used, transform, fallback) = match state.mode {
        RegistrationMode::Off => unreachable!(),
        RegistrationMode::TranslationXY => {
            let (dx, dy) = median_displacement(&pairs, None);
            (
                RegistrationMode::TranslationXY,
                RegistrationTransform::Translation { tx: dx, ty: dy },
                None,
            )
        }
        RegistrationMode::LoadingAxisOnly => {
            let (ax, ay) = state.loading_axis();
            let (dx, dy) = median_displacement(&pairs, Some((ax, ay)));
            (
                RegistrationMode::LoadingAxisOnly,
                RegistrationTransform::Translation { tx: dx, ty: dy },
                None,
            )
        }
        RegistrationMode::CrossAxisOnly => {
            let (ax, ay) = state.loading_axis();
            let (dx, dy) = median_displacement(&pairs, Some((ay, -ax)));
            (
                RegistrationMode::CrossAxisOnly,
                RegistrationTransform::Translation { tx: dx, ty: dy },
                None,
            )
        }
        RegistrationMode::Rigid => {
            if pairs.len() >= 2 {
                let (theta, tx, ty, inliers) = robust_rigid_fit(&pairs);
                outliers_rejected = pairs.len() - inliers.len();
                pairs = inliers;
                (
                    RegistrationMode::Rigid,
                    RegistrationTransform::Rigid { theta, tx, ty },
                    None,
                )
            } else {
                // One pair cannot estimate rotation; fall back to translation.
                let (dx, dy) = median_displacement(&pairs, None);
                (
                    RegistrationMode::TranslationXY,
                    RegistrationTransform::Translation { tx: dx, ty: dy },
                    Some("rigid needs at least 2 valid pairs; degraded to translation".to_owned()),
                )
            }
        }
    };

    // Non-finite translation falls back to Off.
    if matches!(
        mode_used,
        RegistrationMode::TranslationXY
            | RegistrationMode::LoadingAxisOnly
            | RegistrationMode::CrossAxisOnly
    ) {
        let (tx, ty) = transform.translation();
        if !tx.is_finite() || !ty.is_finite() {
            return RegistrationResult::off(
                reference_count,
                vec!["estimated displacement was not finite".to_owned()],
            );
        }
    }

    let rms = rms_residual(&pairs, &transform);
    let mut warnings = Vec::new();
    // Report tracks missing in the current frame.
    let lost = reference
        .points()
        .iter()
        .filter(|(id, _)| current.get(id).is_none())
        .count();
    if lost > 0 {
        warnings.push(format!("{lost} reference track(s) lost in current frame"));
    }
    if outliers_rejected > 0 {
        warnings.push(format!(
            "{outliers_rejected} outlier track(s) rejected by the rigid fit"
        ));
    }

    RegistrationResult {
        mode_used,
        transform,
        valid_count: pairs.len(),
        outliers_rejected,
        reference_count,
        rms_residual_px: Some(rms),
        fallback,
        warnings,
    }
}

/// Coordinate-wise median displacement, optionally projected onto unit axis `(ax, ay)`.
fn median_displacement(pairs: &[Pair], axis: Option<(f64, f64)>) -> (f64, f64) {
    let mut dxs: Vec<f64> = Vec::with_capacity(pairs.len());
    let mut dys: Vec<f64> = Vec::with_capacity(pairs.len());
    match axis {
        None => {
            for (_, r, c) in pairs {
                dxs.push(c.x - r.x);
                dys.push(c.y - r.y);
            }
        }
        Some((ax, ay)) => {
            for (_, r, c) in pairs {
                let dx = c.x - r.x;
                let dy = c.y - r.y;
                let along = dx * ax + dy * ay;
                dxs.push(along * ax);
                dys.push(along * ay);
            }
        }
    }
    (median(&mut dxs), median(&mut dys))
}

/// Maximum number of fit / reject / refit rounds of the rigid estimator.
const RIGID_REJECT_ROUNDS: usize = 3;

/// Pairs whose residual exceeds the median residual by more than this many
/// scaled MADs (1.4826 * MAD, a standard-deviation estimate under normal
/// noise) are rejected as outliers.
pub const RIGID_REJECT_K: f64 = 3.0;

/// Residual scale floor in pixels, so that a near-perfect fit (MAD ~ 0) does
/// not reject pairs over floating-point noise.
const RIGID_MIN_SCALE_PX: f64 = 1e-3;

/// A reference track paired with its position in the current frame.
type Pair = (TrackId, TrackedPoint, TrackedPoint);

/// Reject residuals above `median + RIGID_REJECT_K * 1.4826 * MAD` and refit,
/// up to [`RIGID_REJECT_ROUNDS`] times. Keep at least two pairs.
/// Returns `(theta, tx, ty)` and surviving pairs.
fn robust_rigid_fit(pairs: &[Pair]) -> (f64, f64, f64, Vec<Pair>) {
    let mut inliers: Vec<Pair> = pairs.to_vec();
    let mut fit = rigid_fit(&inliers);
    for _ in 0..RIGID_REJECT_ROUNDS {
        let transform = RegistrationTransform::Rigid {
            theta: fit.0,
            tx: fit.1,
            ty: fit.2,
        };
        let residuals: Vec<f64> = inliers
            .iter()
            .map(|(_, r, c)| {
                let (px, py) = transform.apply(r.x, r.y);
                (px - c.x).hypot(py - c.y)
            })
            .collect();
        let mut sorted = residuals.clone();
        let centre = median(&mut sorted);
        let mut deviations: Vec<f64> = residuals.iter().map(|r| (r - centre).abs()).collect();
        let scale = 1.4826 * median(&mut deviations);
        let limit = centre + RIGID_REJECT_K * scale.max(RIGID_MIN_SCALE_PX);

        let kept: Vec<Pair> = inliers
            .iter()
            .zip(&residuals)
            .filter(|(_, r)| **r <= limit)
            .map(|(pair, _)| pair.clone())
            .collect();
        if kept.len() == inliers.len() || kept.len() < 2 {
            break;
        }
        inliers = kept;
        fit = rigid_fit(&inliers);
    }
    (fit.0, fit.1, fit.2, inliers)
}

/// Least-squares rotation/translation after centring. Rotation is `atan2(Σ cross, Σ dot)`.
/// [`robust_rigid_fit`] wraps this with outlier rejection.
fn rigid_fit(pairs: &[Pair]) -> (f64, f64, f64) {
    let n = pairs.len() as f64;
    let mut cxr = 0.0;
    let mut cyr = 0.0;
    let mut cxc = 0.0;
    let mut cyc = 0.0;
    for (_, r, c) in pairs {
        cxr += r.x;
        cyr += r.y;
        cxc += c.x;
        cyc += c.y;
    }
    cxr /= n;
    cyr /= n;
    cxc /= n;
    cyc /= n;

    let mut dot = 0.0;
    let mut cross = 0.0;
    for (_, r, c) in pairs {
        let rx = r.x - cxr;
        let ry = r.y - cyr;
        let cx = c.x - cxc;
        let cy = c.y - cyc;
        dot += rx * cx + ry * cy;
        cross += rx * cy - ry * cx;
    }
    let theta = cross.atan2(dot);
    let (ct, st) = (theta.cos(), theta.sin());
    let tx = cxc - (ct * cxr - st * cyr);
    let ty = cyc - (st * cxr + ct * cyr);
    (theta, tx, ty)
}

fn rms_residual(pairs: &[Pair], transform: &RegistrationTransform) -> f64 {
    let n = pairs.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for (_, r, c) in pairs {
        let (tx, ty) = transform.apply(r.x, r.y);
        let dx = tx - c.x;
        let dy = ty - c.y;
        sum += dx * dx + dy * dy;
    }
    (sum / n as f64).sqrt()
}

/// Coordinate-wise median shared with stabilization. Sort NaNs last.
pub(super) fn median(values: &mut [f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    let mut sorted = finite;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> TrackedPoint {
        TrackedPoint::with_confidence(x, y, 1.0)
    }

    fn frame(id: u32, pts: &[(&str, f64, f64)]) -> TrackingFrame {
        TrackingFrame::new(
            FrameId(id),
            pts.iter()
                .map(|(n, x, y)| (TrackId::new(*n), point(*x, *y))),
        )
    }

    fn state(mode: RegistrationMode) -> RegistrationState {
        RegistrationState {
            mode,
            reference_frame: Some(FrameId(0)),
            ..Default::default()
        }
    }

    fn ref_from(f: &TrackingFrame) -> ReferenceFrame {
        ReferenceFrame::from_frame(f, &BTreeSet::new(), 0.0)
    }

    const TOL: f64 = 1e-6;

    #[test]
    fn known_translation_is_recovered_exactly() {
        let reference = frame(
            0,
            &[("a", 10.0, 20.0), ("b", 30.0, 40.0), ("c", 50.0, 60.0)],
        );
        let current = frame(
            1,
            &[("a", 15.0, 17.0), ("b", 35.0, 37.0), ("c", 55.0, 57.0)],
        );
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::TranslationXY),
        );
        assert_eq!(result.mode_used, RegistrationMode::TranslationXY);
        let (tx, ty) = result.transform.translation();
        assert!((tx - 5.0).abs() < TOL, "tx={tx}");
        assert!((ty + 3.0).abs() < TOL, "ty={ty}");
        assert_eq!(result.valid_count, 3);
        assert!(result.rms_residual_px.unwrap() < TOL);
    }

    #[test]
    fn strong_outlier_does_not_break_translation() {
        // One gross outlier; the median must ignore it entirely.
        let reference = frame(
            0,
            &[
                ("a", 10.0, 20.0),
                ("b", 30.0, 40.0),
                ("c", 50.0, 60.0),
                ("d", 70.0, 80.0),
            ],
        );
        let current = frame(
            1,
            &[
                ("a", 15.0, 17.0),
                ("b", 35.0, 37.0),
                ("c", 55.0, 57.0),
                ("d", 700.0, 800.0), // wild outlier
            ],
        );
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::TranslationXY),
        );
        let (tx, ty) = result.transform.translation();
        assert!((tx - 5.0).abs() < TOL, "tx={tx}");
        assert!((ty + 3.0).abs() < TOL, "ty={ty}");
    }

    #[test]
    fn rigid_recovers_rotation_and_translation_no_scale() {
        // Reference: a square. Current: rotated 30 deg about its centroid,
        // then shifted.
        let reference = frame(
            0,
            &[
                ("a", 0.0, 0.0),
                ("b", 100.0, 0.0),
                ("c", 100.0, 100.0),
                ("d", 0.0, 100.0),
            ],
        );
        let theta = 30.0_f64.to_radians();
        let (ct, st) = (theta.cos(), theta.sin());
        let (cx, cy) = (50.0, 50.0);
        let (tx0, ty0) = (13.0, -7.0);
        let rot = |x: f64, y: f64| {
            let (dx, dy) = (x - cx, y - cy);
            (cx + ct * dx - st * dy + tx0, cy + st * dx + ct * dy + ty0)
        };
        let current = frame(
            1,
            &[
                ("a", rot(0.0, 0.0).0, rot(0.0, 0.0).1),
                ("b", rot(100.0, 0.0).0, rot(100.0, 0.0).1),
                ("c", rot(100.0, 100.0).0, rot(100.0, 100.0).1),
                ("d", rot(0.0, 100.0).0, rot(0.0, 100.0).1),
            ],
        );
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::Rigid),
        );
        assert_eq!(result.mode_used, RegistrationMode::Rigid);
        let RegistrationTransform::Rigid {
            theta: got_theta,
            tx: got_tx,
            ty: got_ty,
        } = result.transform
        else {
            panic!("expected rigid transform");
        };
        assert!(
            (got_theta - theta).abs() < 1e-9,
            "theta {got_theta} vs {theta}"
        );
        // Rotation is about the centroid: effective translation is `c_ref - R*c_ref + shift`.
        let (rx, ry) = (ct * cx - st * cy, st * cx + ct * cy);
        let ex_tx = cx - rx + tx0;
        let ex_ty = cy - ry + ty0;
        assert!((got_tx - ex_tx).abs() < 1e-9, "tx {got_tx} vs {ex_tx}");
        assert!((got_ty - ex_ty).abs() < 1e-9, "ty {got_ty} vs {ex_ty}");
        assert!(result.rms_residual_px.unwrap() < 1e-9);
        for (id, rp) in reference.points() {
            let cp = current.get(id).unwrap();
            let (px, py) = result.transform.apply(rp.x, rp.y);
            assert!((px - cp.x).abs() < 1e-9);
            assert!((py - cp.y).abs() < 1e-9);
        }
    }

    #[test]
    fn rigid_rejects_gross_outliers_and_refits() {
        // Eight tracks move rigidly (20 deg + shift); two jump elsewhere.
        let theta = 20.0_f64.to_radians();
        let (ct, st) = (theta.cos(), theta.sin());
        let moved = |x: f64, y: f64| (ct * x - st * y + 7.0, st * x + ct * y - 4.0);
        let base: Vec<(String, f64, f64)> = (0..10)
            .map(|i| (format!("t{i}"), 30.0 * i as f64, 17.0 * (i % 4) as f64))
            .collect();
        let reference = TrackingFrame::new(
            FrameId(0),
            base.iter()
                .map(|(n, x, y)| (TrackId::new(n), point(*x, *y))),
        );
        let current = TrackingFrame::new(
            FrameId(1),
            base.iter().enumerate().map(|(i, (n, x, y))| {
                let (mx, my) = moved(*x, *y);
                let (mx, my) = match i {
                    3 => (mx + 400.0, my - 250.0),
                    7 => (mx - 90.0, my + 120.0),
                    _ => (mx, my),
                };
                (TrackId::new(n), point(mx, my))
            }),
        );
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::Rigid),
        );
        let RegistrationTransform::Rigid {
            theta: got_theta,
            tx,
            ty,
        } = result.transform
        else {
            panic!("expected rigid transform");
        };
        assert!((got_theta - theta).abs() < 1e-9, "theta {got_theta}");
        assert!(
            (tx - 7.0).abs() < 1e-6 && (ty + 4.0).abs() < 1e-6,
            "{tx} {ty}"
        );
        assert_eq!(result.outliers_rejected, 2);
        assert_eq!(result.valid_count, 8);
        assert!(result.rms_residual_px.unwrap() < 1e-6);
        assert!(result.warnings.iter().any(|w| w.contains("outlier")));
    }

    #[test]
    fn rigid_without_outliers_rejects_nothing() {
        let reference = frame(
            0,
            &[("a", 0.0, 0.0), ("b", 100.0, 0.0), ("c", 100.0, 100.0)],
        );
        // Small independent noise on each track.
        let current = frame(
            1,
            &[("a", 0.2, -0.1), ("b", 99.9, 0.15), ("c", 100.1, 99.8)],
        );
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::Rigid),
        );
        assert_eq!(result.outliers_rejected, 0);
        assert_eq!(result.valid_count, 3);
    }

    #[test]
    fn to_reference_undoes_the_estimated_motion() {
        let reference = frame(0, &[("a", 10.0, 20.0), ("b", 30.0, 40.0)]);
        let current = frame(1, &[("a", 15.0, 17.0), ("b", 35.0, 37.0)]);
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::TranslationXY),
        );
        for (id, rp) in reference.points() {
            let cp = current.get(id).unwrap();
            let (x, y) = result.to_reference().apply(cp.x, cp.y);
            assert!((x - rp.x).abs() < TOL && (y - rp.y).abs() < TOL);
        }
    }

    #[test]
    fn rigid_preserves_scale_exactly() {
        // Doubling the data must not change the transform's scale.
        let reference = frame(
            0,
            &[("a", 10.0, 20.0), ("b", 30.0, 20.0), ("c", 30.0, 60.0)],
        );
        // "current" = reference scaled 2x about origin + rotated + shifted.
        let theta = 45.0_f64.to_radians();
        let (ct, st) = (theta.cos(), theta.sin());
        let rot = |x: f64, y: f64| (2.0 * (ct * x - st * y) + 5.0, 2.0 * (st * x + ct * y) - 3.0);
        let current = frame(
            1,
            &[
                ("a", rot(10.0, 20.0).0, rot(10.0, 20.0).1),
                ("b", rot(30.0, 20.0).0, rot(30.0, 20.0).1),
                ("c", rot(30.0, 60.0).0, rot(30.0, 60.0).1),
            ],
        );
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::Rigid),
        );
        assert_eq!(result.mode_used, RegistrationMode::Rigid);
        let d_ab_ref = (30.0_f64 - 10.0).hypot(0.0);
        let (ax, ay) = result.transform.apply(10.0, 20.0);
        let (bx, by) = result.transform.apply(30.0, 20.0);
        let d_ab_reg = (bx - ax).hypot(by - ay);
        assert!((d_ab_reg - d_ab_ref).abs() < 1e-9, "scale must stay 1");
        // Scaled data must leave a residual; the rigid fit must not absorb scale.
        let rms = result.rms_residual_px.unwrap();
        assert!(rms > 5.0, "rigid must not absorb the scale, rms={rms}");
    }

    #[test]
    fn missing_points_are_skipped_and_counted() {
        let reference = frame(
            0,
            &[("a", 10.0, 20.0), ("b", 30.0, 40.0), ("c", 50.0, 60.0)],
        );
        let current = frame(1, &[("a", 15.0, 17.0), ("c", 55.0, 57.0)]);
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::TranslationXY),
        );
        assert_eq!(result.valid_count, 2);
        assert_eq!(result.reference_count, 3);
        let (tx, ty) = result.transform.translation();
        assert!((tx - 5.0).abs() < TOL);
        assert!((ty + 3.0).abs() < TOL);
        assert!(result.warnings.iter().any(|w| w.contains("lost")));
    }

    #[test]
    fn low_confidence_points_are_rejected() {
        let mut reference = frame(0, &[("a", 10.0, 20.0), ("b", 30.0, 40.0)]);
        let mut current = frame(1, &[("a", 15.0, 17.0), ("b", 35.0, 37.0)]);
        let ref_pts: Vec<(TrackId, TrackedPoint)> = reference
            .points()
            .iter()
            .map(|(t, p)| {
                if t.as_str() == "a" {
                    (t.clone(), TrackedPoint::with_confidence(p.x, p.y, 0.1))
                } else {
                    (t.clone(), *p)
                }
            })
            .collect();
        reference = TrackingFrame::new(FrameId(0), ref_pts);
        let cur_pts: Vec<(TrackId, TrackedPoint)> = current
            .points()
            .iter()
            .map(|(t, p)| {
                if t.as_str() == "a" {
                    (t.clone(), TrackedPoint::with_confidence(p.x, p.y, 0.1))
                } else {
                    (t.clone(), *p)
                }
            })
            .collect();
        current = TrackingFrame::new(FrameId(1), cur_pts);

        let mut st = state(RegistrationMode::TranslationXY);
        st.min_confidence = 0.5;
        let result = compute_registration(&ref_from(&reference), &current, &st);
        assert_eq!(result.valid_count, 1, "only b is usable");
        let (tx, ty) = result.transform.translation();
        assert!((tx - 5.0).abs() < TOL);
        assert!((ty + 3.0).abs() < TOL);
    }

    #[test]
    fn rigid_falls_back_to_translation_with_one_pair() {
        let reference = frame(0, &[("a", 10.0, 20.0)]);
        let current = frame(1, &[("a", 15.0, 17.0)]);
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::Rigid),
        );
        assert_eq!(result.mode_used, RegistrationMode::TranslationXY);
        assert!(result.fallback.is_some());
        let (tx, ty) = result.transform.translation();
        assert!((tx - 5.0).abs() < TOL);
        assert!((ty + 3.0).abs() < TOL);
    }

    #[test]
    fn empty_frame_yields_off() {
        let reference = frame(0, &[("a", 10.0, 20.0)]);
        let empty = TrackingFrame::new(FrameId(1), vec![]);
        for mode in [
            RegistrationMode::TranslationXY,
            RegistrationMode::LoadingAxisOnly,
            RegistrationMode::CrossAxisOnly,
            RegistrationMode::Rigid,
        ] {
            let result = compute_registration(&ref_from(&reference), &empty, &state(mode));
            assert_eq!(result.mode_used, RegistrationMode::Off, "{mode:?}");
            assert!(result.transform.is_identity());
            assert_eq!(result.valid_count, 0);
        }
    }

    #[test]
    fn off_mode_returns_identity() {
        let reference = frame(0, &[("a", 10.0, 20.0)]);
        let current = frame(1, &[("a", 50.0, 90.0)]);
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::Off),
        );
        assert_eq!(result.mode_used, RegistrationMode::Off);
        assert!(result.transform.is_identity());
        assert_eq!(result.rms_residual_px, None);
    }

    #[test]
    fn chosen_reference_frame_is_the_target() {
        // Use the chosen reference frame, not the first one.
        let f0 = frame(0, &[("a", 100.0, 100.0)]);
        let f3 = frame(3, &[("a", 10.0, 20.0)]);
        let f4 = frame(4, &[("a", 13.0, 22.0)]);
        let reference = ReferenceFrame::from_frame(&f3, &BTreeSet::new(), 0.0);
        let result = compute_registration(&reference, &f4, &state(RegistrationMode::TranslationXY));
        let (tx, ty) = result.transform.translation();
        assert!((tx - 3.0).abs() < TOL, "tx={tx}");
        assert!((ty - 2.0).abs() < TOL, "ty={ty}");
        let _ = (&f0, &f4);
    }

    #[test]
    fn transformed_references_land_on_current() {
        let reference = frame(
            0,
            &[("a", 10.0, 20.0), ("b", 30.0, 40.0), ("c", 50.0, 60.0)],
        );
        let current = frame(
            1,
            &[("a", 15.0, 17.0), ("b", 35.0, 37.0), ("c", 55.0, 57.0)],
        );
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::TranslationXY),
        );
        for (id, rp) in reference.points() {
            let cp = current.get(id).unwrap();
            let (tx, ty) = result.transform.apply(rp.x, rp.y);
            assert!((tx - cp.x).abs() < TOL);
            assert!((ty - cp.y).abs() < TOL);
        }
    }

    #[test]
    fn reference_track_filter_restricts_the_estimate() {
        let reference = frame(0, &[("a", 10.0, 20.0), ("b", 30.0, 40.0)]);
        let current = frame(1, &[("a", 15.0, 17.0), ("b", 200.0, 400.0)]);
        let mut only = BTreeSet::new();
        only.insert(TrackId::new("a"));
        let reference = ReferenceFrame::from_frame(&reference, &only, 0.0);
        let result = compute_registration(
            &reference,
            &current,
            &state(RegistrationMode::TranslationXY),
        );
        assert_eq!(result.valid_count, 1);
        assert_eq!(result.reference_count, 1);
        let (tx, ty) = result.transform.translation();
        assert!((tx - 5.0).abs() < TOL, "tx={tx}");
        assert!((ty + 3.0).abs() < TOL, "ty={ty}");
    }

    #[test]
    fn loading_axis_only_translates_along_axis() {
        // Loading axis = +x. Current moved by (8, 40): only the x component
        // should be removed.
        let reference = frame(0, &[("a", 10.0, 20.0), ("b", 30.0, 40.0)]);
        let current = frame(1, &[("a", 18.0, 60.0), ("b", 38.0, 80.0)]);
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::LoadingAxisOnly),
        );
        let (tx, ty) = result.transform.translation();
        assert!((tx - 8.0).abs() < TOL, "tx={tx}");
        assert!(ty.abs() < TOL, "ty={ty}");
    }

    #[test]
    fn cross_axis_only_translates_perpendicular() {
        // Loading axis = +x: only the y component should be removed.
        let reference = frame(0, &[("a", 10.0, 20.0), ("b", 30.0, 40.0)]);
        let current = frame(1, &[("a", 18.0, 60.0), ("b", 38.0, 80.0)]);
        let result = compute_registration(
            &ref_from(&reference),
            &current,
            &state(RegistrationMode::CrossAxisOnly),
        );
        let (tx, ty) = result.transform.translation();
        assert!(tx.abs() < TOL, "tx={tx}");
        assert!((ty - 40.0).abs() < TOL, "ty={ty}");
    }

    #[test]
    fn loading_axis_at_45_degrees() {
        // Axis at 45 deg (diagonal). Displacement of (10, 10) lies along the
        // axis; a perpendicular displacement is rejected.
        let mut st = state(RegistrationMode::LoadingAxisOnly);
        st.set_loading_axis_angle_deg(45.0);
        let reference = frame(0, &[("a", 0.0, 0.0), ("b", 100.0, 0.0)]);
        let current = frame(1, &[("a", 10.0, 10.0), ("b", 110.0, 10.0)]);
        let result = compute_registration(&ref_from(&reference), &current, &st);
        let (tx, ty) = result.transform.translation();
        assert!((tx - 10.0).abs() < TOL, "tx={tx}");
        assert!((ty - 10.0).abs() < TOL, "ty={ty}");
    }

    #[test]
    fn inverse_transform_round_trips() {
        let t = RegistrationTransform::Rigid {
            theta: 0.7,
            tx: 12.0,
            ty: -5.0,
        };
        for (x, y) in [(0.0, 0.0), (100.0, 200.0), (-3.0, 7.5)] {
            let (fx, fy) = t.apply(x, y);
            let (bx, by) = t.inverse().apply(fx, fy);
            assert!((bx - x).abs() < 1e-9, "x {bx} vs {x}");
            assert!((by - y).abs() < 1e-9, "y {by} vs {y}");
        }
        let tr = RegistrationTransform::Translation { tx: 4.0, ty: -2.0 };
        let (fx, fy) = tr.apply(1.0, 1.0);
        let (bx, by) = tr.inverse().apply(fx, fy);
        assert!((bx - 1.0).abs() < 1e-12);
        assert!((by - 1.0).abs() < 1e-12);
    }

    #[test]
    fn transform_apply_matches_expected_points() {
        let t = RegistrationTransform::Rigid {
            theta: std::f64::consts::FRAC_PI_2,
            tx: 10.0,
            ty: 0.0,
        };
        // 90° rotation maps (1,0) to (0,1), then +tx.
        let (x, y) = t.apply(1.0, 0.0);
        assert!((x - 10.0).abs() < 1e-12);
        assert!((y - 1.0).abs() < 1e-12);
    }
}
