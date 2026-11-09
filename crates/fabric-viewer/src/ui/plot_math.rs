//! Pure plot math for the Curves tab: the plotted y-values (optionally
//! denoised), monotone cubic subdivision for smooth lines, and hit testing.

use egui_plot::PlotTransform;
use fabric_data::{Branch, Curve};

use crate::project::ProjectSlot;
use crate::settings::PlotQuantity;

/// Standard deviation (in samples) of the denoising Gaussian.
const SIGMA: f64 = 2.0;
/// The Gaussian is truncated at this many standard deviations.
const WINDOW_SIGMAS: f64 = 3.0;

/// Gaussian local linear smoothing (3σ window), preserving endpoint slopes and relaxation holds.
pub fn plot_y_values(
    curve: &Curve,
    quantity: PlotQuantity,
    show_relax: bool,
    smooth: bool,
) -> Vec<Option<f64>> {
    let mut y_vals: Vec<Option<f64>> = curve
        .points
        .iter()
        .map(|p| {
            if !show_relax && p.is_relaxation_hold {
                None
            } else {
                quantity.of(p)
            }
        })
        .collect();
    if !smooth {
        return y_vals;
    }

    let indices: Vec<usize> = curve
        .points
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.is_relaxation_hold)
        .filter_map(|(i, _)| y_vals[i].map(|_| i))
        .collect();
    if indices.len() < 3 {
        return y_vals;
    }

    let half = (WINDOW_SIGMAS * SIGMA).ceil() as usize;
    let raw: Vec<f64> = indices.iter().filter_map(|&i| y_vals[i]).collect();
    for (k, &frame_idx) in indices.iter().enumerate() {
        let lo = k.saturating_sub(half);
        let hi = (k + half + 1).min(raw.len());
        let (mut sw, mut swx, mut swx2, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (j, &y) in raw.iter().enumerate().take(hi).skip(lo) {
            let dx = j as f64 - k as f64;
            let w = (-0.5 * (dx / SIGMA).powi(2)).exp();
            sw += w;
            swx += w * dx;
            swx2 += w * dx * dx;
            swy += w * y;
            swxy += w * dx * y;
        }
        let det = sw * swx2 - swx * swx;
        let smoothed = if det.abs() > 1e-12 {
            (swx2 * swy - swx * swxy) / det
        } else {
            swy / sw
        };
        y_vals[frame_idx] = Some(smoothed);
    }
    y_vals
}

/// Fritsch–Carlson tangents: zero at extrema, limited to prevent overshoot.
fn monotone_tangents(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let delta: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
    let mut m = vec![0.0; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        m[i] = if delta[i - 1] * delta[i] <= 0.0 {
            0.0
        } else {
            0.5 * (delta[i - 1] + delta[i])
        };
    }
    for i in 0..n - 1 {
        if delta[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let (a, b) = (m[i] / delta[i], m[i + 1] / delta[i]);
        // Tangents must point the way the data goes (also at the ends).
        let (a, b) = (a.max(0.0), b.max(0.0));
        let norm = a * a + b * b;
        let (a, b) = if norm > 9.0 {
            let tau = 3.0 / norm.sqrt();
            (tau * a, tau * b)
        } else {
            (a, b)
        };
        m[i] = a * delta[i];
        m[i + 1] = b * delta[i];
    }
    m
}

/// Parametric cubic Hermite interpolation, using Fritsch–Carlson tangents per coordinate
/// to avoid overshooting measured values.
pub fn smooth_curve_points(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    const SUBDIVISIONS: usize = 8;
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }
    let xs: Vec<f64> = points.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = points.iter().map(|p| p[1]).collect();
    let (mx, my) = (monotone_tangents(&xs), monotone_tangents(&ys));

    let hermite = |p0: f64, p1: f64, m0: f64, m1: f64, t: f64| {
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
            + (t3 - 2.0 * t2 + t) * m0
            + (-2.0 * t3 + 3.0 * t2) * p1
            + (t3 - t2) * m1
    };
    let mut out = Vec::with_capacity((n - 1) * SUBDIVISIONS + 1);
    for i in 0..n - 1 {
        for s in 0..SUBDIVISIONS {
            let t = s as f64 / SUBDIVISIONS as f64;
            out.push([
                hermite(xs[i], xs[i + 1], mx[i], mx[i + 1], t),
                hermite(ys[i], ys[i + 1], my[i], my[i + 1], t),
            ]);
        }
    }
    out.push(points[n - 1]);
    out
}

/// The visible curve point closest to `pointer` (screen space), within a
/// small pick radius. Returns (project, run, point index).
pub fn nearest_point(
    slots: &[ProjectSlot],
    quantity: PlotQuantity,
    mirror: bool,
    show_relax: bool,
    smooth: bool,
    pointer: egui::Pos2,
    transform: &PlotTransform,
) -> Option<(usize, usize, usize)> {
    const PICK_RADIUS_PX: f32 = 20.0;
    let mut best: Option<(f32, (usize, usize, usize))> = None;
    for (pidx, slot) in slots.iter().enumerate() {
        if !slot.enabled {
            continue;
        }
        for (ridx, entry) in slot.entries.iter().enumerate() {
            if !entry.visible {
                continue;
            }
            let curve = &entry.curve;
            let mirrored = mirror && curve.key.branch == Branch::Release;
            let start = if mirrored {
                slot.release_start_mm(ridx)
            } else {
                0.0
            };
            let y_by_frame = plot_y_values(curve, quantity, show_relax, smooth);
            for (fidx, p) in curve.points.iter().enumerate() {
                let Some(y) = y_by_frame[fidx] else {
                    continue;
                };
                let x = if mirrored {
                    curve.mirrored_displacement(start, p.displacement_mm)
                } else {
                    p.displacement_mm
                };
                let pos = transform.position_from_point(&egui_plot::PlotPoint::new(x, y));
                let d2 = pos.distance_sq(pointer);
                if best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, (pidx, ridx, fidx)));
                }
            }
        }
    }
    best.filter(|(d2, _)| *d2 <= PICK_RADIUS_PX * PICK_RADIUS_PX)
        .map(|(_, hit)| hit)
}

#[cfg(test)]
mod tests {
    use fabric_data::{CurvePoint, Direction, RunKey};

    use super::*;

    fn curve_of(forces: impl Iterator<Item = f64>) -> Curve {
        let points = forces
            .enumerate()
            .map(|(i, f)| CurvePoint {
                frame_id: i as u32,
                displacement_mm: i as f64,
                force_n: Some(f),
                line_load_n_per_m: None,
                is_relaxation_hold: false,
            })
            .collect();
        Curve {
            key: RunKey {
                sample_id: "s".into(),
                direction: Direction::Warp,
                branch: Branch::Loading,
            },
            test_id: "t".into(),
            points,
        }
    }

    fn second_difference_sum(values: &[Option<f64>]) -> f64 {
        values
            .iter()
            .filter_map(|&v| v)
            .collect::<Vec<_>>()
            .windows(3)
            .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
            .sum()
    }

    #[test]
    fn smoothing_denoises_microjumps_and_subdivides_smoothly() {
        // Linear trend with alternating +/- 1.0 micro-jumps.
        let curve = curve_of((0..10).map(|i| i as f64 * 2.0 + if i % 2 == 0 { -1.0 } else { 1.0 }));
        let raw = plot_y_values(&curve, PlotQuantity::ForceN, true, false);
        let smoothed = plot_y_values(&curve, PlotQuantity::ForceN, true, true);
        assert!(
            second_difference_sum(&smoothed) < second_difference_sum(&raw) * 0.1,
            "smoothing should strongly reduce second-difference jitter"
        );

        let series: Vec<[f64; 2]> = smoothed
            .into_iter()
            .enumerate()
            .filter_map(|(i, y)| Some([i as f64, y?]))
            .collect();
        let dense = smooth_curve_points(&series);
        assert!(dense.len() > series.len() * 4);
        assert_eq!(dense.first(), series.first());
        assert_eq!(dense.last(), series.last());
    }

    #[test]
    fn the_gaussian_window_is_truncated_without_changing_the_result_much() {
        // A long noisy ramp: the truncated filter still tracks the trend.
        let curve = curve_of((0..400).map(|i| i as f64 * 0.5 + ((i * 37) % 7) as f64 * 0.1));
        let smoothed = plot_y_values(&curve, PlotQuantity::ForceN, true, true);
        for (i, y) in smoothed.iter().enumerate().skip(10).take(380) {
            let trend = i as f64 * 0.5 + 0.3;
            assert!((y.unwrap() - trend).abs() < 0.4, "frame {i}");
        }
    }

    #[test]
    fn relaxation_holds_are_hidden_or_left_unsmoothed() {
        let mut curve = curve_of([0.0, 1.0, 2.0, 2.0, 2.0, 3.0].into_iter());
        curve.points[3].is_relaxation_hold = true;
        curve.points[4].is_relaxation_hold = true;
        let hidden = plot_y_values(&curve, PlotQuantity::ForceN, false, false);
        assert_eq!(hidden[3], None);
        let shown = plot_y_values(&curve, PlotQuantity::ForceN, true, true);
        assert_eq!(shown[3], Some(2.0), "holds are never denoised");
    }

    #[test]
    fn spline_never_overshoots_a_step_or_a_peak() {
        // A step: a plain Catmull-Rom would ring above 1 and below 0.
        let step: Vec<[f64; 2]> = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            .iter()
            .enumerate()
            .map(|(i, &y)| [i as f64, y])
            .collect();
        for p in smooth_curve_points(&step) {
            assert!((0.0..=1.0).contains(&p[1]), "overshoot at {p:?}");
        }
        // A sharp peak must not exceed the measured maximum.
        let peak: Vec<[f64; 2]> = [0.0, 1.0, 10.0, 1.0, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &y)| [i as f64, y])
            .collect();
        let max = smooth_curve_points(&peak)
            .iter()
            .map(|p| p[1])
            .fold(f64::MIN, f64::max);
        assert!(max <= 10.0 + 1e-9, "peak overshoot {max}");
    }

    #[test]
    fn spline_is_monotone_where_the_data_is_and_interpolates_the_samples() {
        let rising: Vec<[f64; 2]> = [0.0, 0.1, 0.2, 5.0, 5.1, 9.0]
            .iter()
            .enumerate()
            .map(|(i, &y)| [i as f64 * 2.0, y])
            .collect();
        let dense = smooth_curve_points(&rising);
        for w in dense.windows(2) {
            assert!(w[1][1] >= w[0][1] - 1e-12, "y must not decrease: {w:?}");
            assert!(w[1][0] >= w[0][0] - 1e-12, "x must not decrease: {w:?}");
        }
        for (i, p) in rising.iter().enumerate().take(rising.len() - 1) {
            assert_eq!(dense[i * 8], *p, "sample {i} is reproduced exactly");
        }
        assert_eq!(smooth_curve_points(&rising[..2]), rising[..2].to_vec());
    }
}
