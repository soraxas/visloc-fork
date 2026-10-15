//! Absolute trajectory error with timestamp association, whole-trajectory and windowed.

use nalgebra::Point3;
use serde::Serialize;
use visloc_tracking::umeyama_similarity_transform;

use crate::traj::path_length;
use crate::{EvalError, Trajectory};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Alignment {
    /// Rigid: rotation + translation. Use for metric systems (VIO).
    Se3,
    /// Similarity: rotation + translation + scale. Use for monocular visual-only.
    Sim3,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AteResult {
    pub alignment: Alignment,
    pub est_count: usize,
    pub matched: usize,
    pub rmse_m: f64,
    pub mean_m: f64,
    pub median_m: f64,
    pub max_m: f64,
    /// Scale applied to the estimate (1.0 for SE3).
    pub scale: f64,
    pub est_path_m: f64,
    pub gt_path_m: f64,
    pub rmse_pct_of_path: f64,
}

/// For every estimate pose, the nearest ground-truth pose if it lies strictly within
/// `tol_ns`. Several estimates may share one GT pose. Returns `(est_idx, gt_idx)` in
/// estimate order.
pub fn associate(est: &Trajectory, gt: &Trajectory, tol_ns: i64) -> Vec<(usize, usize)> {
    let g = gt.poses();
    if g.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(est.len());
    for (i, e) in est.poses().iter().enumerate() {
        let j = g.partition_point(|p| p.t_ns < e.t_ns);
        let best = [j.checked_sub(1), (j < g.len()).then_some(j)]
            .into_iter()
            .flatten()
            .min_by_key(|&k| ((g[k].t_ns - e.t_ns).abs(), k))
            .expect("gt is non-empty");
        if (g[best].t_ns - e.t_ns).abs() < tol_ns {
            out.push((i, best));
        }
    }
    out
}

/// Align matched positions and return the error statistics.
pub fn ate_points(
    est: &[Point3<f64>],
    gt: &[Point3<f64>],
    alignment: Alignment,
) -> Result<AteResult, EvalError> {
    assert_eq!(est.len(), gt.len());
    if est.len() < 3 {
        return Err(EvalError::TooFewMatches { matched: est.len() });
    }
    let t = umeyama_similarity_transform(est, gt, alignment == Alignment::Sim3)
        .ok_or(EvalError::Degenerate)?;
    let mut errs: Vec<f64> = est
        .iter()
        .zip(gt)
        .map(|(e, g)| (t.apply(e) - g).norm())
        .collect();
    let n = errs.len() as f64;
    let rmse = (errs.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    let mean = errs.iter().sum::<f64>() / n;
    let max = errs.iter().copied().fold(0.0, f64::max);
    errs.sort_by(|a, b| a.total_cmp(b));
    let median = if errs.len() % 2 == 1 {
        errs[errs.len() / 2]
    } else {
        0.5 * (errs[errs.len() / 2 - 1] + errs[errs.len() / 2])
    };
    let gt_path = path_length(gt);
    Ok(AteResult {
        alignment,
        est_count: est.len(),
        matched: est.len(),
        rmse_m: rmse,
        mean_m: mean,
        median_m: median,
        max_m: max,
        scale: t.scale,
        est_path_m: path_length(est),
        gt_path_m: gt_path,
        rmse_pct_of_path: if gt_path > 0.0 {
            100.0 * rmse / gt_path
        } else {
            f64::NAN
        },
    })
}

fn matched_points(
    est: &Trajectory,
    gt: &Trajectory,
    tol_ns: i64,
) -> (Vec<Point3<f64>>, Vec<Point3<f64>>) {
    associate(est, gt, tol_ns)
        .into_iter()
        .map(|(i, j)| (est.poses()[i].position(), gt.poses()[j].position()))
        .unzip()
}

/// Whole-trajectory ATE (position RMSE after alignment).
pub fn ate(
    est: &Trajectory,
    gt: &Trajectory,
    tol_ns: i64,
    alignment: Alignment,
) -> Result<AteResult, EvalError> {
    let (e, g) = matched_points(est, gt, tol_ns);
    let mut r = ate_points(&e, &g, alignment)?;
    r.est_count = est.len();
    Ok(r)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WindowResult {
    pub t0_ns: i64,
    pub matched: usize,
    pub rmse_m: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WindowSummary {
    pub alignment: Alignment,
    pub window_s: f64,
    pub step_s: f64,
    /// Windows with at least `min_matched` associated poses.
    pub scored: usize,
    /// Windows inside the estimate's time span with fewer poses (tracking gaps).
    pub skipped: usize,
    pub median_rmse_m: f64,
    pub p90_rmse_m: f64,
    pub max_rmse_m: f64,
    pub windows: Vec<WindowResult>,
}

/// ATE over sliding windows of `window_s` seconds every `step_s`, each window aligned
/// on its own. Separates local drift from accumulated drift and scale changes.
pub fn windowed_ate(
    est: &Trajectory,
    gt: &Trajectory,
    tol_ns: i64,
    alignment: Alignment,
    window_s: f64,
    step_s: f64,
    min_matched: usize,
) -> WindowSummary {
    let (Some(first), Some(last)) = (est.poses().first(), est.poses().last()) else {
        return summarize(alignment, window_s, step_s, 0, Vec::new());
    };
    let win = (window_s * 1e9).round() as i64;
    let step = ((step_s * 1e9).round() as i64).max(1);
    let mut windows = Vec::new();
    let mut skipped = 0;
    let mut t0 = first.t_ns;
    while t0 + win <= last.t_ns + 1 {
        let (e, g) = matched_points(&est.slice_time(t0, t0 + win), gt, tol_ns);
        match (e.len() >= min_matched.max(3)).then(|| ate_points(&e, &g, alignment)) {
            Some(Ok(r)) => windows.push(WindowResult {
                t0_ns: t0,
                matched: r.matched,
                rmse_m: r.rmse_m,
                scale: r.scale,
            }),
            _ => skipped += 1,
        }
        t0 += step;
    }
    summarize(alignment, window_s, step_s, skipped, windows)
}

fn summarize(
    alignment: Alignment,
    window_s: f64,
    step_s: f64,
    skipped: usize,
    windows: Vec<WindowResult>,
) -> WindowSummary {
    let mut v: Vec<f64> = windows.iter().map(|w| w.rmse_m).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| {
        if v.is_empty() {
            f64::NAN
        } else {
            v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
        }
    };
    WindowSummary {
        alignment,
        window_s,
        step_s,
        scored: windows.len(),
        skipped,
        median_rmse_m: pick(0.5),
        p90_rmse_m: pick(0.9),
        max_rmse_m: v.last().copied().unwrap_or(f64::NAN),
        windows,
    }
}

#[cfg(test)]
mod tests {
    //! Known-answer tests: each feeds an input whose correct score is known exactly.
    use super::*;
    use crate::StampedPose;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    /// A non-planar, non-degenerate path at 20 Hz.
    fn helix(n: usize) -> Trajectory {
        Trajectory::new(
            (0..n)
                .map(|i| {
                    let s = i as f64 * 0.05;
                    StampedPose {
                        t_ns: 1_000_000_000 + i as i64 * 50_000_000,
                        pose: Isometry3::from_parts(
                            Translation3::new(3.0 * s.cos(), 2.0 * s.sin(), 0.3 * s),
                            UnitQuaternion::from_euler_angles(0.1 * s, 0.0, s),
                        ),
                    }
                })
                .collect(),
        )
    }

    fn map(t: &Trajectory, f: impl FnMut(&StampedPose) -> StampedPose) -> Trajectory {
        Trajectory::new(t.poses().iter().map(f).collect())
    }

    const TOL: i64 = 10_000_000;

    #[test]
    fn gt_against_itself_scores_zero() {
        let gt = helix(400);
        for a in [Alignment::Se3, Alignment::Sim3] {
            let r = ate(&gt, &gt, TOL, a).unwrap();
            assert_eq!(r.matched, 400);
            assert!(r.rmse_m < 1e-9, "{a:?} {}", r.rmse_m);
            assert!((r.scale - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn known_similarity_is_recovered() {
        let gt = helix(400);
        let rot = UnitQuaternion::from_euler_angles(0.3, -0.2, 1.1);
        let off = Vector3::new(5.0, -1.0, 2.0);
        // est = 2 * R * gt + t, so aligning est onto gt needs scale 0.5.
        let est = map(&gt, |p| StampedPose {
            t_ns: p.t_ns,
            pose: Isometry3::from_parts(
                Translation3::from(rot * (2.0 * p.pose.translation.vector) + off),
                p.pose.rotation,
            ),
        });
        let sim = ate(&est, &gt, TOL, Alignment::Sim3).unwrap();
        assert!(sim.rmse_m < 1e-9, "{}", sim.rmse_m);
        assert!((sim.scale - 0.5).abs() < 1e-9, "{}", sim.scale);
        let se3 = ate(&est, &gt, TOL, Alignment::Se3).unwrap();
        assert!(
            se3.rmse_m > 0.5,
            "SE3 must not absorb a 2x scale: {}",
            se3.rmse_m
        );
    }

    #[test]
    fn reflection_is_never_absorbed() {
        // A mirrored trajectory is not a similarity of GT: alignment must keep det(R) = +1
        // and leave a large residual.
        let gt = helix(400);
        let est = map(&gt, |p| {
            let v = p.pose.translation.vector;
            StampedPose {
                t_ns: p.t_ns,
                pose: Isometry3::from_parts(Translation3::new(v.x, v.y, -v.z), p.pose.rotation),
            }
        });
        let r = ate(&est, &gt, TOL, Alignment::Sim3).unwrap();
        assert!(r.rmse_m > 0.1, "{}", r.rmse_m);
    }

    #[test]
    fn isotropic_noise_gives_sigma_sqrt3() {
        let gt = helix(5000);
        let sigma = 0.02;
        let mut rng = SmallRng::seed_from_u64(7);
        let mut gauss = || {
            // Box-Muller; rand_distr is not a dependency.
            let (u, v): (f64, f64) = (rng.gen::<f64>().max(1e-300), rng.gen());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let est = map(&gt, |p| StampedPose {
            t_ns: p.t_ns,
            pose: Isometry3::from_parts(
                Translation3::from(
                    p.pose.translation.vector + sigma * Vector3::new(gauss(), gauss(), gauss()),
                ),
                p.pose.rotation,
            ),
        });
        let r = ate(&est, &gt, TOL, Alignment::Se3).unwrap();
        let expected = sigma * 3f64.sqrt();
        assert!(
            (r.rmse_m / expected - 1.0).abs() < 0.05,
            "{} vs {expected}",
            r.rmse_m
        );
    }

    #[test]
    fn association_tolerance_is_strict_and_nearest() {
        let gt = helix(100);
        let shifted = |dt: i64| {
            map(&gt, |p| StampedPose {
                t_ns: p.t_ns + dt,
                pose: p.pose,
            })
        };
        assert_eq!(associate(&shifted(5_000_000), &gt, TOL).len(), 100);
        assert_eq!(associate(&shifted(10_000_000), &gt, TOL).len(), 0);
        assert_eq!(associate(&shifted(-9_999_999), &gt, TOL).len(), 100);
        // 30 ms off a 50 ms grid: nearest is the next GT sample (20 ms away), not the previous.
        let pairs = associate(&shifted(30_000_000), &gt, 25_000_000);
        assert!(pairs.iter().all(|&(i, j)| j == i + 1));
        // Estimate at 200 Hz against 20 Hz GT: only samples within tolerance match.
        let dense = Trajectory::new(
            (0..1000)
                .map(|i| StampedPose {
                    t_ns: 1_000_000_000 + i * 5_000_000,
                    pose: Isometry3::identity(),
                })
                .collect(),
        );
        assert_eq!(
            associate(&dense, &gt, TOL).len(),
            100 * 3 - 1,
            "first GT sample has no earlier neighbour"
        );
    }

    #[test]
    fn too_few_matches_is_an_error() {
        let gt = helix(10);
        let est = gt.first(2);
        assert_eq!(
            ate(&est, &gt, TOL, Alignment::Sim3),
            Err(EvalError::TooFewMatches { matched: 2 })
        );
    }

    #[test]
    fn windows_separate_scale_drift_from_local_error() {
        // Scale jumps 1x -> 3x halfway: each window is a perfect similarity of GT, so
        // windowed Sim3 ATE is ~0 except windows that straddle the jump, while the
        // whole-trajectory Sim3 ATE is large.
        let gt = helix(1200);
        let mid = gt.poses()[600].t_ns;
        let pivot = gt.poses()[600].pose.translation.vector;
        let est = map(&gt, |p| {
            let v = p.pose.translation.vector;
            let v = if p.t_ns < mid {
                v
            } else {
                pivot + 3.0 * (v - pivot)
            };
            StampedPose {
                t_ns: p.t_ns,
                pose: Isometry3::from_parts(Translation3::from(v), p.pose.rotation),
            }
        });
        let whole = ate(&est, &gt, TOL, Alignment::Sim3).unwrap();
        assert!(whole.rmse_m > 1.0, "{}", whole.rmse_m);
        let w = windowed_ate(&est, &gt, TOL, Alignment::Sim3, 5.0, 5.0, 20);
        assert_eq!(w.skipped, 0);
        assert_eq!(
            w.scored, 11,
            "60 s / 5 s windows, the last partial one dropped"
        );
        let clean = w.windows.iter().filter(|r| r.rmse_m < 1e-9).count();
        assert!(clean >= w.scored - 1, "{clean} of {}", w.scored);
        assert!(w.median_rmse_m < 1e-9);
    }

    #[test]
    fn tracking_gaps_are_counted_not_hidden() {
        let gt = helix(1200);
        let est = Trajectory::new(
            gt.poses()
                .iter()
                .copied()
                .filter(|p| !(400..800).contains(&((p.t_ns - 1_000_000_000) / 50_000_000)))
                .collect(),
        );
        let w = windowed_ate(&est, &gt, TOL, Alignment::Sim3, 5.0, 5.0, 20);
        assert_eq!(w.skipped, 4, "20 s gap = 4 empty 5 s windows");
    }
}
