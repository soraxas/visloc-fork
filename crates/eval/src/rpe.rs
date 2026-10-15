//! Relative rotation error over a fixed time gap: scale-free drift of orientation.
//!
//! For associated pairs `(i, j)` with `t_j - t_i ≈ delta`, the error is the angle of
//! `(R_gt_i⁻¹ R_gt_j)⁻¹ (R_est_i⁻¹ R_est_j)`. Any constant world-frame rotation between
//! estimate and ground truth cancels. A constant body-frame offset does not, so both
//! sides must describe the same frame (camera vs camera: convert EuRoC body ground truth
//! with `Trajectory::compose_right(T_BS)`).

use serde::Serialize;

use crate::ate::associate;
use crate::Trajectory;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RotationRpe {
    pub delta_s: f64,
    pub pairs: usize,
    pub median_deg: f64,
    pub p90_deg: f64,
    pub rmse_deg: f64,
    /// `median_deg / delta_s`.
    pub median_deg_per_s: f64,
}

/// Pairs whose actual gap is outside `[delta, 1.5 * delta]` (tracking gaps) are skipped.
pub fn rotation_rpe(est: &Trajectory, gt: &Trajectory, tol_ns: i64, delta_s: f64) -> RotationRpe {
    let pairs = associate(est, gt, tol_ns);
    let delta = (delta_s * 1e9).round() as i64;
    let e = est.poses();
    let g = gt.poses();
    let mut errs = Vec::new();
    let mut j = 0usize;
    for i in 0..pairs.len() {
        let ti = e[pairs[i].0].t_ns;
        if j < i {
            j = i;
        }
        while j < pairs.len() && e[pairs[j].0].t_ns - ti < delta {
            j += 1;
        }
        if j >= pairs.len() {
            break;
        }
        let gap = e[pairs[j].0].t_ns - ti;
        if gap > delta + delta / 2 {
            continue;
        }
        let (ei, gi) = pairs[i];
        let (ej, gj) = pairs[j];
        let d_est = e[ei].pose.rotation.inverse() * e[ej].pose.rotation;
        let d_gt = g[gi].pose.rotation.inverse() * g[gj].pose.rotation;
        errs.push((d_gt.inverse() * d_est).angle().to_degrees());
    }
    summarize(delta_s, errs)
}

fn summarize(delta_s: f64, mut errs: Vec<f64>) -> RotationRpe {
    errs.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| {
        if errs.is_empty() {
            f64::NAN
        } else {
            errs[((q * (errs.len() - 1) as f64).round() as usize).min(errs.len() - 1)]
        }
    };
    let rmse = if errs.is_empty() {
        f64::NAN
    } else {
        (errs.iter().map(|x| x * x).sum::<f64>() / errs.len() as f64).sqrt()
    };
    RotationRpe {
        delta_s,
        pairs: errs.len(),
        median_deg: pick(0.5),
        p90_deg: pick(0.9),
        rmse_deg: rmse,
        median_deg_per_s: pick(0.5) / delta_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StampedPose;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    fn spin(n: usize, extra_yaw_rate: f64) -> Trajectory {
        // 20 Hz, rotating about a tilted axis; `extra_yaw_rate` rad/s of drift about
        // the world z axis is added to the estimate. The relative error is then a
        // conjugate of Rz(rate * delta), so its angle is exactly rate * delta.
        Trajectory::new(
            (0..n)
                .map(|i| {
                    let t = i as f64 * 0.05;
                    let base =
                        UnitQuaternion::from_euler_angles(0.2 * t, 0.1 * (t * 0.7).sin(), 0.5 * t);
                    StampedPose {
                        t_ns: i as i64 * 50_000_000,
                        pose: Isometry3::from_parts(
                            Translation3::new(t, 0.0, 0.0),
                            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), extra_yaw_rate * t)
                                * base,
                        ),
                    }
                })
                .collect(),
        )
    }

    #[test]
    fn identical_is_zero_and_world_rotation_cancels() {
        let gt = spin(400, 0.0);
        assert!(rotation_rpe(&gt, &gt, 10_000_000, 1.0).median_deg < 1e-9);
        let r0 = UnitQuaternion::from_euler_angles(1.0, -0.5, 2.0);
        let rotated = Trajectory::new(
            gt.poses()
                .iter()
                .map(|p| StampedPose {
                    t_ns: p.t_ns,
                    pose: Isometry3::from_parts(p.pose.translation, r0 * p.pose.rotation),
                })
                .collect(),
        );
        let r = rotation_rpe(&rotated, &gt, 10_000_000, 1.0);
        assert!(r.p90_deg < 1e-9, "{}", r.p90_deg);
        assert_eq!(r.pairs, 400 - 20);
    }

    #[test]
    fn known_drift_rate_is_recovered() {
        let gt = spin(400, 0.0);
        let est = spin(400, 0.01); // 0.573 deg/s about world z
        let r = rotation_rpe(&est, &gt, 10_000_000, 2.0);
        assert!(
            (r.median_deg_per_s - 0.01f64.to_degrees()).abs() < 1e-6,
            "{}",
            r.median_deg_per_s
        );
    }

    #[test]
    fn body_frame_offset_is_not_hidden() {
        // The reason ground truth must be converted to the camera frame first.
        let gt = spin(400, 0.0);
        let off = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_euler_angles(0.0, 1.5, 0.0),
        );
        let r = rotation_rpe(&gt.compose_right(&off), &gt, 10_000_000, 1.0);
        assert!(r.median_deg > 1.0, "{}", r.median_deg);
        let r = rotation_rpe(
            &gt.compose_right(&off),
            &gt.compose_right(&off),
            10_000_000,
            1.0,
        );
        assert!(r.median_deg < 1e-9);
    }

    #[test]
    fn gaps_are_skipped() {
        let gt = spin(400, 0.0);
        let est = Trajectory::new(
            gt.poses()
                .iter()
                .copied()
                .filter(|p| !(100..300).contains(&(p.t_ns / 50_000_000)))
                .collect(),
        );
        let r = rotation_rpe(&est, &gt, 10_000_000, 1.0);
        assert_eq!(r.pairs, (100 - 20) + (100 - 20));
    }
}
