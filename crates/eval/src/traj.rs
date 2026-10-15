use nalgebra::{Isometry3, Point3};

/// One pose with its timestamp. `pose` maps the tracked frame (body or camera) to world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StampedPose {
    pub t_ns: i64,
    pub pose: Isometry3<f64>,
}

impl StampedPose {
    pub fn position(&self) -> Point3<f64> {
        Point3::from(self.pose.translation.vector)
    }
}

/// Poses sorted by timestamp (strictly increasing; duplicates keep the first).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectory {
    poses: Vec<StampedPose>,
}

impl Trajectory {
    pub fn new(mut poses: Vec<StampedPose>) -> Self {
        poses.sort_by_key(|p| p.t_ns);
        poses.dedup_by_key(|p| p.t_ns);
        Self { poses }
    }

    pub fn poses(&self) -> &[StampedPose] {
        &self.poses
    }

    pub fn len(&self) -> usize {
        self.poses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Keep poses with `t0_ns <= t < t1_ns`.
    pub fn slice_time(&self, t0_ns: i64, t1_ns: i64) -> Trajectory {
        Trajectory {
            poses: self
                .poses
                .iter()
                .copied()
                .filter(|p| p.t_ns >= t0_ns && p.t_ns < t1_ns)
                .collect(),
        }
    }

    /// Right-compose every pose with `t`: `T_w_b -> T_w_b * t`. With EuRoC's cam0
    /// `T_BS` this turns body (IMU) ground truth into camera poses. The convention was
    /// checked against image-derived relative rotations: 0.72 deg median error, versus
    /// 14.7 deg for the inverse (`docs/fork_changes_vs_upstream.md`, hybrid section).
    pub fn compose_right(&self, t: &Isometry3<f64>) -> Trajectory {
        Trajectory {
            poses: self
                .poses
                .iter()
                .map(|p| StampedPose {
                    t_ns: p.t_ns,
                    pose: p.pose * t,
                })
                .collect(),
        }
    }

    /// Keep the first `n` poses.
    pub fn first(&self, n: usize) -> Trajectory {
        Trajectory {
            poses: self.poses.iter().take(n).copied().collect(),
        }
    }
}

/// Sum of distances between consecutive points.
pub fn path_length(points: &[Point3<f64>]) -> f64 {
    points.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
}
