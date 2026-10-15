//! Trajectory scoring for EuRoC/TUM/KITTI-format runs (binary: `visloc-score`).
//!
//! One scorer for every system (Basalt, DPVO, external references such as
//! ORB-SLAM3) so numbers are comparable: nearest-timestamp association with a
//! strict tolerance, Umeyama SE(3)/Sim(3) alignment
//! (`visloc_tracking::umeyama_similarity_transform`), position RMSE, the same
//! metric over sliding time windows, and rotation RPE over a time gap. EuRoC
//! body-frame ground truth can be converted to the camera frame (`T_BS`) for
//! camera-pose estimates.
//!
//! `visloc_tracking::TrajectoryErrorSummary` matches poses by exact frame id, which
//! does not work for EuRoC (200 Hz ground truth vs 20 Hz camera). This crate adds the
//! timestamp association on top.

pub mod ate;
pub mod io;
pub mod rpe;
pub mod traj;

pub use ate::{associate, ate, windowed_ate, Alignment, AteResult, WindowSummary};
pub use io::{read_euroc_gt, read_euroc_t_bs, read_kitti, read_tum, GtFormat};
pub use rpe::{rotation_rpe, RotationRpe};
pub use traj::{StampedPose, Trajectory};

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    Io(String),
    Parse { line: usize, message: String },
    TooFewMatches { matched: usize },
    Degenerate,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Io(m) => write!(f, "io error: {m}"),
            EvalError::Parse { line, message } => {
                write!(f, "parse error at line {line}: {message}")
            }
            EvalError::TooFewMatches { matched } => {
                write!(f, "too few associated poses ({matched}); need at least 3")
            }
            EvalError::Degenerate => write!(f, "alignment is degenerate (zero spread)"),
        }
    }
}

impl std::error::Error for EvalError {}
