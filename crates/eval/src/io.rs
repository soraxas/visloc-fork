//! Readers for the trajectory formats the benchmarks see.

use std::path::Path;

use nalgebra::{Isometry3, Matrix3, Quaternion, Rotation3, Translation3, UnitQuaternion};

use crate::{EvalError, StampedPose, Trajectory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GtFormat {
    /// `state_groundtruth_estimate0/data.csv`: `t_ns, px, py, pz, qw, qx, qy, qz, ...`
    Euroc,
    /// `t tx ty tz qx qy qz qw`, `t` in seconds or nanoseconds.
    Tum,
    /// 12 values per line (row-major 3x4), timestamps from a `times.txt` or 10 Hz.
    Kitti,
}

/// Parse a timestamp that is either seconds (`1403636579.763555765`, `12`) or integer
/// nanoseconds (`1403636579763555765`). Decimal strings are split at the point so
/// nanoseconds survive exactly (an f64 holds ~1e9 s only to ~100 ns). Integers above
/// 1e12 are taken as nanoseconds, the same rule as the ad hoc `ate.py` this replaces.
pub fn parse_timestamp_ns(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.contains(['e', 'E']) {
        let v: f64 = s.parse().ok()?;
        return Some(if v > 1e12 {
            v.round() as i64
        } else {
            (v * 1e9).round() as i64
        });
    }
    match s.split_once('.') {
        Some((int, frac)) => {
            let secs: i64 = if int.is_empty() { 0 } else { int.parse().ok()? };
            if !frac.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            // ORB-SLAM3's SaveTrajectoryEuRoC writes integer nanoseconds with a decimal
            // suffix ("1403636626113555456.000000"); same >1e12 rule as plain integers.
            if secs.abs() > 1_000_000_000_000 {
                return Some(secs);
            }
            let mut digits: String = frac.chars().take(9).collect();
            while digits.len() < 9 {
                digits.push('0');
            }
            let nanos: i64 = digits.parse().ok()?;
            if int.starts_with('-') {
                Some(secs * 1_000_000_000 - nanos)
            } else {
                Some(secs * 1_000_000_000 + nanos)
            }
        }
        None => {
            let v: i64 = s.parse().ok()?;
            Some(if v > 1_000_000_000_000 {
                v
            } else {
                v * 1_000_000_000
            })
        }
    }
}

/// Nearest rotation to a (rounded or noisy) 3x3 matrix, closed form via SVD. Not
/// nalgebra's iterative `from_matrix`: unbounded it can loop forever, and bounded at
/// 100 iterations it did not converge on an exact 90 degree rotation.
fn nearest_rotation(r: &Matrix3<f64>) -> Result<UnitQuaternion<f64>, EvalError> {
    let degenerate = || EvalError::Parse {
        line: 0,
        message: "degenerate rotation matrix".into(),
    };
    let svd = r
        .try_svd(true, true, f64::EPSILON * 5.0, 10_000)
        .ok_or_else(degenerate)?;
    let (u, vt) = (
        svd.u.ok_or_else(degenerate)?,
        svd.v_t.ok_or_else(degenerate)?,
    );
    let mut m = u * vt;
    if m.determinant() < 0.0 {
        m = u * Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, -1.0)) * vt;
    }
    if !m.iter().all(|x| x.is_finite()) {
        return Err(degenerate());
    }
    Ok(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(m),
    ))
}

fn read(path: &Path) -> Result<String, EvalError> {
    std::fs::read_to_string(path).map_err(|e| EvalError::Io(format!("{}: {e}", path.display())))
}

fn num(field: &str, line: usize) -> Result<f64, EvalError> {
    field.trim().parse().map_err(|_| EvalError::Parse {
        line,
        message: format!("not a number: {field:?}"),
    })
}

fn iso(t: [f64; 3], q: UnitQuaternion<f64>) -> Isometry3<f64> {
    Isometry3::from_parts(Translation3::new(t[0], t[1], t[2]), q)
}

fn unit_quat(
    w: f64,
    x: f64,
    y: f64,
    z: f64,
    line: usize,
) -> Result<UnitQuaternion<f64>, EvalError> {
    UnitQuaternion::try_new(Quaternion::new(w, x, y, z), 1e-12).ok_or(EvalError::Parse {
        line,
        message: "quaternion norm is zero".into(),
    })
}

pub fn parse_tum(text: &str) -> Result<Trajectory, EvalError> {
    let mut poses = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let l = raw.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() != 8 {
            return Err(EvalError::Parse {
                line,
                message: format!("expected 8 fields, got {}", f.len()),
            });
        }
        let t_ns = parse_timestamp_ns(f[0]).ok_or(EvalError::Parse {
            line,
            message: format!("bad timestamp {:?}", f[0]),
        })?;
        let v: Vec<f64> = f[1..]
            .iter()
            .map(|x| num(x, line))
            .collect::<Result<_, _>>()?;
        let q = unit_quat(v[6], v[3], v[4], v[5], line)?;
        poses.push(StampedPose {
            t_ns,
            pose: iso([v[0], v[1], v[2]], q),
        });
    }
    Ok(Trajectory::new(poses))
}

/// Also reads the DPVO demo's `dpvo_trajectory.csv` (same columns, header line
/// `timestamp_ns,tx,...` without `#`).
pub fn parse_euroc_gt(text: &str) -> Result<Trajectory, EvalError> {
    let mut poses = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let l = raw.trim();
        let is_header = poses.is_empty() && l.starts_with(|c: char| c.is_ascii_alphabetic());
        if l.is_empty() || l.starts_with('#') || is_header {
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        if f.len() < 8 {
            return Err(EvalError::Parse {
                line,
                message: format!("expected >= 8 fields, got {}", f.len()),
            });
        }
        let t_ns: i64 = f[0].trim().parse().map_err(|_| EvalError::Parse {
            line,
            message: format!("bad timestamp {:?}", f[0]),
        })?;
        let v: Vec<f64> = f[1..8]
            .iter()
            .map(|x| num(x, line))
            .collect::<Result<_, _>>()?;
        let q = unit_quat(v[3], v[4], v[5], v[6], line)?;
        poses.push(StampedPose {
            t_ns,
            pose: iso([v[0], v[1], v[2]], q),
        });
    }
    Ok(Trajectory::new(poses))
}

/// `times`: one timestamp per line in seconds (KITTI `times.txt`). When absent, pose
/// `i` gets `t = i * 100 ms` (KITTI's nominal 10 Hz), so two KITTI files associate by
/// index.
pub fn parse_kitti(text: &str, times: Option<&str>) -> Result<Trajectory, EvalError> {
    let stamps: Option<Vec<i64>> = match times {
        Some(t) => Some(
            t.lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
                .map(|(i, l)| {
                    parse_timestamp_ns(l).ok_or(EvalError::Parse {
                        line: i + 1,
                        message: format!("bad time {l:?}"),
                    })
                })
                .collect::<Result<_, _>>()?,
        ),
        None => None,
    };
    let mut poses = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let l = raw.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let v: Vec<f64> = l
            .split_whitespace()
            .map(|x| num(x, line))
            .collect::<Result<_, _>>()?;
        if v.len() != 12 {
            return Err(EvalError::Parse {
                line,
                message: format!("expected 12 values, got {}", v.len()),
            });
        }
        let idx = poses.len();
        let t_ns = match &stamps {
            Some(s) => *s.get(idx).ok_or(EvalError::Parse {
                line,
                message: "more poses than timestamps".into(),
            })?,
            None => idx as i64 * 100_000_000,
        };
        let r = Matrix3::new(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]);
        let q = nearest_rotation(&r).map_err(|_| EvalError::Parse {
            line,
            message: "degenerate rotation".into(),
        })?;
        poses.push(StampedPose {
            t_ns,
            pose: iso([v[3], v[7], v[11]], q),
        });
    }
    Ok(Trajectory::new(poses))
}

/// `T_BS` (sensor-to-body) from an EuRoC `sensor.yaml`: the 16 numbers of the
/// `data: [...]` list after the `T_BS:` key, row-major.
pub fn parse_euroc_t_bs(text: &str) -> Result<Isometry3<f64>, EvalError> {
    let start = text.find("T_BS:").ok_or(EvalError::Parse {
        line: 0,
        message: "no T_BS key".into(),
    })?;
    let rest = &text[start..];
    let open = rest.find('[').ok_or(EvalError::Parse {
        line: 0,
        message: "no T_BS data list".into(),
    })?;
    let close = rest[open..].find(']').ok_or(EvalError::Parse {
        line: 0,
        message: "unterminated T_BS data".into(),
    })? + open;
    let v: Vec<f64> = rest[open + 1..close]
        .split(',')
        .map(|x| num(x, 0))
        .collect::<Result<_, _>>()?;
    if v.len() != 16 {
        return Err(EvalError::Parse {
            line: 0,
            message: format!("T_BS has {} values, expected 16", v.len()),
        });
    }
    let r = Matrix3::new(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]);
    Ok(iso([v[3], v[7], v[11]], nearest_rotation(&r)?))
}

pub fn read_euroc_t_bs(path: impl AsRef<Path>) -> Result<Isometry3<f64>, EvalError> {
    parse_euroc_t_bs(&read(path.as_ref())?)
}

pub fn read_tum(path: impl AsRef<Path>) -> Result<Trajectory, EvalError> {
    parse_tum(&read(path.as_ref())?)
}

pub fn read_euroc_gt(path: impl AsRef<Path>) -> Result<Trajectory, EvalError> {
    parse_euroc_gt(&read(path.as_ref())?)
}

pub fn read_kitti(path: impl AsRef<Path>, times: Option<&Path>) -> Result<Trajectory, EvalError> {
    let t = match times {
        Some(p) => Some(read(p)?),
        None => None,
    };
    parse_kitti(&read(path.as_ref())?, t.as_deref())
}

pub fn read_gt(
    path: impl AsRef<Path>,
    format: GtFormat,
    times: Option<&Path>,
) -> Result<Trajectory, EvalError> {
    match format {
        GtFormat::Euroc => read_euroc_gt(path),
        GtFormat::Tum => read_tum(path),
        GtFormat::Kitti => read_kitti(path, times),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_formats() {
        assert_eq!(
            parse_timestamp_ns("1403636579.763555765"),
            Some(1_403_636_579_763_555_765)
        );
        assert_eq!(
            parse_timestamp_ns("1403636579763555765"),
            Some(1_403_636_579_763_555_765)
        );
        assert_eq!(parse_timestamp_ns("12"), Some(12_000_000_000));
        assert_eq!(parse_timestamp_ns("0.5"), Some(500_000_000));
        assert_eq!(parse_timestamp_ns("1.0000000019"), Some(1_000_000_001));
        assert_eq!(parse_timestamp_ns("1.4e9"), Some(1_400_000_000_000_000_000));
        assert_eq!(
            parse_timestamp_ns("1403636626113555456.000000"),
            Some(1_403_636_626_113_555_456)
        );
        assert_eq!(parse_timestamp_ns("abc"), None);
    }

    #[test]
    fn parsers_agree_on_one_pose() {
        let tum = parse_tum("# h\n1.5 1 2 3 0 0 0 1\n").unwrap();
        let euroc = parse_euroc_gt("#h\n1500000000,1,2,3,1,0,0,0,9,9\n").unwrap();
        let kitti = parse_kitti("1 0 0 1 0 1 0 2 0 0 1 3\n", Some("1.5\n")).unwrap();
        assert_eq!(tum.poses(), euroc.poses());
        let (a, b) = (tum.poses()[0], kitti.poses()[0]);
        assert_eq!(a.t_ns, b.t_ns);
        assert!((a.pose.to_homogeneous() - b.pose.to_homogeneous()).norm() < 1e-12);
    }

    #[test]
    fn euroc_t_bs_parses_row_major() {
        let yaml = "T_BS:\n  cols: 4\n  rows: 4\n  data: [0.0, -1.0, 0.0, 1.0,\n 1.0, 0.0, 0.0, 2.0,\n 0.0, 0.0, 1.0, 3.0,\n 0.0, 0.0, 0.0, 1.0]\n";
        let t = parse_euroc_t_bs(yaml).unwrap();
        assert!((t.translation.vector - nalgebra::Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
        // Sensor x axis maps to body y.
        let x = t.rotation * nalgebra::Vector3::x();
        assert!((x - nalgebra::Vector3::y()).norm() < 1e-12);
    }

    #[test]
    fn csv_header_without_hash_is_skipped() {
        let t = parse_euroc_gt("timestamp_ns,tx,ty,tz,qw,qx,qy,qz\n5,1,2,3,1,0,0,0\n").unwrap();
        assert_eq!(t.len(), 1);
        assert!(parse_euroc_gt("5,1,2,3,1,0,0,0\nx,1,2,3,1,0,0,0\n").is_err());
    }

    #[test]
    fn bad_input_is_an_error_not_a_skip() {
        assert!(matches!(
            parse_tum("1 2 3\n"),
            Err(EvalError::Parse { line: 1, .. })
        ));
        assert!(matches!(
            parse_tum("1 0 0 0 0 0 0 0\n"),
            Err(EvalError::Parse { .. })
        ));
        assert!(matches!(
            parse_kitti("1 0 0 0\n", None),
            Err(EvalError::Parse { .. })
        ));
    }
}
