//! Score an estimated trajectory against ground truth.
//!
//! ```text
//! visloc-score --est traj.tum --gt mav0/state_groundtruth_estimate0/data.csv \
//!     [--gt-format euroc|tum|kitti] [--est-format tum|kitti] [--gt-times times.txt]
//!     [--est-times times.txt] [--tol-ms 10] [--first N] [--window-s 10 --step-s 5]
//!     [--gt-to-camera mav0/cam0/sensor.yaml] [--rpe-s 1] [--json out.json]
//! ```
//!
//! Prints SE3 and Sim3 ATE. With `--window-s`, also windowed Sim3 ATE. With `--rpe-s`,
//! rotation RPE over that gap (needs estimate and GT in the same frame: pass
//! `--gt-to-camera` for camera-pose estimates against EuRoC body ground truth).

use std::path::PathBuf;
use std::process::ExitCode;

use serde::Serialize;
use visloc_eval::io::{read_euroc_t_bs, read_gt, GtFormat};
use visloc_eval::{
    ate, rotation_rpe, windowed_ate, Alignment, AteResult, RotationRpe, WindowSummary,
};

#[derive(Serialize)]
struct Report {
    est: String,
    gt: String,
    tol_ms: f64,
    first: Option<usize>,
    se3: AteResult,
    sim3: AteResult,
    windowed_sim3: Option<WindowSummary>,
    rotation_rpe: Option<RotationRpe>,
}

fn parse_format(s: &str) -> Result<GtFormat, String> {
    match s {
        "euroc" => Ok(GtFormat::Euroc),
        "tum" => Ok(GtFormat::Tum),
        "kitti" => Ok(GtFormat::Kitti),
        _ => Err(format!("unknown format {s:?} (euroc|tum|kitti)")),
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut est, mut gt, mut json) = (None, None, None);
    let (mut gt_times, mut est_times) = (None::<PathBuf>, None::<PathBuf>);
    let (mut gt_format, mut est_format) = (GtFormat::Euroc, GtFormat::Tum);
    let (mut tol_ms, mut first, mut window_s, mut step_s) = (10.0f64, None, None, None);
    let (mut gt_to_camera, mut rpe_s) = (None::<PathBuf>, None::<f64>);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--est" => est = Some(PathBuf::from(val()?)),
            "--gt-to-camera" => gt_to_camera = Some(PathBuf::from(val()?)),
            "--rpe-s" => rpe_s = Some(val()?.parse::<f64>().map_err(|e| format!("--rpe-s: {e}"))?),
            "--gt" => gt = Some(PathBuf::from(val()?)),
            "--json" => json = Some(PathBuf::from(val()?)),
            "--gt-format" => gt_format = parse_format(&val()?)?,
            "--est-format" => est_format = parse_format(&val()?)?,
            "--gt-times" => gt_times = Some(PathBuf::from(val()?)),
            "--est-times" => est_times = Some(PathBuf::from(val()?)),
            "--tol-ms" => tol_ms = val()?.parse().map_err(|e| format!("--tol-ms: {e}"))?,
            "--first" => {
                first = Some(
                    val()?
                        .parse::<usize>()
                        .map_err(|e| format!("--first: {e}"))?,
                )
            }
            "--window-s" => {
                window_s = Some(
                    val()?
                        .parse::<f64>()
                        .map_err(|e| format!("--window-s: {e}"))?,
                )
            }
            "--step-s" => {
                step_s = Some(
                    val()?
                        .parse::<f64>()
                        .map_err(|e| format!("--step-s: {e}"))?,
                )
            }
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("visloc-score.rs")
                        .lines()
                        .take(9)
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {a:?}")),
        }
    }
    let est_path = est.ok_or("--est is required")?;
    let gt_path = gt.ok_or("--gt is required")?;
    let mut est_traj =
        read_gt(&est_path, est_format, est_times.as_deref()).map_err(|e| e.to_string())?;
    if let Some(n) = first {
        est_traj = est_traj.first(n);
    }
    let mut gt_traj =
        read_gt(&gt_path, gt_format, gt_times.as_deref()).map_err(|e| e.to_string())?;
    if let Some(p) = &gt_to_camera {
        gt_traj = gt_traj.compose_right(&read_euroc_t_bs(p).map_err(|e| e.to_string())?);
    }
    let tol_ns = (tol_ms * 1e6).round() as i64;

    let se3 = ate(&est_traj, &gt_traj, tol_ns, Alignment::Se3).map_err(|e| e.to_string())?;
    let sim3 = ate(&est_traj, &gt_traj, tol_ns, Alignment::Sim3).map_err(|e| e.to_string())?;
    println!(
        "matched={}/{} path_gt={:.2}m est_path={:.2}m",
        se3.matched, se3.est_count, se3.gt_path_m, se3.est_path_m
    );
    for r in [&sim3, &se3] {
        println!(
            "{:?}: RMSE={:.4}m ({:.2}% of path) median={:.4}m max={:.3}m scale={:.4}",
            r.alignment, r.rmse_m, r.rmse_pct_of_path, r.median_m, r.max_m, r.scale
        );
    }
    let windowed = window_s.map(|w| {
        let s = windowed_ate(
            &est_traj,
            &gt_traj,
            tol_ns,
            Alignment::Sim3,
            w,
            step_s.unwrap_or(w),
            10,
        );
        println!(
            "windowed Sim3 ({w} s): scored={} skipped={} median={:.4}m p90={:.4}m max={:.4}m",
            s.scored, s.skipped, s.median_rmse_m, s.p90_rmse_m, s.max_rmse_m
        );
        s
    });
    let rpe = rpe_s.map(|d| {
        let r = rotation_rpe(&est_traj, &gt_traj, tol_ns, d);
        println!(
            "rotation RPE ({d} s): pairs={} median={:.3}deg p90={:.3}deg ({:.3} deg/s)",
            r.pairs, r.median_deg, r.p90_deg, r.median_deg_per_s
        );
        r
    });
    if let Some(p) = json {
        let report = Report {
            est: est_path.display().to_string(),
            gt: gt_path.display().to_string(),
            tol_ms,
            first,
            se3,
            sim3,
            windowed_sim3: windowed,
            rotation_rpe: rpe,
        };
        let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(&p, text + "\n").map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("visloc-score: {e}");
            ExitCode::from(2)
        }
    }
}
