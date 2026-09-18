# -*- coding: utf-8 -*-
"""
=============================================================================
 1試技の生データから特徴量ベクトルを抽出するスクリプト

 入力:
   - IMU CSV:  data/<S>/<SES>/imu/setNN.csv
   - Pose CSV: data/<S>/<SES>/pose/setNN_yolo_features.csv （YOLO26・優先）
               data/<S>/<SES>/pose/setNN_features.csv      （MediaPipe）
               ※どちらも無ければ骨格特徴なしで処理する
   - meta.json: data/<S>/<SES>/meta.json

 出力:
   - data/<S>/<SES>/features/setNN_features.json (1試技ぶんの特徴量)

 使い方:
   python extract_features.py                              # data/配下を全自動
   python extract_features.py --subject S001
   python extract_features.py --imu path.csv --meta meta.json --set_no 1
   python extract_features.py --force

 依存ライブラリ:
   pip install numpy pandas scipy
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT  = PROJECT_ROOT / "data"
G = 9.80665   # 重力加速度 [m/s^2]


# ===========================================================================
# 信号処理ユーティリティ
# ===========================================================================
def lowpass(x, fs, fc=10.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, fc / nyq, btype="low")
    return filtfilt(b, a, x)


def highpass(x, fs, fc=0.3, order=2):
    nyq = fs / 2.0
    b, a = butter(order, fc / nyq, btype="high")
    return filtfilt(b, a, x)


def integrate_trapz(x, dt):
    """累積台形積分"""
    return np.concatenate([[0.0], np.cumsum((x[:-1] + x[1:]) / 2.0 * dt)])


# ===========================================================================
# レップ検出
# ===========================================================================
def detect_reps(v, fs, expected_reps=None):
    """
    鉛直速度 v から、スクワットのレップ（上昇局面）を検出する。

    単純に上向きピークを拾うと、ラックアウトの持ち上げ動作まで
    レップとして数えてしまう。スクワット1レップは必ず
    「下降 → ボトム → 上昇」の順に起きるため、
    上昇ピークの直前に有意な下降があるものだけを採用する。

    expected_reps を渡すと（meta.json の reps_completed など）、
    候補が多すぎる場合に上位 N 個だけを残す。
    """
    vmax = float(np.max(np.abs(v))) if len(v) else 0.0
    if vmax <= 0.05:
        return np.array([], dtype=int)

    up_thresh   = max(0.12, 0.30 * vmax)   # 上昇とみなす速度
    down_thresh = max(0.08, 0.20 * vmax)   # 下降とみなす速度
    look_back   = int(fs * 3.0)            # 直前3秒の範囲で下降を探す

    cand, _ = find_peaks(v, height=up_thresh, distance=int(fs * 0.6))

    reps = []
    for p in cand:
        lo = max(0, p - look_back)
        window = v[lo:p]
        if len(window) == 0:
            continue
        # 直前に十分な下降があるか（= しゃがんでから立ち上がったか）
        if float(np.min(window)) <= -down_thresh:
            reps.append(p)

    # 下降を伴うものが1つも無ければ、最大ピークだけ残す（ポーズ動作などの保険）
    if not reps and len(cand):
        reps = [int(cand[np.argmax(v[cand])])]

    reps = np.array(sorted(reps), dtype=int)

    # 実測レップ数が分かっている場合は、速度の大きい順に上位 N 個を残す
    if expected_reps and len(reps) > expected_reps:
        top = sorted(reps, key=lambda i: v[i], reverse=True)[:expected_reps]
        reps = np.array(sorted(top), dtype=int)

    return reps


# ===========================================================================
# IMU 特徴量抽出
# ===========================================================================
def compute_imu_features(imu_df: pd.DataFrame, expected_reps=None) -> dict:
    """
    IMU CSV (timestamp_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
    から特徴量を計算

    expected_reps: meta.json の reps_completed（分かっていれば検出を補正する）
    """
    if len(imu_df) < 20:
        return {"imu_n_samples": len(imu_df), "imu_error": "too few samples"}

    t = imu_df["timestamp_ms"].to_numpy() / 1000.0   # [s]
    n = len(t)
    dt_med = float(np.median(np.diff(t)))
    fs = 1.0 / dt_med if dt_med > 0 else 100.0

    ax = imu_df["ax_g"].to_numpy()
    ay = imu_df["ay_g"].to_numpy()
    az = imu_df["az_g"].to_numpy()
    gx = imu_df["gx_dps"].to_numpy()
    gy = imu_df["gy_dps"].to_numpy()
    gz = imu_df["gz_dps"].to_numpy()

    # 重力除去 + LPF
    az_dyn = (az - 1.0) * G                       # [m/s^2]
    az_dyn_lp = lowpass(az_dyn, fs, fc=10.0) if n > int(fs * 0.5) else az_dyn

    # 鉛直速度（1回積分 + ハイパスでドリフト除去）
    v_raw = integrate_trapz(az_dyn_lp, dt_med)
    if n > int(fs * 3):
        v = highpass(v_raw, fs, fc=0.3)
    else:
        v = v_raw - np.mean(v_raw)

    # 各種統計
    peak_v_up   = float(np.max(v))                # 上向きピーク速度
    peak_v_down = float(np.min(v))                # 下向きピーク速度
    mean_v_up   = float(np.mean(v[v > 0])) if np.any(v > 0) else 0.0
    mean_v_down = float(np.mean(v[v < 0])) if np.any(v < 0) else 0.0

    rms_a = float(np.sqrt(np.mean(az_dyn_lp ** 2)))
    peak_a = float(np.max(np.abs(az_dyn_lp)))

    # Jerk（加速度の時間微分）
    jerk = np.diff(az_dyn_lp) / dt_med            # [m/s^3]
    max_jerk = float(np.max(np.abs(jerk)))
    mean_jerk = float(np.mean(np.abs(jerk)))

    # コンセントリック相 / エキセントリック相の時間
    # vが正(上昇) / 負(下降) のフレーム数 × dt
    concentric_time = float(np.sum(v > 0.05) * dt_med)
    eccentric_time  = float(np.sum(v < -0.05) * dt_med)
    total_time = float(t[-1] - t[0])

    # レップ検出
    peaks = detect_reps(v, fs, expected_reps=expected_reps)
    n_reps = int(len(peaks))

    # 各レップのピーク速度
    rep_peak_vs = [float(v[p]) for p in peaks]
    first_rep_peak_v = rep_peak_vs[0] if rep_peak_vs else float("nan")
    last_rep_peak_v  = rep_peak_vs[-1] if rep_peak_vs else float("nan")
    mean_rep_peak_v  = float(np.mean(rep_peak_vs)) if rep_peak_vs else float("nan")

    # Velocity Loss (VL) = (最初のレップピーク - 最終レップピーク) / 最初のレップピーク
    # 1レップのセットでは定義できないため None にする
    if n_reps >= 2 and first_rep_peak_v > 0.01:
        velocity_loss_pct = float(
            (first_rep_peak_v - last_rep_peak_v) / first_rep_peak_v * 100.0
        )
    else:
        velocity_loss_pct = float("nan")

    # 水平方向ブレ（gx, gy積分 → 角度変化、その絶対値の最大）
    roll  = integrate_trapz(gx, dt_med)
    pitch = integrate_trapz(gy, dt_med)
    bar_roll_range  = float(np.max(roll)  - np.min(roll))
    bar_pitch_range = float(np.max(pitch) - np.min(pitch))

    return {
        "imu_n_samples":              int(n),
        "imu_fs_hz":                  round(fs, 2),
        "imu_total_time_s":           round(total_time, 3),
        "imu_concentric_time_s":      round(concentric_time, 3),
        "imu_eccentric_time_s":       round(eccentric_time, 3),

        "imu_peak_velocity_up_mps":   round(peak_v_up,   4),
        "imu_peak_velocity_down_mps": round(peak_v_down, 4),
        "imu_mean_velocity_up_mps":   round(mean_v_up,   4),
        "imu_mean_velocity_down_mps": round(mean_v_down, 4),

        "imu_n_reps_detected":         n_reps,
        "imu_first_rep_peak_v_mps":    round(first_rep_peak_v, 4) if not np.isnan(first_rep_peak_v) else None,
        "imu_last_rep_peak_v_mps":     round(last_rep_peak_v,  4) if not np.isnan(last_rep_peak_v)  else None,
        "imu_mean_rep_peak_v_mps":     round(mean_rep_peak_v,  4) if not np.isnan(mean_rep_peak_v)  else None,
        "imu_velocity_loss_pct":       round(velocity_loss_pct, 2) if not np.isnan(velocity_loss_pct) else None,

        "imu_rms_accel_mps2":          round(rms_a, 4),
        "imu_peak_accel_mps2":         round(peak_a, 4),
        "imu_max_jerk_mps3":           round(max_jerk, 2),
        "imu_mean_jerk_mps3":          round(mean_jerk, 2),

        "imu_bar_roll_range_deg":      round(bar_roll_range, 2),
        "imu_bar_pitch_range_deg":     round(bar_pitch_range, 2),
    }


# ===========================================================================
# Pose 特徴量抽出（現状は visibility と簡易統計のみ。深さ等は後で追加）
# ===========================================================================
def compute_pose_features(pose_df: Optional[pd.DataFrame]) -> dict:
    if pose_df is None or len(pose_df) == 0:
        return {
            "pose_available":           False,
            "pose_n_frames":            0,
            "pose_key_visibility_mean": None,
            "pose_key_visibility_min":  None,
        }

    vis = pose_df["key_visibility"].dropna()
    if len(vis) == 0:
        return {
            "pose_available":           False,
            "pose_n_frames":            int(len(pose_df)),
            "pose_key_visibility_mean": None,
            "pose_key_visibility_min":  None,
        }

    return {
        "pose_available":           True,
        "pose_n_frames":            int(len(pose_df)),
        "pose_n_valid_frames":      int(len(vis)),
        "pose_key_visibility_mean": round(float(vis.mean()), 4),
        "pose_key_visibility_min":  round(float(vis.min()),  4),
        "pose_nan_ratio":           round(1.0 - len(vis) / len(pose_df), 4),
    }


# ===========================================================================
# meta.json から該当セットの情報を引く
# ===========================================================================
def find_set_meta(meta: dict, set_no: int) -> dict:
    sets = meta.get("sets", [])
    for s in sets:
        if int(s.get("set_no", -1)) == int(set_no):
            return s
    return {}


# ===========================================================================
# 1試技ぶんの特徴量を計算
# ===========================================================================
def extract_one_trial(imu_csv: Path, pose_csv: Optional[Path],
                      meta_json: Path, set_no: int) -> dict:
    # メタ（レップ数をレップ検出に使うので先に読む）
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    set_meta = find_set_meta(meta, set_no)

    expected_reps = set_meta.get("reps_completed") or set_meta.get("reps_planned")
    try:
        expected_reps = int(expected_reps) if expected_reps else None
    except (TypeError, ValueError):
        expected_reps = None

    # IMU
    imu_df = pd.read_csv(imu_csv)
    imu_feat = compute_imu_features(imu_df, expected_reps=expected_reps)

    # Pose（あれば）
    pose_df = None
    if pose_csv is not None and pose_csv.exists():
        try:
            pose_df = pd.read_csv(pose_csv)
        except Exception:
            pose_df = None
    pose_feat = compute_pose_features(pose_df)

    meta_feat = {
        "subject_id":      meta.get("subject_id"),
        "session_id":      meta.get("session_id"),
        "date":            meta.get("date"),
        "set_no":          int(set_no),
        "exercise":        set_meta.get("exercise", meta.get("exercise")),
        "weight_kg":       set_meta.get("weight_kg"),
        "reps_planned":    set_meta.get("reps_planned"),
        "reps_completed":  set_meta.get("reps_completed"),
        "rpe":             set_meta.get("rpe"),
        "rest_before_sec": set_meta.get("rest_before_sec"),
        "set_notes":       set_meta.get("notes"),
    }

    return {**meta_feat, **imu_feat, **pose_feat}


# ===========================================================================
# 骨格推定結果の探索
# ===========================================================================
def find_pose_csv(pose_dir: Path, vid_name: Optional[str]):
    """
    骨格推定の結果CSVを探す。モデルによって命名が違うため候補を順に試す。

        set01_yolo_features.csv   … YOLO26版（優先）
        set01_features.csv        … MediaPipe版

    動画名が未設定でも、pose/ の中から setNN に対応するものを拾う。
    """
    if not pose_dir.is_dir():
        return None

    stems = []
    if vid_name:
        stems.append(Path(vid_name).stem)

    for stem in stems:
        for suffix in ("_yolo_features.csv", "_features.csv"):
            cand = pose_dir / f"{stem}{suffix}"
            if cand.exists():
                return cand
    return None


# ===========================================================================
# data/ 配下を走査して全試技を抽出
# ===========================================================================
def find_trials(filter_subject: Optional[str] = None):
    """戻り値: [(imu_csv, pose_csv_or_None, meta_json, set_no, out_json), ...]"""
    results = []
    if not DATA_ROOT.exists():
        return results
    for meta_json in DATA_ROOT.rglob("meta.json"):
        parts = meta_json.relative_to(DATA_ROOT).parts
        if len(parts) < 3:
            continue
        subj, sess = parts[0], parts[1]
        if filter_subject and subj != filter_subject:
            continue
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        sets = meta.get("sets", [])
        for s in sets:
            set_no   = s.get("set_no")
            csv_name = s.get("csv_filename")
            vid_name = s.get("video_filename")
            if set_no is None or csv_name is None:
                continue
            imu_csv  = DATA_ROOT / subj / sess / "imu"   / csv_name
            pose_csv = find_pose_csv(DATA_ROOT / subj / sess / "pose", vid_name)
            feat_dir = DATA_ROOT / subj / sess / "features"
            out_json = feat_dir / f"set{int(set_no):02d}_features.json"
            results.append((imu_csv, pose_csv, meta_json, int(set_no), out_json))
    return results


def is_already_processed(out_json: Path) -> bool:
    return out_json.exists()


# ===========================================================================
# メイン
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Extract features for one or all trials")
    ap.add_argument("--imu", type=str, default=None)
    ap.add_argument("--pose", type=str, default=None)
    ap.add_argument("--meta", type=str, default=None)
    ap.add_argument("--set_no", type=int, default=None)
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # 単一試技モード
    if args.imu:
        if not (args.meta and args.set_no is not None):
            sys.exit("[ERROR] --imu のときは --meta と --set_no が必要")
        imu_csv  = Path(args.imu).resolve()
        pose_csv = Path(args.pose).resolve() if args.pose else None
        meta_json = Path(args.meta).resolve()
        feat = extract_one_trial(imu_csv, pose_csv, meta_json, args.set_no)
        print(json.dumps(feat, indent=2, ensure_ascii=False))
        return

    # バッチモード
    print(f"[BATCH] scanning {DATA_ROOT}")
    trials = find_trials(filter_subject=args.subject)
    if not trials:
        print("[INFO] no trials found")
        print(f"  Expected: {DATA_ROOT}/<subject>/<session>/meta.json + imu/setNN.csv")
        return

    pending = []
    skipped = 0
    for imu_csv, pose_csv, meta_json, set_no, out_json in trials:
        if (not args.force) and is_already_processed(out_json):
            skipped += 1
            continue
        if not imu_csv.exists():
            print(f"  [SKIP] IMU not found: {imu_csv}")
            continue
        pending.append((imu_csv, pose_csv, meta_json, set_no, out_json))

    print(f"[INFO] To process: {len(pending)} trials (skipped: {skipped})")
    for imu_csv, pose_csv, meta_json, set_no, out_json in pending:
        rel = imu_csv.relative_to(DATA_ROOT)
        print(f"\n[RUN] {rel} (set_no={set_no})")
        try:
            feat = extract_one_trial(imu_csv, pose_csv, meta_json, set_no)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(feat, f, indent=2, ensure_ascii=False)
            print(f"  rpe={feat.get('rpe')}, weight={feat.get('weight_kg')}, "
                  f"reps_detected={feat.get('imu_n_reps_detected')}, "
                  f"peak_v={feat.get('imu_peak_velocity_up_mps')}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
