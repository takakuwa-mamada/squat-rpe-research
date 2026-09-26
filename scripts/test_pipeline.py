# -*- coding: utf-8 -*-
"""
=============================================================================
 解析パイプラインのテスト（合成データ）

 実データ（SES004）は1レップのセットしかないので、複数レップのロジック
 （レップ分割・速度低下率・1レップ目→最終レップの変化）と、評価スクリプトが
 最後まで回ることを合成データで確かめる。

 使い方:
   pytest scripts\\test_pipeline.py -q
=============================================================================
"""

import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import extract_features as ef          # noqa: E402
from ingest_remote import parse_label, parse_log_line   # noqa: E402

G = 9.80665
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


# ---------------------------------------------------------------------------
# 合成の挙上軌跡: 立位 → (下降 → 上昇) × n → 立位。上昇時間を段々長くする（疲労）
# ---------------------------------------------------------------------------
def squat_trajectory(fs, n_reps=5, depth=0.45, descent=1.2, ascent0=0.9, ascent_growth=0.15,
                     pause=0.2, rest=0.6, lead=2.0, tail=1.5):
    """戻り値: t, y（立位からの下がり量 [m]、下が正）, 各レップの上昇時間"""
    segs, ascents = [np.zeros(int(lead * fs))], []
    for i in range(n_reps):
        asc = ascent0 * (1 + ascent_growth * i)
        ascents.append(asc)
        td = np.linspace(0, np.pi, int(descent * fs), endpoint=False)
        ta = np.linspace(0, np.pi, int(asc * fs), endpoint=False)
        segs += [depth * (1 - np.cos(td)) / 2, np.full(int(pause * fs), depth),
                 depth * (1 + np.cos(ta)) / 2, np.zeros(int(rest * fs))]
    segs.append(np.zeros(int(tail * fs)))
    y = np.concatenate(segs)
    return np.arange(len(y)) / fs, y, ascents


def make_keypoints(fs=30.0, w=720, h=1280, n_reps=5, knee_cave=True, **kw):
    """正面から撮った人物の keypoints（0〜1 正規化）を作る。肩〜足首 = 0.45 h"""
    t, y, asc = squat_trajectory(fs, n_reps=n_reps, **kw)
    body = 0.45 * h
    rng = np.random.default_rng(0)
    n = len(t)
    cx = 0.5 * w
    rows = {"frame": np.arange(n), "time_s": t}
    drop = y / 0.45 * body * 0.9          # 肩の沈み込み [px]（depth 0.45 m ≒ 0.9 BL 相当に拡大）
    sh_y = 0.25 * h + drop
    hip_y = 0.50 * h + drop * 0.8
    knee_y = np.full(n, 0.60 * h)
    ank_y = np.full(n, 0.70 * h)
    # 膝の開き: レップが進むほどボトム付近で内側に入る
    rep_frac = np.clip(y / max(y.max(), 1e-9), 0, 1)
    progress = np.clip(t / t[-1], 0, 1)
    knee_half = 0.09 * w * (1 - (0.25 * progress * rep_frac if knee_cave else 0))
    pos = {
        "left_shoulder": (cx + 0.12 * w, sh_y), "right_shoulder": (cx - 0.12 * w, sh_y),
        "left_hip": (cx + 0.08 * w, hip_y), "right_hip": (cx - 0.08 * w, hip_y),
        "left_knee": (cx + knee_half, knee_y), "right_knee": (cx - knee_half, knee_y),
        "left_ankle": (cx + 0.09 * w, ank_y), "right_ankle": (cx - 0.09 * w, ank_y),
    }
    for name in COCO:
        x, yy = pos.get(name, (np.full(n, cx), sh_y - 0.08 * h))
        rows[f"{name}_x"] = (x + rng.normal(0, 1.0, n)) / w
        rows[f"{name}_y"] = (yy + rng.normal(0, 1.0, n)) / h
        rows[f"{name}_conf"] = np.full(n, 0.95)
    return pd.DataFrame(rows), asc


def make_imu(fs=100.0, n_reps=5, g_bias=1.05, **kw):
    """バーの鉛直位置から加速度を作り、M5 らしいオフセット・ノイズを載せる"""
    t, y, asc = squat_trajectory(fs, n_reps=n_reps, **kw)
    pos_up = -y
    v_true = np.gradient(pos_up, 1 / fs)
    a_true = np.gradient(v_true, 1 / fs)
    rng = np.random.default_rng(1)
    az = g_bias + a_true / G + rng.normal(0, 0.01, len(t))
    df = pd.DataFrame({
        "timestamp_ms": t * 1000.0,
        "ax_g": rng.normal(0, 0.01, len(t)), "ay_g": rng.normal(0, 0.01, len(t)), "az_g": az,
        "gx_dps": -3.8 + rng.normal(0, 1, len(t)), "gy_dps": -12.9 + rng.normal(0, 1, len(t)),
        "gz_dps": 16.0 + rng.normal(0, 1, len(t)),
    })
    return df, v_true, asc


# ---------------------------------------------------------------------------
# 骨格
# ---------------------------------------------------------------------------
def test_pose_detects_all_reps_and_fatigue():
    kp, asc = make_keypoints(n_reps=5)
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front")
    assert f["pose_available"]
    assert f["pose_n_reps_detected"] == 5
    assert f["pose_velocity_loss_pct"] > 10            # 上昇が遅くなる → 速度低下
    assert f["pose_ascent_time_ratio"] > 1.3            # 最終レップの上昇時間 / 1レップ目
    assert f["pose_knee_ankle_ratio_change"] < 0        # 膝が内側に入っていく
    assert 0.2 < f["pose_shoulder_width_bl"] < 1.0      # 正面


def test_pose_rep_cap_by_reported_reps():
    kp, _ = make_keypoints(n_reps=5)
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front", expected_reps=3)
    assert f["pose_n_reps_detected"] == 3


def test_pose_side_view_leaves_frontal_features_empty():
    kp, _ = make_keypoints(n_reps=3)
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="side")
    assert f["pose_n_reps_detected"] == 3
    assert f["pose_min_knee_ankle_ratio"] is None


def test_pose_missing_video_size():
    kp, _ = make_keypoints(n_reps=2)
    assert ef.compute_pose_features(kp, None, None, 30.0)["pose_available"] is False


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------
def test_imu_detects_reps_and_mcv_close_to_truth():
    imu, v_true, asc = make_imu(n_reps=5)
    f = ef.compute_imu_features(imu)
    assert f["imu_n_reps_detected"] == 5
    assert f["imu_velocity_loss_mcv_pct"] > 10
    # 1レップ目の MCV（真値: 上昇区間の平均速度）と 15% 以内で一致
    true_mcv = 0.45 / asc[0]
    assert abs(f["imu_first_rep_mcv_mps"] - true_mcv) / true_mcv < 0.15
    # 上昇時間も真値の ±20% 以内（0.3 Hz ハイパス時代は大きく短く出ていた）
    assert abs(f["imu_first_rep_concentric_s"] - asc[0]) / asc[0] < 0.2
    assert abs(f["imu_gravity_ref_g"] - 1.05) < 0.01


def test_imu_single_rep_has_no_velocity_loss():
    imu, _, _ = make_imu(n_reps=1)
    f = ef.compute_imu_features(imu, expected_reps=1)
    assert f["imu_n_reps_detected"] == 1
    assert f["imu_velocity_loss_mcv_pct"] is None


# ---------------------------------------------------------------------------
# 協力者のラベル
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("100kg_5rep_RPE8", {"weight_kg": 100, "reps": 5, "rpe": 8}),
    ("102.5kg 3回 RPE9.5", {"weight_kg": 102.5, "reps": 3, "rpe": 9.5}),
    ("100kgx5@8", {"weight_kg": 100, "reps": 5, "rpe": 8}),
    ("IMG_1234", {}),
])
def test_parse_label(name, expected):
    assert parse_label(name) == expected


def test_parse_log_line():
    assert parse_log_line("102.5 3 9.5") == {"weight_kg": 102.5, "reps": 3, "rpe": 9.5}
    assert parse_log_line("# メモ") is None


# ---------------------------------------------------------------------------
# 評価スクリプトが最後まで回る（本人 IMU+動画 12 セッション ＋ 協力者 6 人 × 4 セッション）
# ---------------------------------------------------------------------------
def synthetic_features(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    subjects = {"S001": (12, True)} | {f"S{i:03d}": (4, False) for i in range(2, 8)}
    for sid, (n_sess, has_imu) in subjects.items():
        bias = rng.normal(0, 0.5)                  # 申告の癖（個人差）
        for s in range(1, n_sess + 1):
            for set_no in range(1, 8):
                pct = rng.uniform(65, 92)
                reps = int(rng.integers(1, 8))
                rpe = np.clip(np.round((4 + 0.06 * pct + 0.25 * reps + bias
                                        + rng.normal(0, 0.4)) * 2) / 2, 5, 10)
                mcv = 1.6 - 0.013 * pct - 0.06 * (rpe - 7) + rng.normal(0, 0.03)
                r = {"subject_id": sid, "session_id": f"SES{s:03d}", "date": f"2026-10-{s:02d}",
                     "set_no": set_no, "modality": "imu+video" if has_imu else "video",
                     "weight_kg": pct * 1.5, "pct_1rm": pct, "reps_completed": reps, "rpe": rpe,
                     "pose_first_rep_mcv_bl": mcv / 1.3 + rng.normal(0, 0.03),
                     "pose_velocity_loss_pct": (rpe - 5) * 5 + rng.normal(0, 3) if reps > 1 else np.nan,
                     "pose_mean_ascent_s": 0.5 / max(mcv, 0.1) + rng.normal(0, 0.05),
                     "pose_knee_ankle_ratio_change": -(rpe - 5) * 0.01 + rng.normal(0, 0.01) if reps > 1 else np.nan}
                for c in ef.__dict__.get("POSE_FRONTAL_KEYS", []):
                    r.setdefault(c, rng.normal(0, 1))
                if has_imu:
                    r.update({"imu_first_rep_mcv_mps": mcv, "imu_mean_rep_mcv_mps": mcv - 0.02,
                              "imu_velocity_loss_mcv_pct": (rpe - 5) * 6 if reps > 1 else np.nan,
                              "imu_first_rep_concentric_s": 0.5 / max(mcv, 0.1)})
                rows.append(r)
    return pd.DataFrame(rows)


def test_train_runs_end_to_end(tmp_path):
    csv = tmp_path / "features_all.csv"
    synthetic_features().to_csv(csv, index=False)
    r = subprocess.run([sys.executable, str(SCRIPT_DIR / "train_rpe_model.py"), "--input", str(csv),
                        "--output-dir", str(tmp_path / "models"), "--kmax", "3"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = next((tmp_path / "models").iterdir())
    res = pd.read_csv(out / "results.csv")
    assert {"A_within", "B_loso"} <= set(res["scheme"])
    assert (res["scheme"] == "B_loso").sum() > 0 and "imu_all" not in \
        set(res[res["scheme"] == "B_loso"]["condition"])          # IMU は本人だけ → 汎用評価なし
    curves = pd.read_csv(out / "curves.csv")
    assert {"C_general_plus_k", "D_personal_curve"} <= set(curves["scheme"])
    assert (out / "report.md").exists() and (out / "best.pkl").exists()
    # 同じセッションが学習とテストにまたがっていない（A: フォールド = 被験者/セッション）
    pred = pd.read_csv(out / "predictions.csv")
    a = pred[pred["scheme"] == "A_within"]
    assert (a["fold"] == a["subject_id"] + "/" + a["session_id"]).all()
