# -*- coding: utf-8 -*-
"""
=============================================================================
 骨格推定モデル比較スクリプト (MediaPipe vs YOLO26)

 同じ動画に対して両モデルの結果を比較し、以下を定量・可視化する:
   1. 検出失敗率 (NaN率)
   2. key_visibility の平均・最小値
   3. 関節角度の波形の比較
   4. trunk_lean の差異
   5. 異常値率（trunk_lean > 80度などの物理的にあり得ない値）

 入力:
   data/<S>/<SES>/pose/setNN_features.csv         (MediaPipe版)
   data/<S>/<SES>/pose/setNN_yolo_features.csv    (YOLO26版)

 出力:
   data/<S>/<SES>/pose/setNN_comparison.png       (時系列比較プロット)
   pose_comparison_summary.csv                    (定量サマリ)

 使い方:
   python compare_pose_models.py                  # data/配下の全試技
   python compare_pose_models.py --subject S001
   python compare_pose_models.py --set squat_RPE6 # 動画名指定

 依存ライブラリ:
   pip install numpy pandas matplotlib
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT  = PROJECT_ROOT / "data"

# 物理的に異常な値の閾値
TRUNK_LEAN_ABNORMAL_DEG = 80.0   # 80度以上の前傾は人間にあり得ない
KNEE_ANGLE_ABNORMAL_DEG = 20.0   # 20度未満の膝角度は物理的に困難


# ===========================================================================
# ペア検出
# ===========================================================================
def find_pairs(filter_subject: Optional[str] = None,
               filter_set: Optional[str] = None) -> List[Tuple[str, Path, Path, Path]]:
    """
    戻り値: [(stem, mp_csv, yolo_csv, out_png), ...]
       両方の features.csv が揃っているものだけ返す
    """
    pairs = []
    if not DATA_ROOT.exists():
        return pairs
    for mp_csv in DATA_ROOT.rglob("*_features.csv"):
        # YOLO版は除外
        if "_yolo_" in mp_csv.name:
            continue
        if not mp_csv.parent.name == "pose":
            continue
        stem = mp_csv.name.replace("_features.csv", "")
        yolo_csv = mp_csv.parent / f"{stem}_yolo_features.csv"
        if not yolo_csv.exists():
            continue
        parts = mp_csv.relative_to(DATA_ROOT).parts
        if len(parts) >= 3:
            subj = parts[0]
            if filter_subject and subj != filter_subject:
                continue
        if filter_set and filter_set not in stem:
            continue
        out_png = mp_csv.parent / f"{stem}_comparison.png"
        pairs.append((stem, mp_csv, yolo_csv, out_png))
    return pairs


# ===========================================================================
# 定量指標
# ===========================================================================
def compute_metrics(df: pd.DataFrame, name: str) -> dict:
    n_total = len(df)
    if n_total == 0:
        return {"model": name, "n_total": 0}

    valid_mask = df["key_visibility"].notna()
    n_valid = int(valid_mask.sum())
    nan_ratio = float(1.0 - n_valid / n_total)

    # 異常値率
    trunk = df["trunk_lean_deg"].dropna()
    abnormal_trunk_ratio = float((trunk.abs() > TRUNK_LEAN_ABNORMAL_DEG).mean()) if len(trunk) > 0 else float("nan")

    knee_l = df["knee_angle_left"].dropna()
    knee_r = df["knee_angle_right"].dropna()
    abnormal_knee_l_ratio = float((knee_l < KNEE_ANGLE_ABNORMAL_DEG).mean()) if len(knee_l) > 0 else float("nan")
    abnormal_knee_r_ratio = float((knee_r < KNEE_ANGLE_ABNORMAL_DEG).mean()) if len(knee_r) > 0 else float("nan")

    vis = df["key_visibility"].dropna()

    asym = df["asymmetry_knee"].dropna()
    huge_asym_ratio = float((asym > 30.0).mean()) if len(asym) > 0 else float("nan")

    return {
        "model":               name,
        "n_total":             int(n_total),
        "n_valid":             n_valid,
        "nan_ratio":           round(nan_ratio, 4),
        "key_vis_mean":        round(float(vis.mean()), 4) if len(vis) > 0 else None,
        "key_vis_min":         round(float(vis.min()),  4) if len(vis) > 0 else None,
        "abnormal_trunk_pct":  round(abnormal_trunk_ratio * 100, 2) if not np.isnan(abnormal_trunk_ratio) else None,
        "abnormal_knee_l_pct": round(abnormal_knee_l_ratio * 100, 2) if not np.isnan(abnormal_knee_l_ratio) else None,
        "abnormal_knee_r_pct": round(abnormal_knee_r_ratio * 100, 2) if not np.isnan(abnormal_knee_r_ratio) else None,
        "huge_asymmetry_pct":  round(huge_asym_ratio * 100, 2) if not np.isnan(huge_asym_ratio) else None,
    }


# ===========================================================================
# 比較プロット
# ===========================================================================
def make_comparison_plot(df_mp: pd.DataFrame, df_yolo: pd.DataFrame,
                         out_path: Path, title: str = ""):
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    t_mp   = df_mp["time_s"].to_numpy()
    t_yolo = df_yolo["time_s"].to_numpy()

    # (1) key_visibility
    axes[0].plot(t_mp,   df_mp["key_visibility"],   lw=1.2, label="MediaPipe", color="C0", alpha=0.8)
    axes[0].plot(t_yolo, df_yolo["key_visibility"], lw=1.2, label="YOLO26",    color="C1", alpha=0.8)
    axes[0].axhline(0.5, color="red", lw=0.5, linestyle="--")
    axes[0].set_ylabel("key_visibility")
    axes[0].set_title(f"Pose model comparison - {title}")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    # (2) 左膝角度
    axes[1].plot(t_mp,   df_mp["knee_angle_left"],   lw=1.2, label="MediaPipe", color="C0", alpha=0.8)
    axes[1].plot(t_yolo, df_yolo["knee_angle_left"], lw=1.2, label="YOLO26",    color="C1", alpha=0.8)
    axes[1].axhline(KNEE_ANGLE_ABNORMAL_DEG, color="red", lw=0.5, linestyle="--",
                    label=f"abnormal threshold ({KNEE_ANGLE_ABNORMAL_DEG} deg)")
    axes[1].set_ylabel("left knee angle [deg]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    # (3) 右膝角度
    axes[2].plot(t_mp,   df_mp["knee_angle_right"],   lw=1.2, label="MediaPipe", color="C0", alpha=0.8)
    axes[2].plot(t_yolo, df_yolo["knee_angle_right"], lw=1.2, label="YOLO26",    color="C1", alpha=0.8)
    axes[2].axhline(KNEE_ANGLE_ABNORMAL_DEG, color="red", lw=0.5, linestyle="--")
    axes[2].set_ylabel("right knee angle [deg]")
    axes[2].grid(alpha=0.3)

    # (4) 体幹前傾
    axes[3].plot(t_mp,   df_mp["trunk_lean_deg"],   lw=1.2, label="MediaPipe", color="C0", alpha=0.8)
    axes[3].plot(t_yolo, df_yolo["trunk_lean_deg"], lw=1.2, label="YOLO26",    color="C1", alpha=0.8)
    axes[3].axhline( TRUNK_LEAN_ABNORMAL_DEG, color="red", lw=0.5, linestyle="--",
                     label=f"abnormal ({TRUNK_LEAN_ABNORMAL_DEG} deg)")
    axes[3].axhline(-TRUNK_LEAN_ABNORMAL_DEG, color="red", lw=0.5, linestyle="--")
    axes[3].set_ylabel("trunk lean [deg]")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="lower right")

    # (5) 腰の高さ
    axes[4].plot(t_mp,   df_mp["hip_y"],   lw=1.2, label="MediaPipe", color="C0", alpha=0.8)
    axes[4].plot(t_yolo, df_yolo["hip_y"], lw=1.2, label="YOLO26",    color="C1", alpha=0.8)
    axes[4].set_ylabel("hip_y (image coord)")
    axes[4].set_xlabel("time [s]")
    axes[4].grid(alpha=0.3)
    axes[4].invert_yaxis()
    axes[4].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ===========================================================================
# サマリテキスト出力
# ===========================================================================
def print_comparison(stem: str, m_mp: dict, m_yolo: dict):
    print(f"\n  === {stem} ===")
    print(f"  {'metric':28s}  {'MediaPipe':>12s}  {'YOLO26':>12s}  {'delta':>10s}")
    print("  " + "-" * 70)
    for key in ["nan_ratio", "key_vis_mean", "key_vis_min",
                "abnormal_trunk_pct", "abnormal_knee_l_pct",
                "abnormal_knee_r_pct", "huge_asymmetry_pct"]:
        v_mp = m_mp.get(key)
        v_yo = m_yolo.get(key)
        if v_mp is None or v_yo is None:
            continue
        delta = v_yo - v_mp
        better = ""
        if key in ("nan_ratio", "abnormal_trunk_pct",
                   "abnormal_knee_l_pct", "abnormal_knee_r_pct",
                   "huge_asymmetry_pct"):
            # 小さい方が良い
            if delta < 0:
                better = "YOLO win"
            elif delta > 0:
                better = "MP win"
        else:
            # 大きい方が良い
            if delta > 0:
                better = "YOLO win"
            elif delta < 0:
                better = "MP win"
        print(f"  {key:28s}  {v_mp:12.4f}  {v_yo:12.4f}  {delta:+10.4f}  {better}")


# ===========================================================================
# メイン
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Compare MediaPipe vs YOLO26 pose results")
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--set", type=str, default=None, help="filter by stem name substring")
    ap.add_argument("--summary-out", type=str, default=str(PROJECT_ROOT / "outputs" / "pose_comparison_summary.csv"))
    args = ap.parse_args()

    pairs = find_pairs(filter_subject=args.subject, filter_set=args.set)
    if not pairs:
        print("[INFO] no comparable pairs found")
        print("       run both pose_extract.py and pose_extract_yolo.py first")
        return

    print(f"[INFO] {len(pairs)} pairs to compare")

    summary_rows = []
    for stem, mp_csv, yolo_csv, out_png in pairs:
        df_mp   = pd.read_csv(mp_csv)
        df_yolo = pd.read_csv(yolo_csv)

        m_mp   = compute_metrics(df_mp,   "MediaPipe")
        m_yolo = compute_metrics(df_yolo, "YOLO26")
        m_mp["stem"]   = stem
        m_yolo["stem"] = stem

        print_comparison(stem, m_mp, m_yolo)
        summary_rows.append(m_mp)
        summary_rows.append(m_yolo)

        try:
            make_comparison_plot(df_mp, df_yolo, out_png, title=stem)
            print(f"  [plot] {out_png.relative_to(PROJECT_ROOT) if out_png.is_relative_to(PROJECT_ROOT) else out_png}")
        except Exception as e:
            print(f"  [WARN] plot failed: {e}")

    summary_df = pd.DataFrame(summary_rows)
    col_order = ["stem", "model", "n_total", "n_valid", "nan_ratio",
                 "key_vis_mean", "key_vis_min",
                 "abnormal_trunk_pct", "abnormal_knee_l_pct", "abnormal_knee_r_pct",
                 "huge_asymmetry_pct"]
    col_order = [c for c in col_order if c in summary_df.columns]
    summary_df = summary_df[col_order]
    out_csv = Path(args.summary_out).resolve()
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[OK] summary saved: {out_csv}")


if __name__ == "__main__":
    main()
