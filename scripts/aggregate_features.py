# -*- coding: utf-8 -*-
"""
=============================================================================
 全試技の特徴量JSONを1つのCSVに集約するスクリプト

 入力: data/<S>/<SES>/features/setNN_features.json （extract_features.py の出力）
 出力: features_all.csv （ルートに保存。Stage 3 機械学習の入力）

 使い方:
   python aggregate_features.py
   python aggregate_features.py --out my_features.csv
   python aggregate_features.py --subject S001
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT  = PROJECT_ROOT / "data"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "features_all.csv"


def find_feature_jsons(filter_subject=None):
    results = []
    if not DATA_ROOT.exists():
        return results
    for p in DATA_ROOT.rglob("features/set*_features.json"):
        parts = p.relative_to(DATA_ROOT).parts
        if len(parts) < 4:
            continue
        subj = parts[0]
        if filter_subject and subj != filter_subject:
            continue
        results.append(p)
    return sorted(results)


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-trial features into one CSV")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--subject", type=str, default=None)
    args = ap.parse_args()

    files = find_feature_jsons(filter_subject=args.subject)
    if not files:
        print(f"[INFO] no feature JSONs under {DATA_ROOT}")
        print("       run extract_features.py first")
        return

    rows = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            # "_" で始まるキーはレップごとの詳細（リスト）なので CSV に入れない
            d = {k: v for k, v in d.items() if not k.startswith("_")}
            d["_source"] = str(fp.relative_to(DATA_ROOT))
            rows.append(d)
        except Exception as e:
            print(f"  [WARN] skip {fp}: {e}")

    df = pd.DataFrame(rows)
    # 並び順を整える: メタ → IMU → Pose
    meta_cols = [
        "subject_id", "session_id", "date", "set_no", "modality", "camera_view",
        "exercise", "weight_kg", "pct_1rm", "reps_planned", "reps_completed",
        "rpe", "rest_before_sec", "set_notes",
    ]
    imu_cols  = sorted([c for c in df.columns if c.startswith("imu_")])
    pose_cols = sorted([c for c in df.columns if c.startswith("pose_")])
    other_cols = [c for c in df.columns
                  if c not in meta_cols + imu_cols + pose_cols + ["_source"]]
    final_cols = meta_cols + imu_cols + pose_cols + other_cols + ["_source"]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # subject_id, set_no でソート
    if "subject_id" in df.columns and "set_no" in df.columns:
        df = df.sort_values(
            by=["subject_id", "session_id", "set_no"],
            na_position="last"
        ).reset_index(drop=True)

    out_path = Path(args.out).resolve()
    df.to_csv(out_path, index=False)
    print(f"[OK] saved: {out_path}")
    print(f"     rows: {len(df)}, cols: {len(df.columns)}")
    print(f"     RPE distribution:")
    if "rpe" in df.columns:
        rpe_counts = df["rpe"].value_counts().sort_index()
        for rpe, n in rpe_counts.items():
            print(f"       RPE {rpe}: {n}")


if __name__ == "__main__":
    main()
