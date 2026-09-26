# -*- coding: utf-8 -*-
"""
=============================================================================
 解析パイプラインを1本で回し、データの品質を報告するスクリプト

   骨格推定（未処理の動画だけ）→ 特徴量（未処理の試技だけ）→ 集計 → 品質レポート

 計測・取り込みのあとはこれだけ実行すればよい:
   本人（IMU＋動画）: python scripts\\split_session.py ... → python scripts\\run_pipeline.py
   協力者（動画のみ）: python scripts\\ingest_remote.py    → python scripts\\run_pipeline.py

 使い方:
   python scripts\\run_pipeline.py                    # 全被験者
   python scripts\\run_pipeline.py --subject S002
   python scripts\\run_pipeline.py --force-features   # 特徴量を全部作り直す（コードを直したとき）
   python scripts\\run_pipeline.py --report-only      # 品質レポートだけ
   python scripts\\run_pipeline.py --no-video         # 骨格を描いた確認用動画を作らない（速い）

 出力:
   outputs/features_all.csv      1行 = 1セット（学習の入力）
   outputs/quality_report.csv    1行 = 1セット、要確認の理由つき
=============================================================================
"""

import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
FEATURES_CSV = PROJECT_ROOT / "outputs" / "features_all.csv"
QUALITY_CSV = PROJECT_ROOT / "outputs" / "quality_report.csv"

# 学習を回し始める目安（被験者1人あたりの試技数）
TARGET_TRIALS_PER_SUBJECT = 30


def run(script: str, *args) -> bool:
    cmd = [sys.executable, str(SCRIPT_DIR / script), *args]
    print(f"\n{'=' * 64}\n[RUN] {script} {' '.join(args)}\n{'=' * 64}", flush=True)
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode == 0


# ---------------------------------------------------------------------------
# 品質チェック
# ---------------------------------------------------------------------------
def _num(row, key):
    v = row.get(key)
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def check_row(row) -> list:
    """1セットぶんの要確認事項を返す"""
    issues = []
    if _num(row, "rpe") is None:
        issues.append("RPE 未入力")
    if _num(row, "weight_kg") is None:
        issues.append("重量 未入力")
    reps = _num(row, "reps_completed")
    if reps is None:
        issues.append("回数 未入力")

    # 骨格
    if not bool(row.get("pose_available")):
        issues.append(f"骨格なし（{row.get('pose_error') or '動画/骨格推定を確認'}）")
    else:
        nan_ratio = _num(row, "pose_nan_ratio")
        vis = _num(row, "pose_key_visibility_mean")
        ankle = _num(row, "pose_ankle_y_max_norm")
        shw = _num(row, "pose_shoulder_width_bl")
        pr = _num(row, "pose_n_reps_detected")
        if nan_ratio is not None and nan_ratio > 0.2:
            issues.append(f"人物を検出できないフレームが {nan_ratio:.0%}")
        if vis is not None and vis < 0.6:
            issues.append(f"キーポイントの信頼度が低い（{vis:.2f}）")
        if ankle is not None and ankle > 0.97:
            issues.append("足元が画面の下端で切れている")
        if shw is not None and shw < 0.12:
            issues.append(f"正面から撮れていない可能性（肩幅/体高 {shw:.2f}）")
        if reps is not None and pr is not None and pr != reps:
            issues.append(f"骨格のレップ検出 {int(pr)} ≠ 申告 {int(reps)}")

    # IMU（ある場合）
    if row.get("modality") == "imu+video" or _num(row, "imu_n_samples") is not None:
        az = _num(row, "imu_az_median_g")
        fs = _num(row, "imu_fs_hz")
        ir = _num(row, "imu_n_reps_detected")
        if az is not None and az < 0.8:
            issues.append(f"M5 の向きが違う可能性（az 中央値 {az:.2f} g。画面を真上に）")
        if fs is not None and fs < 80:
            issues.append(f"IMU の受信レートが低い（{fs:.0f} Hz）")
        if reps is not None and ir is not None and ir != reps:
            issues.append(f"IMU のレップ検出 {int(ir)} ≠ 申告 {int(reps)}")
    return issues


def quality_report(subject=None):
    if not FEATURES_CSV.exists():
        print("[INFO] features_all.csv がありません")
        return
    df = pd.read_csv(FEATURES_CSV)
    if subject:
        df = df[df["subject_id"] == subject]
    if len(df) == 0:
        print("[INFO] 対象の試技がありません")
        return

    df["issues"] = [" / ".join(check_row(r)) for r in df.to_dict("records")]
    cols = [c for c in ["subject_id", "session_id", "date", "set_no", "modality", "weight_kg",
                        "reps_completed", "rpe", "imu_n_reps_detected", "pose_n_reps_detected",
                        "imu_first_rep_mcv_mps", "pose_first_rep_mcv_bl", "issues"] if c in df.columns]
    df[cols].to_csv(QUALITY_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 64}\n 品質レポート（詳細: {QUALITY_CSV.relative_to(PROJECT_ROOT)}）\n{'=' * 64}")
    for (sid, ses), g in df.groupby(["subject_id", "session_id"], sort=True):
        n_bad = int((g["issues"] != "").sum())
        mark = "OK " if n_bad == 0 else "要確認"
        print(f"\n[{mark}] {sid}/{ses}  {g['date'].iloc[0]}  {len(g)}セット  "
              f"RPE {sorted(g['rpe'].dropna().unique().tolist())}")
        for _, r in g[g["issues"] != ""].iterrows():
            print(f"      set{int(r['set_no']):02d}: {r['issues']}")

    # データの集まり具合
    print(f"\n{'=' * 64}\n データの集まり具合（目安: 1人 {TARGET_TRIALS_PER_SUBJECT} 試技、RPE 6〜10 が満遍なく）\n{'=' * 64}")
    ok = df[df["issues"] == ""]
    for sid, g in df.groupby("subject_id"):
        good = ok[ok["subject_id"] == sid]
        dist = good["rpe"].value_counts().sort_index()
        dist_txt = "  ".join(f"{k:g}:{v}" for k, v in dist.items())
        print(f"  {sid}: セッション {g['session_id'].nunique()}  試技 {len(g)}（問題なし {len(good)}"
              f" / 目安まで あと {max(0, TARGET_TRIALS_PER_SUBJECT - len(good))}）  RPE分布 {dist_txt}")
    print(f"  合計: 被験者 {df['subject_id'].nunique()} 人、試技 {len(df)}（問題なし {len(ok)}）")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="骨格推定 → 特徴量 → 集計 → 品質レポート")
    ap.add_argument("--subject", type=str, default=None, help="例: S002（省略時は全員）")
    ap.add_argument("--force-features", action="store_true", help="特徴量を全部作り直す")
    ap.add_argument("--report-only", action="store_true", help="品質レポートだけ出す")
    ap.add_argument("--no-video", action="store_true", help="骨格を描いた確認用動画を作らない")
    args = ap.parse_args()

    if not args.report_only:
        subj = ["--subject", args.subject] if args.subject else []
        pose_args = subj + (["--no-video"] if args.no_video else [])
        if not run("pose_extract_yolo.py", *pose_args):
            print("[WARN] 骨格推定でエラーがありました（続行します）")
        if not run("extract_features.py", *subj, *(["--force"] if args.force_features else [])):
            print("[WARN] 特徴量抽出でエラーがありました（続行します）")
        if not run("aggregate_features.py"):
            sys.exit("[ERROR] 集計に失敗しました")

    quality_report(args.subject)


if __name__ == "__main__":
    main()
