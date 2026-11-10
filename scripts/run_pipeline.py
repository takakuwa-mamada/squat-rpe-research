# -*- coding: utf-8 -*-
"""
=============================================================================
 解析パイプラインを1本で回し、データの品質を報告するスクリプト

   本人の計測の取り込み（未取り込みの受信フォルダを日ごとに自動で SESnnn に）
   → 骨格推定（未処理の動画だけ）→ 特徴量（未処理の試技だけ）→ 集計 → 品質レポート

 計測のあとはこれだけ実行すればよい:
   本人（IMU＋動画）: 動画を data\\_sessions\\session_…\\ の直下にコピー → python scripts\\run_pipeline.py
   協力者（動画のみ）: python scripts\\ingest_remote.py → python scripts\\run_pipeline.py

 使い方:
   python scripts\\run_pipeline.py                    # 全被験者
   python scripts\\run_pipeline.py --subject S002
   python scripts\\run_pipeline.py --force-features   # 特徴量を全部作り直す（コードを直したとき）
   python scripts\\run_pipeline.py --report-only      # 品質レポートだけ
   python scripts\\run_pipeline.py --no-video         # 骨格を描いた確認用動画を作らない（速い）
   python scripts\\run_pipeline.py --no-split         # 本人の計測の自動取り込みをしない

 出力:
   outputs/features_all.csv      1行 = 1セット（学習の入力）
   outputs/quality_report.csv    1行 = 1セット、要確認の理由つき。
                                 excluded_from = 学習から外すデータ（IMU / 動画）。
                                 基準は quality_checks.py（train_rpe_model.py も同じ基準で外す）
=============================================================================
"""

import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd

from quality_checks import check_set, has_imu


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
# 品質チェック（基準は quality_checks.py。学習から外すかどうかも同じ基準で決まる）
# ---------------------------------------------------------------------------
def check_row(row) -> list:
    """1セットぶんの要確認事項（文字列のリスト）"""
    return [i.text for i in check_set(row)]


def excluded_from(row) -> str:
    """学習から外すデータ: "IMU" / "動画" / "IMU・動画" / ""（どちらも使える）"""
    parts = {i.part for i in check_set(row) if i.exclude}
    return "・".join(name for part, name in (("imu", "IMU"), ("video", "動画")) if part in parts)


def collection_summary(df) -> list:
    """被験者ごとのデータの集まり具合を、学習に使えるセットで数える（df には excluded_from の列が要る）。
    要確認が付いただけのセットは学習に使うので数に入れる（2026-11-10 修正。それまでは要確認が 1 つも
    付いていないセットだけを数えていて、学習に使える数より少なく出ていた）。RPE が未入力のセットは数えない。
    戻り値: [{subject_id, sessions, trials, video, imu（IMU の無い人は None）, rpe_dist（動画が使えるセットの RPE）}, ...]"""
    out = []
    for sid, g in df.groupby("subject_id"):
        labeled = g["rpe"].notna()
        ex = g["excluded_from"].fillna("").astype(str)
        with_imu = pd.Series([has_imu(r) for r in g.to_dict("records")], index=g.index)
        video_ok = labeled & ~ex.str.contains("動画")
        imu_ok = labeled & with_imu & ~ex.str.contains("IMU")
        dist = g.loc[video_ok, "rpe"].value_counts().sort_index()
        out.append({"subject_id": sid, "sessions": int(g["session_id"].nunique()), "trials": int(len(g)),
                    "video": int(video_ok.sum()), "imu": int(imu_ok.sum()) if with_imu.any() else None,
                    "rpe_dist": {float(k): int(v) for k, v in dist.items()}})
    return out


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

    recs = df.to_dict("records")
    df["issues"] = [" / ".join(check_row(r)) for r in recs]
    df["excluded_from"] = [excluded_from(r) for r in recs]
    cols = [c for c in ["subject_id", "session_id", "date", "set_no", "modality", "weight_kg",
                        "reps_completed", "rpe", "imu_n_reps_detected", "pose_n_reps_detected",
                        "imu_first_rep_mcv_mps", "pose_first_rep_mcv_bl", "imu_rep_source", "sync_offset_s",
                        "sync_r", "sync_margin", "issues", "excluded_from"]
            if c in df.columns]
    df[cols].to_csv(QUALITY_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 64}\n 品質レポート（詳細: {QUALITY_CSV.relative_to(PROJECT_ROOT)}）\n{'=' * 64}")
    for (sid, ses), g in df.groupby(["subject_id", "session_id"], sort=True):
        n_bad = int((g["issues"] != "").sum())
        mark = "OK " if n_bad == 0 else "要確認"
        print(f"\n[{mark}] {sid}/{ses}  {g['date'].iloc[0]}  {len(g)}セット  "
              f"RPE {sorted(g['rpe'].dropna().unique().tolist())}")
        for _, r in g[g["issues"] != ""].iterrows():
            ex = f"　→ 学習から除外: {r['excluded_from']}" if r["excluded_from"] else ""
            print(f"      set{int(r['set_no']):02d}: {r['issues']}{ex}")

    # データの集まり具合（学習に使えるセットで数える。要確認が付いただけのセットは学習に使う）
    print(f"\n{'=' * 64}\n データの集まり具合（目安: 1人 {TARGET_TRIALS_PER_SUBJECT} 試技、RPE 6〜10 が満遍なく。"
          f"学習に使えるセットで数える）\n{'=' * 64}")
    for s in collection_summary(df):
        usable = f"動画 {s['video']}" + (f"・IMU {s['imu']}" if s["imu"] is not None else "")
        dist_txt = "  ".join(f"{k:g}:{v}" for k, v in s["rpe_dist"].items())
        print(f"  {s['subject_id']}: セッション {s['sessions']}  試技 {s['trials']}（学習に使える: {usable}"
              f" / 目安まで あと {max(0, TARGET_TRIALS_PER_SUBJECT - s['video'])}）  RPE分布 {dist_txt}")
    n_flag = int((df["issues"] != "").sum())
    print(f"  合計: 被験者 {df['subject_id'].nunique()} 人、試技 {len(df)}"
          f"（要確認の付いたセット {n_flag}。「学習から除外」と出ていなければ学習に使う）")
    n_imu = int(df["excluded_from"].str.contains("IMU").sum())
    n_vid = int(df["excluded_from"].str.contains("動画").sum())
    if n_imu or n_vid:
        print(f"  学習から除外: IMU {n_imu} セット、動画 {n_vid} セット"
              "（そのデータを使う条件だけ。train_rpe_model.py も同じ基準で外す）")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="骨格推定 → 特徴量 → 集計 → 品質レポート")
    ap.add_argument("--subject", type=str, default=None, help="例: S002（省略時は全員）")
    ap.add_argument("--force-features", action="store_true", help="特徴量を全部作り直す")
    ap.add_argument("--report-only", action="store_true", help="品質レポートだけ出す")
    ap.add_argument("--no-video", action="store_true", help="骨格を描いた確認用動画を作らない")
    ap.add_argument("--no-split", action="store_true",
                    help="本人の計測（data/_sessions）の自動取り込みをしない")
    args = ap.parse_args()

    if not args.report_only:
        # 0. 本人の計測の取り込み（新しい受信フォルダがあれば、日ごとに次の SES 番号で取り込む）
        if not args.no_split and not run("split_session.py"):
            print("[WARN] 計測の取り込みでエラーがありました（続行します）")
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
