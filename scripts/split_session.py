# -*- coding: utf-8 -*-
"""
=============================================================================
 セッション分割スクリプト

 multi_imu_receiver.py が出力した「連続記録 + マーカー」を、
 解析パイプラインが期待するセット単位の構造へ変換する。

 【入力】 data/_sessions/session_YYYYMMDD_HHMMSS/
              raw_BAR.csv, raw_TRUNK.csv, ...   デバイスごとの連続データ
              markers.csv                       set_no, start_s, end_s, rpe
              session_info.json
              130.mp4  など                     動画（フォルダ直下でよい）

 【出力】 data/<subject>/<session>/
              imu/set01_BAR.csv, set01_TRUNK.csv, ...
              videos/set01.mp4                  直下の動画を自動で配置
              pose/                             （空）
              meta.json                         RPE・重量を反映

 【特徴】
  ・セッションフォルダ直下に置いた動画を自動で見つけ、videos/setNN.mp4 に配置する
  ・動画のファイル名から重量を推定する（130.mp4 → weight_kg = 130）
  ・複数のセッションを1つにまとめられる（受信を何度も起動し直した場合に使う）

 【使い方】
  # 最新セッション1つを変換
  python split_session.py --subject S001 --session SES004

  # _sessions/ にある全セッションを1つにまとめる
  python split_session.py --subject S001 --session SES004 --all

  # まとめる対象を指定（順番どおりに set01, set02, ... になる）
  python split_session.py --subject S001 --session SES004 \
      --src data/_sessions/session_20260918_004509 \
      --src data/_sessions/session_20260918_005350 \
      --src data/_sessions/session_20260918_010423

  # 一部を除外してまとめる（フォルダ名の一部で指定）
  python split_session.py --subject S001 --session SES004 --all --skip 013048

 依存ライブラリ: pip install pandas
=============================================================================
"""

import re
import sys
import json
import shutil
import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT = PROJECT_ROOT / "data"
SESSIONS_ROOT = DATA_ROOT / "_sessions"

VIDEO_EXT = (".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI")


# ---------------------------------------------------------------------------
# セッションフォルダの探索
# ---------------------------------------------------------------------------
def list_sessions() -> list:
    if not SESSIONS_ROOT.exists():
        return []
    return sorted([p for p in SESSIONS_ROOT.iterdir()
                   if p.is_dir() and p.name.startswith("session_")])


def resolve_sources(args) -> list:
    """--src / --all / 省略 のいずれかから、処理対象セッションの一覧を決める"""
    if args.src:
        srcs = [Path(s).resolve() for s in args.src]
    elif args.all:
        srcs = list_sessions()
    else:
        srcs = list_sessions()[-1:]          # 省略時は最新1つ

    if args.skip:
        srcs = [s for s in srcs
                if not any(k in s.name for k in args.skip)]

    if not srcs:
        sys.exit("[ERROR] 処理対象のセッションが見つかりません。\n"
                 f"        {SESSIONS_ROOT} を確認してください。")
    for s in srcs:
        if not s.exists():
            sys.exit(f"[ERROR] セッションが存在しません: {s}")
    return srcs


# ---------------------------------------------------------------------------
# セッションフォルダ直下の動画を集める
# ---------------------------------------------------------------------------
def find_videos_in(session_dir: Path) -> list:
    """フォルダ直下の動画をファイル名順で返す（サブフォルダは見ない）"""
    vids = [p for p in session_dir.iterdir()
            if p.is_file() and p.suffix in VIDEO_EXT
            and "_annotated" not in p.stem]
    # videos/ サブフォルダがあればそこも見る
    sub = session_dir / "videos"
    if sub.is_dir():
        vids += [p for p in sub.iterdir()
                 if p.is_file() and p.suffix in VIDEO_EXT
                 and "_annotated" not in p.stem]
    return sorted(vids, key=lambda p: p.name)


def guess_weight(stem: str):
    """ファイル名から重量を推定する。 130.mp4 → 130 ／ sq_135rpe9 → 135"""
    for m in re.finditer(r"\d+", stem):
        v = int(m.group())
        if 20 <= v <= 400:          # バーベルとして現実的な範囲
            return v
    return None


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="連続記録をセット単位に分割し、動画も自動配置する")
    ap.add_argument("--subject", required=True, help="例: S001")
    ap.add_argument("--session", required=True, help="例: SES004")
    ap.add_argument("--src", action="append", default=None,
                    help="セッションフォルダ（複数指定可）。省略時は最新1つ")
    ap.add_argument("--all", action="store_true",
                    help="_sessions/ の全セッションを1つにまとめる")
    ap.add_argument("--skip", action="append", default=None,
                    help="除外するセッション（フォルダ名の一部）")
    ap.add_argument("--margin", type=float, default=0.3,
                    help="切り出しの前後マージン[秒] (default: 0.3)")
    ap.add_argument("--exercise", type=str, default="Squat")
    ap.add_argument("--date", type=str, default=None)
    ap.add_argument("--move-video", action="store_true",
                    help="動画をコピーではなく移動する")
    args = ap.parse_args()

    srcs = resolve_sources(args)

    print(f"[対象] {len(srcs)} セッション")
    for s in srcs:
        print(f"   - {s.name}")
    print()

    out_dir = DATA_ROOT / args.subject / args.session
    imu_dir = out_dir / "imu"
    vid_dir = out_dir / "videos"
    for d in (imu_dir, vid_dir, out_dir / "pose"):
        d.mkdir(parents=True, exist_ok=True)

    sets_meta = []
    all_devices = set()
    date_str = args.date
    set_no = 0

    for src in srcs:
        mpath = src / "markers.csv"
        if not mpath.exists():
            print(f"  [SKIP] markers.csv がありません: {src.name}")
            continue
        markers = pd.read_csv(mpath)
        if len(markers) == 0:
            print(f"  [SKIP] マーカーが空です: {src.name}")
            continue

        # デバイスごとの連続データ
        raws = {}
        for f in sorted(src.glob("raw_*.csv")):
            dev = f.stem.replace("raw_", "")
            df = pd.read_csv(f)
            if "recv_time_s" not in df.columns:
                print(f"  [SKIP] {f.name}: recv_time_s 列なし")
                continue
            raws[dev] = df
            all_devices.add(dev)
        if not raws:
            print(f"  [SKIP] raw_*.csv がありません: {src.name}")
            continue

        # 日付（最初に見つかったものを採用）
        if date_str is None:
            info = src / "session_info.json"
            if info.exists():
                started = json.loads(info.read_text(encoding="utf-8")).get("started_at", "")
                date_str = started.split(" ")[0] if started else None

        # 直下の動画
        videos = find_videos_in(src)
        n_sets_here = len(markers)
        if videos and len(videos) != n_sets_here:
            print(f"  [WARN] {src.name}: 動画 {len(videos)}本 / セット {n_sets_here}件 "
                  f"— 数が合わないため先頭から順に割り当てます")

        print(f"  == {src.name}  ({len(raws)} devices, {n_sets_here} sets, "
              f"{len(videos)} videos)")

        for i, (_, m) in enumerate(markers.iterrows()):
            set_no += 1
            t0 = float(m["start_s"]) - args.margin
            t1 = float(m["end_s"]) + args.margin

            # --- IMU の切り出し ---
            n_dev = 0
            for dev, df in raws.items():
                seg = df[(df["recv_time_s"] >= t0) & (df["recv_time_s"] <= t1)].copy()
                if len(seg) == 0:
                    continue
                seg.insert(0, "timestamp_ms",
                           ((seg["recv_time_s"] - float(m["start_s"])) * 1000.0).round(1))
                seg.to_csv(imu_dir / f"set{set_no:02d}_{dev}.csv", index=False)
                n_dev += 1

            # --- 動画の配置 ---
            video_name = None
            weight = None
            if i < len(videos):
                srcv = videos[i]
                video_name = f"set{set_no:02d}{srcv.suffix.lower()}"
                dstv = vid_dir / video_name
                if args.move_video:
                    shutil.move(str(srcv), str(dstv))
                else:
                    shutil.copy2(str(srcv), str(dstv))
                weight = guess_weight(srcv.stem)

            # --- RPE ---
            rpe = m.get("rpe")
            rpe_val = float(rpe) if pd.notna(rpe) and str(rpe).strip() != "" else None

            sets_meta.append({
                "set_no": set_no,
                "csv_filename": f"set{set_no:02d}_BAR.csv",
                "video_filename": video_name,
                "exercise": args.exercise,
                "weight_kg": weight,
                "reps_planned": None,
                "reps_completed": None,
                "rpe": rpe_val,
                "rest_before_sec": None,
                "duration_s": float(m["duration_s"]),
                "source_session": src.name,
                "notes": "",
            })

            wtxt = f"{weight}kg" if weight else "重量?"
            vtxt = video_name if video_name else "動画なし"
            print(f"     set{set_no:02d}  {float(m['duration_s']):>5.1f}s  "
                  f"{n_dev}dev  {wtxt:>7}  RPE={rpe_val if rpe_val is not None else '未入力'}  "
                  f"[{vtxt}]")

    if not sets_meta:
        sys.exit("\n[ERROR] 変換できるセットがありませんでした。")

    # --- meta.json ---
    meta = {
        "session_id": args.session,
        "subject_id": args.subject,
        "date": date_str or "",
        "source_sessions": [s.name for s in srcs],
        "devices": sorted(all_devices),
        "sampling_hz": 100,
        "exercise": args.exercise,
        "notes": "",
        "sets": sets_meta,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- サマリ ---
    n_vid = sum(1 for s in sets_meta if s["video_filename"])
    n_w   = sum(1 for s in sets_meta if s["weight_kg"])
    n_rpe = sum(1 for s in sets_meta if s["rpe"] is not None)

    print()
    print("=" * 64)
    print(f"[OK] {out_dir}")
    print(f"     セット     : {len(sets_meta)}")
    print(f"     IMU CSV    : {len(list(imu_dir.glob('*.csv')))} ファイル")
    print(f"     動画       : {n_vid} / {len(sets_meta)}")
    print(f"     重量判定   : {n_w} / {len(sets_meta)}  （動画名から自動推定）")
    print(f"     RPE        : {n_rpe} / {len(sets_meta)}")
    print()
    print("  次にやること:")
    if n_w < len(sets_meta):
        print(f"   1. meta.json の weight_kg（null の箇所）を埋める")
    else:
        print(f"   1. meta.json の reps_completed を埋める（重量は自動で入っています）")
    print(f"   2. python scripts\\pose_extract_yolo.py --subject {args.subject}")
    print(f"   3. python scripts\\extract_features.py --subject {args.subject}")
    print(f"   4. python scripts\\aggregate_features.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
