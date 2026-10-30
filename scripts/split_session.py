# -*- coding: utf-8 -*-
"""
=============================================================================
 セッション分割スクリプト

 multi_imu_receiver.py が出力した「連続記録 + マーカー」を、
 解析パイプラインが期待するセット単位の構造へ変換する。

 【入力】 data/_sessions/session_YYYYMMDD_HHMMSS[_90kg_3rep]/
              raw_BAR.csv, raw_TRUNK.csv, ...   デバイスごとの連続データ
              markers.csv                       set_no, start_s, end_s, rpe, weight_kg, reps
              session_info.json
              130.mp4  など                     動画（フォルダ直下でよい）

 【出力】 data/<subject>/<session>/
              imu/set01_BAR.csv, set01_TRUNK.csv, ...
              videos/set01.mp4                  直下の動画を自動で配置
              pose/                             （空）
              meta.json                         RPE・重量を反映

 【自動モード（--session を付けない。run_pipeline.py が最初に呼ぶ）】
  ・まだどの meta.json にも入っていない受信フォルダを探す
  ・開始時刻が 3 時間以内に続くフォルダを「同じジムの日」として 1 セッションにまとめる
    （受信スクリプトを何回か起動し直した日も 1 セッションになる。--gap-hours で変更）
  ・セッション番号は自動（その被験者の最後の SESnnn の次）。被験者は既定 S001（本人）
  ・動画がまだ入っていないフォルダを含む日は取り込まない（動画を入れてから再実行）
    動画なしで取り込むなら --allow-no-video
  ・セッション（日）を分けるのは評価のため: 同じ日のセットが学習とテストにまたがると
    体調・疲れがそろっているぶん精度が良く見えてしまう。学習曲線も日の単位で数える

 【その他の特徴】
  ・セッションフォルダ直下に置いた動画を自動で見つけ、videos/setNN.mp4 に配置する
  ・重量・回数は 受信時に入れた値（markers.csv の weight_kg, reps 列）→ 動画のファイル名
    （130.mp4 / 130kg_3rep.mp4 → 130 kg, 3回）の順に使う。--weights / --reps は全部に優先
  ・再実行しても、meta.json に手で入れた値（重量・回数・RPE・メモ）は残す

 【使い方】
  # いつもはこれ（run_pipeline.py の中で自動で呼ばれるので、普段は打たなくてよい）
  python scripts\\split_session.py
  python scripts\\split_session.py --dry-run          # 何をするかだけ表示

  # 手動で指定する（番号を決めて作り直す・別の被験者に入れる など）
  python scripts\\split_session.py --subject S001 --session SES004 \\
      --src data/_sessions/session_20261024_221707_70kg_5rep \\
      --src data/_sessions/session_20261024_222123_80kg_5rep
  python scripts\\split_session.py --subject S001 --session SES004 --weights 100,110 --reps 5,3

 依存ライブラリ: pip install pandas
=============================================================================
"""

import re
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from ingest_remote import parse_label


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT = PROJECT_ROOT / "data"
SESSIONS_ROOT = DATA_ROOT / "_sessions"

VIDEO_EXT = (".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI")
GAP_HOURS = 3.0          # これより間が空いたら別の日（別セッション）とみなす


# ---------------------------------------------------------------------------
# セッションフォルダの探索
# ---------------------------------------------------------------------------
def list_sessions(root: Path = None) -> list:
    root = root or SESSIONS_ROOT
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir()
                   if p.is_dir() and p.name.startswith("session_")])


def resolve_sources(args) -> list:
    """--src / --all / 省略 のいずれかから、処理対象セッションの一覧を決める（手動モード）"""
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


def session_start(folder: Path):
    """受信を始めた時刻（フォルダ名 session_YYYYMMDD_HHMMSS → 無ければ session_info.json）"""
    m = re.match(r"session_(\d{8})_(\d{6})", folder.name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    info = folder / "session_info.json"
    if info.exists():
        try:
            started = json.loads(info.read_text(encoding="utf-8")).get("started_at", "")
            return datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
        except (ValueError, json.JSONDecodeError):
            pass
    return datetime.fromtimestamp(folder.stat().st_mtime)


def imported_folders(data_root: Path = None) -> set:
    """どれかの meta.json の source_sessions に入っている受信フォルダ名"""
    data_root = data_root or DATA_ROOT
    done = set()
    for mj in data_root.glob("*/*/meta.json"):
        if mj.parts[-3].startswith("_"):
            continue
        try:
            done.update(json.loads(mj.read_text(encoding="utf-8")).get("source_sessions", []))
        except Exception:
            continue
    return done


def group_visits(folders_with_time: list, gap_hours: float = GAP_HOURS) -> list:
    """
    [(フォルダ, 開始時刻), ...] を、間が gap_hours 以内で続くものごとにまとめる（時刻順）。
    戻り値: [[フォルダ, ...], ...]
    """
    groups, last = [], None
    for folder, t in sorted(folders_with_time, key=lambda x: x[1]):
        if last is None or (t - last).total_seconds() > gap_hours * 3600:
            groups.append([])
        groups[-1].append(folder)
        last = t
    return groups


def next_session_number(subject: str, data_root: Path = None) -> int:
    subj_dir = (data_root or DATA_ROOT) / subject
    nums = [int(m.group(1)) for p in subj_dir.glob("SES*")
            if p.is_dir() and (m := re.fullmatch(r"SES(\d{3})", p.name))] if subj_dir.is_dir() else []
    return max(nums, default=0) + 1


def has_markers(folder: Path) -> bool:
    mpath = folder / "markers.csv"
    if not mpath.exists():
        return False
    try:
        return len(pd.read_csv(mpath)) > 0
    except Exception:
        return False


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


def guess_label(stem: str) -> dict:
    """
    動画のファイル名から重量・回数を読む。
      130.mp4 → 130 kg ／ 130kg_3rep.mp4 → 130 kg, 3回 ／ IMG_0130.MOV → 読まない
    """
    lab = parse_label(stem)
    if "weight_kg" not in lab and re.fullmatch(r"\d+(?:\.\d+)?", stem):
        v = float(stem)
        if 20 <= v <= 400:          # バーベルとして現実的な範囲
            lab["weight_kg"] = v
    return lab


def parse_list(text, cast, n_sets, name):
    """--weights 100,110,120 のようなリストを読む（個数はセット数と一致させる）"""
    if text is None:
        return None
    vals = [cast(x) for x in text.replace("、", ",").split(",") if x.strip()]
    if len(vals) != n_sets:
        sys.exit(f"[ERROR] --{name} の個数 {len(vals)} がセット数 {n_sets} と合いません")
    return vals


def merge_manual_edits(new_sets: list, old_meta_path: Path) -> list:
    """既存の meta.json に手で入れた値を引き継ぐ（新しい値が空のとき、または RPE を直したとき）"""
    if not old_meta_path.exists():
        return new_sets
    try:
        old = json.loads(old_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return new_sets
    old_by_no = {int(s.get("set_no", -1)): s for s in old.get("sets", [])}
    for s in new_sets:
        o = old_by_no.get(s["set_no"])
        if not o:
            continue
        for key in ("weight_kg", "reps_planned", "reps_completed", "rest_before_sec", "notes"):
            if s.get(key) in (None, "") and o.get(key) not in (None, ""):
                s[key] = o[key]
        # RPE は meta.json を手で直した値（8.5 など）を優先する
        if o.get("rpe") is not None and o.get("rpe") != s.get("rpe"):
            print(f"     set{s['set_no']:02d}: RPE は meta.json の値 {o['rpe']} を残します"
                  f"（markers.csv は {s.get('rpe')}）")
            s["rpe"] = o["rpe"]
    return new_sets


# ---------------------------------------------------------------------------
# 変換（受信フォルダ群 → data/<subject>/<session>/）
# ---------------------------------------------------------------------------
def convert(args, subject: str, session: str, srcs: list):
    """受信フォルダ（1つ以上）を1つのセッションに変換する。戻り値: 集計の dict（失敗時 None）"""
    print(f"[{subject}/{session}] {len(srcs)} フォルダ")
    for s in srcs:
        print(f"   - {s.name}")

    out_dir = DATA_ROOT / subject / session
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
            print(f"  [SKIP] markers.csv がありません（受信を q で終えていない？）: {src.name}")
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
            date_str = session_start(src).strftime("%Y-%m-%d")

        # 直下の動画
        videos = find_videos_in(src)
        n_sets_here = len(markers)
        if videos and len(videos) != n_sets_here:
            print(f"  [WARN] {src.name}: 動画 {len(videos)}本 / セット {n_sets_here}件 "
                  f"— 数が合わないため先頭から順に割り当てます")

        print(f"  == {src.name}  ({len(raws)} devices, {n_sets_here} sets, "
              f"{len(videos)} videos)")

        prev_end = None
        for i, (_, m) in enumerate(markers.iterrows()):
            set_no += 1
            rest = round(float(m["start_s"]) - prev_end, 1) if prev_end is not None else None
            prev_end = float(m["end_s"])
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
            reps = None
            if i < len(videos):
                srcv = videos[i]
                video_name = f"set{set_no:02d}{srcv.suffix.lower()}"
                dstv = vid_dir / video_name
                if args.move_video:
                    shutil.move(str(srcv), str(dstv))
                else:
                    shutil.copy2(str(srcv), str(dstv))
                lab = guess_label(srcv.stem)
                weight = lab.get("weight_kg")
                reps = lab.get("reps")

            # --- 受信時に入れた重量・回数（markers.csv の列。あればファイル名より優先）---
            mw, mr = m.get("weight_kg"), m.get("reps")
            if mw is not None and pd.notna(mw) and str(mw).strip() != "":
                weight = float(mw)
            if mr is not None and pd.notna(mr) and str(mr).strip() != "":
                reps = int(float(mr))

            # --- RPE ---
            rpe = m.get("rpe")
            rpe_val = float(rpe) if pd.notna(rpe) and str(rpe).strip() != "" else None

            sets_meta.append({
                "set_no": set_no,
                "csv_filename": f"set{set_no:02d}_{'BAR' if 'BAR' in raws else sorted(raws)[0]}.csv",
                "video_filename": video_name,
                "exercise": args.exercise,
                "weight_kg": weight,
                "reps_planned": None,
                "reps_completed": reps,
                "rpe": rpe_val,
                "rest_before_sec": rest,
                "duration_s": float(m["duration_s"]),
                "source_session": src.name,
                "notes": "",
            })

            wtxt = f"{weight:g}kg" if weight else "重量?"
            rtxt = f"{reps}回" if reps else "回数?"
            vtxt = video_name if video_name else "動画なし"
            print(f"     set{set_no:02d}  {float(m['duration_s']):>5.1f}s  "
                  f"{n_dev}dev  {wtxt:>7} {rtxt:>4}  RPE={rpe_val if rpe_val is not None else '未入力'}  "
                  f"[{vtxt}]")

    if not sets_meta:
        print("  [ERROR] 変換できるセットがありませんでした。")
        return None

    # --- 手入力の値を引き継ぎ、コマンドラインの指定で上書き ---
    sets_meta = merge_manual_edits(sets_meta, out_dir / "meta.json")
    weights = parse_list(getattr(args, "weights", None), float, len(sets_meta), "weights")
    reps_list = parse_list(getattr(args, "reps", None), int, len(sets_meta), "reps")
    for k, s in enumerate(sets_meta):
        if weights:
            s["weight_kg"] = weights[k]
        if reps_list:
            s["reps_completed"] = reps_list[k]

    # --- meta.json ---
    meta = {
        "session_id": session,
        "subject_id": subject,
        "date": date_str or "",
        "source_sessions": [s.name for s in srcs],
        "devices": sorted(all_devices),
        "sampling_hz": 100,
        "modality": "imu+video",
        "camera_view": args.view,
        "exercise": args.exercise,
        "notes": "",
        "sets": sets_meta,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "dir": out_dir, "sets": len(sets_meta),
        "video": sum(1 for s in sets_meta if s["video_filename"]),
        "weight": sum(1 for s in sets_meta if s["weight_kg"]),
        "reps": sum(1 for s in sets_meta if s["reps_completed"]),
        "rpe": sum(1 for s in sets_meta if s["rpe"] is not None),
    }
    print(f"  [OK] {out_dir.relative_to(PROJECT_ROOT)}  セット {summary['sets']}  "
          f"動画 {summary['video']}  重量 {summary['weight']}  回数 {summary['reps']}  RPE {summary['rpe']}")
    return summary


def print_next_steps(subject: str, session: str, summary: dict):
    n = summary["sets"]
    if summary["weight"] < n or summary["reps"] < n:
        print(f"  → 重量・回数が足りません（推測では埋めない）。meta.json を直接編集するか:")
        print(f"     python scripts\\split_session.py --subject {subject} --session {session} "
              f"--src ... --weights 100,110,... --reps 5,3,...")


# ---------------------------------------------------------------------------
# 自動モード
# ---------------------------------------------------------------------------
def auto_import(args) -> int:
    """まだ取り込んでいない受信フォルダを日ごとにまとめ、次の番号のセッションとして取り込む"""
    done = imported_folders()
    pending = [f for f in list_sessions() if f.name not in done]
    if args.skip:
        pending = [f for f in pending if not any(k in f.name for k in args.skip)]
    if not pending:
        print("[split] 新しい計測はありません（data/_sessions はすべて取り込み済み）")
        return 0

    usable = []
    for f in pending:
        if not has_markers(f):
            print(f"[split] [SKIP] {f.name}: markers.csv が無いか空（受信を q で終えていない？）")
            continue
        usable.append((f, session_start(f)))

    groups = group_visits(usable, args.gap_hours)
    num = next_session_number(args.subject)
    n_done = 0
    for g in groups:
        no_video = [f.name for f in g if not find_videos_in(f)]
        day = session_start(g[0]).strftime("%Y-%m-%d %H:%M")
        if no_video and not args.allow_no_video:
            print(f"\n[split] [待ち] {day} の計測（{len(g)} フォルダ）: 動画が入っていないフォルダがあります")
            for n in no_video:
                print(f"           - {n}")
            print("         → 動画をそのフォルダの直下にコピーしてから、もう一度実行してください")
            continue
        session = f"SES{num:03d}"
        print(f"\n[split] {day} の計測 → {args.subject}/{session}")
        if args.dry_run:
            for f in g:
                print(f"   - {f.name}（動画 {len(find_videos_in(f))} 本）")
            num += 1
            continue
        summary = convert(args, args.subject, session, g)
        if summary is not None:
            print_next_steps(args.subject, session, summary)
            num += 1
            n_done += 1
    if args.dry_run:
        print("\n[split] --dry-run のため何も書き込んでいません")
    return n_done


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="連続記録をセット単位に分割し、動画も自動配置する（--session なしで自動モード）")
    ap.add_argument("--subject", default="S001", help="被験者ID（既定: S001 = 本人）")
    ap.add_argument("--session", default=None,
                    help="例: SES004。省略すると自動モード（未取り込みの計測を日ごとに自動で番号付け）")
    ap.add_argument("--src", action="append", default=None,
                    help="（手動モード）セッションフォルダ（複数指定可）。省略時は最新1つ")
    ap.add_argument("--all", action="store_true",
                    help="（手動モード）_sessions/ の全セッションを1つにまとめる")
    ap.add_argument("--skip", action="append", default=None,
                    help="除外するセッション（フォルダ名の一部）")
    ap.add_argument("--gap-hours", type=float, default=GAP_HOURS,
                    help=f"（自動モード）この時間以上あいたら別の日とみなす（既定 {GAP_HOURS:g}）")
    ap.add_argument("--allow-no-video", action="store_true",
                    help="（自動モード）動画が無いフォルダも取り込む")
    ap.add_argument("--dry-run", action="store_true", help="（自動モード）何をするかだけ表示")
    ap.add_argument("--margin", type=float, default=0.3,
                    help="切り出しの前後マージン[秒] (default: 0.3)")
    ap.add_argument("--exercise", type=str, default="Squat")
    ap.add_argument("--date", type=str, default=None)
    ap.add_argument("--move-video", action="store_true",
                    help="動画をコピーではなく移動する")
    ap.add_argument("--weights", type=str, default=None,
                    help="（手動モード）セット順の重量 [kg]（例: 100,110,120）")
    ap.add_argument("--reps", type=str, default=None,
                    help="（手動モード）セット順の完了回数（例: 5,3,2）")
    ap.add_argument("--view", type=str, default="front", choices=["front", "side", "oblique"],
                    help="撮影方向（既定: front）")
    args = ap.parse_args()

    if args.session is None:
        if args.src or args.all or args.weights or args.reps:
            sys.exit("[ERROR] --src / --all / --weights / --reps を使うときは --session も指定してください")
        auto_import(args)
        return

    srcs = resolve_sources(args)
    summary = convert(args, args.subject, args.session, srcs)
    if summary is None:
        sys.exit(1)
    print_next_steps(args.subject, args.session, summary)
    print(f"  次: python scripts\\run_pipeline.py --subject {args.subject}")


if __name__ == "__main__":
    main()
