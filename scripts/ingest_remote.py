# -*- coding: utf-8 -*-
"""
=============================================================================
 協力者（遠隔・動画のみ）のデータを取り込むスクリプト

 【置き方】 Google Drive の共有フォルダを、そのまま data/_remote/ に置く
   data/_remote/<名前>/プロフィール.txt           （初回だけ。無くても取り込める）
   data/_remote/<名前>/<日付フォルダ>/*.mp4|*.mov   （1セット = 1本）
   data/_remote/<名前>/<日付フォルダ>/記録.txt      （ファイル名にラベルが無いとき）

   日付フォルダ名は 20261005 / 2026-10-05 / 2026_10_05 など、日付が読めればよい。

 【ラベル（重量・回数・RPE）】 次のどちらか
   (1) ファイル名:  100kg_5rep_RPE8.mp4 / 102.5kg 3回 RPE9.5.mov / 100kgx5@8.mp4
   (2) 記録.txt:    撮影した順に 1行1セット「重量 回数 RPE」
                       100 5 8
                       102.5 3 9.5
   撮影順は動画の撮影日時（メタデータ）→ 無ければファイル名の順。

 【プロフィール.txt】 例（1行1項目、順不同）
     身長 175
     体重 80
     1RM 150        ← スクワットの 1RM（推定でよい）
     歴 5           ← トレーニング歴（年）

 【出力】
   data/<Sxxx>/<SESnnn>/videos/setNN.<拡張子>   元の動画はそのまま残す（コピー）
   data/<Sxxx>/<SESnnn>/meta.json               camera_view=front, modality=video
   data/subjects.json                           名前 ↔ 被験者ID、プロフィール

 【使い方】
   python scripts\\ingest_remote.py              # 未取り込みの日付フォルダをすべて取り込む
   python scripts\\ingest_remote.py --dry-run    # 何をするかだけ表示（コピーしない）
   python scripts\\ingest_remote.py --force      # 取り込み済みも作り直す（同じ SES 番号を使う）
   → 続けて python scripts\\run_pipeline.py

 依存: ffprobe（撮影日時の取得。無ければファイル名順になる）
=============================================================================
"""

import re
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT = PROJECT_ROOT / "data"
REMOTE_ROOT = DATA_ROOT / "_remote"
SUBJECTS_JSON = DATA_ROOT / "subjects.json"

VIDEO_EXT = {".mp4", ".mov", ".m4v"}
LOG_NAMES = ("記録.txt", "記録.csv", "record.txt", "log.txt")
PROFILE_NAMES = ("プロフィール.txt", "profile.txt")

RE_WEIGHT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:kg|ｋｇ|キロ)", re.I)
RE_REPS = re.compile(r"(?:(\d+)\s*(?:reps?|回|レップ))|(?:[x×]\s*(\d+))", re.I)
RE_RPE = re.compile(r"(?:rpe\s*(\d+(?:[.,]\d+)?))|(?:@\s*(\d+(?:[.,]\d+)?))", re.I)
RE_DATE = re.compile(r"(20\d{2})[-_./]?(\d{2})[-_./]?(\d{2})")


# ---------------------------------------------------------------------------
# ラベル・プロフィールの読み取り
# ---------------------------------------------------------------------------
def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_label(text: str) -> dict:
    """文字列から 重量・回数・RPE を読む（見つかったものだけ返す）"""
    out = {}
    m = RE_WEIGHT.search(text)
    if m:
        out["weight_kg"] = _num(m.group(1))
    m = RE_REPS.search(text)
    if m:
        out["reps"] = int(m.group(1) or m.group(2))
    m = RE_RPE.search(text)
    if m:
        out["rpe"] = _num(m.group(1) or m.group(2))
    return out


def parse_log_line(line: str) -> Optional[dict]:
    """記録.txt の1行。単位付きでも、数字3つ（重量 回数 RPE）でもよい"""
    lab = parse_label(line)
    if len(lab) == 3:
        return lab
    nums = re.findall(r"\d+(?:[.,]\d+)?", line)
    if len(nums) >= 3:
        return {"weight_kg": _num(nums[0]), "reps": int(float(nums[1].replace(",", "."))),
                "rpe": _num(nums[2])}
    return None


def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def read_log(folder: Path):
    """記録.txt を読む。戻り値: (ラベルのリスト or None, ファイル名)"""
    for name in LOG_NAMES:
        p = folder / name
        if p.exists():
            rows = []
            for line in read_text_any(p).splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lab = parse_log_line(line)
                if lab:
                    rows.append(lab)
            return rows, name
    return None, None


def read_profile(person_dir: Path) -> dict:
    """プロフィール.txt を読む（身長・体重・1RM・トレーニング歴・性別）"""
    for name in PROFILE_NAMES:
        p = person_dir / name
        if not p.exists():
            continue
        prof = {}
        for line in read_text_any(p).splitlines():
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            num = re.search(r"\d+(?:[.,]\d+)?", line.replace("1RM", "").replace("1rm", ""))
            val = _num(num.group()) if num else None
            if "身長" in line or low.startswith("height"):
                prof["height_cm"] = val
            elif "体重" in line or low.startswith("weight") or low.startswith("body"):
                prof["body_mass_kg"] = val
            elif "1rm" in low or "max" in low:
                prof["squat_1rm_kg"] = val
            elif "歴" in line or "year" in low:
                prof["training_years"] = val
            elif "性別" in line or low.startswith("sex"):
                prof["sex"] = line.split(maxsplit=1)[-1] if len(line.split()) > 1 else None
        return {k: v for k, v in prof.items() if v is not None}
    return {}


# ---------------------------------------------------------------------------
# 動画の撮影日時・長さ（ffprobe）
# ---------------------------------------------------------------------------
def probe_video(path: Path) -> dict:
    """撮影日時（datetime or None）と長さ[s]。LINE 経由などでメタデータが消えていれば None"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:format_tags=creation_time,com.apple.quicktime.creationdate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30)
        fmt = json.loads(r.stdout or "{}").get("format", {})
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"created": None, "duration": None}
    tags = fmt.get("tags", {}) or {}
    raw = tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
    created = None
    if raw:
        try:
            created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            created = None
    dur = fmt.get("duration")
    return {"created": created, "duration": float(dur) if dur else None}


# ---------------------------------------------------------------------------
# 被験者・セッションの番号付け
# ---------------------------------------------------------------------------
def load_subjects() -> dict:
    subs = {"subjects": {}}
    if SUBJECTS_JSON.exists():
        subs = json.loads(SUBJECTS_JSON.read_text(encoding="utf-8"))
    # 本人（S001）も登録しておく（プロフィールは手で埋める）
    if (DATA_ROOT / "S001").is_dir() and "S001" not in subs["subjects"]:
        subs["subjects"]["S001"] = {"source": "self", "name": "本人",
                                    "profile": {"height_cm": None, "squat_1rm_kg": None}}
    return subs


def save_subjects(subs: dict):
    SUBJECTS_JSON.write_text(json.dumps(subs, indent=2, ensure_ascii=False), encoding="utf-8")


def subject_id_for(subs: dict, person: str) -> str:
    """協力者フォルダ名 → 被験者ID（無ければ S002, S003, ... を新しく振る）"""
    source = f"_remote/{person}"
    for sid, info in subs["subjects"].items():
        if info.get("source") == source:
            return sid
    used = {int(m.group(1)) for k in list(subs["subjects"]) + [p.name for p in DATA_ROOT.iterdir()]
            if (m := re.fullmatch(r"S(\d{3})", k))}
    sid = f"S{max(used | {1}) + 1:03d}"
    subs["subjects"][sid] = {"source": source, "name": person, "profile": {}}
    return sid


def find_ingested(sid: str, source_key: str) -> Optional[Path]:
    """この日付フォルダを取り込み済みのセッションがあれば、そのフォルダを返す"""
    subj_dir = DATA_ROOT / sid
    if not subj_dir.is_dir():
        return None
    for mj in subj_dir.glob("SES*/meta.json"):
        try:
            if source_key in json.loads(mj.read_text(encoding="utf-8")).get("source_sessions", []):
                return mj.parent
        except Exception:
            continue
    return None


def next_session_id(sid: str) -> str:
    subj_dir = DATA_ROOT / sid
    nums = [int(m.group(1)) for p in subj_dir.glob("SES*") if (m := re.fullmatch(r"SES(\d{3})", p.name))] \
        if subj_dir.is_dir() else []
    return f"SES{max(nums, default=0) + 1:03d}"


# ---------------------------------------------------------------------------
# 1つの日付フォルダを取り込む
# ---------------------------------------------------------------------------
def check_label(lab: dict) -> Optional[str]:
    if not (20 <= lab["weight_kg"] <= 500):
        return f"重量 {lab['weight_kg']} kg が範囲外"
    if not (1 <= lab["reps"] <= 30):
        return f"回数 {lab['reps']} が範囲外"
    if not (5 <= lab["rpe"] <= 10) or (lab["rpe"] * 2) % 1:
        return f"RPE {lab['rpe']} が不正（5〜10、0.5刻み）"
    return None


def ingest_folder(sid: str, person: str, folder: Path, dry_run: bool, force: bool) -> Optional[str]:
    source_key = f"_remote/{person}/{folder.name}"
    done = find_ingested(sid, source_key)
    if done and not force:
        return None                                         # 取り込み済み

    videos = [p for p in folder.iterdir()
              if p.is_file() and p.suffix.lower() in VIDEO_EXT and "_annotated" not in p.stem]
    if not videos:
        return None
    info = {v: probe_video(v) for v in videos}
    have_time = all(info[v]["created"] for v in videos)
    videos.sort(key=(lambda v: (info[v]["created"], v.name)) if have_time else (lambda v: v.name))

    # ラベル: ファイル名 → 足りなければ 記録.txt
    labels = [parse_label(v.stem) for v in videos]
    log_rows, log_name = read_log(folder)
    errors = []
    for i, v in enumerate(videos):
        if len(labels[i]) == 3:
            continue
        if log_rows is None:
            errors.append(f"{v.name}: ファイル名にラベルが無く、記録.txt もありません")
        elif len(log_rows) != len(videos):
            errors.append(f"{log_name} の行数 {len(log_rows)} と動画の本数 {len(videos)} が合いません")
            break
        else:
            labels[i] = log_rows[i]
    for i, v in enumerate(videos):
        if len(labels[i]) == 3 and (msg := check_label(labels[i])):
            errors.append(f"{v.name}: {msg}")
    if errors:
        print(f"  [要確認] {source_key}（取り込みませんでした）")
        for e in errors:
            print(f"      - {e}")
        return None

    # 日付
    m = RE_DATE.search(folder.name)
    if m:
        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    elif have_time:
        date_str = info[videos[0]]["created"].date().isoformat()
    else:
        date_str = ""

    ses = done.name if done else next_session_id(sid)
    out_dir = DATA_ROOT / sid / ses
    order_note = "撮影日時の順" if have_time else "ファイル名の順（撮影日時が読めませんでした）"
    print(f"  == {source_key} → {sid}/{ses}  ({len(videos)}セット, {date_str}, {order_note})")

    sets = []
    for i, v in enumerate(videos, start=1):
        lab = labels[i - 1]
        rest = None
        if have_time and i >= 2:
            prev = videos[i - 2]
            if info[prev]["duration"]:
                rest = round((info[v]["created"] - info[prev]["created"]).total_seconds()
                             - info[prev]["duration"])
        video_name = f"set{i:02d}{v.suffix.lower()}"
        sets.append({
            "set_no": i,
            "csv_filename": None,
            "video_filename": video_name,
            "exercise": "Squat",
            "weight_kg": lab["weight_kg"],
            "reps_planned": None,
            "reps_completed": lab["reps"],
            "rpe": lab["rpe"],
            "rest_before_sec": rest if rest is None or rest >= 0 else None,
            "duration_s": round(info[v]["duration"], 2) if info[v]["duration"] else None,
            "source_video": v.name,
            "recorded_at": info[v]["created"].isoformat() if info[v]["created"] else None,
            "notes": "",
        })
        print(f"     set{i:02d}  {lab['weight_kg']:>6g}kg  {lab['reps']:>2}回  RPE {lab['rpe']:<4g}  ← {v.name}")

    if dry_run:
        return ses

    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(videos, start=1):
        shutil.copy2(str(v), str(out_dir / "videos" / sets[i - 1]["video_filename"]))
    meta = {
        "session_id": ses,
        "subject_id": sid,
        "date": date_str,
        "source_sessions": [source_key],
        "devices": [],
        "sampling_hz": None,
        "modality": "video",
        "camera_view": "front",
        "exercise": "Squat",
        "notes": "",
        "sets": sets,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return ses


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="協力者の動画（data/_remote/）を取り込む")
    ap.add_argument("--dry-run", action="store_true", help="何をするかだけ表示する")
    ap.add_argument("--force", action="store_true", help="取り込み済みの日付フォルダも作り直す")
    ap.add_argument("--person", type=str, default=None, help="この協力者フォルダだけ")
    args = ap.parse_args()

    if not REMOTE_ROOT.is_dir():
        REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {REMOTE_ROOT} を作りました。協力者の共有フォルダをここに置いてください。")

    subs = load_subjects()
    n_new = 0
    for person_dir in sorted(p for p in REMOTE_ROOT.iterdir() if p.is_dir()):
        person = person_dir.name
        if args.person and person != args.person:
            continue
        sid = subject_id_for(subs, person)
        prof = read_profile(person_dir)
        if prof:
            subs["subjects"][sid]["profile"].update(prof)
        print(f"\n[{sid}] {person}  プロフィール: {subs['subjects'][sid]['profile'] or '未提出'}")

        date_dirs = sorted((p for p in person_dir.iterdir() if p.is_dir()),
                           key=lambda p: (RE_DATE.search(p.name).group(0) if RE_DATE.search(p.name) else "", p.name))
        for folder in date_dirs:
            if ingest_folder(sid, person, folder, args.dry_run, args.force):
                n_new += 1

    if not args.dry_run:
        save_subjects(subs)
    print()
    print("=" * 64)
    print(f"[OK] 取り込んだセッション: {n_new}" + ("（dry-run: 何もコピーしていません）" if args.dry_run else ""))
    if n_new and not args.dry_run:
        print("  次: python scripts\\run_pipeline.py   （骨格推定 → 特徴量 → 集計 → 品質レポート）")
    print("=" * 64)


if __name__ == "__main__":
    main()
