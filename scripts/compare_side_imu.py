# -*- coding: utf-8 -*-
"""
=============================================================================
 横から撮った動画（プレートの中心）と IMU（バー）のレップ単位の一致度
 （前から撮った動画の速度が IMU より約 9 % 速く出る（×1.09）原因の切り分け。CLAUDE.md §3・§8 の 7）

 横から撮るとプレートがまん丸に写り、その中心はバーの軸そのもの。物差しはプレートの直径（45 cm）で、
 身長からは換算しない。なので「プレートの中心の上下」をバーの本当の上下として IMU と比べる。
   横 ÷ IMU ≈ 1.00 → IMU は正しい。×1.09 は前の動画の側（肩の点はバーではない・身長 × 0.779 の換算）
   横 ÷ IMU ≈ 1.09 → IMU が約 8 % 小さく出ている
 比べ方は compare_video_imu.py と同じ（同じ時間軸に置き、両方の波形に同じ定義を当てる）。

 置き方（チートシート「横から撮るセット」）:
   受信は python scripts\\multi_imu_receiver.py --outdir data\\_side → data/_side/session_…/（raw_*.csv・markers.csv）
   横の動画はその session_… の直下に置く（ファイル名の順 = セットの順。1 セット 1 本）
   data/_ で始まるフォルダなので、run_pipeline.py の取り込み・学習には入らない

 1 セットの処理:
   1. IMU を markers.csv の SPACE〜SPACE（前後 0.3 s）で切り出す（split_session.py と同じ）
   2. プレートを追う（bar_path_extract.py: Hough 円で自動検出 → CSRT 追跡＋Hough 補正）→ session_…/plate/
      物差し: Hough で測ったプレートの半径の中央値 × 2 = プレートの直径（--plate-diameter、既定 450 mm）
      位置は肩の点と同じ前処理（0.3 s 以下の見失いだけ直線でつなぎ、6 Hz のローパス: extract_features._fill_smooth）
   3. 時刻合わせ: プレートの上向きの速度と IMU の速度を重ね、相関がいちばん高いずれ（extract_features.estimate_offset。
      前の動画と同じく r ≥ 0.6・margin ≥ 0.05 のときだけ使う）
   4. レップの区切り: プレートの沈み込みで（extract_features.detect_pose_reps と同じ区切り方）
   5. レップごとに 上昇時間・MCV・ピーク・上昇の変位・深さ（compare_video_imu.rep_metrics）
      レップの前後でプレートを 0.2 s より長く見失ったレップは比べない（ラック・セーフティバーに隠れたなど）

 出力（outputs/。列の vid_ は横の動画、imu_ は IMU）:
   rep_compare_side_imu.csv          1 行 = 1 レップ
   rep_compare_side_imu_summary.csv  指標ごとの差・95 % 一致限界・r・倍率
   rep_compare_side_imu_sets.csv     1 行 = 1 セット（MCV の倍率、物差し、時刻合わせなど）
   fig_bland_altman_side_imu.png
 判定: セットごとの MCV の倍率（横 ÷ IMU）の平均と 95 % 信頼区間（t 分布。同じセットのレップは独立でないので
       セットを単位にする）が、1.00 と 前の動画の倍率（outputs/rep_compare_video_imu_summary.csv）のどちらを含むか。
       セットごとの倍率の SD が前の動画と同じ 0.032 なら、6 セットで ±0.034

 使い方:
   python scripts\\compare_side_imu.py                  # data/_side の全部
   python scripts\\compare_side_imu.py --force          # プレートの追跡をやり直す（見つけた位置はそのまま）
   python scripts\\compare_side_imu.py --reselect-roi   # プレートの位置から探し直す

 依存: opencv-python（CSRT）numpy pandas scipy matplotlib tqdm
       （bar_path_extract.py・extract_features.py・compare_video_imu.py を使う）
=============================================================================
"""

import re
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import extract_features as ef                      # noqa: E402
import compare_video_imu as cvi                    # noqa: E402  （matplotlib を Agg にしてから）
import bar_path_extract as bpe                     # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
SIDE_ROOT = PROJECT_ROOT / "data" / "_side"
OUT_DIR = PROJECT_ROOT / "outputs"
FRONT_SUMMARY = OUT_DIR / "rep_compare_video_imu_summary.csv"

VIDEO_EXT = (".mp4", ".mov", ".avi")
IMU_MARGIN_S = 0.3        # IMU の切り出しの前後の余裕 [s]（split_session.py の --margin と同じ）
PLATE_DIAMETER_MM = 450.0 # 20 kg プレート（バンパープレートも 450 mm）
REP_SCALE_M = 1.25        # レップの区切り: 沈み込み [m] をこれで割り、骨格（肩〜足首の高さ = 1）と同じしきい値で区切る
                          # （沈み込み 0.08 ≒ 10 cm より大きいものだけ。ラックアウト・ラックインの数 cm は入らない）
MAX_LOST_S = 0.2          # レップの前後でプレートを見失い、直線でつないだ時間がこれより長いレップは比べない
SUBJECT = "S001"          # 横から撮るのは本人だけ（IMU は本人だけ）

SIDE_NOTE = ("図の見方\n"
             "・1 点 = 1 レップ（1 回のしゃがみ）\n"
             "・同じレップを 2 通りで測った:\n"
             "    横の動画 = プレートの中心の上下の動き\n"
             "      （プレートの直径 45 cm を物差しに m に換算）\n"
             "    IMU = バーにつけたセンサー\n"
             "・縦軸 = 横の動画 − IMU（+ なら動画の方が大きい）\n"
             "・横軸 = その値の大きさ（2 つの平均）\n"
             "・実線 = ずれの平均（0 から離れるほど、\n"
             "    いつも同じ向きにずれている）\n"
             "・破線 = 95 % のレップが入る範囲")


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def find_side_sessions(root: Path = SIDE_ROOT) -> list:
    """markers.csv のある受信フォルダ（data/_side/session_…）を名前の順で"""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "markers.csv").exists())


def session_date(folder: Path):
    """フォルダ名 session_YYYYMMDD_HHMMSS… から日付（読めなければ None）"""
    m = re.match(r"session_(\d{4})(\d{2})(\d{2})", folder.name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def load_raw_imu(folder: Path):
    """raw_<DEV>.csv（BAR があればそれ、無ければ名前の順で最初）。戻り値: (DEV, 表) か (None, None)"""
    raws = sorted(folder.glob("raw_*.csv"))
    if not raws:
        return None, None
    pick = next((p for p in raws if p.stem == "raw_BAR"), raws[0])
    df = pd.read_csv(pick)
    if "recv_time_s" not in df.columns:
        return None, None
    return pick.stem.replace("raw_", ""), df


def cut_imu(raw: pd.DataFrame, start_s: float, end_s: float, margin: float = IMU_MARGIN_S) -> pd.DataFrame:
    """SPACE〜SPACE（前後 margin 秒）を切り出し、先頭に timestamp_ms =（受信時刻 − start_s）× 1000（split_session.py と同じ）"""
    seg = raw[(raw["recv_time_s"] >= start_s - margin) & (raw["recv_time_s"] <= end_s + margin)].copy()
    seg.insert(0, "timestamp_ms", ((seg["recv_time_s"] - start_s) * 1000.0).round(1))
    return seg.reset_index(drop=True)


def _int_or_none(x):
    try:
        return None if pd.isna(x) else int(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# プレート
# ---------------------------------------------------------------------------
def track_plate(video: Path, out_dir: Path, plate_mm: float, force=False, reselect=False, allow_gui=True):
    """
    プレートの追跡（bar_path_extract.py）。済んでいれば保存したものを読む。
    戻り値: (フレームごとの表, summary の dict) か None（プレートが見つからない）
    """
    csv, summ = out_dir / f"{video.stem}_bar_path.csv", out_dir / f"{video.stem}_bar_summary.json"
    if force or reselect or not (csv.exists() and summ.exists()):
        try:
            res = bpe.process_one_video(video, out_dir, plate_diameter_mm=plate_mm, save_annotated=False,
                                        reselect_roi=reselect, allow_gui=allow_gui)
        except Exception as e:                      # 追跡できたコマが少なすぎる など
            res = {"status": "error", "reason": str(e)}
        if res.get("status") != "ok":
            print(f"    [SKIP] プレートを追えませんでした（{res.get('reason')}）")
            return None
    return pd.read_csv(csv), json.loads(summ.read_text(encoding="utf-8"))


def plate_motion(path_df: pd.DataFrame, mm_per_px: float, fps: float) -> dict:
    """
    プレートの中心の上下の位置 z [m]（上が正。立っているときがほぼ 0）と速度 v [m/s]。時刻 t は動画の先頭からの秒。
    前処理は肩の点と同じ（0.3 s 以下の見失いだけ直線でつなぎ、6 Hz のローパス）。
    lost = プレートを見失ったコマ（追跡の始まる前を含む）
    """
    lost = ~path_df["tracked"].astype(bool).to_numpy()
    y = np.where(lost, np.nan, path_df["y_px"].to_numpy(dtype=float))
    ys = ef._fill_smooth(y, fps)
    z = pd.Series(-ys * mm_per_px / 1000.0).interpolate(limit_direction="both").to_numpy()
    z = z - np.nanpercentile(z, 95)
    t = path_df["frame"].to_numpy(dtype=float) / fps
    return {"t": t, "z": z, "v": np.gradient(z) * fps, "lost": lost}


def longest_run(mask) -> int:
    """True が続くいちばん長い長さ（コマ数）"""
    best = cur = 0
    for x in np.asarray(mask, dtype=bool):
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# 1 セット・1 フォルダ
# ---------------------------------------------------------------------------
def compare_set(path_df, summary, imu_df, plate_mm, expected_reps=None):
    """
    1 セットのレップごとの比較。戻り値: (レップの行のリスト, セットの情報 dict)
    行の列は compare_video_imu.compare_set と同じ（vid_ = 横の動画）に、見失った秒数 lost_s を足したもの。
    """
    fps = float(summary.get("fps") or 30.0)
    # 物差し: 保存した追跡は保存したときの直径で換算してあるので、今の直径に合わせ直す
    mm_per_px = float(summary["mm_per_px"]) * plate_mm / float(summary.get("plate_diameter_mm") or plate_mm)
    info = {"fps": fps, "mm_per_px": round(mm_per_px, 4), "scale_source": summary.get("scale_source"),
            "tracked_ratio": summary.get("tracked_ratio")}
    pm = plate_motion(path_df, mm_per_px, fps)
    reps = ef.detect_pose_reps(-pm["z"] / REP_SCALE_M, fps, expected_reps=expected_reps)
    info["n_reps_detected"] = len(reps)
    if not reps:
        info["reason"] = "プレートの動きにレップが見つからない"
        return [], info

    # 時刻合わせ（前の動画と同じく、セットの区間だけを使う）
    lo = pm["t"][reps[0][1]] - ef.SET_PRE_S
    hi = pm["t"][reps[-1][1]] + ef.SET_POST_S
    sel = (pm["t"] >= lo) & (pm["t"] <= hi)
    imu_c = ef.imu_on_device_clock(imu_df)
    t_i = imu_c["timestamp_ms"].to_numpy() / 1000.0
    dt = float(np.median(np.diff(t_i)))
    _, _, _, _, v_hp, _ = ef.imu_vertical_velocity(imu_c, 1.0 / dt if dt > 0 else 100.0, dt)
    sync = ef.estimate_offset(pm["t"][sel], pm["v"][sel], t_i, v_hp)
    info.update(sync)
    if not sync or sync["sync_r"] < ef.SYNC_R_MIN or sync["sync_margin"] < ef.SYNC_MARGIN_MIN:
        info["reason"] = "時刻合わせが合わない（別のセットの動画か、IMU が途中で止まった？）"
        return [], info
    off = sync["sync_offset_s"]

    windows = [(pm["t"][s] + off, pm["t"][b] + off, pm["t"][e] + off) for s, b, e in reps]
    imu = cvi.imu_motion(imu_df, windows)
    rows, n_lost = [], 0
    for i, (s, b, e) in enumerate(reps):
        ts, te = pm["t"][s], pm["t"][e]
        j0, j1 = np.searchsorted(pm["t"], [ts - cvi.STAND_PRE_S, te + cvi.WIN_POST_S])
        lost_s = longest_run(pm["lost"][j0:j1]) / fps
        if lost_s > MAX_LOST_S:
            n_lost += 1
            continue
        mv = cvi.rep_metrics(pm["t"], pm["z"], pm["v"], ts, te)
        mi = cvi.rep_metrics(imu["t"], imu["z"], imu["v"], ts + off, te + off)
        if mv is None or mi is None:
            continue
        row = {"rep": i + 1}
        for k, _, _, scale in cvi.METRICS:
            row[f"imu_{k}"] = mi[k] * scale
            row[f"vid_{k}"] = mv[k] * scale
            row[f"diff_{k}"] = (mv[k] - mi[k]) * scale
        row["bottom_gap_s"] = mi["bottom_t"] - off - mv["bottom_t"]   # 時刻合わせの確認（0 に近いはず）
        row["lost_s"] = lost_s
        rows.append(row)
    info["n_compared"] = len(rows)
    info["n_lost_skipped"] = n_lost
    return rows, info


def compare_session(folder: Path, plate_mm=PLATE_DIAMETER_MM, force=False, reselect=False, allow_gui=True):
    """
    1 つの受信フォルダ（data/_side/session_…）のレップごとの比較。
    動画はフォルダ直下をファイル名の順に、markers.csv のセットへ先頭から割り当てる（split_session.py と同じ）。
    戻り値: (レップの行のリスト, セットごとの情報のリスト)
    """
    markers = pd.read_csv(folder / "markers.csv")
    dev, raw = load_raw_imu(folder)
    videos = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT),
                    key=lambda p: p.name)
    print(f"[{folder.name}] セット {len(markers)}・動画 {len(videos)} 本・IMU {dev or 'なし'}")
    if raw is None:
        print("  [SKIP] raw_*.csv（受信時刻 recv_time_s のある IMU）がありません")
        return [], []
    if len(videos) != len(markers):
        print(f"  [注意] 動画 {len(videos)} 本とセット {len(markers)} の数が合いません。先頭から順に割り当てます")

    date = session_date(folder)
    all_rows, infos = [], []
    for (_, m), video in zip(markers.iterrows(), videos):
        set_no = int(m["set_no"])
        expected = _int_or_none(m.get("reps"))
        base = {"subject_id": SUBJECT, "session_id": folder.name, "set_no": set_no, "date": date,
                "weight_kg": m.get("weight_kg"), "rpe": m.get("rpe")}
        info = {"session_id": folder.name, "set_no": set_no, "video": video.name, "reps_planned": expected}
        print(f"  set{set_no:02d} ← {video.name}")
        tracked = track_plate(video, folder / "plate", plate_mm, force=force, reselect=reselect, allow_gui=allow_gui)
        if tracked is None:
            infos.append({**info, "reason": "プレートを追えない"})
            continue
        imu = cut_imu(raw, float(m["start_s"]), float(m["end_s"]))
        rows, sinfo = compare_set(*tracked, imu, plate_mm, expected_reps=expected)
        info.update(sinfo)
        infos.append(info)
        for r in rows:
            r.update({**base, "sync_r": info.get("sync_r"), "sync_margin": info.get("sync_margin")})
        all_rows += rows
        msg = (f"    {len(rows)} レップ比べた（レップ {info.get('n_reps_detected')}・申告 {expected}"
               f"・見失いで外した {info.get('n_lost_skipped', 0)}）")
        if info.get("sync_r") is not None:
            msg += f"  時刻合わせ r {info['sync_r']:.2f}・margin {info['sync_margin']:.2f}"
        msg += f"  物差し {info['mm_per_px']:.2f} mm/px（{info['scale_source']}）"
        print(msg)
        if info.get("reason"):
            print(f"    [SKIP] {info['reason']}")
        if info.get("scale_source") != "hough_radius":
            print("    [注意] 物差しが最初の枠の大きさから（プレートの縁を 10 コマ未満しか確かめられなかった）")
    return all_rows, infos


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def front_ratio(path: Path = FRONT_SUMMARY):
    """前から撮った動画 ÷ IMU の MCV の倍率（compare_video_imu.py の出力。無ければ None）"""
    try:
        s = pd.read_csv(path)
        return float(s.loc[s["metric"] == "mcv", "ratio"].iloc[0])
    except Exception:
        return None


def judge(ratios, front=None) -> dict:
    """
    セットごとの MCV の倍率（横 ÷ IMU）の平均と 95 % 信頼区間（t 分布）と、どちらの見方に合うか。
    front: 前から撮った動画 ÷ IMU の倍率（1.09 など）
    """
    r = np.asarray(ratios, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    out = {"n": n, "mean": float(r.mean()) if n else float("nan")}
    if n < 2:
        out["verdict"] = "セットが足りない（2 セット以上要る）"
        return out
    half = float(stats.t.ppf(0.975, n - 1) * r.std(ddof=1) / np.sqrt(n))
    lo, hi = out["mean"] - half, out["mean"] + half
    out.update({"ci_low": lo, "ci_high": hi})
    has1 = lo <= 1.0 <= hi
    hasf = front is not None and lo <= front <= hi
    if front is None:
        out["verdict"] = ("1.00 を含む: IMU とプレートは合っている" if has1
                          else "1.00 を含まない: IMU とプレートがずれている")
    elif has1 and hasf:
        out["verdict"] = f"まだ決められない（1.00 と {front:.2f} の両方を含む。セットを増やす）"
    elif has1:
        out["verdict"] = (f"IMU はほぼ正しい。前の動画の ×{front:.2f} は動画の側から"
                          "（肩の点はバーではない・身長 × 0.779 の換算）")
    elif hasf:
        out["verdict"] = f"IMU が小さく出ている（前の動画の ×{front:.2f} とほぼ同じだけ）"
    elif hi < 1.0:
        out["verdict"] = "横の動画の方が IMU より小さい: プレートの直径（--plate-diameter）が合っているか確かめる"
    else:
        out["verdict"] = (f"1.00 とも {front:.2f} とも違う: IMU が {100 * (out['mean'] - 1):.0f} % ほど小さく出ていて、"
                          "前の動画にも残りのずれがある可能性")
    return out


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="横から撮った動画（プレート）と IMU のレップ単位の一致度")
    ap.add_argument("--root", type=str, default=str(SIDE_ROOT), help="受信フォルダの置き場（既定 data/_side）")
    ap.add_argument("--plate-diameter", type=float, default=PLATE_DIAMETER_MM,
                    help=f"プレートの直径 [mm]（既定 {PLATE_DIAMETER_MM:.0f} = 20 kg プレート）")
    ap.add_argument("--force", action="store_true", help="プレートの追跡をやり直す")
    ap.add_argument("--reselect-roi", action="store_true", help="プレートの位置から探し直す")
    args = ap.parse_args()

    sessions = find_side_sessions(Path(args.root))
    if not sessions:
        print(f"[INFO] {args.root} に受信フォルダ（markers.csv のある session_…）がありません")
        print("  受信: python scripts\\multi_imu_receiver.py --outdir data\\_side")
        print("  横の動画はその session_… の直下に置く（ファイル名の順 = セットの順）")
        return

    all_rows, all_info = [], []
    for folder in sessions:
        rows, info = compare_session(folder, plate_mm=args.plate_diameter,
                                     force=args.force, reselect=args.reselect_roi)
        all_rows += rows
        all_info += info
    if not all_rows:
        sys.exit("[ERROR] 比べられるレップがありません")

    lead = ["subject_id", "session_id", "set_no", "rep", "date", "weight_kg", "rpe"]
    reps = pd.DataFrame(all_rows)
    reps = reps[lead + [c for c in reps.columns if c not in lead]]
    summ = cvi.summarize(reps)
    info_cols = ["session_id", "set_no", "video", "reps_planned", "n_reps_detected", "n_lost_skipped",
                 "sync_offset_s", "mm_per_px", "scale_source", "tracked_ratio"]
    info_df = pd.DataFrame(all_info)
    sets = cvi.per_set(reps).merge(info_df[[c for c in info_cols if c in info_df.columns]],
                                   on=["session_id", "set_no"], how="left")

    OUT_DIR.mkdir(exist_ok=True)
    reps.to_csv(OUT_DIR / "rep_compare_side_imu.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(OUT_DIR / "rep_compare_side_imu_summary.csv", index=False, encoding="utf-8-sig")
    sets.to_csv(OUT_DIR / "rep_compare_side_imu_sets.csv", index=False, encoding="utf-8-sig")
    cvi.fig_bland_altman(reps, summ, OUT_DIR / "fig_bland_altman_side_imu.png",
                         title="横から撮った動画（プレートの中心）と IMU（バー）のレップごとの一致（Bland–Altman 図、参考値）",
                         note=SIDE_NOTE, vid_name="横の動画")

    print(f"\n[一致度] レップ {len(reps)}・セット {len(sets)}（参考値。列の vid_ は横の動画）")
    for s in summ.itertuples():
        if s.n >= 2:
            print(f"  {s.name:<6} IMU {s.imu_mean:.3f} / 横 {s.vid_mean:.3f} {s.unit}  差 {s.bias:+.3f}"
                  f"（95 % 一致限界 {s.loa_low:+.3f}〜{s.loa_high:+.3f}）  r {s.r:.2f}  倍率 {s.ratio:.2f}")
    print(f"  時刻合わせの確認: 底の時刻の差 中央値 {reps['bottom_gap_s'].median():+.3f} s"
          f"（{reps['bottom_gap_s'].min():+.2f}〜{reps['bottom_gap_s'].max():+.2f}）")

    front = front_ratio()
    j = judge(sets["mcv_ratio"], front)
    print("\n[判定] MCV の倍率（横の動画 ÷ IMU、セットごと）")
    for r in sets.itertuples():
        print(f"  {r.session_id} set{int(r.set_no):02d}: {r.mcv_ratio:.3f}（{r.n_reps} レップ）")
    if "ci_low" in j:
        print(f"  平均 {j['mean']:.3f}（95 % 信頼区間 {j['ci_low']:.3f}〜{j['ci_high']:.3f}、{j['n']} セット）")
    else:
        print(f"  平均 {j['mean']:.3f}（{j['n']} セット）")
    if front is not None:
        print(f"  前から撮った動画 ÷ IMU は {front:.3f}（{FRONT_SUMMARY.name}）")
    print(f"  → {j['verdict']}")
    print(f"\n[OK] {OUT_DIR / 'rep_compare_side_imu.csv'} ほか（summary / sets / 図）")
    print("  プレートの見つけ方の確認: data\\_side\\session_…\\plate\\<動画名>_bar_roi_preview.png"
          "（緑の円がプレートの縁に重なっているか）")


if __name__ == "__main__":
    main()
