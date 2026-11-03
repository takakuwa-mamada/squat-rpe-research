# -*- coding: utf-8 -*-
"""
=============================================================================
 動画（肩の上下動）と IMU（バー）のレップ単位の一致度（RQ2: 動画のバー速度の妥当性）

 時刻合わせ（extract_features.sync_video_imu）が合ったセットで、動画のレップ i と IMU のレップ i を
 同じ時間軸に置き、両方の速度の波形に「同じ定義」を当てて比べる。
   上昇区間 : 底より後で、速度がピークの 5 %（最低 0.02 m/s）を超えて続く区間（extract_features と同じ）
   MCV      : 上昇区間の平均速度 [m/s]
   ピーク   : 上昇区間の最大速度 [m/s]
   上昇の変位: 上昇区間の終わりの位置 − 始まりの位置 [cm]（MCV ≒ 上昇の変位 / 上昇時間）
   深さ     : しゃがみ始めの直前 0.3 s の位置の中央値 − 底（区間の 1 パーセンタイル）[cm]
              （1 フレームのキーポイントの跳びに引きずられないよう、最大・最小は使わない）
 動画の速度・位置は 肩の上下 [px] → m（身長 × 0.779 = 立位の肩〜足首、を体の大きさ BL [px] で割った倍率）。
 IMU は extract_features と同じ速度（レップごとにロックアウトで速度 0・変位 0 に補正）を使い、位置はその積分。
 特徴量 JSON の値（_imu_reps / _pose_reps）は区間の決め方が両者で少し違うので、比較には使わず参考列として残す。

 対象: features_all.csv のうち、imu_rep_source = video（時刻合わせが合った）で、
       quality_checks.exclusion_reasons で IMU・動画のどちらも学習から外れていないセット

 出力（outputs/）:
   rep_compare_video_imu.csv          1 行 = 1 レップ（IMU と動画の値、差）
   rep_compare_video_imu_summary.csv  指標ごとの n・平均・差（バイアス）・SD・95 % 一致限界・r・倍率
   rep_compare_video_imu_sets.csv     1 行 = 1 セット（MCV の倍率、セット内の差の SD、速度低下の一致）
   fig_bland_altman_video_imu.png     Bland–Altman（上昇時間・MCV・ピーク・上昇の変位・深さ）
   fig_identity_video_imu.png         IMU と動画の散布図（y = x の線つき）
 注意: 1 人・1 セットの中の複数レップは独立ではないので、一致限界はやや狭めに出る（参考値）。

 使い方:
   python scripts\\compare_video_imu.py                  # 全被験者
   python scripts\\compare_video_imu.py --subject S001

 依存: numpy pandas scipy matplotlib（extract_features.py・quality_checks.py を使う）
=============================================================================
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import extract_features as ef                      # noqa: E402
from quality_checks import exclusion_reasons       # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
FEATURES_CSV = PROJECT_ROOT / "outputs" / "features_all.csv"
OUT_DIR = PROJECT_ROOT / "outputs"

ASC_FRAC = 0.05      # 上昇区間: 速度がピークのこの割合を超えている間（extract_features.concentric_phase と同じ）
ASC_MIN = 0.02       # 〃 の下限 [m/s]
STAND_PRE_S = 0.3    # 深さの基準（立位）: レップの始まりの直前この秒数の位置の中央値
WIN_POST_S = 0.8     # 上昇の終わりを探す余裕: レップの終わりのこの秒数後まで
PEAK_POST_S = 0.2    # 上昇のピークを探すのは レップの終わりのこの秒数後まで（レップの中に限る）。
                     # 広くすると、切っていない動画のラックインでのキーポイントの跳び（S002 set01: 0.1 s で 24 cm、
                     # 2.9 m/s）をピークとして拾った（2026-10-30）

# 指標: (列名, 表示名, 単位, IMU と動画の値に掛ける倍率)
METRICS = [
    ("ascent_s", "上昇時間", "s", 1.0),
    ("mcv", "MCV", "m/s", 1.0),
    ("peak", "ピーク速度", "m/s", 1.0),
    ("rom", "上昇の変位", "cm", 100.0),
    ("depth", "深さ", "cm", 100.0),
]
# 図に書く説明（何の秒数・速さ・距離か）
METRIC_DESC = {
    "ascent_s": "いちばん下から立ち上がりきるまでの秒数",
    "mcv": "立ち上がる間の平均の速さ（上向き）",
    "peak": "立ち上がる間でいちばん速い瞬間の速さ",
    "rom": "立ち上がる間に上がった距離",
    "depth": "しゃがむ前の立った位置から、いちばん下までの距離",
}
FIG_NOTE = ("図の見方\n"
            "・1 点 = 1 レップ（1 回のしゃがみ）\n"
            "・同じレップを 2 通りで測った:\n"
            "    動画 = 肩の点の上下の動き（身長から m に換算）\n"
            "    IMU = バーにつけたセンサー（基準）\n"
            "・縦軸 = 動画 − IMU（+ なら動画の方が大きい）\n"
            "・横軸 = その値の大きさ（2 つの平均）\n"
            "・実線 = ずれの平均（0 から離れるほど、\n"
            "    動画がいつも同じ向きにずれている）\n"
            "・破線 = 95 % のレップが入る範囲\n"
            "    （1 レップごとの誤差の幅）")


# ---------------------------------------------------------------------------
# 波形
# ---------------------------------------------------------------------------
def video_motion(kp_df, width, height, fps, height_cm, expected_reps=None):
    """
    動画の肩の上下の位置 z [m]（上が正、立位 = 0）と速度 v [m/s]。時刻 t は動画の先頭からの秒。
    体の大きさ BL と立位の高さは extract_features と同じく、セットの芯の区間で測る。
    """
    fs = ef._pose_fs(kp_df, fps)
    i0, i1, c0, c1 = ef._set_region(kp_df, width, height, fs, expected_reps)
    kp = kp_df.iloc[i0:i1].reset_index(drop=True)
    pts = ef._load_points(kp, width, height, fs, ef._other_person_frames(kp_df, width, height, fs)[i0:i1])
    sh_y = ef._nanmean([pts["left_shoulder"][1], pts["right_shoulder"][1]])
    an_y = ef._nanmean([pts["left_ankle"][1], pts["right_ankle"][1]])
    cs = slice(c0 - i0, c1 - i0)
    body_len = ef._nanstat(lambda a: np.percentile(a, 95), (an_y - sh_y)[cs])
    stand_y = ef._nanstat(lambda a: np.percentile(a, 5), sh_y[cs])
    if np.isnan(body_len) or body_len < 50 or not height_cm:
        return None
    m_per_px = ef.SHOULDER_ANKLE_RATIO * float(height_cm) / 100.0 / body_len
    z = pd.Series(-(sh_y - stand_y) * m_per_px).interpolate(limit_direction="both").to_numpy()
    return {"t": (i0 + np.arange(len(z))) / fs, "z": z, "v": np.gradient(z) * fs, "m_per_px": m_per_px}


def imu_motion(imu_df, windows):
    """IMU の速度 v [m/s]（extract_features と同じもの）と、その積分の位置 z [m]。時刻 t は IMU の秒"""
    m = ef.imu_rep_velocity(imu_df, rep_windows=windows)
    return {"t": m["t"], "z": ef.integrate_trapz(m["v"], m["dt"]), "v": m["v"]}


def rep_metrics(t, z, v, ts, te):
    """
    1 レップ（区間 [ts, te]、その波形の時刻）の上昇時間・MCV・ピーク・深さ。動画にも IMU にも同じ定義で当てる。
    区間が波形の外なら None。
    """
    n = len(t)
    lo, hi = int(np.searchsorted(t, ts - STAND_PRE_S)), int(np.searchsorted(t, te + WIN_POST_S))
    i_s, i_e = int(np.searchsorted(t, ts)), int(np.searchsorted(t, te))
    if t[0] > ts or t[-1] < te or i_e - i_s < 3:
        return None
    hi = min(hi, n)
    hp = min(hi, int(np.searchsorted(t, te + PEAK_POST_S)))
    b = i_s + int(np.nanargmin(z[i_s:i_e + 1]))                 # 底
    if hp - b < 3:
        return None
    p = b + int(np.nanargmax(v[b:hp]))                          # 上昇のピーク（レップの中）
    thr = max(ASC_MIN, ASC_FRAC * float(v[p]))
    s, e = p, p
    while s > b and v[s - 1] > thr:
        s -= 1
    while e < hi - 1 and v[e + 1] > thr:
        e += 1
    stand = float(np.nanmedian(z[lo:i_s + 1]))
    bottom = float(np.nanpercentile(z[i_s:i_e + 1], 1))
    return {"ascent_s": float(t[e] - t[s]), "mcv": float(np.nanmean(v[s:e + 1])),
            "peak": float(v[p]), "rom": float(z[e] - z[s]), "depth": stand - bottom,
            "bottom_t": float(t[b])}


def compare_set(feat: dict, kp_df, width, height, fps, imu_df, height_cm, expected_reps=None) -> list:
    """
    1 セットのレップごとの比較。feat は特徴量 JSON（sync_offset_s と _pose_reps を使う）。
    戻り値: [{rep, imu_*, vid_*, imu_*_json, vid_*_json}, ...]（比べられたレップだけ）
    """
    off = feat.get("sync_offset_s")
    pose_reps = feat.get("_pose_reps") or []
    if off is None or not pose_reps:
        return []
    vid = video_motion(kp_df, width, height, fps, height_cm, expected_reps)
    if vid is None:
        return []
    windows = [(r["start_t_s"] + off, r["bottom_t_s"] + off, r["end_t_s"] + off) for r in pose_reps]
    imu = imu_motion(imu_df, windows)
    imu_json = feat.get("_imu_reps") or []
    m_per_bl = ef.SHOULDER_ANKLE_RATIO * float(height_cm) / 100.0

    rows = []
    for i, r in enumerate(pose_reps):
        mv = rep_metrics(vid["t"], vid["z"], vid["v"], r["start_t_s"], r["end_t_s"])
        mi = rep_metrics(imu["t"], imu["z"], imu["v"], r["start_t_s"] + off, r["end_t_s"] + off)
        if mv is None or mi is None:
            continue
        row = {"rep": i + 1}
        for k, _, _, scale in METRICS:
            row[f"imu_{k}"] = mi[k] * scale
            row[f"vid_{k}"] = mv[k] * scale
            row[f"diff_{k}"] = (mv[k] - mi[k]) * scale
        row["bottom_gap_s"] = mi["bottom_t"] - off - mv["bottom_t"]   # 時刻合わせの確認（0 に近いはず）
        # 参考: 特徴量 JSON の値（区間の決め方が少し違う）
        row["vid_mcv_json"] = r.get("mcv_bl", np.nan) * m_per_bl if r.get("mcv_bl") is not None else np.nan
        row["imu_mcv_json"] = imu_json[i]["mcv"] if i < len(imu_json) and len(imu_json) == len(pose_reps) else np.nan
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
def agreement(a, b) -> dict:
    """b − a のバイアス・SD・95 % 一致限界・相関（a = IMU = 基準、b = 動画）"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return {"n": int(len(a))}
    d = b - a
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"n": int(len(a)), "imu_mean": float(a.mean()), "vid_mean": float(b.mean()),
            "bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "r": r, "ratio": float(b.mean() / a.mean()) if a.mean() != 0 else float("nan")}


def summarize(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, name, unit, _ in METRICS:
        rows.append({"metric": k, "name": name, "unit": unit, **agreement(reps[f"imu_{k}"], reps[f"vid_{k}"])})
    return pd.DataFrame(rows)


def per_set(reps: pd.DataFrame) -> pd.DataFrame:
    """セットごと: MCV の倍率（動画 / IMU）、セット内の差の SD、速度低下（1 回目 → 最後の MCV）の一致"""
    rows = []
    for key, g in reps.groupby(["subject_id", "session_id", "set_no"], sort=True):
        g = g.sort_values("rep")
        vl = lambda s: (s.iloc[0] - s.iloc[-1]) / s.iloc[0] * 100.0 if len(s) >= 2 and s.iloc[0] > 0 else np.nan
        rows.append({
            "subject_id": key[0], "session_id": key[1], "set_no": key[2], "date": g["date"].iloc[0],
            "rpe": g["rpe"].iloc[0], "n_reps": len(g),
            "mcv_ratio": g["vid_mcv"].mean() / g["imu_mcv"].mean(),
            "rom_ratio": g["vid_rom"].mean() / g["imu_rom"].mean(),
            "imu_rom_cm": g["imu_rom"].mean(), "vid_rom_cm": g["vid_rom"].mean(),
            "imu_depth_cm": g["imu_depth"].mean(), "vid_depth_cm": g["vid_depth"].mean(),
            "mcv_diff_mean": g["diff_mcv"].mean(),
            "mcv_diff_sd_within": g["diff_mcv"].std(ddof=1) if len(g) >= 2 else np.nan,
            "imu_vl_pct": vl(g["imu_mcv"]), "vid_vl_pct": vl(g["vid_mcv"]),
            "sync_r": g["sync_r"].iloc[0], "sync_margin": g["sync_margin"].iloc[0],
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["vl_diff_pt"] = out["vid_vl_pct"] - out["imu_vl_pct"]
    return out


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def _colors(reps):
    subs = sorted(reps["subject_id"].unique())
    return {s: f"C{i}" for i, s in enumerate(subs)}


def fig_bland_altman(reps, summ, out, title=None, note=FIG_NOTE, vid_name="動画"):
    """title・note・vid_name は compare_side_imu.py（横から撮った動画のプレート）が差し替える"""
    col = _colors(reps)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.02, 0.98, note, va="top", ha="left", fontsize=10, transform=axes.flat[-1].transAxes)
    for ax, (k, name, unit, _) in zip(axes.flat, METRICS):
        s = summ[summ["metric"] == k].iloc[0]
        for sid, g in reps.groupby("subject_id"):
            ax.scatter((g[f"imu_{k}"] + g[f"vid_{k}"]) / 2, g[f"diff_{k}"], s=16, alpha=0.75,
                       color=col[sid], label=sid)
        title = f"{name} [{unit}]: {METRIC_DESC[k]}"
        if s.get("n", 0) >= 2:
            for y, ls in ((s["bias"], "-"), (s["loa_low"], "--"), (s["loa_high"], "--")):
                ax.axhline(y, color="k", lw=0.9, ls=ls)
            title += (f"\nずれの平均 {s['bias']:+.3f} {unit}、95 % のレップは {s['loa_low']:+.3f}〜{s['loa_high']:+.3f}"
                      f"（n={int(s['n'])}）")
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel(f"その値の大きさ（IMU と{vid_name}の平均）[{unit}]")
        ax.set_ylabel(f"{vid_name} − IMU [{unit}]")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title or "動画（肩）と IMU（バー）のレップごとの一致（Bland–Altman 図、参考値）")
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_identity(reps, summ, out):
    col = _colors(reps)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes.flat[-1].axis("off")
    for ax, (k, name, unit, _) in zip(axes.flat, METRICS):
        s = summ[summ["metric"] == k].iloc[0]
        for sid, g in reps.groupby("subject_id"):
            ax.scatter(g[f"imu_{k}"], g[f"vid_{k}"], s=16, alpha=0.75, color=col[sid], label=sid)
        lo = np.nanmin(reps[[f"imu_{k}", f"vid_{k}"]].to_numpy())
        hi = np.nanmax(reps[[f"imu_{k}", f"vid_{k}"]].to_numpy())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        r_txt = f"r = {s['r']:.2f}" if s.get("n", 0) >= 2 and not np.isnan(s.get("r", np.nan)) else ""
        ax.set_title(f"{name} [{unit}]  {r_txt}\n{METRIC_DESC[k]}", fontsize=10)
        ax.set_xlabel(f"IMU [{unit}]")
        ax.set_ylabel(f"動画 [{unit}]")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("IMU と動画（破線: y = x、参考値）")
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def eligible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """時刻合わせが合い、IMU・動画のどちらも学習から外れていないセット"""
    keep = []
    for row in df.to_dict("records"):
        reasons = exclusion_reasons(row)
        keep.append(row.get("imu_rep_source") == "video" and not reasons["imu"] and not reasons["video"])
    return df[np.array(keep, dtype=bool)]


def main():
    ap = argparse.ArgumentParser(description="動画と IMU のレップ単位の一致度（RQ2）")
    ap.add_argument("--subject", type=str, default=None, help="例: S001（省略時は全員）")
    ap.add_argument("--input", type=str, default=str(FEATURES_CSV))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.subject:
        df = df[df["subject_id"] == args.subject]
    use = eligible_rows(df)
    print(f"[対象] {len(use)} / {len(df)} セット（時刻合わせが合い、IMU・動画とも学習に使えるもの）")

    trials = {(m.parent.parent.name, m.parent.name, s): (imu, kp, vid, m, out)
              for imu, kp, vid, m, s, out in ef.find_trials(filter_subject=args.subject)}
    all_rows = []
    for row in use.to_dict("records"):
        key = (row["subject_id"], row["session_id"], int(row["set_no"]))
        if key not in trials:
            print(f"  [SKIP] ファイルが見つかりません: {key}")
            continue
        imu_csv, kp_csv, video, meta_json, out_json = trials[key]
        feat = json.loads(out_json.read_text(encoding="utf-8"))
        prof = ef.load_subject_profile(row["subject_id"])
        if not prof.get("height_cm"):
            print(f"  [SKIP] {key}: subjects.json に身長がありません")
            continue
        w, h, fps = ef.load_video_info(kp_csv, video)
        rows = compare_set(feat, pd.read_csv(kp_csv), w, h, fps, pd.read_csv(imu_csv),
                           prof["height_cm"], expected_reps=row.get("reps_completed"))
        for r in rows:
            r.update({"subject_id": key[0], "session_id": key[1], "set_no": key[2], "date": row.get("date"),
                      "weight_kg": row.get("weight_kg"), "rpe": row.get("rpe"),
                      "sync_r": row.get("sync_r"), "sync_margin": row.get("sync_margin")})
        print(f"  {key[0]}/{key[1]} set{key[2]:02d}: {len(rows)} レップ（申告 {row.get('reps_completed')}）")
        all_rows += rows

    if not all_rows:
        sys.exit("[ERROR] 比べられるレップがありません")
    lead = ["subject_id", "session_id", "set_no", "rep", "date", "weight_kg", "rpe"]
    reps = pd.DataFrame(all_rows)
    reps = reps[lead + [c for c in reps.columns if c not in lead]]
    summ = summarize(reps)
    sets = per_set(reps)

    OUT_DIR.mkdir(exist_ok=True)
    reps.to_csv(OUT_DIR / "rep_compare_video_imu.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(OUT_DIR / "rep_compare_video_imu_summary.csv", index=False, encoding="utf-8-sig")
    sets.to_csv(OUT_DIR / "rep_compare_video_imu_sets.csv", index=False, encoding="utf-8-sig")
    fig_bland_altman(reps, summ, OUT_DIR / "fig_bland_altman_video_imu.png")
    fig_identity(reps, summ, OUT_DIR / "fig_identity_video_imu.png")

    print(f"\n[一致度] レップ {len(reps)}・セット {len(sets)}・被験者 {reps['subject_id'].nunique()} 人（参考値）")
    for s in summ.itertuples():
        if s.n >= 2:
            print(f"  {s.name:<6} IMU {s.imu_mean:.3f} / 動画 {s.vid_mean:.3f} {s.unit}  差 {s.bias:+.3f}"
                  f"（95 % 一致限界 {s.loa_low:+.3f}〜{s.loa_high:+.3f}）  r {s.r:.2f}  倍率 {s.ratio:.2f}")
    vl = sets.dropna(subset=["imu_vl_pct", "vid_vl_pct"])
    if len(sets):
        print(f"  MCV の倍率（セットごと）{sets['mcv_ratio'].min():.2f}〜{sets['mcv_ratio'].max():.2f}、"
              f"セット内の差の SD 中央値 {sets['mcv_diff_sd_within'].median():.3f} m/s")
    if len(vl) >= 2:
        a = agreement(vl["imu_vl_pct"], vl["vid_vl_pct"])
        print(f"  速度低下（{a['n']} セット）IMU {a['imu_mean']:.1f} / 動画 {a['vid_mean']:.1f} %  "
              f"差 {a['bias']:+.1f} ポイント（95 % 一致限界 {a['loa_low']:+.1f}〜{a['loa_high']:+.1f}）  r {a['r']:.2f}")
    print(f"  時刻合わせの確認: 底の時刻の差 中央値 {reps['bottom_gap_s'].median():+.3f} s"
          f"（{reps['bottom_gap_s'].min():+.2f}〜{reps['bottom_gap_s'].max():+.2f}）")
    print(f"\n[OK] {OUT_DIR / 'rep_compare_video_imu.csv'} ほか（summary / sets / 図 2 枚）")


if __name__ == "__main__":
    main()
