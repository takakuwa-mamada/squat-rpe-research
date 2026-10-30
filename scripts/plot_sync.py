# -*- coding: utf-8 -*-
"""
=============================================================================
 動画と IMU の時刻合わせの確認図（1 セット 1 枚）

 extract_features.sync_video_imu が選んだずれが本当に合っているかを目で確かめる。
   上段: ずれ [s] ごとの相関 r。選んだずれ（●）、1 秒以上離れたずれでいちばん高い相関（▲）。
         ● − ▲ が margin。r ≥ 0.6 かつ margin ≥ 0.05 のときだけ時刻合わせを使う
   中段: 速度 [m/s]。動画（肩、選んだずれで IMU の時刻に移したもの）と IMU（時刻合わせに使う v_hp と、
         レップごとにロックアウトで補正した速度）。橙の帯 = 動画で数えたレップの区間
   下段: 位置 [m]（立位 = 0）。動画の肩と、IMU の補正後の速度の積分。比較スクリプトの深さの外れ値の確認用
 時刻合わせを使わなかったセット（imu_rep_source = imu）も、いちばん相関の高いずれで描く（題に「使わない」）。

 出力: outputs/sync_plots/<被験者>_<セッション>_setNN.png

 使い方:
   python scripts\\plot_sync.py                  # 全被験者
   python scripts\\plot_sync.py --subject S002

 依存: numpy pandas matplotlib（extract_features.py・compare_video_imu.py を使う）
=============================================================================
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import extract_features as ef          # noqa: E402
import compare_video_imu as cv         # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
OUT_DIR = PROJECT_ROOT / "outputs" / "sync_plots"


def sync_curve(kp_df, w, h, fps, imu_df, t_range):
    """sync_video_imu と同じ入力で、ずれごとの相関を返す: (ずれ [s], r)"""
    fs_v = ef._pose_fs(kp_df, fps)
    t_v, v_v = ef._shoulder_up_velocity(kp_df, w, h, fs_v)
    if t_v is None:
        return None, None
    if t_range is not None and None not in t_range:
        sel = (t_v >= t_range[0]) & (t_v <= t_range[1])
        if sel.sum() >= 10:
            t_v, v_v = t_v[sel], v_v[sel]
    u = ef.imu_on_device_clock(imu_df)
    t_i = u["timestamp_ms"].to_numpy() / 1000.0
    dt = float(np.median(np.diff(t_i)))
    _, _, _, _, v_hp, _ = ef.imu_vertical_velocity(u, 1.0 / dt, dt)
    return ef.offset_curve(t_v, v_v, t_i, v_hp)


def imu_hp(imu_df):
    """時刻合わせに使う IMU の速度 v_hp（M5 の時計）"""
    u = ef.imu_on_device_clock(imu_df)
    t = u["timestamp_ms"].to_numpy() / 1000.0
    dt = float(np.median(np.diff(t)))
    _, _, _, _, v_hp, _ = ef.imu_vertical_velocity(u, 1.0 / dt, dt)
    return t, v_hp


def plot_one(key, feat, kp_df, w, h, fps, imu_df, height_cm, expected_reps, out_png):
    shifts, rs = sync_curve(kp_df, w, h, fps, imu_df, (feat.get("pose_set_start_s"), feat.get("pose_set_end_s")))
    if shifts is None:
        return None
    i = int(np.argmax(rs))
    off = float(shifts[i])
    far = np.abs(shifts - off) > 1.0 + 0.5 / ef.SYNC_RATE_HZ
    j = int(np.argmax(np.where(far, rs, -np.inf))) if far.any() else None
    used = feat.get("imu_rep_source") == "video"

    vid = cv.video_motion(kp_df, w, h, fps, height_cm, expected_reps)
    pose_reps = feat.get("_pose_reps") or []
    windows = [(r["start_t_s"] + off, r["bottom_t_s"] + off, r["end_t_s"] + off) for r in pose_reps]
    imu = cv.imu_motion(imu_df, windows if used else None)
    t_hp, v_hp = imu_hp(imu_df)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax = axes[0]
    ax.plot(shifts, rs, lw=0.8, color="C0")
    ax.axhline(ef.SYNC_R_MIN, color="gray", ls="--", lw=0.8)
    ax.plot(off, rs[i], "o", color="C3", label=f"選んだずれ {off:+.2f} s（r {rs[i]:.2f}）")
    if j is not None:
        ax.plot(shifts[j], rs[j], "^", color="C1",
                label=f"1 秒以上離れた最大 {shifts[j]:+.2f} s（r {rs[j]:.2f}、margin {rs[i] - rs[j]:.2f}）")
    ax.set_xlabel("ずれ [s]（IMU の時刻 = 動画の時刻 + ずれ）")
    ax.set_ylabel("相関 r")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_hp, v_hp, lw=0.7, color="gray", label="IMU（時刻合わせに使う v_hp）")
    ax.plot(imu["t"], imu["v"], lw=1.0, color="C0", label="IMU（ロックアウトで補正）")
    if vid is not None:
        ax.plot(vid["t"] + off, vid["v"], lw=1.0, color="C3", label="動画（肩、ずれを足した時刻）")
    for ws, _, we in windows:
        ax.axvspan(ws, we, color="orange", alpha=0.15)
    ax.set_ylabel("速度 [m/s]（上が正）")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    z_imu = ef.integrate_trapz(imu["v"], float(np.median(np.diff(imu["t"]))))
    if windows:   # 積分は記録の先頭（歩き出しの前）から始まるので、1 回目の直前 0.5 s の立位を 0 にそろえる
        pre = (imu["t"] >= windows[0][0] - 0.5) & (imu["t"] <= windows[0][0])
        if pre.any():
            z_imu = z_imu - float(np.median(z_imu[pre]))
    ax.plot(imu["t"], z_imu, lw=1.0, color="C0", label="IMU（補正後の速度の積分）")
    if vid is not None:
        ax.plot(vid["t"] + off, vid["z"], lw=1.0, color="C3", label="動画（肩）")
    for ws, _, we in windows:
        ax.axvspan(ws, we, color="orange", alpha=0.15)
    ax.set_ylabel("位置 [m]（立位 = 0）")
    ax.set_xlabel("IMU の時刻 [s]（SPACE から）")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    # 中段・下段はセットの区間（＋前後 3 s）に絞る
    if windows:
        for a in axes[1:]:
            a.set_xlim(windows[0][0] - 3.0, windows[-1][2] + 3.0)

    tag = "使う" if used else "使わない（IMU だけでレップを数えた）"
    fig.suptitle(f"{key[0]}/{key[1]} set{key[2]:02d}  RPE {feat.get('rpe')}  {feat.get('weight_kg')} kg × "
                 f"{feat.get('reps_completed')}  時刻合わせ: r {feat.get('sync_r')}・margin {feat.get('sync_margin')} → {tag}")
    plt.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"set": f"{key[0]}/{key[1]}/set{key[2]:02d}", "offset_s": round(off, 2), "r": round(float(rs[i]), 3),
            "second_s": round(float(shifts[j]), 2) if j is not None else None,
            "margin": round(float(rs[i] - rs[j]), 3) if j is not None else None, "used": used}


def main():
    ap = argparse.ArgumentParser(description="動画と IMU の時刻合わせの確認図")
    ap.add_argument("--subject", type=str, default=None, help="例: S001（省略時は全員）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for imu_csv, kp_csv, video, meta_json, set_no, out_json in ef.find_trials(filter_subject=args.subject):
        key = (meta_json.parent.parent.name, meta_json.parent.name, set_no)
        if imu_csv is None or not imu_csv.exists() or kp_csv is None or not out_json.exists():
            continue
        feat = json.loads(out_json.read_text(encoding="utf-8"))
        prof = ef.load_subject_profile(key[0])
        w, h, fps = ef.load_video_info(kp_csv, video)
        if not w or not prof.get("height_cm"):
            print(f"  [SKIP] {key}: 動画の大きさか身長がありません")
            continue
        out_png = OUT_DIR / f"{key[0]}_{key[1]}_set{set_no:02d}.png"
        r = plot_one(key, feat, pd.read_csv(kp_csv), w, h, fps, pd.read_csv(imu_csv), prof["height_cm"],
                     feat.get("reps_completed"), out_png)
        if r is None:
            print(f"  [SKIP] {key}: 波形を重ねられません")
            continue
        rows.append(r)
        print(f"  {r['set']}: ずれ {r['offset_s']:+.2f} s  r {r['r']:.2f}  margin {r['margin']}  "
              f"（2 番目 {r['second_s']} s）  {'使う' if r['used'] else '使わない'}")
    print(f"\n[OK] {OUT_DIR} に {len(rows)} 枚")


if __name__ == "__main__":
    main()
