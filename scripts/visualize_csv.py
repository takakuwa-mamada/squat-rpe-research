# -*- coding: utf-8 -*-
"""
=============================================================================
 計測CSVの可視化スクリプト
 ・記録済みCSV（YYYYMMDD_HHMMSS.csv）を読み込み、以下を可視化する
   1. 加速度 ax, ay, az の時系列
   2. 角速度 gx, gy, gz の時系列
   3. 鉛直方向 az から計算した「速度」（1回積分）
   4. 速度から計算した「位置（バーの上下動）」（2回積分 + ドリフト補正）
   5. 角速度の積分から計算した「姿勢角」（バーの傾き）
   6. レップ単位の自動検出と各レップの統計値
 ・使い方:
     python visualize_csv.py                      # 最新CSVを自動選択
     python visualize_csv.py 20260519_151023.csv  # ファイル指定
=============================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

G = 9.80665                                  # 重力加速度 [m/s^2]
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CSV選択
# ---------------------------------------------------------------------------
def select_csv():
    if len(sys.argv) >= 2:
        target = sys.argv[1]
        path = target if os.path.isabs(target) else os.path.join(DATA_DIR, target)
        if not os.path.exists(path):
            sys.exit(f"[ERROR] ファイルが見つかりません: {path}")
        return path
    # 指定なし → 最新のYYYYMMDD_HHMMSS.csv
    cands = sorted(glob.glob(os.path.join(DATA_DIR, "20*_*.csv")))
    if not cands:
        sys.exit(f"[ERROR] {DATA_DIR} にCSVがありません")
    return cands[-1]


# ---------------------------------------------------------------------------
# 信号処理ユーティリティ
# ---------------------------------------------------------------------------
def highpass(x, fs, fc=0.3, order=2):
    """ ドリフト除去用ハイパスフィルタ """
    b, a = butter(order, fc / (fs / 2.0), btype="high")
    return filtfilt(b, a, x)


def lowpass(x, fs, fc=10.0, order=4):
    """ ノイズ除去用ローパスフィルタ """
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def integrate(x, dt):
    """ 累積台形積分 """
    return np.concatenate([[0.0], np.cumsum((x[:-1] + x[1:]) / 2.0 * dt)])


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    csv_path = select_csv()
    print(f"[INFO] 読み込み: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"timestamp_ms", "ax_g", "ay_g", "az_g",
                "gx_dps", "gy_dps", "gz_dps"}
    if not required.issubset(df.columns):
        sys.exit(f"[ERROR] 必要な列が足りません: {required - set(df.columns)}")

    n = len(df)
    if n < 50:
        sys.exit(f"[ERROR] サンプル数が少なすぎます: n={n}")

    t = df["timestamp_ms"].to_numpy() / 1000.0    # [s]
    # サンプリング周波数の推定（中央値で安定化）
    dt_arr = np.diff(t)
    dt_med = float(np.median(dt_arr))
    fs = 1.0 / dt_med
    print(f"[INFO] サンプル数 n={n}, 推定サンプリング周波数 fs={fs:.1f} Hz")

    ax = df["ax_g"].to_numpy()
    ay = df["ay_g"].to_numpy()
    az = df["az_g"].to_numpy()
    gx = df["gx_dps"].to_numpy()
    gy = df["gy_dps"].to_numpy()
    gz = df["gz_dps"].to_numpy()

    # ---- 速度・位置の推定 -------------------------------------------------
    # az から重力を除いた動的加速度 [m/s^2]
    a_dyn = (az - 1.0) * G
    # ノイズ除去（10Hzローパス）
    a_dyn_lp = lowpass(a_dyn, fs, fc=10.0)
    # 1回積分 → 速度
    v_raw = integrate(a_dyn_lp, dt_med)
    # ハイパスでドリフト除去
    v = highpass(v_raw, fs, fc=0.3) if n > int(fs * 3) else v_raw
    # 2回積分 → 位置
    p_raw = integrate(v, dt_med)
    p = highpass(p_raw, fs, fc=0.3) if n > int(fs * 3) else p_raw

    # ---- 姿勢角（角速度の積分） -------------------------------------------
    # deg/s を そのまま [s] で積分 → [deg]
    roll  = integrate(gx, dt_med)
    pitch = integrate(gy, dt_med)
    yaw   = integrate(gz, dt_med)

    # ---- レップ検出（速度のゼロクロス + ピーク） --------------------------
    # 立ち上がり（上向き速度）のピークを「レップの上昇相」と見なす
    # 閾値: 最大値の30%以上
    if np.max(np.abs(v)) > 0.05:
        thresh = max(0.1, 0.3 * np.max(np.abs(v)))
        peaks, _ = find_peaks(v, height=thresh, distance=int(fs * 0.6))
    else:
        peaks = np.array([], dtype=int)

    # 各レップの統計
    rep_stats = []
    for i, p_idx in enumerate(peaks, start=1):
        # レップ範囲: ピーク前後 ±0.5秒くらいを取る（簡易）
        i0 = max(0, p_idx - int(fs * 0.7))
        i1 = min(n - 1, p_idx + int(fs * 0.5))
        seg_v = v[i0:i1]
        seg_a = a_dyn_lp[i0:i1]
        rep_stats.append({
            "rep": i,
            "t_peak": t[p_idx],
            "peak_v": float(v[p_idx]),
            "mean_v": float(np.mean(seg_v[seg_v > 0])) if np.any(seg_v > 0) else 0.0,
            "rms_a":  float(np.sqrt(np.mean(seg_a ** 2))),
            "duration": t[i1] - t[i0],
        })

    print(f"[INFO] 検出されたレップ数: {len(rep_stats)}")
    if rep_stats:
        print("\n  rep | t_peak[s] | peak_v[m/s] | mean_v[m/s] | RMS_a[m/s^2] | dur[s]")
        print("  " + "-" * 70)
        for r in rep_stats:
            print(f"  {r['rep']:>3} | {r['t_peak']:>9.2f} | {r['peak_v']:>+11.3f} | "
                  f"{r['mean_v']:>+11.3f} | {r['rms_a']:>12.3f} | {r['duration']:>6.2f}")
        print()

    # =====================================================================
    # 可視化
    # =====================================================================
    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(f"Squat IMU Analysis  -  {os.path.basename(csv_path)}",
                 fontsize=13, fontweight="bold")

    # (1) 加速度 3軸
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, ax, lw=0.8, label="ax", alpha=0.7)
    ax1.plot(t, ay, lw=0.8, label="ay", alpha=0.7)
    ax1.plot(t, az, lw=1.2, label="az", color="C3")
    ax1.set_ylabel("acc [g]")
    ax1.set_title("Acceleration (3 axes)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", ncol=3)

    # (2) 角速度 3軸
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(t, gx, lw=0.8, label="gx (roll rate)",  alpha=0.7)
    ax2.plot(t, gy, lw=0.8, label="gy (pitch rate)", alpha=0.7)
    ax2.plot(t, gz, lw=0.8, label="gz (yaw rate)",   alpha=0.7)
    ax2.set_ylabel("gyro [deg/s]")
    ax2.set_title("Angular velocity (3 axes)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", ncol=3)

    # (3) 速度（鉛直方向）+ レップマーカー
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(t, v, lw=1.2, color="C2", label="vertical velocity")
    ax3.axhline(0, color="gray", lw=0.5)
    for r in rep_stats:
        ax3.axvline(r["t_peak"], color="red", lw=0.6, alpha=0.6, linestyle="--")
        ax3.annotate(f"#{r['rep']}\n{r['peak_v']:+.2f}",
                     xy=(r["t_peak"], r["peak_v"]),
                     xytext=(0, 8), textcoords="offset points",
                     fontsize=8, ha="center", color="red")
    ax3.set_ylabel("velocity [m/s]")
    ax3.set_title("Vertical velocity (integrated, drift-corrected) — red dashed = detected rep peaks")
    ax3.grid(alpha=0.3)

    # (4) 位置（バー上下動）
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(t, p * 100.0, lw=1.2, color="C4", label="vertical position [cm]")
    ax4.axhline(0, color="gray", lw=0.5)
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("position [cm]")
    ax4.set_title("Vertical bar displacement (double-integrated, drift-corrected — accuracy degrades over long durations)")
    ax4.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    # =====================================================================
    # 別ウィンドウ: 姿勢角
    # =====================================================================
    fig2 = plt.figure(figsize=(12, 4))
    fig2.suptitle(f"Bar orientation (integrated gyro)  -  {os.path.basename(csv_path)}",
                  fontsize=12)
    plt.plot(t, roll,  lw=1.0, label="roll  (∫gx)")
    plt.plot(t, pitch, lw=1.0, label="pitch (∫gy)")
    plt.plot(t, yaw,   lw=1.0, label="yaw   (∫gz)")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # =====================================================================
    # 別ウィンドウ: 速度-加速度 位相プロット（おまけ）
    # =====================================================================
    if len(rep_stats) > 0:
        fig3 = plt.figure(figsize=(6, 6))
        fig3.suptitle(f"Phase plot: velocity vs acceleration  -  {os.path.basename(csv_path)}",
                      fontsize=12)
        plt.plot(v, a_dyn_lp, lw=0.7, alpha=0.7)
        plt.axhline(0, color="gray", lw=0.4)
        plt.axvline(0, color="gray", lw=0.4)
        plt.xlabel("vertical velocity [m/s]")
        plt.ylabel("vertical acceleration [m/s^2]")
        plt.grid(alpha=0.3)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
