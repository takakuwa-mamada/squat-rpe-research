# -*- coding: utf-8 -*-
"""
=============================================================================
 パワーリフティング スクワット計測システム（受信側）
 M5StickC Plus からのIMUデータをUDPで受信し、
   1. az（鉛直方向加速度）をリアルタイムプロット
   2. 挙上速度（az積分）をリアルタイム計算・画面表示
   3. RMS加速度をリアルタイム計算・画面表示
   4. 受信データを CSV に自動保存
   5. |az| が閾値を超えたら「試技開始」「試技終了」を判定
      終了時にピーク速度・平均速度・RMSをターミナルに出力

 著者: masaki（大学院修士研究 - RPE推定）
 動作環境: Python 3.x （macOS / Windows どちらでもOK）
 必要ライブラリ: numpy, matplotlib
   pip install numpy matplotlib
=============================================================================
"""

import socket
import csv
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================== ユーザ設定 =====================================
UDP_IP        = "0.0.0.0"     # 全インターフェースで受信
UDP_PORT      = 5005          # M5側 PC_PORT と一致させる
SAMPLE_HZ     = 100           # 想定サンプリング周波数 [Hz]
PLOT_WINDOW_S = 8             # プロットに表示する時間幅 [秒]
THRESH_G      = 0.3           # 試技判定の閾値 [g]（|az - 1g| ではなく |az| そのもの）
G             = 9.80665       # 重力加速度 [m/s^2]
SAVE_DIR      = os.path.dirname(os.path.abspath(__file__))  # CSV保存先（このスクリプトと同じフォルダ）
# ============================================================================


# ---------------------------------------------------------------------------
# 共有データ（受信スレッド <-> プロットスレッド）
# ---------------------------------------------------------------------------
BUFFER_LEN = SAMPLE_HZ * PLOT_WINDOW_S          # リングバッファ長
ts_buf = deque(maxlen=BUFFER_LEN)               # 時刻 [s]
az_buf = deque(maxlen=BUFFER_LEN)               # az [g]
ax_buf = deque(maxlen=BUFFER_LEN)               # ax [g]
ay_buf = deque(maxlen=BUFFER_LEN)               # ay [g]
buf_lock = threading.Lock()

# 試技ごとの一時保存（積分や統計算出用）
class TrialState:
    def __init__(self):
        self.in_trial = False                   # 現在試技中か
        self.start_ts = None                    # 試技開始時刻 [s]
        self.az_samples = []                    # 試技中の az [g]
        self.v_samples  = []                    # 試技中の速度 [m/s]
        self.peak_v = 0.0
        self.last_t = None                      # 直前サンプルの時刻
        self.velocity = 0.0                     # 現在の速度（az積分） [m/s]
        # 全期間にわたるRMS（プロット下の数値表示用）
        self.rms_all = 0.0

trial = TrialState()
trial_lock = threading.Lock()

# CSV書き出し
csv_filename = datetime.now().strftime("%Y%m%d_%H%M%S.csv")
csv_path = os.path.join(SAVE_DIR, csv_filename)
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp_ms", "ax_g", "ay_g", "az_g",
                     "gx_dps", "gy_dps", "gz_dps"])
csv_lock = threading.Lock()

stop_flag = threading.Event()


# ---------------------------------------------------------------------------
# UDP 受信スレッド
# ---------------------------------------------------------------------------
def udp_receiver():
    """ UDPでCSV1行を受信し、バッファ追加・CSV書き出し・試技判定を行う """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except OSError as e:
        print(f"[FATAL] UDPポート {UDP_PORT} のバインドに失敗: {e}")
        print("       他のアプリで同じポートを使っていないか確認してください。")
        stop_flag.set()
        return

    sock.settimeout(1.0)
    print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")
    print(f"[CSV] saving to {csv_path}")

    while not stop_flag.is_set():
        try:
            data, _addr = sock.recvfrom(1024)
        except socket.timeout:
            continue
        except OSError:
            break

        line = data.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        # CSV1行 = timestamp_ms, ax, ay, az, gx, gy, gz
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            ts_ms = float(parts[0])
            ax = float(parts[1]); ay = float(parts[2]); az = float(parts[3])
            gx = float(parts[4]); gy = float(parts[5]); gz = float(parts[6])
        except ValueError:
            continue

        # CSV保存
        with csv_lock:
            csv_writer.writerow([int(ts_ms), ax, ay, az, gx, gy, gz])

        t_s = ts_ms / 1000.0

        # リングバッファに追加
        with buf_lock:
            ts_buf.append(t_s)
            ax_buf.append(ax)
            ay_buf.append(ay)
            az_buf.append(az)

        # 試技判定・速度積分
        update_trial(t_s, ax, ay, az)


# ---------------------------------------------------------------------------
# 試技判定 & 速度積分
# ---------------------------------------------------------------------------
def update_trial(t_s, ax, ay, az):
    """
    |az| が閾値を超えたら試技開始、下回ったら試技終了とする。
    速度は az を時間積分（重力1gを引いた値 × G）して計算する。
    """
    global trial
    with trial_lock:
        # 全期間のRMS（プロット下に表示） … 直近バッファから計算
        with buf_lock:
            if len(az_buf) > 0:
                az_arr = np.array(az_buf)
                trial.rms_all = float(np.sqrt(np.mean(az_arr ** 2)))

        # az の絶対値（重力1gを差し引いた "動き" 成分） [g]
        az_dyn = az - 1.0
        moving = abs(az_dyn) > THRESH_G

        # --- 試技開始判定 ---
        if not trial.in_trial and moving:
            trial.in_trial = True
            trial.start_ts = t_s
            trial.az_samples = []
            trial.v_samples = []
            trial.peak_v = 0.0
            trial.velocity = 0.0
            trial.last_t = t_s
            print(f"[TRIAL] >>> 試技開始  t={t_s:.2f}s")

        # --- 試技中の処理 ---
        if trial.in_trial:
            # 速度積分（台形近似）
            if trial.last_t is not None:
                dt = t_s - trial.last_t
                if dt > 0:
                    # 重力成分を引いて m/s^2 に変換
                    a_ms2 = az_dyn * G
                    trial.velocity += a_ms2 * dt
            trial.last_t = t_s

            trial.az_samples.append(az_dyn)
            trial.v_samples.append(trial.velocity)
            if abs(trial.velocity) > abs(trial.peak_v):
                trial.peak_v = trial.velocity

        # --- 試技終了判定 ---
        # 動きが止まって 0.3秒経過したら終了とみなす
        if trial.in_trial and not moving:
            if (t_s - trial.last_t) >= 0.3 or len(trial.az_samples) > SAMPLE_HZ * 10:
                # 終了処理
                az_arr = np.array(trial.az_samples) if trial.az_samples else np.array([0.0])
                v_arr  = np.array(trial.v_samples)  if trial.v_samples  else np.array([0.0])
                rms = float(np.sqrt(np.mean(az_arr ** 2))) * G  # [m/s^2]
                avg_v = float(np.mean(v_arr))
                peak_v = float(trial.peak_v)
                duration = t_s - trial.start_ts
                print("[TRIAL] <<< 試技終了")
                print(f"        duration     : {duration:.2f} s")
                print(f"        peak velocity: {peak_v:+.3f} m/s")
                print(f"        avg  velocity: {avg_v:+.3f} m/s")
                print(f"        RMS accel    : {rms:.3f} m/s^2")
                print("-" * 50)
                trial.in_trial = False
                trial.velocity = 0.0
                trial.last_t = None


# ---------------------------------------------------------------------------
# matplotlib リアルタイムプロット
# ---------------------------------------------------------------------------
def make_plot():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.canvas.manager.set_window_title("Squat IMU Live (az)")
    (line_az,) = ax.plot([], [], lw=1.5, label="az [g]")
    ax.axhline(1.0 + THRESH_G, color="red", linestyle="--", lw=0.8, label="threshold")
    ax.axhline(1.0 - THRESH_G, color="red", linestyle="--", lw=0.8)
    ax.set_xlim(0, PLOT_WINDOW_S)
    ax.set_ylim(-2.0, 4.0)
    ax.set_xlabel("time [s] (recent window)")
    ax.set_ylabel("acceleration [g]")
    ax.set_title("M5StickC Plus - Vertical Acceleration")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    text_status = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    def update(_frame):
        with buf_lock:
            if len(ts_buf) < 2:
                return line_az, text_status
            ts = np.array(ts_buf)
            az = np.array(az_buf)

        t0 = ts[-1] - PLOT_WINDOW_S
        ax.set_xlim(max(0, t0), max(PLOT_WINDOW_S, ts[-1]))
        line_az.set_data(ts, az)

        with trial_lock:
            status = "TRIAL" if trial.in_trial else "idle "
            v_now  = trial.velocity
            rms_all = trial.rms_all
        text_status.set_text(
            f"status  : {status}\n"
            f"velocity: {v_now:+.3f} m/s\n"
            f"RMS(az) : {rms_all:.3f} g\n"
            f"samples : {len(ts_buf)}"
        )
        return line_az, text_status

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return ani


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Squat IMU UDP Receiver")
    print(f"  UDP        : {UDP_IP}:{UDP_PORT}")
    print(f"  CSV出力先   : {csv_path}")
    print(f"  閾値        : |az-1g| > {THRESH_G} g で試技と判定")
    print("  停止: グラフウィンドウを閉じる or Ctrl-C")
    print("=" * 60)

    th = threading.Thread(target=udp_receiver, daemon=True)
    th.start()

    try:
        make_plot()   # プロットを閉じるまでブロック
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C 検出")
    finally:
        stop_flag.set()
        time.sleep(0.3)
        with csv_lock:
            csv_file.flush()
            csv_file.close()
        print(f"[INFO] CSV保存完了: {csv_path}")


if __name__ == "__main__":
    main()
