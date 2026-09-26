# -*- coding: utf-8 -*-
"""
=============================================================================
 マルチIMU 受信・記録スクリプト

 複数の M5StickC Plus (m5stick_multi_sender.ino) から送られてくる
 IMU データを同時に受信し、デバイスごとに CSV へ保存する。

 【設計方針】
  ・デバイス側は常時ストリーミング。計測の開始/停止は PC 側で制御する。
    → 5台のボタンを押して回る必要がない。
  ・セッション全体を連続記録し、セットの区切りは「マーカー」で記録する。
    → 取りこぼしが起きない。セット分割は後処理で行う。
  ・デバイス間の時刻同期は PC 側の受信時刻 (recv_time_s) を基準とする。
  ・PC の居場所を知らせる合図（ビーコン）を UDP 5006 に1秒ごとにブロードキャストする。
    ファーム v4 はこれを受け取って送信先を自動で決めるので、テザリング等で
    PC の IP が変わってもファームを書き直さなくてよい（--no-beacon で無効）。

 【受信フォーマット】
    DEVICE_ID,boot_ms,ax,ay,az,gx,gy,gz

 【操作方法】（グラフウィンドウにフォーカスを当てて操作）
    SPACE : セット開始 / セット終了 をトグル
    6〜9  : 直前に終了したセットに RPE を付与（6,7,8,9）
    0     : 同上で RPE 10
    5     : 同上で RPE 5
    u     : 直前のマーカーを取り消し
    q     : 終了

    ※ RPE 8.5 などの中間値は、生成された markers.csv を後から編集する

 【出力】
    data/_sessions/session_YYYYMMDD_HHMMSS/
        raw_BAR.csv       … デバイスごとの連続データ
        raw_TRUNK.csv
        ...
        markers.csv       … set_no, start_s, end_s, rpe
        session_info.json … 受信統計・デバイス一覧

 使い方:
    python multi_imu_receiver.py
    python multi_imu_receiver.py --port 5005
    python multi_imu_receiver.py --outdir data/_sessions

 依存ライブラリ:
    pip install numpy matplotlib
=============================================================================
"""

import os
import sys
import csv
import json
import socket
import argparse
import threading
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR

# =========================== 設定 ===========================================
UDP_IP        = "0.0.0.0"
UDP_PORT      = 5005
BEACON_PORT   = 5006        # PC の居場所を M5 に知らせる合図の送信先ポート
BEACON_MSG    = b"SQUATPC"
PLOT_WINDOW_S = 10.0        # プロットに表示する時間幅 [秒]
EXPECTED_HZ   = 100         # 想定サンプリング周波数（統計表示用）

# デバイスごとの表示色（未登録のIDは自動で色を割り当てる）
DEVICE_COLORS = {
    "BAR":    "#065A82",
    "TRUNK":  "#E76F51",
    "PELVIS": "#2A9D8F",
    "THIGH":  "#9B5DE5",
    "SHANK":  "#F4A261",
}
FALLBACK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                   "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
# ============================================================================


class SessionRecorder:
    """デバイスごとの CSV 書き出しと、セットマーカーの管理"""

    CSV_HEADER = ["recv_time_s", "boot_ms",
                  "ax_g", "ay_g", "az_g",
                  "gx_dps", "gy_dps", "gz_dps"]

    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = {}         # device_id -> file object
        self._writers = {}       # device_id -> csv writer
        self._lock = threading.Lock()
        self.counts = defaultdict(int)
        self.first_recv = {}
        self.last_recv = {}

        # マーカー
        self.markers = []        # [{set_no, start_s, end_s, rpe}]
        self.recording_set = False
        self.current_start = None
        self.set_no = 0

        self.t0 = time.time()    # セッション基準時刻

    # ---- データ書き込み ----
    def write(self, device_id, recv_time_s, boot_ms, vals):
        with self._lock:
            if device_id not in self._writers:
                path = self.dir / f"raw_{device_id}.csv"
                f = open(path, "w", newline="", encoding="utf-8")
                w = csv.writer(f)
                w.writerow(self.CSV_HEADER)
                self._files[device_id] = f
                self._writers[device_id] = w
                print(f"  [NEW DEVICE] {device_id}  -> {path.name}")
                self.first_recv[device_id] = recv_time_s

            self._writers[device_id].writerow(
                [f"{recv_time_s:.4f}", boot_ms] + [f"{v:.5f}" for v in vals]
            )
            self.counts[device_id] += 1
            self.last_recv[device_id] = recv_time_s

    # ---- マーカー操作 ----
    def toggle_set(self, now_s):
        if not self.recording_set:
            self.set_no += 1
            self.current_start = now_s
            self.recording_set = True
            return ("start", self.set_no, now_s)
        else:
            self.markers.append({
                "set_no": self.set_no,
                "start_s": round(self.current_start, 4),
                "end_s": round(now_s, 4),
                "duration_s": round(now_s - self.current_start, 2),
                "rpe": None,
            })
            self.recording_set = False
            dur = now_s - self.current_start
            return ("end", self.set_no, dur)

    def set_rpe(self, rpe):
        if not self.markers:
            return None
        self.markers[-1]["rpe"] = rpe
        return self.markers[-1]["set_no"]

    def undo_last(self):
        if self.recording_set:
            self.recording_set = False
            self.set_no -= 1
            return ("cancel_start", None)
        if self.markers:
            m = self.markers.pop()
            self.set_no -= 1
            return ("remove", m["set_no"])
        return (None, None)

    # ---- 終了処理 ----
    def close(self):
        with self._lock:
            for f in self._files.values():
                f.flush()
                f.close()

        # markers.csv
        mpath = self.dir / "markers.csv"
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["set_no", "start_s", "end_s", "duration_s", "rpe"])
            for m in self.markers:
                w.writerow([m["set_no"], m["start_s"], m["end_s"],
                            m["duration_s"], m["rpe"] if m["rpe"] is not None else ""])

        # session_info.json
        info = {
            "session_dir": self.dir.name,
            "started_at": datetime.fromtimestamp(self.t0).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(time.time() - self.t0, 1),
            "devices": {},
            "n_sets": len(self.markers),
        }
        for dev, n in self.counts.items():
            span = self.last_recv.get(dev, 0) - self.first_recv.get(dev, 0)
            info["devices"][dev] = {
                "n_samples": n,
                "duration_s": round(span, 1),
                "mean_hz": round(n / span, 1) if span > 0.5 else None,
            }
        with open(self.dir / "session_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        return info, mpath


# ---------------------------------------------------------------------------
# 受信スレッド
# ---------------------------------------------------------------------------
class Receiver(threading.Thread):
    def __init__(self, recorder, buffers, buf_lock, port, stop_flag):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.buffers = buffers
        self.buf_lock = buf_lock
        self.port = port
        self.stop_flag = stop_flag
        self.bad_lines = 0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((UDP_IP, self.port))
        except OSError as e:
            print(f"[FATAL] ポート {self.port} をバインドできません: {e}")
            print("        他のアプリが使用中でないか確認してください。")
            self.stop_flag.set()
            return
        sock.settimeout(0.5)
        print(f"[UDP] listening on {UDP_IP}:{self.port}")

        t0 = self.recorder.t0
        while not self.stop_flag.is_set():
            try:
                data, _addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            recv_time_s = time.time() - t0

            # 1パケットに複数行入っている場合にも対応
            for line in data.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 8:
                    self.bad_lines += 1
                    continue
                device_id = parts[0].strip()
                try:
                    boot_ms = int(float(parts[1]))
                    vals = [float(x) for x in parts[2:8]]
                except ValueError:
                    self.bad_lines += 1
                    continue

                self.recorder.write(device_id, recv_time_s, boot_ms, vals)

                with self.buf_lock:
                    b = self.buffers[device_id]
                    b["t"].append(recv_time_s)
                    b["az"].append(vals[2])

        sock.close()


# ---------------------------------------------------------------------------
# ビーコン（PC の居場所を M5 に知らせる）
# ---------------------------------------------------------------------------
def broadcast_addresses():
    """ブロードキャスト先: 255.255.255.255 と、各ネットワークのブロードキャストアドレス"""
    addrs = {"255.255.255.255"}
    try:
        import ipaddress
        import psutil
        for lst in psutil.net_if_addrs().values():
            for a in lst:
                if a.family == socket.AF_INET and a.netmask and not a.address.startswith(("127.", "169.254.")):
                    net = ipaddress.IPv4Network(f"{a.address}/{a.netmask}", strict=False)
                    addrs.add(str(net.broadcast_address))
    except Exception:
        pass
    return sorted(addrs)


class Beacon(threading.Thread):
    """1秒ごとに "SQUATPC,<受信ポート>" をブロードキャストする（ファーム v4 が受け取る）"""
    def __init__(self, data_port, stop_flag):
        super().__init__(daemon=True)
        self.msg = BEACON_MSG + f",{data_port}".encode()
        self.stop_flag = stop_flag

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        addrs, last_refresh = [], 0.0
        while not self.stop_flag.is_set():
            if time.time() - last_refresh > 10.0:          # ネットワークの切り替えに追従
                addrs, last_refresh = broadcast_addresses(), time.time()
            for addr in addrs:
                try:
                    sock.sendto(self.msg, (addr, BEACON_PORT))
                except OSError:
                    pass
            self.stop_flag.wait(1.0)
        sock.close()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Multi-IMU UDP receiver")
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--no-beacon", action="store_true",
                    help="PC の居場所を知らせる合図を送らない（ファームの pc_ip 固定で使う）")
    ap.add_argument("--outdir", type=str,
                    default=str(PROJECT_ROOT / "data" / "_sessions"))
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outdir).resolve() / f"session_{stamp}"
    recorder = SessionRecorder(out_dir)

    maxlen = int(PLOT_WINDOW_S * EXPECTED_HZ * 1.5)
    buffers = defaultdict(lambda: {"t": deque(maxlen=maxlen),
                                   "az": deque(maxlen=maxlen)})
    buf_lock = threading.Lock()
    stop_flag = threading.Event()

    print("=" * 68)
    print(" マルチIMU 受信・記録")
    print("=" * 68)
    print(f"  出力先 : {out_dir}")
    print(f"  ポート : {args.port}")
    print()
    print("  [SPACE] セット開始 / 終了")
    print("  [6-9]   直前のセットに RPE 6〜9 を付与     [0] RPE 10   [5] RPE 5")
    print("  [u]     直前のマーカーを取り消し")
    print("  [q]     終了")
    print()
    print("  ※ グラフウィンドウにフォーカスを当てて操作してください")
    print("=" * 68)

    rx = Receiver(recorder, buffers, buf_lock, args.port, stop_flag)
    rx.start()
    if not args.no_beacon:
        Beacon(args.port, stop_flag).start()
        print(f"[BEACON] PC の居場所を UDP {BEACON_PORT} に送信中: {', '.join(broadcast_addresses())}")

    # ---------------- プロット ----------------
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        fig.canvas.manager.set_window_title("Multi-IMU Live  [SPACE]set  [6-9]RPE  [q]quit")
    except Exception:
        pass

    ax.set_xlabel("time [s] (session)")
    ax.set_ylabel("az  [g]")
    ax.set_ylim(-2.5, 4.0)
    ax.grid(alpha=0.3)
    ax.set_title("Multi-IMU  vertical acceleration", fontsize=13)

    lines = {}
    color_idx = [0]

    status_text = ax.text(
        0.01, 0.98, "", transform=ax.transAxes, fontsize=11,
        va="top", ha="left", family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999999"),
    )
    set_text = ax.text(
        0.99, 0.98, "", transform=ax.transAxes, fontsize=13,
        va="top", ha="right", family="monospace", fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999999"),
    )

    set_spans = []   # 記録済みセットの背景

    def get_color(dev):
        if dev in DEVICE_COLORS:
            return DEVICE_COLORS[dev]
        c = FALLBACK_COLORS[color_idx[0] % len(FALLBACK_COLORS)]
        color_idx[0] += 1
        DEVICE_COLORS[dev] = c
        return c

    def update(_frame):
        with buf_lock:
            snapshot = {d: (list(b["t"]), list(b["az"])) for d, b in buffers.items()}

        now = time.time() - recorder.t0
        t_lo = max(0.0, now - PLOT_WINDOW_S)
        ax.set_xlim(t_lo, max(PLOT_WINDOW_S, now))

        for dev, (ts, az) in snapshot.items():
            if dev not in lines:
                (ln,) = ax.plot([], [], lw=1.4, label=dev, color=get_color(dev))
                lines[dev] = ln
                ax.legend(loc="lower left", ncol=min(5, len(lines)), fontsize=10)
            lines[dev].set_data(ts, az)

        # 記録済みセットを背景で表示
        for m in recorder.markers[len(set_spans):]:
            sp = ax.axvspan(m["start_s"], m["end_s"], color="#2A9D8F", alpha=0.13)
            set_spans.append(sp)

        # ステータス
        rows = []
        for dev in sorted(snapshot.keys()):
            n = recorder.counts[dev]
            span = recorder.last_recv.get(dev, 0) - recorder.first_recv.get(dev, 0)
            hz = (n / span) if span > 0.5 else 0.0
            age = now - recorder.last_recv.get(dev, now)
            mark = "  " if age < 1.0 else "!!"
            rows.append(f"{mark}{dev:<7s} {n:>7d}  {hz:5.1f}Hz")
        if not rows:
            rows = ["  (待機中: デバイスからの受信なし)"]
        if rx.bad_lines:
            rows.append(f"  bad lines: {rx.bad_lines}")
        status_text.set_text("\n".join(rows))

        if recorder.recording_set:
            el = now - recorder.current_start
            set_text.set_text(f"● REC  SET {recorder.set_no}   {el:5.1f}s")
            set_text.set_color("#C62828")
        else:
            last = recorder.markers[-1] if recorder.markers else None
            if last:
                r = last["rpe"] if last["rpe"] is not None else "-"
                set_text.set_text(f"idle   last: SET{last['set_no']}  RPE {r}")
            else:
                set_text.set_text("idle")
            set_text.set_color("#333333")

        return list(lines.values()) + [status_text, set_text]

    # ---------------- キー操作 ----------------
    def on_key(event):
        now = time.time() - recorder.t0
        k = (event.key or "").lower()

        if k == " ":
            kind, no, val = recorder.toggle_set(now)
            if kind == "start":
                print(f"\n[SET {no}] ▶ 開始   t={val:.1f}s")
            else:
                print(f"[SET {no}] ■ 終了   duration={val:.1f}s   → RPE を数字キーで入力")

        elif k in ("5", "6", "7", "8", "9", "0"):
            rpe = 10.0 if k == "0" else float(k)
            no = recorder.set_rpe(rpe)
            if no is None:
                print("  (まだ完了したセットがありません)")
            else:
                print(f"  SET {no} → RPE {rpe:g} を記録")

        elif k == "u":
            kind, no = recorder.undo_last()
            if kind == "cancel_start":
                print("  セット開始を取り消しました")
            elif kind == "remove":
                print(f"  SET {no} のマーカーを削除しました")
            else:
                print("  取り消すマーカーがありません")

        elif k == "q":
            print("\n[INFO] 終了します...")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    ani = FuncAnimation(fig, update, interval=100,
                        blit=False, cache_frame_data=False)
    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        time.sleep(0.4)
        info, mpath = recorder.close()

        print()
        print("=" * 68)
        print(" セッション終了")
        print("=" * 68)
        print(f"  保存先     : {out_dir}")
        print(f"  記録時間   : {info['duration_s']} 秒")
        print(f"  セット数   : {info['n_sets']}")
        print()
        if info["devices"]:
            print(f"  {'device':<10s} {'samples':>9s} {'mean Hz':>9s}")
            print("  " + "-" * 30)
            for dev, d in sorted(info["devices"].items()):
                hz = d["mean_hz"] if d["mean_hz"] is not None else 0
                print(f"  {dev:<10s} {d['n_samples']:>9d} {hz:>9.1f}")
        else:
            print("  受信データなし")
        print()
        for m in recorder.markers:
            r = m["rpe"] if m["rpe"] is not None else "未入力"
            print(f"  SET {m['set_no']:>2d}  {m['duration_s']:>6.1f}s   RPE {r}")
        print()
        print(f"  マーカー   : {mpath}")
        print("  ※ RPE の中間値(8.5等)や未入力分は markers.csv を編集してください")
        print("=" * 68)


if __name__ == "__main__":
    main()
