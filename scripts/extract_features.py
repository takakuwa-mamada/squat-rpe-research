# -*- coding: utf-8 -*-
"""
=============================================================================
 1試技（1セット）の生データから特徴量ベクトルを抽出するスクリプト

 入力:
   - IMU CSV:   data/<S>/<SES>/imu/setNN_BAR.csv        （無ければ動画のみの試技として扱う）
   - 骨格:      data/<S>/<SES>/pose/setNN_yolo_keypoints.csv（YOLO26。無ければ骨格特徴なし）
                data/<S>/<SES>/pose/setNN_yolo_info.json    （動画の縦横サイズ。無ければ動画から読む）
   - meta.json: data/<S>/<SES>/meta.json
   - 被験者情報: data/subjects.json（身長・1RM。あれば %1RM と m/s 換算に使う）

 出力:
   - data/<S>/<SES>/features/setNN_features.json（1試技ぶんの特徴量）
     "_" で始まるキー（レップごとの詳細）は aggregate_features.py で CSV に入れない

 特徴量:
   imu_*   バー IMU（速度・加速度・レップ単位の MCV/MPV・速度低下率）
   pose_*  骨格（正面撮影用）: レップ分割、テンポ、深さ、肩の上下動から見たバー速度、
           膝の開き（膝の内側への入り）、腰の横ブレ、肩・腰の傾き、体幹の側方傾斜、
           1レップ目 → 最終レップの変化（疲労によるフォームの崩れ）
           距離は「立位の肩〜足首の高さ」= 1 BL（body length）で正規化する

 使い方:
   python scripts\\extract_features.py                    # data/ 配下を全自動
   python scripts\\extract_features.py --subject S001
   python scripts\\extract_features.py --force            # 既存の JSON も作り直す
   python scripts\\extract_features.py --meta data\\S001\\SES004\\meta.json --set_no 1 \\
       --imu data\\S001\\SES004\\imu\\set01_BAR.csv --pose data\\S001\\SES004\\pose\\set01_yolo_keypoints.csv

 依存ライブラリ:
   pip install numpy pandas scipy opencv-python
=============================================================================
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT  = PROJECT_ROOT / "data"
SUBJECTS_JSON = DATA_ROOT / "subjects.json"
G = 9.80665   # 重力加速度 [m/s^2]

# 骨格
POSE_CONF_MIN = 0.3      # これ未満の信頼度のキーポイントは欠損扱い
POSE_LPF_HZ   = 6.0      # キーポイント軌跡のローパス
# 身長に対する「肩峰〜足関節」の高さの比（Drillis & Contini: 肩峰高 0.818H − 足関節高 0.039H）
SHOULDER_ANKLE_RATIO = 0.779


# ===========================================================================
# 信号処理ユーティリティ
# ===========================================================================
def lowpass(x, fs, fc=10.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, fc / nyq, btype="low")
    return filtfilt(b, a, x)


def highpass(x, fs, fc=0.3, order=2):
    """カットオフが低いと端の歪みが長く続くので、端を十分に（1/fc 秒ぶん）折り返して延長する"""
    nyq = fs / 2.0
    b, a = butter(order, fc / nyq, btype="high")
    return filtfilt(b, a, x, padtype="odd", padlen=min(len(x) - 1, int(fs / fc)))


def integrate_trapz(x, dt):
    """累積台形積分"""
    return np.concatenate([[0.0], np.cumsum((x[:-1] + x[1:]) / 2.0 * dt)])


def _r(x, nd=4):
    """JSON 用に丸める。NaN / None は None"""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(xf) or np.isinf(xf):
        return None
    return round(xf, nd)


def _nanmean(arrs):
    """複数配列の要素ごとの平均（片方だけ欠損ならもう片方を使う）"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.vstack(arrs), axis=0)


def _nanstat(fn, x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(fn(x))


# ===========================================================================
# IMU: レップ検出
# ===========================================================================
def detect_reps(v, fs, expected_reps=None):
    """
    鉛直速度 v から、スクワットのレップ（上昇局面）を検出する。

    単純に上向きピークを拾うと、ラックアウトの持ち上げ動作まで
    レップとして数えてしまう。スクワット1レップは必ず
    「下降 → ボトム → 上昇」の順に起きるため、
    上昇ピークの直前に有意な下降があるものだけを採用する。

    expected_reps を渡すと（meta.json の reps_completed など）、
    候補が多すぎる場合に上位 N 個だけを残す。
    """
    vmax = float(np.max(np.abs(v))) if len(v) else 0.0
    if vmax <= 0.05:
        return np.array([], dtype=int)

    up_thresh   = max(0.12, 0.30 * vmax)   # 上昇とみなす速度
    down_thresh = max(0.08, 0.20 * vmax)   # 下降とみなす速度
    look_back   = int(fs * 3.0)            # 直前3秒の範囲で下降を探す

    cand, _ = find_peaks(v, height=up_thresh, distance=int(fs * 0.6))

    reps = []
    for p in cand:
        lo = max(0, p - look_back)
        window = v[lo:p]
        if len(window) == 0:
            continue
        # 直前に十分な下降があるか（= しゃがんでから立ち上がったか）
        if float(np.min(window)) <= -down_thresh:
            reps.append(p)

    # 下降を伴うものが1つも無ければ、最大ピークだけ残す（ポーズ動作などの保険）
    if not reps and len(cand):
        reps = [int(cand[np.argmax(v[cand])])]

    reps = np.array(sorted(reps), dtype=int)

    # 実測レップ数が分かっている場合は、速度の大きい順に上位 N 個を残す
    if expected_reps and len(reps) > expected_reps:
        top = sorted(reps, key=lambda i: v[i], reverse=True)[:expected_reps]
        reps = np.array(sorted(top), dtype=int)

    return reps


def imu_vertical_velocity(imu_df: pd.DataFrame, fs: float, dt: float):
    """
    バーの鉛直加速度と速度を求める。

    1. 鉛直加速度 = 3軸合成加速度 − 重力の実測値
       重力は「加速度の揺れが小さく、ジャイロ（オフセット除去後）も小さい」静かな区間の中央値。
       → センサの向きの誤差や、静止時の値のずれ（SES004 では 1.05 g 前後）の影響を受けにくい
    2. 静止点を探す: 加速度が静かで重力の実測値に近く、ハイパス（0.15 Hz）後の速度もほぼ 0 の区間
       （一定速度の下降は加速度だけ見ると静止に見えるが、速度が大きいので選ばれない）
    3. 速度 = 生の積分 − 静止点どうしを直線でつないだドリフト（ZUPT）
       ハイパスの速度をそのまま使うと、遅いレップ（周期がカットオフに近い）の波形が削られて
       MCV を過小評価する（合成データの 8 レップ目で約 25% 低く出た）
       静止点が 2 つ未満のときだけ、ハイパスの速度を使う
    戻り値: (a_vert [m/s^2], v [m/s], 重力の実測値 [g], 生の積分, ハイパス後の速度, 静止点)
    """
    ax = imu_df["ax_g"].to_numpy(); ay = imu_df["ay_g"].to_numpy(); az = imu_df["az_g"].to_numpy()
    n = len(az)
    acc = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyr = imu_df[["gx_dps", "gy_dps", "gz_dps"]].to_numpy(dtype=float)
    gyr = np.linalg.norm(gyr - np.median(gyr, axis=0), axis=1)   # M5 のジャイロはオフセットが大きい
    long_enough = n > int(fs * 0.5)
    acc_lp = lowpass(acc, fs, fc=10.0) if long_enough else acc
    gyr_lp = lowpass(gyr, fs, fc=5.0) if long_enough else gyr

    acc_sd = pd.Series(acc_lp).rolling(max(3, int(0.2 * fs)), center=True, min_periods=3).std().to_numpy()
    quiet = (acc_sd < 0.02) & (gyr_lp < 10.0)
    g_ref = float(np.median(acc_lp[quiet])) if quiet.sum() > int(0.2 * fs) else float(np.median(acc_lp))

    a_vert = (acc_lp - g_ref) * G
    v_raw = integrate_trapz(a_vert, dt)
    v_hp = highpass(v_raw, fs, fc=0.15) if n > int(fs * 3) else v_raw - np.mean(v_raw)

    # 遅いレップの上昇中も加速度の揺れは小さいが、加速・減速の局面では重力からずれるので除ける
    still = quiet & (np.abs(acc_lp - g_ref) < 0.03) & (np.abs(v_hp) < 0.1)
    centers, anchors, i = [], [], 0
    min_len = max(3, int(0.1 * fs))
    while i < n:
        if still[i]:
            j = i
            while j + 1 < n and still[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                centers.append((i + j) / 2.0)
                anchors.append(float(np.mean(v_raw[i:j + 1])))
            i = j + 1
        else:
            i += 1
    if len(centers) >= 2:
        v = v_raw - np.interp(np.arange(n), centers, anchors)
    else:
        v = v_hp
    return a_vert, v, g_ref, v_raw, v_hp, [int(c) for c in centers]


def select_reps(v_hp, fs, dt, expected_reps=None, min_rom=0.15):
    """
    おおまかな速度 v_hp からスクワットのレップを選ぶ。
    戻り値: [(下降の始まり, 下降のピーク, 上昇のピーク), ...]（時刻順）

    「直前に下降がある上昇ピーク」のうち、上昇の変位（ROM）が min_rom [m] 以上のものだけを採用する。
    ラックアウト・ラックイン・歩き出しは上下動が数 cm なので外れる。
    レップ数が分かっていれば、ROM の大きい順に N 個残す（速度の大きい順だとラックインを拾う）。
    """
    n = len(v_hp)
    cands = []
    for pk in detect_reps(v_hp, fs, expected_reps=None):
        lo = max(0, pk - int(3.0 * fs))
        dmin = lo + int(np.argmin(v_hp[lo:pk + 1]))
        ds = dmin
        while ds > 0 and v_hp[ds - 1] < -0.03:
            ds -= 1
        zb = dmin
        while zb < pk and v_hp[zb] <= 0:
            zb += 1
        ze = pk
        while ze < n - 1 and v_hp[ze + 1] > 0:
            ze += 1
        rom = float(np.sum(v_hp[zb:ze + 1]) * dt)
        if rom >= min_rom:
            cands.append((ds, dmin, int(pk), rom))
    if expected_reps and len(cands) > expected_reps:
        cands = sorted(sorted(cands, key=lambda c: c[3], reverse=True)[:expected_reps])
    return [(ds, dmin, pk) for ds, dmin, pk, _ in cands]


def solve_lockout(v_raw, a0, pk, fs, dt):
    """
    1レップ分のドリフトを解く。立位（a0: 速度 0）から始まり、ロックアウト T で
    「速度 0 かつ 変位 0（立位の高さに戻る）」になるように、一定のずれ（センサのバイアス）を引く。
    T は上昇ピーク以降で変位の残りが最小になる最初の時刻。
    戻り値: (T, 補正後の速度 v[a0:T+1])
    """
    lo, hi = pk + max(1, int(0.05 * fs)), min(len(v_raw) - 1, pk + int(2.5 * fs))
    results = []
    for T in range(lo, hi + 1):
        seg = v_raw[a0:T + 1] - v_raw[a0]
        L = len(seg) - 1
        if L <= 0:
            continue
        vc = seg - (seg[-1] / L) * np.arange(L + 1)
        disp = float(np.sum((vc[:-1] + vc[1:]) / 2.0) * dt)
        results.append((abs(disp), T, vc))
    if not results:
        seg = v_raw[a0:pk + 1] - v_raw[a0]
        return pk, seg
    best = min(r[0] for r in results)
    T, vc = next((r[1], r[2]) for r in results if r[0] <= best + 0.005)
    return T, vc


def concentric_phase(v, a, p):
    """
    レップの上昇ピーク p を含むコンセントリック区間 [s, e] と、推進局面の終わり pe を返す。
      s, e : 速度がピークの 5%（最低 0.02 m/s）を超えて続く区間の両端
             （0 を境にすると、残ったわずかなずれで静止中まで上昇に数えてしまう）
      pe   : ピーク以降で加速度が −g を下回る直前（MPV の定義: 減速が重力より大きい局面を除く）
    """
    n = len(v)
    thr = max(0.02, 0.05 * float(v[p]))
    s = p
    while s > 0 and v[s - 1] > thr:
        s -= 1
    e = p
    while e < n - 1 and v[e + 1] > thr:
        e += 1
    pe = e
    for i in range(p, e + 1):
        if a[i] < -G:
            pe = max(s, i - 1)
            break
    return s, e, pe


# ===========================================================================
# IMU 特徴量抽出
# ===========================================================================
def compute_imu_features(imu_df: pd.DataFrame, expected_reps=None) -> dict:
    """
    IMU CSV (timestamp_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
    から特徴量を計算する。

    expected_reps: meta.json の reps_completed（分かっていれば検出を補正する）
    注意: 3軸合成加速度を使うので向きの誤差には強いが、水平方向の加速度が大きいと誤差になる。
          roll/pitch はジャイロの単純積分で
          ドリフトを含むため、学習には使わない（train_rpe_model.py で除外）。
    """
    if len(imu_df) < 20:
        return {"imu_n_samples": len(imu_df), "imu_error": "too few samples"}

    t = imu_df["timestamp_ms"].to_numpy() / 1000.0   # [s]
    n = len(t)
    dt_med = float(np.median(np.diff(t)))
    fs = 1.0 / dt_med if dt_med > 0 else 100.0

    gx = imu_df["gx_dps"].to_numpy()
    gy = imu_df["gy_dps"].to_numpy()

    # 鉛直加速度（重力は静止区間で実測）と速度（静止点の間でドリフト補正）
    az_dyn_lp, v, g_ref, v_raw, v_hp, anchors = imu_vertical_velocity(imu_df, fs, dt_med)

    # レップを選び、レップごとに「ロックアウトで速度0・変位0」でドリフトを解き直す
    # （挙上の直後はすぐラックへ歩くので、ロックアウト後に静止区間が無いことが多い）
    rep_idx = select_reps(v_hp, fs, dt_med, expected_reps=expected_reps)
    v = v.copy()
    peaks, prev_T = [], None
    for ds, dmin, pk in rep_idx:
        before = [c for c in anchors if c <= ds]
        a0 = before[-1] if before else ds
        if prev_T is not None and prev_T > a0:
            a0 = prev_T
        T, vc = solve_lockout(v_raw, a0, pk, fs, dt_med)
        v[a0:T + 1] = vc
        prev_T = T
        peaks.append(int(dmin + np.argmax(v[dmin:T + 1])))
    peaks = np.array(peaks, dtype=int)

    # 各種統計
    peak_v_up   = float(np.max(v))                # 上向きピーク速度
    peak_v_down = float(np.min(v))                # 下向きピーク速度
    mean_v_up   = float(np.mean(v[v > 0])) if np.any(v > 0) else 0.0
    mean_v_down = float(np.mean(v[v < 0])) if np.any(v < 0) else 0.0

    rms_a = float(np.sqrt(np.mean(az_dyn_lp ** 2)))
    peak_a = float(np.max(np.abs(az_dyn_lp)))

    # Jerk（加速度の時間微分）
    jerk = np.diff(az_dyn_lp) / dt_med            # [m/s^3]
    max_jerk = float(np.max(np.abs(jerk)))
    mean_jerk = float(np.mean(np.abs(jerk)))

    # コンセントリック相 / エキセントリック相の時間
    concentric_time = float(np.sum(v > 0.05) * dt_med)
    eccentric_time  = float(np.sum(v < -0.05) * dt_med)
    total_time = float(t[-1] - t[0])

    n_reps = int(len(peaks))

    # レップごとの指標（ピーク速度・MCV・MPV・コンセントリック時間）
    reps = []
    for p in peaks:
        s, e, pe = concentric_phase(v, az_dyn_lp, int(p))
        reps.append({
            "peak_t_s": float(t[p]),
            "start_t_s": float(t[s]),
            "end_t_s": float(t[e]),
            "peak_v": float(v[p]),
            "mcv": float(np.mean(v[s:e + 1])),
            "mpv": float(np.mean(v[s:pe + 1])),
            "concentric_s": float(t[e] - t[s]),
        })

    def first(key):
        return reps[0][key] if reps else float("nan")

    def last(key):
        return reps[-1][key] if reps else float("nan")

    def mean(key):
        return float(np.mean([r[key] for r in reps])) if reps else float("nan")

    # Velocity Loss (VL) = (最初のレップ − 最終レップ) / 最初のレップ
    # 1レップのセットでは定義できないため None にする
    def vloss(key):
        if n_reps >= 2 and first(key) > 0.01:
            return (first(key) - last(key)) / first(key) * 100.0
        return float("nan")

    # 水平方向ブレ（gx, gy 積分 → 角度変化。ドリフト込みなので参考値）
    roll  = integrate_trapz(gx, dt_med)
    pitch = integrate_trapz(gy, dt_med)

    return {
        "imu_n_samples":              int(n),
        "imu_fs_hz":                  round(fs, 2),
        "imu_gravity_ref_g":          round(g_ref, 4),
        # M5 の向きの確認用（画面が真上なら +1 g 前後）
        "imu_az_median_g":            round(float(np.median(imu_df["az_g"])), 3),
        "imu_total_time_s":           round(total_time, 3),
        "imu_concentric_time_s":      round(concentric_time, 3),
        "imu_eccentric_time_s":       round(eccentric_time, 3),

        "imu_peak_velocity_up_mps":   round(peak_v_up,   4),
        "imu_peak_velocity_down_mps": round(peak_v_down, 4),
        "imu_mean_velocity_up_mps":   round(mean_v_up,   4),
        "imu_mean_velocity_down_mps": round(mean_v_down, 4),

        "imu_n_reps_detected":         n_reps,
        "imu_first_rep_peak_v_mps":    _r(first("peak_v")),
        "imu_last_rep_peak_v_mps":     _r(last("peak_v")),
        "imu_mean_rep_peak_v_mps":     _r(mean("peak_v")),
        "imu_velocity_loss_pct":       _r(vloss("peak_v"), 2),

        # VBT の標準指標（レップ単位）
        "imu_first_rep_mcv_mps":       _r(first("mcv")),
        "imu_last_rep_mcv_mps":        _r(last("mcv")),
        "imu_mean_rep_mcv_mps":        _r(mean("mcv")),
        "imu_first_rep_mpv_mps":       _r(first("mpv")),
        "imu_mean_rep_mpv_mps":        _r(mean("mpv")),
        "imu_velocity_loss_mcv_pct":   _r(vloss("mcv"), 2),
        "imu_first_rep_concentric_s":  _r(first("concentric_s"), 3),
        "imu_mean_rep_concentric_s":   _r(mean("concentric_s"), 3),

        "imu_rms_accel_mps2":          round(rms_a, 4),
        "imu_peak_accel_mps2":         round(peak_a, 4),
        "imu_max_jerk_mps3":           round(max_jerk, 2),
        "imu_mean_jerk_mps3":          round(mean_jerk, 2),

        "imu_bar_roll_range_deg":      round(float(np.max(roll) - np.min(roll)), 2),
        "imu_bar_pitch_range_deg":     round(float(np.max(pitch) - np.min(pitch)), 2),

        "_imu_reps": [{k: round(val, 4) for k, val in r.items()} for r in reps],
    }


# ===========================================================================
# 骨格: 前処理
# ===========================================================================
POSE_POINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle"]


def _fill_smooth(sig, fs, max_gap_s=0.3):
    """短い欠損を線形補間し、ローパスで平滑化する（長い欠損は NaN のまま残す）"""
    s = pd.Series(sig, dtype=float).interpolate(
        limit=max(1, int(max_gap_s * fs)), limit_area="inside")
    arr = s.to_numpy()
    ok = ~np.isnan(arr)
    if ok.sum() > max(15, int(fs * 0.5)) and fs > 2.5 * POSE_LPF_HZ:
        filled = pd.Series(arr).ffill().bfill().to_numpy()
        sm = lowpass(filled, fs, fc=POSE_LPF_HZ)
        sm[~ok] = np.nan
        return sm
    return arr


def _load_points(kp_df, w, h, fs):
    """キーポイントをピクセル座標（縦横比を保った座標）に戻し、平滑化して返す"""
    pts = {}
    for name in POSE_POINTS:
        x = kp_df[f"{name}_x"].to_numpy(dtype=float) * w
        y = kp_df[f"{name}_y"].to_numpy(dtype=float) * h
        c = kp_df[f"{name}_conf"].to_numpy(dtype=float)
        bad = ~(c >= POSE_CONF_MIN)
        x[bad] = np.nan
        y[bad] = np.nan
        pts[name] = (_fill_smooth(x, fs), _fill_smooth(y, fs))
    return pts


def _tilt_deg(xl, yl, xr, yr):
    """左右2点を結ぶ線の水平からの傾き [deg]（右が下がると正）"""
    return np.degrees(np.arctan2(yr - yl, np.abs(xr - xl)))


# ===========================================================================
# 骨格: レップ分割
# ===========================================================================
def detect_pose_reps(depth, fs, expected_reps=None):
    """
    depth: 立位からの肩の下がり量 [BL]（下が正）
    戻り値: [(start, bottom, end), ...] フレーム番号

    ラックアウトの歩き出しでは肩はほとんど沈まないので、
    沈み込みの大きさ（prominence）でしゃがみ動作だけを拾う。
    """
    d = pd.Series(depth).interpolate(limit_direction="both").to_numpy()
    if np.all(np.isnan(d)):
        return []
    peaks, props = find_peaks(d, prominence=0.08, distance=max(1, int(fs * 0.8)))
    if len(peaks) == 0:
        return []
    prom = props["prominences"]
    keep = prom >= max(0.08, 0.5 * float(np.max(prom)))
    peaks, prom = peaks[keep], prom[keep]
    if expected_reps and len(peaks) > expected_reps:
        idx = np.argsort(prom)[::-1][:expected_reps]
        peaks = np.sort(peaks[idx])

    reps = []
    bounds = [0] + list(peaks) + [len(d) - 1]
    for k, b in enumerate(peaks):
        lo, hi = bounds[k], bounds[k + 2]
        base_before = float(np.min(d[lo:b + 1]))
        base_after  = float(np.min(d[b:hi + 1]))
        thr_b = base_before + 0.1 * (d[b] - base_before)
        thr_a = base_after  + 0.1 * (d[b] - base_after)
        s = b
        while s > lo and d[s - 1] > thr_b:
            s -= 1
        e = b
        while e < hi and d[e + 1] > thr_a:
            e += 1
        reps.append((int(s), int(b), int(e)))
    return reps


# ===========================================================================
# 骨格特徴量
# ===========================================================================
POSE_MOTION_KEYS = [
    "pose_n_reps_detected",
    "pose_first_rep_mcv_bl", "pose_last_rep_mcv_bl", "pose_mean_rep_mcv_bl",
    "pose_first_rep_peak_v_bl", "pose_mean_rep_peak_v_bl", "pose_velocity_loss_pct",
    "pose_mean_descent_s", "pose_mean_ascent_s", "pose_mean_bottom_pause_s",
    "pose_ascent_time_ratio",
    "pose_mean_depth_bl", "pose_depth_change_bl", "pose_mean_hip_knee_dy_bl",
    "pose_mean_sticking_ratio",
    "pose_first_rep_mcv_mps_est", "pose_mean_rep_peak_v_mps_est",
]
POSE_FRONTAL_KEYS = [
    "pose_min_knee_ankle_ratio", "pose_knee_ankle_ratio_change",
    "pose_max_lateral_shift_bl", "pose_lateral_shift_change_bl",
    "pose_max_shoulder_tilt_deg", "pose_shoulder_tilt_change_deg",
    "pose_max_hip_tilt_deg", "pose_max_trunk_side_lean_deg",
]


def _pose_empty(reason, n_frames=0):
    d = {"pose_available": False, "pose_n_frames": int(n_frames), "pose_error": reason}
    for k in POSE_MOTION_KEYS + POSE_FRONTAL_KEYS:
        d[k] = None
    return d


def compute_pose_features(kp_df: Optional[pd.DataFrame], width=None, height=None, fps=None,
                          view="front", expected_reps=None, height_cm=None) -> dict:
    """
    骨格キーポイント（setNN_yolo_keypoints.csv）から運動特徴を計算する。

    view      : "front"（正面）なら膝の開き・横ブレ・傾きも出す。それ以外は共通の特徴のみ
    height_cm : 身長が分かれば BL/s を m/s に換算した推定値も出す
    """
    if kp_df is None or len(kp_df) == 0:
        return _pose_empty("no keypoints")
    n = len(kp_df)
    if not width or not height:
        return _pose_empty("video size unknown", n)

    if not fps or fps <= 0:
        dt = np.median(np.diff(kp_df["time_s"].to_numpy()))
        fps = 1.0 / dt if dt > 0 else 30.0
    fs = float(fps)

    # 品質
    valid = kp_df["left_hip_x"].notna().to_numpy()
    conf = np.vstack([kp_df[f"{p}_conf"].to_numpy(dtype=float) for p in POSE_POINTS])
    vis = conf.mean(axis=0)[valid]
    quality = {
        "pose_available":           True,
        "pose_n_frames":            int(n),
        "pose_nan_ratio":           _r(1.0 - valid.mean()),
        "pose_key_visibility_mean": _r(vis.mean() if len(vis) else np.nan),
        "pose_key_visibility_min":  _r(vis.min() if len(vis) else np.nan),
        # 全身が画面に収まっているか（足首・頭が画面端で切れていない）
        "pose_ankle_y_max_norm":    _r(np.nanpercentile(
            kp_df[["left_ankle_y", "right_ankle_y"]].to_numpy(dtype=float), 99)
            if valid.any() else np.nan),
    }

    pts = _load_points(kp_df, width, height, fs)
    lsx, lsy = pts["left_shoulder"];  rsx, rsy = pts["right_shoulder"]
    lhx, lhy = pts["left_hip"];       rhx, rhy = pts["right_hip"]
    lkx, lky = pts["left_knee"];      rkx, rky = pts["right_knee"]
    lax, lay = pts["left_ankle"];     rax, ray = pts["right_ankle"]

    sh_x, sh_y = _nanmean([lsx, rsx]), _nanmean([lsy, rsy])
    hip_x, hip_y = _nanmean([lhx, rhx]), _nanmean([lhy, rhy])
    kn_y = _nanmean([lky, rky])
    an_x, an_y = _nanmean([lax, rax]), _nanmean([lay, ray])

    # 体の大きさ: 立位の肩〜足首の高さ（ピクセル）= 1 BL
    body_len = _nanstat(lambda a: np.percentile(a, 95), an_y - sh_y)
    quality["pose_body_len_px"] = _r(body_len, 1)
    # 肩幅 / 体の高さ: 正面なら 0.2 以上、真横なら 0.1 未満になる（撮影方向の確認用）
    quality["pose_shoulder_width_bl"] = _r(_nanstat(np.median, np.abs(lsx - rsx)) / body_len
                                           if body_len and not np.isnan(body_len) else np.nan, 3)
    if np.isnan(body_len) or body_len < 50:
        d = _pose_empty("body too small or not detected", n)
        d.update(quality)
        d["pose_available"] = False
        return d

    # 肩の沈み込み（≒バーの鉛直位置）と上向き速度
    stand_y = _nanstat(lambda a: np.percentile(a, 5), sh_y)
    depth = (sh_y - stand_y) / body_len                 # [BL] 下が正
    v_up = -np.gradient(depth) * fs                     # [BL/s] 上が正

    reps_idx = detect_pose_reps(depth, fs, expected_reps=expected_reps)

    # 正面用の時系列
    frontal = (view == "front")
    knee_ankle = np.abs(lkx - rkx) / np.abs(lax - rax)  # 膝の開き / 足首の開き（小さいほど膝が内側）
    lateral = (hip_x - an_x) / body_len                 # 腰の横位置（足首中点基準）
    sh_tilt = _tilt_deg(lsx, lsy, rsx, rsy)
    hip_tilt = _tilt_deg(lhx, lhy, rhx, rhy)
    trunk_side = np.degrees(np.arctan2(sh_x - hip_x, hip_y - sh_y))

    def rel_absmax(sig, s, e):
        """レップ開始時点を基準にした変化量の絶対値の最大（カメラの傾きを打ち消す）"""
        seg = sig[s:e + 1]
        ref = _nanstat(np.median, sig[s:min(e, s + max(2, int(0.2 * fs))) + 1])
        return _nanstat(np.max, np.abs(seg - ref))

    reps = []
    for s, b, e in reps_idx:
        seg_v = v_up[b:e + 1]
        exc = depth[b] - _nanstat(np.min, depth[s:b + 1])
        # ボトムでの停止時間: 沈み込みが最深の 95% 以上の区間
        thr = depth[b] - 0.05 * exc
        l, r = b, b
        while l > s and depth[l - 1] >= thr:
            l -= 1
        while r < e and depth[r + 1] >= thr:
            r += 1
        # スティッキングポイント: 上昇中に速度が一度落ち込む度合い（最小/最大）
        sv = np.nan_to_num(seg_v, nan=0.0)
        vp, _ = find_peaks(sv, prominence=0.03)
        sticking = (float(np.min(sv[vp[0]:vp[-1] + 1]) / np.max(sv))
                    if len(vp) >= 2 and np.max(sv) > 0 else 1.0)
        rep = {
            "start_t_s": s / fs, "bottom_t_s": b / fs, "end_t_s": e / fs,
            "descent_s": (b - s) / fs, "ascent_s": (e - b) / fs, "pause_s": (r - l) / fs,
            "mcv_bl": _nanstat(np.mean, seg_v), "peak_v_bl": _nanstat(np.max, seg_v),
            "depth_bl": float(depth[b]),
            "hip_knee_dy_bl": float((hip_y[b] - kn_y[b]) / body_len),   # 正: 腰が膝より下
            "sticking_ratio": sticking,
        }
        if frontal:
            rep.update({
                "knee_ankle_min": _nanstat(np.min, knee_ankle[b:e + 1]),
                "lateral_shift_bl": rel_absmax(lateral, s, e),
                "shoulder_tilt_deg": rel_absmax(sh_tilt, s, e),
                "hip_tilt_deg": rel_absmax(hip_tilt, s, e),
                "trunk_side_lean_deg": rel_absmax(trunk_side, s, e),
            })
        reps.append(rep)

    nr = len(reps)

    def col(k):
        return np.array([r.get(k, np.nan) for r in reps], dtype=float)

    def first(k):
        return col(k)[0] if nr else np.nan

    def last(k):
        return col(k)[-1] if nr else np.nan

    def mean(k):
        return _nanstat(np.mean, col(k)) if nr else np.nan

    def change(k):
        return last(k) - first(k) if nr >= 2 else np.nan

    m_per_bl = SHOULDER_ANKLE_RATIO * float(height_cm) / 100.0 if height_cm else np.nan
    vl = ((first("mcv_bl") - last("mcv_bl")) / first("mcv_bl") * 100.0
          if nr >= 2 and first("mcv_bl") > 0.01 else np.nan)

    feat = {
        "pose_n_reps_detected":        nr,
        "pose_first_rep_mcv_bl":       _r(first("mcv_bl")),
        "pose_last_rep_mcv_bl":        _r(last("mcv_bl")),
        "pose_mean_rep_mcv_bl":        _r(mean("mcv_bl")),
        "pose_first_rep_peak_v_bl":    _r(first("peak_v_bl")),
        "pose_mean_rep_peak_v_bl":     _r(mean("peak_v_bl")),
        "pose_velocity_loss_pct":      _r(vl, 2),
        "pose_mean_descent_s":         _r(mean("descent_s"), 3),
        "pose_mean_ascent_s":          _r(mean("ascent_s"), 3),
        "pose_mean_bottom_pause_s":    _r(mean("pause_s"), 3),
        "pose_ascent_time_ratio":      _r(last("ascent_s") / first("ascent_s")
                                          if nr >= 2 and first("ascent_s") > 0 else np.nan, 3),
        "pose_mean_depth_bl":          _r(mean("depth_bl")),
        "pose_depth_change_bl":        _r(change("depth_bl")),
        "pose_mean_hip_knee_dy_bl":    _r(mean("hip_knee_dy_bl")),
        "pose_mean_sticking_ratio":    _r(mean("sticking_ratio"), 3),
        "pose_first_rep_mcv_mps_est":  _r(first("mcv_bl") * m_per_bl),
        "pose_mean_rep_peak_v_mps_est": _r(mean("peak_v_bl") * m_per_bl),
    }
    if frontal:
        feat.update({
            "pose_min_knee_ankle_ratio":     _r(_nanstat(np.min, col("knee_ankle_min")) if nr else np.nan),
            "pose_knee_ankle_ratio_change":  _r(change("knee_ankle_min")),
            "pose_max_lateral_shift_bl":     _r(_nanstat(np.max, col("lateral_shift_bl")) if nr else np.nan),
            "pose_lateral_shift_change_bl":  _r(change("lateral_shift_bl")),
            "pose_max_shoulder_tilt_deg":    _r(_nanstat(np.max, col("shoulder_tilt_deg")) if nr else np.nan, 2),
            "pose_shoulder_tilt_change_deg": _r(change("shoulder_tilt_deg"), 2),
            "pose_max_hip_tilt_deg":         _r(_nanstat(np.max, col("hip_tilt_deg")) if nr else np.nan, 2),
            "pose_max_trunk_side_lean_deg":  _r(_nanstat(np.max, col("trunk_side_lean_deg")) if nr else np.nan, 2),
        })
    else:
        feat.update({k: None for k in POSE_FRONTAL_KEYS})

    feat["_pose_reps"] = [{k: _r(val, 4) for k, val in r.items()} for r in reps]
    return {**quality, **feat}


# ===========================================================================
# 補助: meta / 被験者情報 / 動画サイズ
# ===========================================================================
def find_set_meta(meta: dict, set_no: int) -> dict:
    for s in meta.get("sets", []):
        if int(s.get("set_no", -1)) == int(set_no):
            return s
    return {}


def load_subject_profile(subject_id: Optional[str]) -> dict:
    """data/subjects.json から被験者のプロフィール（身長・1RM など）を読む"""
    if not subject_id or not SUBJECTS_JSON.exists():
        return {}
    try:
        subs = json.loads(SUBJECTS_JSON.read_text(encoding="utf-8")).get("subjects", {})
        return subs.get(subject_id, {}).get("profile", {}) or {}
    except Exception:
        return {}


def load_video_info(kp_csv: Optional[Path], video: Optional[Path]):
    """(width, height, fps)。pose_extract_yolo.py の info JSON → 無ければ動画から読む"""
    if kp_csv is not None:
        info = kp_csv.with_name(kp_csv.name.replace("_yolo_keypoints.csv", "_yolo_info.json"))
        if info.exists():
            d = json.loads(info.read_text(encoding="utf-8"))
            return d.get("width"), d.get("height"), d.get("fps")
    if video is not None and video.exists():
        import cv2
        cap = cv2.VideoCapture(str(video))
        w, h, fps = cap.get(3), cap.get(4), cap.get(5)
        cap.release()
        if w and h:
            return int(w), int(h), float(fps) if fps else None
    return None, None, None


# ===========================================================================
# 1試技ぶんの特徴量を計算
# ===========================================================================
def extract_one_trial(imu_csv: Optional[Path], kp_csv: Optional[Path],
                      meta_json: Path, set_no: int, video: Optional[Path] = None) -> dict:
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    set_meta = find_set_meta(meta, set_no)
    profile = load_subject_profile(meta.get("subject_id"))

    expected_reps = set_meta.get("reps_completed") or set_meta.get("reps_planned")
    try:
        expected_reps = int(expected_reps) if expected_reps else None
    except (TypeError, ValueError):
        expected_reps = None

    # IMU（あれば）
    imu_feat = {}
    if imu_csv is not None and imu_csv.exists():
        imu_feat = compute_imu_features(pd.read_csv(imu_csv), expected_reps=expected_reps)

    # 骨格（あれば）
    view = set_meta.get("camera_view") or meta.get("camera_view") or "front"
    kp_df = None
    if kp_csv is not None and kp_csv.exists():
        try:
            kp_df = pd.read_csv(kp_csv)
        except Exception:
            kp_df = None
    w, h, fps = load_video_info(kp_csv, video)
    pose_feat = compute_pose_features(kp_df, w, h, fps, view=view,
                                      expected_reps=expected_reps,
                                      height_cm=profile.get("height_cm"))

    weight = set_meta.get("weight_kg")
    one_rm = profile.get("squat_1rm_kg")
    pct_1rm = (float(weight) / float(one_rm) * 100.0) if weight and one_rm else None

    meta_feat = {
        "subject_id":      meta.get("subject_id"),
        "session_id":      meta.get("session_id"),
        "date":            meta.get("date"),
        "set_no":          int(set_no),
        "modality":        "imu+video" if imu_feat else "video",
        "camera_view":     view,
        "exercise":        set_meta.get("exercise", meta.get("exercise")),
        "weight_kg":       weight,
        "pct_1rm":         _r(pct_1rm, 2),
        "reps_planned":    set_meta.get("reps_planned"),
        "reps_completed":  set_meta.get("reps_completed"),
        "rpe":             set_meta.get("rpe"),
        "rest_before_sec": set_meta.get("rest_before_sec"),
        "set_notes":       set_meta.get("notes"),
    }
    return {**meta_feat, **imu_feat, **pose_feat}


# ===========================================================================
# data/ 配下を走査して全試技を抽出
# ===========================================================================
def find_pose_keypoints(pose_dir: Path, vid_name: Optional[str], set_no: int) -> Optional[Path]:
    """pose/ から setNN に対応する YOLO の keypoints CSV を探す"""
    if not pose_dir.is_dir():
        return None
    stems = []
    if vid_name:
        stems.append(Path(vid_name).stem)
    stems.append(f"set{int(set_no):02d}")
    for stem in stems:
        cand = pose_dir / f"{stem}_yolo_keypoints.csv"
        if cand.exists():
            return cand
    return None


def find_trials(filter_subject: Optional[str] = None):
    """戻り値: [(imu_csv or None, kp_csv or None, video or None, meta_json, set_no, out_json), ...]"""
    results = []
    if not DATA_ROOT.exists():
        return results
    for meta_json in sorted(DATA_ROOT.rglob("meta.json")):
        parts = meta_json.relative_to(DATA_ROOT).parts
        if len(parts) != 3 or parts[0].startswith("_"):
            continue
        subj, sess = parts[0], parts[1]
        if filter_subject and subj != filter_subject:
            continue
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        sess_dir = DATA_ROOT / subj / sess
        for s in meta.get("sets", []):
            set_no = s.get("set_no")
            if set_no is None:
                continue
            csv_name = s.get("csv_filename")
            vid_name = s.get("video_filename")
            imu_csv = sess_dir / "imu" / csv_name if csv_name else None
            video = sess_dir / "videos" / vid_name if vid_name else None
            kp_csv = find_pose_keypoints(sess_dir / "pose", vid_name, set_no)
            out_json = sess_dir / "features" / f"set{int(set_no):02d}_features.json"
            results.append((imu_csv, kp_csv, video, meta_json, int(set_no), out_json))
    return results


# ===========================================================================
# メイン
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="1試技ごとの特徴量を抽出する")
    ap.add_argument("--imu", type=str, default=None, help="単一試技: IMU CSV")
    ap.add_argument("--pose", type=str, default=None, help="単一試技: setNN_yolo_keypoints.csv")
    ap.add_argument("--meta", type=str, default=None, help="単一試技: meta.json")
    ap.add_argument("--set_no", type=int, default=None, help="単一試技: セット番号")
    ap.add_argument("--subject", type=str, default=None, help="例: S001")
    ap.add_argument("--force", action="store_true", help="既存の JSON も作り直す")
    args = ap.parse_args()

    # 単一試技モード
    if args.meta:
        if args.set_no is None:
            sys.exit("[ERROR] --meta のときは --set_no が必要")
        imu_csv = Path(args.imu).resolve() if args.imu else None
        kp_csv = Path(args.pose).resolve() if args.pose else None
        feat = extract_one_trial(imu_csv, kp_csv, Path(args.meta).resolve(), args.set_no)
        print(json.dumps(feat, indent=2, ensure_ascii=False))
        return

    # バッチモード
    print(f"[BATCH] scanning {DATA_ROOT}")
    trials = find_trials(filter_subject=args.subject)
    if not trials:
        print("[INFO] 試技が見つかりません")
        print(f"  想定: {DATA_ROOT}/<被験者>/<セッション>/meta.json")
        return

    pending, skipped = [], 0
    for tr in trials:
        imu_csv, kp_csv, video, meta_json, set_no, out_json = tr
        if (not args.force) and out_json.exists():
            skipped += 1
            continue
        has_imu = imu_csv is not None and imu_csv.exists()
        if not has_imu and kp_csv is None:
            print(f"  [SKIP] IMU も骨格もありません: {meta_json.parent.name} set{set_no:02d}"
                  f"（先に pose_extract_yolo.py を実行）")
            continue
        pending.append(tr)

    print(f"[INFO] 処理対象: {len(pending)} 試技 (処理済みでスキップ: {skipped})")
    for imu_csv, kp_csv, video, meta_json, set_no, out_json in pending:
        rel = out_json.parent.parent.relative_to(DATA_ROOT)
        print(f"\n[RUN] {rel} set{set_no:02d}")
        try:
            feat = extract_one_trial(imu_csv, kp_csv, meta_json, set_no, video=video)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(feat, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  RPE={feat.get('rpe')}  重量={feat.get('weight_kg')}  "
                  f"レップ(IMU/骨格/申告)={feat.get('imu_n_reps_detected', '-')}/"
                  f"{feat.get('pose_n_reps_detected', '-')}/{feat.get('reps_completed')}  "
                  f"MCV(IMU)={feat.get('imu_first_rep_mcv_mps', '-')}  "
                  f"MCV(骨格,BL/s)={feat.get('pose_first_rep_mcv_bl', '-')}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
