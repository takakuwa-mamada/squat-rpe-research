# -*- coding: utf-8 -*-
"""
=============================================================================
 解析パイプラインのテスト（合成データ）

 実データ（SES004）は1レップのセットしかないので、複数レップのロジック
 （レップ分割・速度低下率・1レップ目→最終レップの変化）と、評価スクリプトが
 最後まで回ることを合成データで確かめる。

 使い方:
   pytest scripts\\test_pipeline.py -q
=============================================================================
"""

import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import extract_features as ef          # noqa: E402
from ingest_remote import parse_label, parse_log_line   # noqa: E402
import ingest_remote as ir             # noqa: E402

G = 9.80665
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


# ---------------------------------------------------------------------------
# 合成の挙上軌跡: 立位 → (下降 → 上昇) × n → 立位。上昇時間を段々長くする（疲労）
# ---------------------------------------------------------------------------
def squat_trajectory(fs, n_reps=5, depth=0.45, descent=1.2, ascent0=0.9, ascent_growth=0.15,
                     pause=0.2, rest=0.6, lead=2.0, tail=1.5):
    """戻り値: t, y（立位からの下がり量 [m]、下が正）, 各レップの上昇時間"""
    segs, ascents = [np.zeros(int(lead * fs))], []
    for i in range(n_reps):
        asc = ascent0 * (1 + ascent_growth * i)
        ascents.append(asc)
        td = np.linspace(0, np.pi, int(descent * fs), endpoint=False)
        ta = np.linspace(0, np.pi, int(asc * fs), endpoint=False)
        segs += [depth * (1 - np.cos(td)) / 2, np.full(int(pause * fs), depth),
                 depth * (1 + np.cos(ta)) / 2, np.zeros(int(rest * fs))]
    segs.append(np.zeros(int(tail * fs)))
    y = np.concatenate(segs)
    return np.arange(len(y)) / fs, y, ascents


def make_keypoints(fs=30.0, w=720, h=1280, n_reps=5, knee_cave=True, **kw):
    """正面から撮った人物の keypoints（0〜1 正規化）を作る。肩〜足首 = 0.45 h"""
    t, y, asc = squat_trajectory(fs, n_reps=n_reps, **kw)
    body = 0.45 * h
    rng = np.random.default_rng(0)
    n = len(t)
    cx = 0.5 * w
    rows = {"frame": np.arange(n), "time_s": t}
    drop = y / 0.45 * body * 0.9          # 肩の沈み込み [px]（depth 0.45 m ≒ 0.9 BL 相当に拡大）
    sh_y = 0.25 * h + drop
    hip_y = 0.50 * h + drop * 0.8
    knee_y = np.full(n, 0.60 * h)
    ank_y = np.full(n, 0.70 * h)
    # 膝の開き: レップが進むほどボトム付近で内側に入る
    rep_frac = np.clip(y / max(y.max(), 1e-9), 0, 1)
    progress = np.clip(t / t[-1], 0, 1)
    knee_half = 0.09 * w * (1 - (0.25 * progress * rep_frac if knee_cave else 0))
    pos = {
        "left_shoulder": (cx + 0.12 * w, sh_y), "right_shoulder": (cx - 0.12 * w, sh_y),
        "left_hip": (cx + 0.08 * w, hip_y), "right_hip": (cx - 0.08 * w, hip_y),
        "left_knee": (cx + knee_half, knee_y), "right_knee": (cx - knee_half, knee_y),
        "left_ankle": (cx + 0.09 * w, ank_y), "right_ankle": (cx - 0.09 * w, ank_y),
    }
    for name in COCO:
        x, yy = pos.get(name, (np.full(n, cx), sh_y - 0.08 * h))
        rows[f"{name}_x"] = (x + rng.normal(0, 1.0, n)) / w
        rows[f"{name}_y"] = (yy + rng.normal(0, 1.0, n)) / h
        rows[f"{name}_conf"] = np.full(n, 0.95)
    return pd.DataFrame(rows), asc


def make_imu(fs=100.0, n_reps=5, g_bias=1.05, **kw):
    """バーの鉛直位置から加速度を作り、M5 らしいオフセット・ノイズを載せる"""
    t, y, asc = squat_trajectory(fs, n_reps=n_reps, **kw)
    pos_up = -y
    v_true = np.gradient(pos_up, 1 / fs)
    a_true = np.gradient(v_true, 1 / fs)
    rng = np.random.default_rng(1)
    az = g_bias + a_true / G + rng.normal(0, 0.01, len(t))
    df = pd.DataFrame({
        "timestamp_ms": t * 1000.0,
        "ax_g": rng.normal(0, 0.01, len(t)), "ay_g": rng.normal(0, 0.01, len(t)), "az_g": az,
        "gx_dps": -3.8 + rng.normal(0, 1, len(t)), "gy_dps": -12.9 + rng.normal(0, 1, len(t)),
        "gz_dps": 16.0 + rng.normal(0, 1, len(t)),
    })
    return df, v_true, asc


# ---------------------------------------------------------------------------
# 骨格
# ---------------------------------------------------------------------------
def test_pose_detects_all_reps_and_fatigue():
    kp, asc = make_keypoints(n_reps=5)
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front")
    assert f["pose_available"]
    assert f["pose_n_reps_detected"] == 5
    assert f["pose_velocity_loss_pct"] > 10            # 上昇が遅くなる → 速度低下
    assert f["pose_ascent_time_ratio"] > 1.3            # 最終レップの上昇時間 / 1レップ目
    assert f["pose_knee_ankle_ratio_change"] < 0        # 膝が内側に入っていく
    assert 0.2 < f["pose_shoulder_width_bl"] < 1.0      # 正面


def test_pose_survives_low_confidence_gap():
    """人の検出が弱いフレームが 1 秒続いても、骨格特徴がまるごと欠損しない（2026-10-30 の不具合）"""
    kp, _ = make_keypoints(n_reps=3)
    gap = (kp["time_s"] > 1.0) & (kp["time_s"] < 2.0)          # 立位の 1 秒間だけ信頼度を下げる
    for c in [c for c in kp.columns if c.endswith("_conf")]:
        kp.loc[gap, c] = 0.05
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front")
    assert f["pose_available"] is True
    assert f["pose_n_reps_detected"] == 3
    assert f["pose_first_rep_mcv_bl"] is not None


def test_fill_smooth_leaves_long_gap_empty():
    """0.3 秒より長い欠損は全部空のまま（2026-10-31 まで、最初の 8 コマが直線で埋まっていた）"""
    sig = np.linspace(0.0, 100.0, 120)
    gap = sig.copy()
    gap[40:54] = np.nan                                   # 14 コマ = 0.47 秒（10/29 set01 の横切りと同じ長さ）
    out = ef._fill_smooth(gap, 29.97)
    assert np.isnan(out[40:54]).all()
    assert not np.isnan(out[:40]).any() and not np.isnan(out[54:]).any()


def test_fill_smooth_bridges_gap_up_to_0_3_s():
    """0.3 秒（29.97 fps で 9 コマ）までの欠損はつなぐ（2026-10-31 まで、切り捨てで 8 コマ = 0.27 秒までだった）"""
    sig = np.linspace(0.0, 100.0, 120)
    gap = sig.copy()
    gap[40:49] = np.nan
    out = ef._fill_smooth(gap, 29.97)
    assert not np.isnan(out).any()
    assert np.allclose(out[40:49], sig[40:49], atol=0.5)


def test_fill_smooth_no_step_after_long_gap():
    """長い欠損の直後のコマが、平滑化で欠損の前の値に引っ張られない（以前は一定の動きで 1 コマ分以上ずれた）"""
    sig = np.linspace(0.0, 100.0, 120)                    # 1 コマ 0.84 ずつ動く
    gap = sig.copy()
    gap[40:54] = np.nan
    out = ef._fill_smooth(gap, 29.97)
    assert abs(out[54] - sig[54]) < 0.3


def test_pose_rep_cap_by_reported_reps():
    kp, _ = make_keypoints(n_reps=5)
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front", expected_reps=3)
    assert f["pose_n_reps_detected"] == 3


def test_pose_side_view_leaves_frontal_features_empty():
    kp, _ = make_keypoints(n_reps=3)
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="side")
    assert f["pose_n_reps_detected"] == 3
    assert f["pose_min_knee_ankle_ratio"] is None


def test_pose_missing_video_size():
    kp, _ = make_keypoints(n_reps=2)
    assert ef.compute_pose_features(kp, None, None, 30.0)["pose_available"] is False


# ---------------------------------------------------------------------------
# IMU
# ---------------------------------------------------------------------------
def test_imu_detects_reps_and_mcv_close_to_truth():
    imu, v_true, asc = make_imu(n_reps=5)
    f = ef.compute_imu_features(imu)
    assert f["imu_n_reps_detected"] == 5
    assert f["imu_velocity_loss_mcv_pct"] > 10
    # 1レップ目の MCV（真値: 上昇区間の平均速度）と 15% 以内で一致
    true_mcv = 0.45 / asc[0]
    assert abs(f["imu_first_rep_mcv_mps"] - true_mcv) / true_mcv < 0.15
    # 上昇時間も真値の ±20% 以内（0.3 Hz ハイパス時代は大きく短く出ていた）
    assert abs(f["imu_first_rep_concentric_s"] - asc[0]) / asc[0] < 0.2
    assert abs(f["imu_gravity_ref_g"] - 1.05) < 0.01


def make_imu_double_hump(fs=100.0, depths=(0.52, 0.55, 0.56), descent=1.0, ascent=1.4, rest=2.5,
                         lead=20.0, g_bias=1.05):
    """
    上昇中の速度に山が2つあるレップ（切り返し直後に小さい山 → スティッキングポイントで減速 → 大きい山）。
    2026-10-15 の実データ（SES005）では、この2つの山が別々のレップとして数えられ、
    深いレップが2回入って、いちばん浅いレップが落ちた。それを再現するため1レップ目だけ少し浅くする。
    """
    s = np.linspace(0, 1, int(ascent * fs), endpoint=False)
    v_shape = (0.35                                         # 谷でも上昇は続く（速度は正のまま）
               + 0.55 * np.exp(-((s - 0.2) / 0.1) ** 2)     # 切り返し直後の山
               + 1.00 * np.exp(-((s - 0.72) / 0.13) ** 2))  # スティッキングポイントを越えた後の山
    v_shape *= np.sin(np.pi * s)                            # 始まりと終わりは速度 0
    td = np.linspace(0, np.pi, int(descent * fs), endpoint=False)
    segs = [np.zeros(int(lead * fs))]
    for depth in depths:
        up = np.cumsum(v_shape)
        up = depth * up / up[-1]                            # 0 → depth まで上がる
        segs += [depth * (1 - np.cos(td)) / 2, np.full(int(0.2 * fs), depth),
                 depth - up, np.zeros(int(rest * fs))]
    y = np.concatenate(segs + [np.zeros(int(3.0 * fs))])   # 立位からの下がり量 [m]
    t = np.arange(len(y)) / fs
    a_true = np.gradient(np.gradient(-y, 1 / fs), 1 / fs)
    rng = np.random.default_rng(2)
    az = g_bias + a_true / G + rng.normal(0, 0.01, len(t))
    return pd.DataFrame({
        "timestamp_ms": t * 1000.0,
        "ax_g": rng.normal(0, 0.01, len(t)), "ay_g": rng.normal(0, 0.01, len(t)), "az_g": az,
        "gx_dps": -3.8 + rng.normal(0, 1, len(t)), "gy_dps": -12.9 + rng.normal(0, 1, len(t)),
        "gz_dps": 16.0 + rng.normal(0, 1, len(t)),
    })


def test_split_groups_receiver_folders_into_gym_visits():
    """受信フォルダは開始時刻が 3 時間以内に続けば同じ日（セッション）。日付をまたいでもよい"""
    import split_session as ss
    from datetime import datetime
    t = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M")
    items = [("a", t("2026-09-18 00:45")), ("b", t("2026-09-17 23:50")),
             ("c", t("2026-10-20 20:03")), ("d", t("2026-10-20 20:10")), ("e", t("2026-10-20 23:30"))]
    assert ss.group_visits(items, gap_hours=3) == [["b", "a"], ["c", "d"], ["e"]]


def test_split_finds_unimported_folders_and_next_number(tmp_path):
    import split_session as ss
    (tmp_path / "S001" / "SES001").mkdir(parents=True)
    (tmp_path / "S001" / "SES001" / "meta.json").write_text(
        json.dumps({"source_sessions": ["session_20261015_151052"]}), encoding="utf-8")
    (tmp_path / "S001" / "SES003").mkdir()
    assert ss.imported_folders(tmp_path) == {"session_20261015_151052"}
    assert ss.next_session_number("S001", tmp_path) == 4      # 抜けがあっても最後の番号の次
    assert ss.next_session_number("S009", tmp_path) == 1


def test_underscore_folders_are_skipped(tmp_path, monkeypatch):
    """data/_archive（外したセッション）・_sessions など _ で始まるフォルダは集計・骨格推定の対象にしない"""
    import aggregate_features as af
    for sub in ("S001/SES001", "_archive/split_1029/SES004"):
        (tmp_path / sub / "features").mkdir(parents=True)
        (tmp_path / sub / "features" / "set01_features.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(af, "DATA_ROOT", tmp_path)
    assert [p.relative_to(tmp_path).parts[0] for p in af.find_feature_jsons()] == ["S001"]

    pytest.importorskip("ultralytics")
    import pose_extract_yolo as px
    for sub in ("S001/SES001/videos", "_archive/split_1029/SES004/videos", "_sessions/session_x"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
        (tmp_path / sub / "set01.mp4").write_bytes(b"")
    monkeypatch.setattr(px, "DATA_ROOT", tmp_path)
    assert [v.relative_to(tmp_path).parts[0] for v, _ in px.find_videos()] == ["S001"]
    assert [v.relative_to(tmp_path).parts[0] for v, _ in px.find_videos("_sessions")] == ["_sessions"]


def test_imu_double_hump_ascent_counts_each_rep_once():
    imu = make_imu_double_hump()
    f = ef.compute_imu_features(imu, expected_reps=3)
    peaks = [r["peak_t_s"] for r in f["_imu_reps"]]
    assert f["imu_n_reps_detected"] == 3
    assert len(set(peaks)) == 3                            # 同じレップが2回入らない
    assert min(np.diff(peaks)) > 2.5                       # 3つとも別のレップ（間隔 ≒ 5.1 s）


def sticking_stop_trajectory(fs=100.0, n_reps=3, depth=0.55, descent=1.0, ascent=2.0, rest=2.5, lead=20.0,
                             humps=(0.9, 1.0)):
    """
    上がる途中で一度止まり、少し下がってから上がりきるレップ（速度が最も遅い所 −0.06 m/s、0 未満が 0.24 秒）。
    11/10 set03 の 4 回目（RPE 8）では、上がる途中でバーの端の IMU の速度が 0.25 秒ほど −0.05 m/s まで落ちた
    （原因は分かっていない: バーが左右に揺れたためかどうかは確かめられていない）。
    humps: 切り返し直後の山と、止まった後の山の高さ。既定はいちばん速い所が止まった後（上昇の後半）にある。
           (1.6, 0.6) にすると、いちばん速い所が止まる前（上昇の前半）に来る
    戻り値: t, y（立位からの下がり量 [m]、下が正）, 各レップの区間 [(始まり, 底, 終わり) [s]], 底からの上がり量 [m]
    """
    s = np.linspace(0, 1, int(ascent * fs), endpoint=False)
    v_shape = (humps[0] * np.exp(-((s - 0.18) / 0.09) ** 2)  # 切り返し直後の山
               - 0.06 * np.exp(-((s - 0.42) / 0.06) ** 2)    # 止まって少し下がる
               + humps[1] * np.exp(-((s - 0.72) / 0.13) ** 2))   # 上がりきる
    v_shape *= np.sin(np.pi * s)                             # 始まりと終わりは速度 0
    up = np.cumsum(v_shape)
    up = depth * up / up[-1]                                 # 0 → depth まで上がる
    td = np.linspace(0, np.pi, int(descent * fs), endpoint=False)
    segs, windows, t0 = [np.zeros(int(lead * fs))], [], lead
    for _ in range(n_reps):
        segs += [depth * (1 - np.cos(td)) / 2, np.full(int(0.1 * fs), depth), depth - up, np.zeros(int(rest * fs))]
        windows.append((t0, t0 + descent + 0.05, t0 + descent + 0.1 + ascent))
        t0 += descent + 0.1 + ascent + rest
    y = np.concatenate(segs + [np.zeros(int(3.0 * fs))])
    return np.arange(len(y)) / fs, y, windows, up


def true_rise(up, fs):
    """合成の上昇の真の上昇時間 [s] と MCV [m/s]: 底のあと初めてピークの 5 % を超えてから、最後に超えている所まで"""
    vt = np.gradient(up) * fs
    above = np.where(vt > 0.05 * vt.max())[0]
    return (above[-1] - above[0]) / fs, float(vt[above[0]:above[-1] + 1].mean())


def test_imu_rise_counts_from_bottom_through_mid_rise_stop():
    """
    上がる途中で速度が 0 を下回っても、上昇は底から数える（2026-11-12 修正）。
    それまでは「速度のピークからさかのぼって、しきい値以下になった所」を上がり始めにしていたので、
    止まった所より前が切れていた（11/10 set03 の 4 回目: IMU の上昇時間 0.93 秒、動画 1.97 秒）。
    この合成データでは、真の 1.67 秒・0.325 m/s に対して 0.8 秒・0.47 m/s と出ていた。
    """
    fs = 100.0
    t, y, windows, up = sticking_stop_trajectory(fs)
    true_s, true_mcv = true_rise(up, fs)
    a_true = np.gradient(np.gradient(-y, 1 / fs), 1 / fs)
    rng = np.random.default_rng(3)
    imu = pd.DataFrame({
        "timestamp_ms": t * 1000.0,
        "ax_g": rng.normal(0, 0.01, len(t)), "ay_g": rng.normal(0, 0.01, len(t)),
        "az_g": 1.05 + a_true / G + rng.normal(0, 0.01, len(t)),
        "gx_dps": -3.8 + rng.normal(0, 1, len(t)), "gy_dps": -12.9 + rng.normal(0, 1, len(t)),
        "gz_dps": 16.0 + rng.normal(0, 1, len(t)),
    })
    f = ef.compute_imu_features(imu, expected_reps=3, rep_windows=windows)
    assert f["imu_n_reps_detected"] == 3
    for r in f["_imu_reps"]:
        assert abs(r["concentric_s"] - true_s) < 0.15
        assert abs(r["mcv"] - true_mcv) / true_mcv < 0.1


def test_compare_rep_metrics_counts_rise_from_bottom_through_mid_rise_stop():
    """動画と IMU の比較（compare_video_imu.rep_metrics。横からの比較も使う）も、上昇は底から数える（2026-11-12 修正）"""
    import compare_video_imu as cv
    fs = 100.0
    t, y, windows, up = sticking_stop_trajectory(fs)
    true_s, true_mcv = true_rise(up, fs)
    z = -y                                                   # 上が正
    v = np.gradient(z) * fs
    for ts, _, te in windows:
        m = cv.rep_metrics(t, z, v, ts, te)
        assert abs(m["ascent_s"] - true_s) < 0.05
        assert abs(m["mcv"] - true_mcv) / true_mcv < 0.03


def imu_of_vertical_motion(t, y, seed=3):
    """上下にだけ動くバーの IMU（y = 立位からの下がり量 [m]、下が正。静止で 1.05 g、センサの雑音つき）"""
    dt = float(t[1] - t[0])
    a_true = np.gradient(np.gradient(-y, dt), dt)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp_ms": t * 1000.0,
        "ax_g": rng.normal(0, 0.01, len(t)), "ay_g": rng.normal(0, 0.01, len(t)),
        "az_g": 1.05 + a_true / G + rng.normal(0, 0.01, len(t)),
        "gx_dps": -3.8 + rng.normal(0, 1, len(t)), "gy_dps": -12.9 + rng.normal(0, 1, len(t)),
        "gz_dps": 16.0 + rng.normal(0, 1, len(t)),
    })


def windows_ending_at_nine_tenths(windows, up, fs):
    """
    レップの終わりを、動画のレップの終わりと同じ「沈み込みの 1 割まで戻った時刻」にした区間（detect_pose_reps の決め方）。
    windows の終わりは立ち上がりきった時刻なので、そこから上昇の長さを引いて、9 割まで上がった時刻を足す
    """
    k90 = int(np.argmax(up >= 0.9 * up[-1]))
    return [(ts, tb, te - len(up) / fs + k90 / fs) for ts, tb, te in windows]


def test_rise_end_continues_through_stop_below_nine_tenths_of_the_height():
    """
    上がり終わりの決め方（extract_features.rise_end。IMU の特徴量にも、動画と IMU の比較にも同じものを使う）:
    バーが「底から、レップの中でいちばん高い所まで」の 9 割に届く前に速度がしきい値を下回っても、上がり終わりにしない。
    その先で速度がもう一度しきい値を超える所から続けて、9 割を超えたあとで下回る直前を上がり終わりにする。
    """
    fs = 100.0
    h1 = 0.9 * np.sin(np.pi * np.linspace(0, 1, 60, endpoint=False))        # 前半の山（いちばん速い。高さの約 6 割まで）
    h2 = 0.5 * np.sin(np.pi * np.linspace(0, 1, 80, endpoint=False))        # 止まりかけた後の山
    v = np.concatenate([np.zeros(10), h1, np.full(25, -0.02), h2, np.zeros(60)])
    z = np.cumsum(v) / fs
    b, p = 10, 10 + int(np.argmax(h1))
    thr = max(0.02, 0.05 * float(v[p]))
    first_end, second_end = 10 + 60 - 1, 10 + 60 + 25 + 80 - 1
    assert (z[first_end] - z[b]) / (z.max() - z[b]) < 0.7                   # 前半の山の終わりは、まだ高さの 7 割より下
    e = ef.rise_end(v, z, b, p, thr, stop=len(v) - 1, limit=len(v) - 1)
    assert second_end - 5 <= e <= second_end                                # 後ろの山の終わりまで数える
    # 続きを探すのは limit（レップの終わり）まで。その先にしか続きが無ければ、前半の山の終わりのまま
    e_cut = ef.rise_end(v, z, b, p, thr, stop=len(v) - 1, limit=first_end + 10)
    assert first_end - 5 <= e_cut <= first_end


def test_rise_end_does_not_run_on_to_a_bump_after_the_top():
    """
    立ち上がりきった後の小さな揺れは上昇に数えない。実データの IMU では、上がり終わりの 0.18〜0.27 秒後に
    0.05〜0.13 m/s の上向きの速度が 0.04〜0.18 秒続くレップが 146 レップ中 28 レップあった（2026-11-16、ロックアウトまでの範囲）。
    「レップの終わりの手前で、最後にしきい値を超えていた所」を上がり終わりにすると、そこまで延びてしまう。
    """
    fs = 100.0
    rise = 0.8 * np.sin(np.pi * np.linspace(0, 1, 120, endpoint=False))     # 1.2 秒の上昇
    bump = 0.10 * np.sin(np.pi * np.linspace(0, 1, 15, endpoint=False))     # 0.15 秒・0.10 m/s の揺れ
    v = np.concatenate([np.zeros(10), rise, np.zeros(20), bump, np.zeros(50)])
    z = np.cumsum(v) / fs
    b, p = 10, 10 + int(np.argmax(rise))
    thr = max(0.02, 0.05 * float(v[p]))
    e = ef.rise_end(v, z, b, p, thr, stop=len(v) - 1, limit=len(v) - 1)
    assert 10 + 110 <= e <= 10 + 119                                        # 上昇の山の終わり（揺れの手前）


def test_concentric_phase_counts_past_the_stop_only_when_rep_end_is_given():
    """
    concentric_phase は、底とレップの終わり（limit）が分かっているときに、止まりかけの先まで上昇を数える。
    limit が無いときは以前の決め方（いちばん速い所のあと、速度がしきい値を下回る直前）のまま
    """
    fs = 100.0
    h1 = 0.9 * np.sin(np.pi * np.linspace(0, 1, 60, endpoint=False))
    h2 = 0.5 * np.sin(np.pi * np.linspace(0, 1, 80, endpoint=False))
    v = np.concatenate([np.zeros(10), h1, np.full(25, -0.02), h2, np.zeros(60)])
    a = np.gradient(v) * fs
    b, p = 10, 10 + int(np.argmax(h1))
    first_end, second_end = 10 + 60 - 1, 10 + 60 + 25 + 80 - 1
    s, e, _ = ef.concentric_phase(v, a, p, b, limit=len(v) - 1)
    assert s <= b + 5 and second_end - 5 <= e <= second_end
    _, e_old, _ = ef.concentric_phase(v, a, p, b)
    assert first_end - 5 <= e_old <= first_end


def test_imu_rise_runs_to_the_top_when_fastest_point_comes_before_mid_rise_stop():
    """
    いちばん速い所が上昇の前半にあり、その後で止まりかけるレップでも、上昇は立ち上がりきるまで数える（2026-11-16 修正）。
    それまでは「いちばん速い所のあと、速度がしきい値を下回る直前」を上がり終わりにしていたので、止まりかけた所で
    上昇が終わったと数えていた（この合成データでは、真の 1.65 秒に対して 0.55 秒）。
    レップは動画の区切りで取る（実データの 29 セット中 28 セットと同じ）。IMU だけで数えるときは、上がる途中で速度が
    0 を下回るレップがあるとレップの数え方の方が先に崩れる（3 回が 2 回になる。別の弱点で、ここでは扱わない）
    """
    fs = 100.0
    t, y, windows, up = sticking_stop_trajectory(fs, humps=(1.6, 0.6))
    true_s, true_mcv = true_rise(up, fs)
    assert np.argmax(np.gradient(up)) < 0.3 * len(up)        # この合成データの前提: いちばん速い所は上昇の前半
    imu = imu_of_vertical_motion(t, y, seed=3)
    f = ef.compute_imu_features(imu, expected_reps=3, rep_windows=windows_ending_at_nine_tenths(windows, up, fs))
    assert f["imu_n_reps_detected"] == 3
    for i, r in enumerate(f["_imu_reps"]):
        assert abs(r["concentric_s"] - true_s) < 0.15, \
            f"{i + 1} 回目: 上昇時間 {r['concentric_s']:.2f} 秒（真値 {true_s:.2f}）"
        assert abs(r["mcv"] - true_mcv) / true_mcv < 0.1, f"{i + 1} 回目: MCV {r['mcv']:.3f} m/s（真値 {true_mcv:.3f}）"


@pytest.mark.parametrize("lead,ascent", [(20.0, 2.5), (20.0, 3.0), (20.0, 3.5), (10.0, 2.5), (3.0, 4.0), (3.0, 5.0)])
def test_imu_lockout_is_not_taken_at_mid_rise_stop(lead, ascent):
    """
    IMU のロックアウト（立ち上がりきって、速度 0・変位 0 とみなす時刻。ここで速度のずれを解く）を、上がる途中で
    止まりかけた所に取り違えない（2026-11-16 修正。動画のレップ区間があるとき）。それまでは、いちばん速い所が
    上昇の前半にあるレップで、次の 2 つのときに取り違えていた。
      ・レップの前に長く静止していたとき（lead 10〜20 秒の 1 回目: 上昇時間 2.06 秒 → 0.82〜0.99 秒 など）。
        速度のずれを解く区間が長いと、止まりかけた所の小さな下向きの速度が、残りの高さを打ち消す大きさの変位に化ける
      ・上昇が長く、ロックアウトが「いちばん速い所の 2.5 秒後」までに来ないとき（上昇 4〜5 秒: 3.29 秒 → 1.3〜1.6 秒 など）
    動画のレップの終わり（肩が沈み込みの 1 割まで戻った時刻）の 0.3 秒前より前はロックアウトにせず、
    探す範囲は少なくともその 1.5 秒後までとる。
    """
    fs = 100.0
    t, y, windows, up = sticking_stop_trajectory(fs, ascent=ascent, lead=lead, humps=(1.6, 0.6))
    true_s, true_mcv = true_rise(up, fs)
    imu = imu_of_vertical_motion(t, y, seed=3)
    f = ef.compute_imu_features(imu, expected_reps=3, rep_windows=windows_ending_at_nine_tenths(windows, up, fs))
    assert f["imu_n_reps_detected"] == 3
    for i, r in enumerate(f["_imu_reps"]):
        assert abs(r["concentric_s"] - true_s) / true_s < 0.08, \
            f"{i + 1} 回目: 上昇時間 {r['concentric_s']:.2f} 秒（真値 {true_s:.2f}）"
        assert abs(r["mcv"] - true_mcv) / true_mcv < 0.1, f"{i + 1} 回目: MCV {r['mcv']:.3f} m/s（真値 {true_mcv:.3f}）"


def test_compare_rep_metrics_runs_to_the_top_when_fastest_point_comes_before_mid_rise_stop():
    """動画と IMU の比較（compare_video_imu.rep_metrics。横からの比較も使う）も、立ち上がりきるまで数える（2026-11-16 修正）"""
    import compare_video_imu as cv
    fs = 100.0
    t, y, windows, up = sticking_stop_trajectory(fs, humps=(1.6, 0.6))
    true_s, true_mcv = true_rise(up, fs)
    z = -y                                                   # 上が正
    v = np.gradient(z) * fs
    for ts, _, te in windows_ending_at_nine_tenths(windows, up, fs):
        m = cv.rep_metrics(t, z, v, ts, te)
        assert abs(m["ascent_s"] - true_s) < 0.05, f"上昇時間 {m['ascent_s']:.2f} 秒（真値 {true_s:.2f}）"
        assert abs(m["mcv"] - true_mcv) / true_mcv < 0.03
        assert m["rom"] > 0.95 * m["depth"]                  # 上がった距離が深さとほぼ同じ（途中で切れていない）


def test_imu_single_rep_has_no_velocity_loss():
    imu, _, _ = make_imu(n_reps=1)
    f = ef.compute_imu_features(imu, expected_reps=1)
    assert f["imu_n_reps_detected"] == 1
    assert f["imu_velocity_loss_mcv_pct"] is None


def make_imu_short_rest(seed, rest=0.5, n_reps=8, fs=100.0, depth=0.45, descent=1.0, pause=0.1,
                        ascent0=0.7, growth=0.05, sway=0.03, lead=3.0, tail=2.0):
    """
    回と回の間が短いセットの IMU（立ち上がって rest 秒で次の回へ）。バーはしゃがむ途中で前後に sway [m] 動く。
    鉛直加速度は 3 軸の合成の大きさから出すので、横の加速度が小さな誤差になり、立位で「変位 0」がぴったりにはならない（実データと同じ）。
    戻り値: IMU, 各レップの区間 [(始まり, 底, 終わり) [s]]（動画のレップ区間を IMU の時刻に直したものに当たる）, 上昇時間, 深さ [m]
    """
    segs, windows, ascents, t0 = [np.zeros(int(lead * fs))], [], [], lead
    for i in range(n_reps):
        n_d, n_p, n_a, n_r = (int(x * fs) for x in (descent, pause, ascent0 * (1 + growth * i), rest))
        td = np.linspace(0, np.pi, n_d, endpoint=False)
        ta = np.linspace(0, np.pi, n_a, endpoint=False)
        segs += [depth * (1 - np.cos(td)) / 2, np.full(n_p, depth), depth * (1 + np.cos(ta)) / 2, np.zeros(n_r)]
        windows.append((t0, t0 + (n_d + n_p / 2) / fs, t0 + (n_d + n_p + n_a) / fs))
        ascents.append(n_a / fs)
        t0 += (n_d + n_p + n_a + n_r) / fs
    y = np.concatenate(segs + [np.zeros(int(tail * fs))])          # 立位からの下がり量 [m]
    t = np.arange(len(y)) / fs
    a_true = np.gradient(np.gradient(-y, 1 / fs), 1 / fs)
    x_h = sway * np.sin(np.pi * np.clip(y / depth, 0, 1)) ** 2     # 前後の動き（立位と底で 0、途中で sway）
    a_h = np.gradient(np.gradient(x_h, 1 / fs), 1 / fs)
    rng = np.random.default_rng(seed)
    imu = pd.DataFrame({
        "timestamp_ms": t * 1000.0,
        "ax_g": a_h / G + rng.normal(0, 0.01, len(t)), "ay_g": rng.normal(0, 0.01, len(t)),
        "az_g": 1.05 + a_true / G + rng.normal(0, 0.01, len(t)),
        "gx_dps": -3.8 + rng.normal(0, 1, len(t)), "gy_dps": -12.9 + rng.normal(0, 1, len(t)),
        "gz_dps": 16.0 + rng.normal(0, 1, len(t)),
    })
    return imu, windows, ascents, depth


@pytest.mark.parametrize("use_video_windows", [True, False])
def test_imu_mcv_stays_right_when_rest_between_reps_is_short(use_video_windows):
    """
    回と回の間が 0.5 秒のセットでも、どのレップの MCV も真値に近い（動画の区切りを使うときも、IMU だけで数えるときも）。
    ロックアウト（速度 0・変位 0 とみなす時刻）を上昇ピークの後 2.5 秒の中だけで探していたころは、次のレップのしゃがみの中に取り違え、
    そのレップと次のレップの速度が崩れていた（2026-11-06。11/6 の 80 kg×8 で IMU の 4 回目が 0.083 m/s、動画は 0.545 m/s）。
    種によって起きたり起きなかったりするので、10 通り回す。
    """
    for seed in range(10):
        imu, windows, asc, depth = make_imu_short_rest(seed)
        f = ef.compute_imu_features(imu, expected_reps=len(asc), rep_windows=windows if use_video_windows else None)
        assert f["imu_n_reps_detected"] == len(asc), f"種 {seed}"
        for i, (r, a) in enumerate(zip(f["_imu_reps"], asc)):
            true_mcv = depth / a
            assert abs(r["mcv"] - true_mcv) / true_mcv < 0.15, \
                f"種 {seed} の {i + 1} 回目: MCV {r['mcv']:.3f} m/s（真値 {true_mcv:.3f}）"


# ---------------------------------------------------------------------------
# 協力者のラベル
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("100kg_5rep_RPE8", {"weight_kg": 100, "reps": 5, "rpe": 8}),
    ("102.5kg 3回 RPE9.5", {"weight_kg": 102.5, "reps": 3, "rpe": 9.5}),
    ("100kgx5@8", {"weight_kg": 100, "reps": 5, "rpe": 8}),
    ("01_100㎏ 5回 ＲＰＥ８", {"weight_kg": 100, "reps": 5, "rpe": 8}),             # スマホで打った1行を貼った名前
    ("１０２．５ｋｇ　３回　ＲＰＥ９．５", {"weight_kg": 102.5, "reps": 3, "rpe": 9.5}),
    ("IMG_1234", {}),
])
def test_parse_label(name, expected):
    assert parse_label(name) == expected


def test_parse_log_line():
    assert parse_log_line("102.5 3 9.5") == {"weight_kg": 102.5, "reps": 3, "rpe": 9.5}
    assert parse_log_line("# メモ") is None


def test_read_profile_pasted_from_line(tmp_path):
    """LINE で送られた4行を貼ったプロフィール（全角の数字・空白が混ざっていても読める）"""
    (tmp_path / "プロフィール.txt").write_text("身長　１７５\n体重 80\n１ＲＭ　１５０\n歴 5\n", encoding="utf-8")
    assert ir.read_profile(tmp_path) == {"height_cm": 175, "body_mass_kg": 80, "squat_1rm_kg": 150,
                                         "training_years": 5}


# ---------------------------------------------------------------------------
# 協力者の取り込み（LINE で受け取った動画は、動画の中の日時が送った時刻になる）
# ---------------------------------------------------------------------------
def _remote_folder(tmp_path, monkeypatch, folder_name, sent):
    """data/_remote/協力者A/<folder_name>/ に空の動画を作る。sent = {ファイル名: 動画の中の日時 or None}"""
    data = tmp_path / "data"
    folder = data / "_remote" / "協力者A" / folder_name
    folder.mkdir(parents=True)
    for name in sent:
        (folder / name).write_bytes(b"")
    monkeypatch.setattr(ir, "DATA_ROOT", data)
    monkeypatch.setattr(ir, "probe_video", lambda p: {
        "created": datetime.fromisoformat(sent[p.name]) if sent[p.name] else None, "duration": 60.0})
    return data, folder


def test_ingest_line_videos_follow_number_prefix(tmp_path, monkeypatch):
    """番号付きの名前は番号の順。動画の中の日時（送った時刻）は順番・撮影時刻・休憩時間に使わない"""
    sent = {"02_105kg 5回 RPE8.5.mp4": "2026-11-06T10:00:05+00:00",    # 03 → 02 → 01 の順に送った
            "01_100kg_5rep_RPE8.mp4": "2026-11-06T10:00:09+00:00",
            "03_110kgx5@9.5.mp4": "2026-11-06T10:00:00+00:00"}
    data, folder = _remote_folder(tmp_path, monkeypatch, "20261105", sent)
    ses = ir.ingest_folder("S003", "協力者A", folder, dry_run=False, force=False)
    meta = json.loads((data / "S003" / ses / "meta.json").read_text(encoding="utf-8"))
    assert meta["date"] == "2026-11-05"
    assert [s["source_video"] for s in meta["sets"]] == sorted(sent)
    assert [(s["weight_kg"], s["reps_completed"], s["rpe"]) for s in meta["sets"]] == \
        [(100, 5, 8), (105, 5, 8.5), (110, 5, 9.5)]
    assert all(s["recorded_at"] is None and s["rest_before_sec"] is None for s in meta["sets"])


def test_ingest_number_gap_only_warns(tmp_path, monkeypatch, capsys):
    """番号が飛んでいても（02 の置き忘れなど）取り込む。ラベルは1本ずつ名前にあるので、注意を出すだけ"""
    sent = {"01_100kg 5回 RPE8.mp4": None, "03_110kg 5回 RPE9.mp4": None}
    data, folder = _remote_folder(tmp_path, monkeypatch, "20261105", sent)
    ses = ir.ingest_folder("S003", "協力者A", folder, dry_run=False, force=False)
    meta = json.loads((data / "S003" / ses / "meta.json").read_text(encoding="utf-8"))
    assert [s["rpe"] for s in meta["sets"]] == [8, 9]
    assert "[注意] 番号が 1 から順に続いていません（1, 3）" in capsys.readouterr().out


@pytest.mark.parametrize("folder_name,sent", [
    ("1回目", {"01_100kg 5回 RPE8.mp4": None, "02_105kg 5回 RPE9.mp4": None}),          # 日付が読めない
    ("20261105", {"01_100kg 5回 RPE8.mp4": None, "105kg 5回 RPE9.mp4": None}),        # 番号の無い動画が混ざる
    ("20261105", {"01_100kg 5回 RPE8.mp4": None, "01_105kg 5回 RPE9.mp4": None}),     # 番号が重なる
    ("20261105", {"815051398.929315.mp4": "2026-10-30T11:09:59+00:00"}),                # LINE の名前のまま
])
def test_ingest_rejects_unclear_order(tmp_path, monkeypatch, folder_name, sent):
    data, folder = _remote_folder(tmp_path, monkeypatch, folder_name, sent)
    assert ir.ingest_folder("S003", "協力者A", folder, dry_run=False, force=False) is None
    assert not (data / "S003").exists()


def test_ingest_without_numbers_keeps_recording_time_order(tmp_path, monkeypatch):
    """番号が無ければ今までどおり撮影日時の順（撮影日時の残った元の動画を受け取ったとき）"""
    sent = {"IMG_0002.MOV": "2026-11-05T18:05:00+00:00", "IMG_0001.MOV": "2026-11-05T18:10:00+00:00"}
    data, folder = _remote_folder(tmp_path, monkeypatch, "20261105", sent)
    (folder / "記録.txt").write_text("100 5 8\n105 5 9\n", encoding="utf-8")
    ses = ir.ingest_folder("S003", "協力者A", folder, dry_run=False, force=False)
    meta = json.loads((data / "S003" / ses / "meta.json").read_text(encoding="utf-8"))
    assert [s["source_video"] for s in meta["sets"]] == ["IMG_0002.MOV", "IMG_0001.MOV"]
    assert [s["weight_kg"] for s in meta["sets"]] == [100, 105]
    assert meta["sets"][1]["rest_before_sec"] == 240           # 5 分 − 1 本目の長さ 60 s


# ---------------------------------------------------------------------------
# 評価スクリプトが最後まで回る（本人 IMU+動画 12 セッション ＋ 協力者 6 人 × 4 セッション）
# ---------------------------------------------------------------------------
def synthetic_features(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    subjects = {"S001": (12, True)} | {f"S{i:03d}": (4, False) for i in range(2, 8)}
    for sid, (n_sess, has_imu) in subjects.items():
        bias = rng.normal(0, 0.5)                  # 申告の癖（個人差）
        for s in range(1, n_sess + 1):
            for set_no in range(1, 8):
                pct = rng.uniform(65, 92)
                reps = int(rng.integers(1, 8))
                rpe = np.clip(np.round((4 + 0.06 * pct + 0.25 * reps + bias
                                        + rng.normal(0, 0.4)) * 2) / 2, 5, 10)
                mcv = 1.6 - 0.013 * pct - 0.06 * (rpe - 7) + rng.normal(0, 0.03)
                r = {"subject_id": sid, "session_id": f"SES{s:03d}", "date": f"2026-10-{s:02d}",
                     "set_no": set_no, "modality": "imu+video" if has_imu else "video",
                     "weight_kg": pct * 1.5, "pct_1rm": pct, "reps_completed": reps, "rpe": rpe,
                     "pose_first_rep_mcv_bl": mcv / 1.3 + rng.normal(0, 0.03),
                     "pose_velocity_loss_pct": (rpe - 5) * 5 + rng.normal(0, 3) if reps > 1 else np.nan,
                     "pose_mean_ascent_s": 0.5 / max(mcv, 0.1) + rng.normal(0, 0.05),
                     "pose_knee_ankle_ratio_change": -(rpe - 5) * 0.01 + rng.normal(0, 0.01) if reps > 1 else np.nan}
                for c in ef.__dict__.get("POSE_FRONTAL_KEYS", []):
                    r.setdefault(c, rng.normal(0, 1))
                if has_imu:
                    r.update({"imu_first_rep_mcv_mps": mcv, "imu_mean_rep_mcv_mps": mcv - 0.02,
                              "imu_velocity_loss_mcv_pct": (rpe - 5) * 6 if reps > 1 else np.nan,
                              "imu_first_rep_concentric_s": 0.5 / max(mcv, 0.1)})
                # 最後・平均のレップの速度は、1 回目の速度と速度低下から作る（乱数は使わないので、ほかの列の値は変わらない）
                vl = r["pose_velocity_loss_pct"] if reps > 1 else 0.0
                r["pose_last_rep_mcv_bl"] = r["pose_first_rep_mcv_bl"] * (1 - vl / 100)
                r["pose_mean_rep_mcv_bl"] = (r["pose_first_rep_mcv_bl"] + r["pose_last_rep_mcv_bl"]) / 2
                if has_imu:
                    vl = r["imu_velocity_loss_mcv_pct"] if reps > 1 else 0.0
                    r["imu_last_rep_mcv_mps"] = mcv * (1 - vl / 100)
                rows.append(r)
    return pd.DataFrame(rows)


def test_train_runs_end_to_end(tmp_path):
    csv = tmp_path / "features_all.csv"
    synthetic_features().to_csv(csv, index=False)
    r = subprocess.run([sys.executable, str(SCRIPT_DIR / "train_rpe_model.py"), "--input", str(csv),
                        "--output-dir", str(tmp_path / "models"), "--kmax", "3"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = next((tmp_path / "models").iterdir())
    res = pd.read_csv(out / "results.csv")
    assert {"A_within", "B_loso"} <= set(res["scheme"])
    assert (res["scheme"] == "B_loso").sum() > 0 and "imu_all" not in \
        set(res[res["scheme"] == "B_loso"]["condition"])          # IMU は本人だけ → 汎用評価なし
    curves = pd.read_csv(out / "curves.csv")
    assert {"C_general_plus_k", "D_personal_curve"} <= set(curves["scheme"])
    # 曲線は線形回帰（無ければリッジ）で描く（HGB は学習が 10 セット未満だと平均を答えるだけになる。2026-11-05）
    cm = curves.groupby("condition")["model"].first()
    assert (cm["context"], cm["video_velocity"], cm["video_vbt"], cm["video_all"], cm["imu+video"],
            cm["mean_rpe"]) == ("linear", "linear", "ridge", "ridge", "ridge", "mean")
    assert (out / "report.md").exists() and (out / "best.pkl").exists()
    # 同じセッションが学習とテストにまたがっていない（A: フォールド = 被験者/セッション）
    pred = pd.read_csv(out / "predictions.csv")
    a = pred[pred["scheme"] == "A_within"]
    assert (a["fold"] == a["subject_id"] + "/" + a["session_id"]).all()
    # RPE 5 未満が無いデータでは「全 RPE」だけで学習。RPE 6 未満も無ければ評価は全セットだけ
    feats = synthetic_features()
    assert set(res["train_set"]) == {"all"}
    assert set(res["eval_set"]) == ({"all", "rpe>=6"} if (feats["rpe"] < 6).any() else {"all"})
    # 比べる相手（2026-11-05）: 平均を答えるだけの基準、最後のレップの速度だけ、速度の指標一式 ＋ 回数
    new = {"mean_rpe", "imu_lrv", "imu_vbt", "video_lrv", "video_vbt"}
    assert new <= set(res[res["scheme"] == "A_within"]["condition"])
    b_conds = set(res[res["scheme"] == "B_loso"]["condition"])
    assert {"mean_rpe", "video_lrv", "video_vbt"} <= b_conds and not {"imu_lrv", "imu_vbt"} & b_conds
    # 基準の予測は、そのフォールドの学習データ（本人のほかのセッション）の RPE の平均
    base = a[(a["condition"] == "mean_rpe") & (a["fold"] == "S001/SES001")]
    expect = feats.loc[(feats["subject_id"] == "S001") & (feats["session_id"] != "SES001"), "rpe"].mean()
    assert len(base) == 7 and np.allclose(base["pred"], expect)
    # 合成データでは、速度の指標一式（速度低下を含む）の方が平均を答えるだけより当たる
    mae = res[(res["scheme"] == "A_within") & (res["eval_set"] == "all")].set_index("condition")["mae"]
    assert mae["video_vbt"] < mae["mean_rpe"] and mae["imu_vbt"] < mae["mean_rpe"]
    # 基準は最良モデルの候補にしない・保存もしない。条件の意味を report.md に書く
    import joblib
    assert joblib.load(out / "best.pkl")["condition"] != "mean_rpe"
    assert not (out / "mean_rpe__mean.pkl").exists() and (out / "video_vbt__ridge.pkl").exists()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## 条件の説明" in report and "最後のレップの速度だけ" in report
    # RPE 5 未満のセットが無いデータでは、主結果の学習は全 RPE。保存するモデルも全部で学習し、予測は 1〜10 に収める
    assert "主結果は「学習: 全 RPE／" in report
    best = joblib.load(out / "best.pkl")
    assert best["train_min_rpe"] is None and best["pred_range"] == [1.0, 10.0]


def test_train_mean_baseline_predicts_training_mean():
    """比べる基準（mean_rpe）は特徴を使わず、学習データの RPE の平均を答える"""
    import train_rpe_model as trm
    train = pd.DataFrame({"rpe": [6.0, 7.0, 8.0, 9.0]})
    test = pd.DataFrame({"rpe": [5.0, 10.0]})
    res, use = trm.fit_predict("mean", [], train, test)
    assert use == [] and np.allclose(res[0], 7.5)
    assert trm.fit_predict("linear", [], train, test)[0] is None      # 列の無いふつうのモデルは今までどおり学習しない
    conds = trm.build_conditions(synthetic_features(), with_context=True)
    assert {"mean_rpe", "imu_lrv", "imu_vbt", "video_lrv", "video_vbt"} <= set(conds)
    assert conds["mean_rpe"]["baseline"] and "mean_rpe+ctx" not in conds


def test_curve_model_avoids_hgb_on_small_training_sets():
    """学習曲線は線形回帰（無ければリッジ）で描く。HGB は学習が 10 セット未満だと分けられず、平均を答えるだけになるため"""
    import train_rpe_model as trm
    assert trm.curve_model(["linear", "hgb"]) == "linear"          # context
    assert trm.curve_model(["ridge", "rf", "hgb"]) == "ridge"      # 特徴すべて
    assert trm.curve_model(["ridge", "hgb"]) == "ridge"            # --with-context の *+ctx
    assert trm.curve_model(["mean"]) == "mean"
    # 8 セット（本人の 2 セッション分ほど）。速いほど RPE が低い
    train = pd.DataFrame({"v": np.linspace(0.30, 0.60, 8)})
    train["rpe"] = 10 - 10 * (train["v"] - 0.30)
    test = pd.DataFrame({"v": [0.32, 0.58], "rpe": [9.8, 7.2]})
    hgb = trm.fit_predict("hgb", ["v"], train, test)[0][0]
    assert np.allclose(hgb, train["rpe"].mean())                   # 分けられず平均を答える
    ridge = trm.fit_predict(trm.curve_model(["ridge", "rf", "hgb"]), ["v"], train, test)[0][0]
    assert ridge[0] > ridge[1] + 1                                 # 速さの違いを拾う


def test_train_compares_training_rpe_range(tmp_path):
    """RPE 5 未満の軽いセットがあると「全 RPE」と「RPE 5 以上だけ」の2通りで学習し、同じテストで比べる"""
    df = synthetic_features(seed=1)
    rng = np.random.default_rng(3)
    light = rng.random(len(df)) < 0.15                      # 15% を軽いセット（RPE 3〜4.5）にする
    df.loc[light, "rpe"] = rng.choice([3.0, 3.5, 4.0, 4.5], light.sum())
    df.loc[light, "pct_1rm"] = rng.uniform(45, 60, light.sum())
    for c in ("pose_first_rep_mcv_bl", "imu_first_rep_mcv_mps"):
        df.loc[light & df[c].notna(), c] += 0.25            # 軽いので速い
    csv = tmp_path / "features_all.csv"
    df.to_csv(csv, index=False)
    import os
    # 画面への出力を UTF-8 で受け取る（何もしないと端末の文字コードで出るので、日本語の行を確かめられない）
    r = subprocess.run([sys.executable, str(SCRIPT_DIR / "train_rpe_model.py"), "--input", str(csv),
                        "--output-dir", str(tmp_path / "models"), "--kmax", "2"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = next((tmp_path / "models").iterdir())
    res = pd.read_csv(out / "results.csv")
    assert set(res["train_set"]) == {"all", "rpe>=5"}
    assert set(res["eval_set"]) == {"all", "rpe>=6"}
    # テストに使うセットは学習の範囲によらず同じ（n が一致する）
    key = ["scheme", "condition", "model", "eval_set"]
    n = res.pivot_table(index=key, columns="train_set", values="n")
    assert (n["all"] == n["rpe>=5"]).all()
    # RPE 5 以上だけで学習したモデルは 5 未満を予測しない
    pred = pd.read_csv(out / "predictions.csv")
    assert pred.loc[pred["train_set"] == "rpe>=5", "pred"].min() >= 5.0
    curves = pd.read_csv(out / "curves.csv")
    assert {"all", "rpe>=5"} <= set(curves["train_set"])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "学習データの範囲の比較" in report
    # 主結果（表・図・保存するモデル）は「RPE 5 以上だけで学習」（2026-11-17 ユーザー決定。RPE 4 以下は目安で付けていて、
    # 数字の違いに意味がないので学習に使わない）。「全 RPE で学習」は比較用の表に残る
    assert "主結果は「学習: RPE 5 以上／評価: RPE 6 以上」" in report
    assert "主結果（学習: RPE 5 以上／評価: RPE 6 以上）" in r.stdout
    assert "（学習: RPE 5 以上／評価: RPE 6 以上）" in report.split("## B. 汎用")[1].splitlines()[0]   # 表の見出しにも範囲
    import joblib
    import train_rpe_model as trm
    best = joblib.load(out / "best.pkl")
    main = res[(res["scheme"] == "B_loso") & (res["train_set"] == "rpe>=5") & (res["eval_set"] == "rpe>=6")
               & (res["condition"] != "mean_rpe")].sort_values("mae").iloc[0]
    assert (best["condition"], best["model_name"]) == (main["condition"], main["model"])
    # 保存するモデルも RPE 5 以上のセットだけで学習し、5 未満は予測しない
    assert best["train_min_rpe"] == 5.0 and best["pred_range"] == [5.0, 10.0]
    full = trm.mark_unusable(pd.read_csv(csv))
    elig = trm.eligible(full, trm.build_conditions(full, False)["video_vbt"])
    assert joblib.load(out / "video_vbt__ridge.pkl")["n_train"] == int((elig["rpe"] >= 5).sum()) < len(elig)


def test_predict_keeps_predictions_in_the_trained_range():
    """
    保存したモデルで予測するとき（predict_rpe.py）も、評価のときと同じ範囲に収める。RPE 5 以上だけで学習したモデルは
    5 未満を予測しない（5 と出たら「5 以下」の意味）。範囲を持たない古いモデルは今までどおり、そのまま返す
    """
    import predict_rpe as pr

    class Fixed:
        def predict(self, X):
            return np.array([3.2, 7.0, 11.5])[:len(X)]

    df = pd.DataFrame({"v": [0.9, 0.5, 0.1], "rpe": [4.0, 7.0, 10.0]})
    new = {"model": Fixed(), "feature_cols": ["v"], "pred_range": [5.0, 10.0]}
    assert list(pr.predict_dataframe(new, df)["predicted_rpe"]) == [5.0, 7.0, 10.0]
    assert pr.predict_single(new, {"v": 0.9}) == 5.0
    old = {"model": Fixed(), "feature_cols": ["v"]}
    assert list(pr.predict_dataframe(old, df)["predicted_rpe"]) == [3.2, 7.0, 11.5]
    assert pr.predict_single(old, {"v": 0.9}) == 3.2


# ---------------------------------------------------------------------------
# 学習に使えないセット（データの欠け・レップの取り違え）を外す（品質レポートと同じ基準）
# ---------------------------------------------------------------------------
def test_quality_marks_unusable_data():
    from quality_checks import check_set, exclusion_reasons
    import run_pipeline as rp
    ok = {"rpe": 8, "weight_kg": 90, "reps_completed": 8, "modality": "imu+video",
          "pose_available": True, "pose_n_reps_detected": 8, "pose_first_rep_mcv_bl": 0.4,
          "pose_ankle_y_max_norm": 0.8, "pose_shoulder_width_bl": 0.25, "pose_nan_ratio": 0.0,
          "pose_key_visibility_mean": 0.99, "pose_velocity_loss_pct": 20.0,
          "imu_n_samples": 6000, "imu_fs_hz": 100.0, "imu_az_median_g": 1.04,
          "imu_n_reps_detected": 8, "imu_first_rep_mcv_mps": 0.5, "imu_velocity_loss_mcv_pct": 22.0}
    assert exclusion_reasons(ok) == {"imu": [], "video": []} and check_set(ok) == []
    # IMU が 1 回取りこぼした → IMU だけ外す（動画は使う）
    r = exclusion_reasons(ok | {"imu_n_reps_detected": 7})
    assert r["imu"] and not r["video"]
    assert rp.excluded_from(ok | {"imu_n_reps_detected": 7}) == "IMU"
    # 回数は合っているが、最後のレップが 1 回目より 25 % 速い（ラックインを数えた）→ IMU を外す
    assert exclusion_reasons(ok | {"imu_velocity_loss_mcv_pct": -24.9})["imu"]
    # 送信が途中で止まった（IMU のレップ 0）→ IMU を外す
    assert exclusion_reasons(ok | {"imu_n_reps_detected": 0, "imu_first_rep_mcv_mps": None})["imu"]
    # 足元が切れている → 動画だけ外す
    r = exclusion_reasons(ok | {"pose_ankle_y_max_norm": 1.0})
    assert r["video"] and not r["imu"]
    assert rp.excluded_from(ok | {"pose_ankle_y_max_norm": 1.0, "imu_n_reps_detected": 7}) == "IMU・動画"
    # キーポイントの信頼度が低い・M5 の向きは要確認として出すだけ（学習には使う）
    low = ok | {"pose_key_visibility_mean": 0.5, "imu_az_median_g": 0.5}
    assert exclusion_reasons(low) == {"imu": [], "video": []} and len(check_set(low)) == 2
    # 動画だけの協力者には IMU の確認をしない
    vid = {k: v for k, v in ok.items() if not k.startswith("imu_")} | {"modality": "video"}
    assert exclusion_reasons(vid) == {"imu": [], "video": []}


def test_collection_summary_counts_usable_sets():
    """集まり具合は学習に使えるセットで数える。要確認が付いただけのセット（学習には使う）を数から落とさない"""
    import run_pipeline as rp
    ok = {"subject_id": "S001", "session_id": "SES001", "rpe": 6, "weight_kg": 90, "reps_completed": 8,
          "modality": "imu+video", "pose_available": True, "pose_n_reps_detected": 8, "pose_first_rep_mcv_bl": 0.4,
          "pose_ankle_y_max_norm": 0.8, "pose_shoulder_width_bl": 0.25, "pose_nan_ratio": 0.0,
          "pose_key_visibility_mean": 0.99, "pose_velocity_loss_pct": 20.0,
          "imu_n_samples": 6000, "imu_fs_hz": 100.0, "imu_az_median_g": 1.04,
          "imu_n_reps_detected": 8, "imu_first_rep_mcv_mps": 0.5, "imu_velocity_loss_mcv_pct": 22.0}
    rows = [ok | {"set_no": 1},
            ok | {"set_no": 2, "rpe": 8, "pose_other_person_frames": 8},     # 人が横切った: 要確認だけ（学習に使う）
            ok | {"set_no": 3, "rpe": 8, "imu_n_reps_detected": 7},          # IMU だけ外れる
            ok | {"set_no": 4, "rpe": 9, "pose_ankle_y_max_norm": 1.0}]      # 動画だけ外れる
    vid = ({k: v for k, v in ok.items() if not k.startswith("imu_")}
           | {"subject_id": "S003", "modality": "video", "set_no": 1, "rpe": 7})   # 動画だけの協力者
    df = pd.DataFrame(rows + [vid])
    df["excluded_from"] = [rp.excluded_from(r) for r in df.to_dict("records")]
    s = {x["subject_id"]: x for x in rp.collection_summary(df)}
    assert (s["S001"]["trials"], s["S001"]["video"], s["S001"]["imu"]) == (4, 3, 3)
    assert s["S001"]["rpe_dist"] == {6.0: 1, 8.0: 2}        # 動画が使える 3 セットの RPE
    assert (s["S003"]["video"], s["S003"]["imu"]) == (1, None)


def test_train_excludes_unusable_sets(tmp_path):
    """データが欠けているセットは、そのデータ（IMU / 動画）を使う条件の学習・テストから外す"""
    df = synthetic_features(seed=2)
    df["pose_available"] = True
    df["pose_n_reps_detected"] = df["reps_completed"]
    imu_rows = df["modality"] == "imu+video"
    df.loc[imu_rows, "imu_n_reps_detected"] = df.loc[imu_rows, "reps_completed"]
    s1 = df.index[df["subject_id"] == "S001"]
    bad_imu = [s1[0], s1[1]]                     # IMU の取りこぼし ／ ラックインを数えた
    df.loc[bad_imu[0], "imu_n_reps_detected"] = df.loc[bad_imu[0], "reps_completed"] + 1
    df.loc[bad_imu[1], "imu_velocity_loss_mcv_pct"] = -25.0
    bad_vid = [s1[2], df.index[df["subject_id"] == "S003"][0]]   # 足元が切れている ／ 骨格のレップ数が違う
    df.loc[bad_vid[0], "pose_ankle_y_max_norm"] = 1.0
    df.loc[bad_vid[1], "pose_n_reps_detected"] = df.loc[bad_vid[1], "reps_completed"] + 1
    csv = tmp_path / "features_all.csv"
    df.to_csv(csv, index=False)
    r = subprocess.run([sys.executable, str(SCRIPT_DIR / "train_rpe_model.py"), "--input", str(csv),
                        "--output-dir", str(tmp_path / "models"), "--kmax", "2"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = next((tmp_path / "models").iterdir())
    pred = pd.read_csv(out / "predictions.csv")
    ids = lambda idx: {(df.at[i, "subject_id"], df.at[i, "session_id"], int(df.at[i, "set_no"])) for i in idx}
    got = lambda p: set(zip(p["subject_id"], p["session_id"], p["set_no"].astype(int)))
    uses_imu = pred["condition"].str.contains("imu")
    uses_video = pred["condition"].str.contains("video")
    assert not got(pred[uses_imu]) & ids(bad_imu)            # IMU を使う条件には入らない
    assert not got(pred[uses_video]) & ids(bad_vid)          # 動画を使う条件には入らない
    assert ids(bad_imu) <= got(pred[pred["condition"] == "video_all"])   # 動画の条件には残る
    assert ids(bad_vid[:1]) <= got(pred[pred["condition"] == "imu_all"]) # IMU の条件には残る
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "学習から外したセット" in report and "足元が画面の下端で切れている" in report


# ---------------------------------------------------------------------------
# 切っていない動画・申告より多いしゃがみ・動画と IMU の時刻合わせ（2026-10-30）
# ---------------------------------------------------------------------------
def with_waiting_near_rack(kp, pre_s=20.0, post_s=6.0, fs=30.0):
    """切っていない動画: セットの前後に、ラックの近く（カメラに近い）で立っている時間を足す。
    近いので大きく写る（肩が高く、足首は画面の下端）"""
    first = kp.iloc[0]
    rng = np.random.default_rng(5)

    def near(n):
        rows = {}
        for name in COCO:
            rows[f"{name}_x"] = 0.5 + (first[f"{name}_x"] - 0.5) * 1.88 + rng.normal(0, 0.001, n)
            rows[f"{name}_y"] = 0.15 + (first[f"{name}_y"] - 0.25) * 1.88 + rng.normal(0, 0.001, n)
            rows[f"{name}_conf"] = np.full(n, 0.95)
        return pd.DataFrame(rows)

    out = pd.concat([near(int(pre_s * fs)), kp.drop(columns=["frame", "time_s"]), near(int(post_s * fs))],
                    ignore_index=True)
    out.insert(0, "time_s", np.arange(len(out)) / fs)
    out.insert(0, "frame", np.arange(len(out)))
    return out


def test_pose_untrimmed_video_uses_only_the_set():
    """切っていない動画（協力者・切り忘れ）でも、セットの区間だけで体の大きさ・足元・レップの区切りを出す
    （2026-10-24 set04 では 1 回目が待ち時間まで伸び、骨格の速度低下が 58 % になっていた）"""
    kp, _ = make_keypoints(n_reps=5)
    a = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front", expected_reps=5)
    b = ef.compute_pose_features(with_waiting_near_rack(kp), 720, 1280, 30.0, view="front", expected_reps=5)
    assert b["pose_n_reps_detected"] == 5
    assert max(r["descent_s"] for r in b["_pose_reps"]) < 2.0          # 1 回目が待ち時間まで伸びない
    assert abs(b["pose_first_rep_mcv_bl"] / a["pose_first_rep_mcv_bl"] - 1) < 0.05
    assert abs(b["pose_velocity_loss_pct"] - a["pose_velocity_loss_pct"]) < 3
    assert b["pose_ankle_y_max_norm"] < 0.97                            # 足元の確認もセットの区間だけ
    assert abs(b["_pose_reps"][0]["bottom_t_s"] - a["_pose_reps"][0]["bottom_t_s"] - 20.0) < 0.1


def test_pose_keeps_first_reps_when_more_than_reported():
    """申告より多くしゃがんでいたら、時刻順に最初の N 回を残す（RPE 10 で潰れた最後のレップを外す）"""
    kp, _ = make_keypoints(n_reps=6)
    full = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front")
    f = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front", expected_reps=5)
    assert f["pose_n_reps_detected"] == 5
    assert [r["bottom_t_s"] for r in f["_pose_reps"]] == [r["bottom_t_s"] for r in full["_pose_reps"][:5]]


def imu_with_walkout_and_rack(n_reps=5, lead=6.0, tail=4.0):
    """IMU（SPACE〜SPACE）: レップの前後に、歩き出し・ラックインの小さな上下動（4 cm）が入る"""
    imu, v_true, asc = make_imu(n_reps=n_reps, lead=lead, tail=tail)
    t = imu["timestamp_ms"].to_numpy() / 1000.0
    bump = lambda t0: 0.04 * np.exp(-((t - t0) / 0.3) ** 2)
    pos = bump(2.0) + bump(t[-1] - 1.5)
    imu["az_g"] += np.gradient(np.gradient(pos, t), t) / G
    return imu, asc


def test_sync_video_imu_and_take_imu_reps_from_video(tmp_path):
    """切った動画（1 回目の 1 秒前から）と IMU（6 秒前から）のずれ 5 秒を波形で求め、
    IMU のレップを動画のレップ区間で取る。ファイルの日時は使わない（LINE で保存した動画でもよい）"""
    kp, asc = make_keypoints(n_reps=5, lead=1.0, tail=1.0)
    imu, _ = imu_with_walkout_and_rack()
    sync = ef.sync_video_imu(kp, 720, 1280, 30.0, imu)
    assert abs(sync["sync_offset_s"] - 5.0) < 0.05
    assert sync["sync_r"] >= ef.SYNC_R_MIN and sync["sync_margin"] >= ef.SYNC_MARGIN_MIN

    # 1 試技の通し: 骨格 → 時刻合わせ → IMU のレップは動画の区間で
    (tmp_path / "pose").mkdir()
    (tmp_path / "imu").mkdir()
    kp.to_csv(tmp_path / "pose" / "set01_yolo_keypoints.csv", index=False)
    (tmp_path / "pose" / "set01_yolo_info.json").write_text(
        json.dumps({"width": 720, "height": 1280, "fps": 30.0}), encoding="utf-8")
    imu.to_csv(tmp_path / "imu" / "set01_BAR.csv", index=False)
    meta = {"session_id": "SES001", "subject_id": "S999", "date": "2026-11-01",
            "sets": [{"set_no": 1, "reps_completed": 5, "rpe": 8, "weight_kg": 100}]}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    f = ef.extract_one_trial(tmp_path / "imu" / "set01_BAR.csv", tmp_path / "pose" / "set01_yolo_keypoints.csv",
                             tmp_path / "meta.json", 1)
    assert f["imu_rep_source"] == "video" and f["imu_n_reps_detected"] == 5
    assert abs(f["sync_offset_s"] - 5.0) < 0.05
    true_mcv = 0.45 / asc[0]
    assert abs(f["imu_first_rep_mcv_mps"] - true_mcv) / true_mcv < 0.15
    # 各レップの IMU の上昇ピークは、動画のそのレップの区間（IMU の時刻に直したもの）の中にある
    for rv, ri in zip(f["_pose_reps"], f["_imu_reps"]):
        assert rv["bottom_t_s"] + 5.0 - 0.3 <= ri["peak_t_s"] <= rv["end_t_s"] + 5.0 + 0.3


def test_imu_falls_back_when_video_does_not_match():
    """動画が別のセットのもの（回数・リズムが違う）なら時刻合わせを使わず、IMU だけで数える"""
    kp, _ = make_keypoints(n_reps=3, lead=1.0, tail=1.0, descent=0.8, rest=2.0)
    imu, _ = imu_with_walkout_and_rack(n_reps=5)
    sync = ef.sync_video_imu(kp, 720, 1280, 30.0, imu)
    ok = (sync.get("sync_r") or 0) >= ef.SYNC_R_MIN and (sync.get("sync_margin") or 0) >= ef.SYNC_MARGIN_MIN
    assert not ok


def test_quality_notes_failed_sync_without_excluding():
    """時刻合わせが不確かで IMU だけで数えたセットは、要確認として出すが学習からは外さない"""
    from quality_checks import check_set, exclusion_reasons
    row = {"rpe": 8, "weight_kg": 90, "reps_completed": 5, "modality": "imu+video", "pose_available": True,
           "pose_n_reps_detected": 5, "pose_first_rep_mcv_bl": 0.4, "imu_n_reps_detected": 5,
           "imu_first_rep_mcv_mps": 0.5, "imu_rep_source": "imu", "sync_r": 0.42, "sync_margin": 0.01}
    assert any("時刻合わせ" in i.text and not i.exclude for i in check_set(row))
    assert exclusion_reasons(row) == {"imu": [], "video": []}


def with_passerby(kp, t0=1.0, dur=0.47, shoulder_conf=0.9, hip_y=0.5):
    """カメラの前を人が横切る（2026-10-29 set01・10-20 set02 と同じ形）: dur 秒だけ追跡がカメラの近くの人に移る。
    その人の肩は画面の最上部、腰は hip_y、足首は画面の下端（信頼度 0）。最後の 2 コマは肩が見えず腰だけ残る。
    shoulder_conf を 0.3 未満にすると、肩が見えず腰・膝だけ見える横切り（10/20 set02）"""
    out = kp.copy()
    idx = np.where((out["time_s"] >= t0) & (out["time_s"] < t0 + dur))[0]
    rng = np.random.default_rng(7)
    for name in COCO:
        part = name.split("_")[-1]
        y = {"shoulder": 0.04, "hip": hip_y, "knee": 0.75, "ankle": 0.98}.get(part, 0.02)
        conf = {"shoulder": shoulder_conf, "hip": 0.8, "knee": 0.5, "ankle": 0.0}.get(part, 0.3)
        out.loc[idx, f"{name}_x"] = (0.75 if name.startswith("left") else 0.45) + rng.normal(0, 0.02, len(idx))
        out.loc[idx, f"{name}_y"] = y + rng.normal(0, 0.01, len(idx))
        out.loc[idx, f"{name}_conf"] = conf
    for c in ("left_shoulder_conf", "right_shoulder_conf"):
        out.loc[idx[-2:], c] = 0.1
    return out, idx


def test_pose_ignores_frames_tracking_a_passerby():
    """追跡が横切った人に移ったコマ（腰・足首も）を使わない。本人の速い動きのコマは弾かない。
    横切った人の足首で「足元切れ」と誤判定しない（10/20 set02 で発生）"""
    kp, _ = make_keypoints(n_reps=5)
    assert not ef._other_person_frames(kp, 720, 1280, 30.0).any()        # しゃがみの速い動きは弾かない
    bad, idx = with_passerby(kp)
    other = ef._other_person_frames(bad, 720, 1280, 30.0)
    assert set(np.where(other)[0]) == set(idx)                            # 肩が見えない最後の 2 コマも
    a = ef.compute_pose_features(kp, 720, 1280, 30.0, view="front", expected_reps=5)
    b = ef.compute_pose_features(bad, 720, 1280, 30.0, view="front", expected_reps=5)
    assert b["pose_other_person_frames"] == len(idx)
    assert b["pose_n_reps_detected"] == 5
    assert abs(b["pose_first_rep_mcv_bl"] / a["pose_first_rep_mcv_bl"] - 1) < 0.03
    assert abs(b["pose_mean_depth_bl"] / a["pose_mean_depth_bl"] - 1) < 0.03
    assert b["pose_ankle_y_max_norm"] < 0.97
    from quality_checks import check_set
    row = {"rpe": 8, "weight_kg": 90, "reps_completed": 5, "modality": "video", **b}
    assert any("別の人" in i.text and not i.exclude for i in check_set(row))


def test_pose_passerby_seen_only_by_hips():
    """横切った人の肩が見えず腰・膝だけ見える（10/20 set02 と同じ形）ときも、そのコマを使わない"""
    kp, _ = make_keypoints(n_reps=5)
    bad, idx = with_passerby(kp, shoulder_conf=0.02, hip_y=0.38)
    assert set(np.where(ef._other_person_frames(bad, 720, 1280, 30.0))[0]) == set(idx)
    b = ef.compute_pose_features(bad, 720, 1280, 30.0, view="front", expected_reps=5)
    assert b["pose_other_person_frames"] == len(idx)
    assert b["pose_ankle_y_max_norm"] < 0.97


def test_pose_left_right_swap_is_not_another_person():
    """本人の肩・腰の左右のラベルが入れ替わっただけのコマ（10/29 set04 に 1 コマずつ 2 回）は別の人としない"""
    kp, _ = make_keypoints(n_reps=5)
    sw = kp.copy()
    idx = np.where((sw["time_s"] >= 2.0) & (sw["time_s"] < 2.5))[0]
    for part in ("shoulder", "hip"):
        for c in ("x", "y", "conf"):
            cols = [f"left_{part}_{c}", f"right_{part}_{c}"]
            sw.loc[idx, cols] = sw.loc[idx, cols[::-1]].to_numpy()
    assert not ef._other_person_frames(sw, 720, 1280, 30.0).any()


def test_quality_notes_imu_without_still_period():
    """IMU に静止区間がないセットは要確認として出すが、学習からは外さない（2026-10-31 ユーザー判断）"""
    from quality_checks import check_set, exclusion_reasons
    row = {"rpe": 9, "weight_kg": 160, "reps_completed": 1, "modality": "imu+video", "pose_available": True,
           "pose_n_reps_detected": 1, "pose_first_rep_mcv_bl": 0.3, "imu_n_reps_detected": 1,
           "imu_first_rep_mcv_mps": 0.3, "imu_rep_source": "video", "imu_n_still": 0}
    assert any("静止区間" in i.text and not i.exclude for i in check_set(row))
    assert exclusion_reasons(row) == {"imu": [], "video": []}
    assert not any("静止区間" in i.text for i in check_set({**row, "imu_n_still": 2}))


# ---------------------------------------------------------------------------
# IMU の時刻は M5 の時計で・動画と IMU のレップ単位の比較（2026-10-31）
# ---------------------------------------------------------------------------
def m5_like_receive(imu):
    """実機のような受信: M5 の時計（boot_ms）は 10 ms ごと。PC の受信時刻は Wi-Fi で揺れ、
    遅れが 0.9 ms ずつ増えて 10 個ごとに戻る（間隔 10.9 ms × 9 + 1.9 ms → 中央値では 92 Hz に見える）。
    2 秒ごとに 4 個欠ける"""
    out = imu.copy()
    k = np.arange(len(out))
    out.insert(1, "boot_ms", 123456 + k * 10)
    out["timestamp_ms"] = out["timestamp_ms"] + (k % 10) * 0.9
    return out[(k % 200) < 196].reset_index(drop=True)


def test_imu_uses_device_clock_not_receive_time():
    """受信時刻の揺れ・欠けがあっても、M5 の時計で測れば速度・上昇時間は変わらない
    （受信間隔の中央値を時間刻みにしていた頃は、実データで速度がセットごとに −15〜+11 % ずれていた）"""
    imu, _, asc = make_imu(n_reps=5)
    clean = ef.compute_imu_features(imu)
    got = ef.compute_imu_features(m5_like_receive(imu))
    assert got["imu_n_reps_detected"] == 5
    assert abs(got["imu_first_rep_mcv_mps"] / clean["imu_first_rep_mcv_mps"] - 1) < 0.02
    assert abs(got["imu_last_rep_mcv_mps"] / clean["imu_last_rep_mcv_mps"] - 1) < 0.02
    assert abs(got["imu_first_rep_concentric_s"] - clean["imu_first_rep_concentric_s"]) < 0.03
    assert 97.0 < got["imu_fs_hz"] < 99.0                     # 受信レート: 2 % 欠けた分だけ下がる
    assert got["imu_n_still"] >= 1                            # 1 回目の前に立っている時間がある
    assert abs(got["_imu_reps"][0]["peak_t_s"] - clean["_imu_reps"][0]["peak_t_s"]) < 0.03   # 時刻の基準は同じ


def test_compare_video_imu_agrees_on_true_scale():
    """肩の上下動とバーが同じ動き・換算の倍率も正しいなら、動画と IMU の MCV・上昇時間は一致する。
    身長を 10 % 大きく入れると、動画の MCV も 10 % 大きく出る（倍率のずれがそのまま出る）"""
    import compare_video_imu as cv
    kp, asc = make_keypoints(n_reps=5, lead=1.0, tail=1.0)
    imu, _ = imu_with_walkout_and_rack()
    imu = m5_like_receive(imu)
    feat = {**ef.compute_pose_features(kp, 720, 1280, 30.0, view="front", expected_reps=5),
            **ef.sync_video_imu(kp, 720, 1280, 30.0, imu)}
    height_cm = 0.45 / 0.9 / ef.SHOULDER_ANKLE_RATIO * 100     # make_keypoints の肩の沈み 0.9 BL = 0.45 m
    rows = pd.DataFrame(cv.compare_set(feat, kp, 720, 1280, 30.0, imu, height_cm, expected_reps=5))
    assert len(rows) == 5
    s = cv.summarize(rows).set_index("metric")
    assert abs(s.loc["mcv", "ratio"] - 1) < 0.05
    assert abs(s.loc["ascent_s", "bias"]) < 0.05
    # 底の時刻がそろう（合成の底は 0.2 s 止まるので、その中のどこを底とするかは揺れる）
    assert rows["bottom_gap_s"].abs().median() < 0.07 and rows["bottom_gap_s"].abs().max() < 0.25
    big = pd.DataFrame(cv.compare_set(feat, kp, 720, 1280, 30.0, imu, height_cm * 1.1, expected_reps=5))
    assert abs(big["vid_mcv"].mean() / rows["vid_mcv"].mean() - 1.1) < 0.01


# ---------------------------------------------------------------------------
# 横から撮った動画（プレートの中心）と IMU（2026-11-03）
# ---------------------------------------------------------------------------
def side_truth(n_reps=5):
    """横から撮るセットの合成: 時刻 t [s]（受信時刻 − 1000 s = 動画の時刻）と、プレートの中心（バー）の
    上下の位置 [m]（上が正）。ラックアウト・ラックインの小さな上下動（4 cm）つき"""
    t, y, asc = squat_trajectory(1000.0, n_reps=n_reps, lead=6.0, tail=4.0)
    bump = lambda t0: 0.04 * np.exp(-((t - t0) / 0.3) ** 2)
    return t, -y + bump(2.0) + bump(t[-1] - 1.5), asc


def side_receiver_folder(folder, t, pos, reps=5):
    """受信フォルダ（raw_BAR.csv・markers.csv）。SPACE は動画の 0.5 秒目と、終わりの 0.5 秒前"""
    ti = np.arange(0.0, t[-1], 0.01)
    a = np.gradient(np.gradient(np.interp(ti, t, pos), 0.01), 0.01)
    rng = np.random.default_rng(2)
    n = len(ti)
    pd.DataFrame({"recv_time_s": 1000.0 + ti, "boot_ms": 123456 + np.arange(n) * 10,
                  "ax_g": rng.normal(0, 0.01, n), "ay_g": rng.normal(0, 0.01, n),
                  "az_g": 1.05 + a / G + rng.normal(0, 0.01, n),
                  "gx_dps": -3.8 + rng.normal(0, 1, n), "gy_dps": -12.9 + rng.normal(0, 1, n),
                  "gz_dps": 16.0 + rng.normal(0, 1, n)}).to_csv(folder / "raw_BAR.csv", index=False)
    start, end = 1000.5, 1000.0 + t[-1] - 0.5
    pd.DataFrame([{"set_no": 1, "start_s": start, "end_s": end, "duration_s": end - start,
                   "rpe": None, "weight_kg": 60, "reps": reps}]).to_csv(folder / "markers.csv", index=False)


def write_plate_video(path, t, pos, fps=30.0, r_px=40.0, w=360, h=640):
    """横から撮った合成の動画: プレート（直径 2 r_px = 450 mm。内側に輪と穴）が pos のとおりに上下する"""
    import cv2
    px_per_m = 2 * r_px / 0.45
    bg = np.random.default_rng(3).integers(165, 205, (h, w, 3), dtype=np.uint8)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sub = lambda v: int(round(v * 16))                       # 1/16 px 単位（shift=4）
    for k in range(int(t[-1] * fps)):
        img = bg.copy()
        c = (sub(w / 2), sub(0.35 * h - np.interp(k / fps, t, pos) * px_per_m))
        cv2.circle(img, c, sub(r_px), (40, 40, 40), -1, cv2.LINE_AA, 4)
        cv2.circle(img, c, sub(0.55 * r_px), (110, 110, 110), 3, cv2.LINE_AA, 4)
        cv2.circle(img, c, sub(0.15 * r_px), (225, 225, 225), -1, cv2.LINE_AA, 4)
        out.write(img)
    out.release()


def test_compare_side_imu_plate_matches_imu(tmp_path):
    """横の動画のプレートの中心と IMU が同じ動きなら、波形で時刻が合い（動画は SPACE の 0.5 秒前から）、
    MCV・上昇時間も一致する。プレートの直径を 10 % 大きく入れると、横の MCV も 10 % 大きく出る（物差しがそのまま効く）。
    日本語を含むフォルダでも、プレートの確認画像が書ける（cv2.imwrite は書けない）"""
    import compare_side_imu as cs
    folder = tmp_path / "横から" / "session_20261105_180000_60kg_5rep"
    folder.mkdir(parents=True)
    t, pos, _ = side_truth()
    side_receiver_folder(folder, t, pos)
    write_plate_video(folder / "side01.mp4", t, pos)
    rows, info = cs.compare_session(folder, plate_mm=450.0, allow_gui=False)
    assert len(rows) == 5 and info[0]["n_reps_detected"] == 5
    assert abs(info[0]["sync_offset_s"] + 0.5) < 0.05
    assert info[0]["sync_r"] >= ef.SYNC_R_MIN and info[0]["sync_margin"] >= ef.SYNC_MARGIN_MIN
    reps = pd.DataFrame(rows)
    assert abs(reps["vid_mcv"].mean() / reps["imu_mcv"].mean() - 1) < 0.05
    assert abs(reps["diff_ascent_s"].mean()) < 0.05
    assert (folder / "plate" / "side01_bar_roi_preview.png").exists()
    big, _ = cs.compare_session(folder, plate_mm=495.0, allow_gui=False)   # 追跡は保存したものを使い、物差しだけ変わる
    assert abs(pd.DataFrame(big)["vid_mcv"].mean() / reps["vid_mcv"].mean() - 1.1) < 0.01


def test_compare_side_imu_skips_reps_where_plate_was_hidden(tmp_path):
    """プレートを 0.2 s より長く見失った（ラックなどに隠れた）レップは比べない。0.1 s ならつないで比べる"""
    import compare_side_imu as cs
    t, pos, asc = side_truth()
    fps, mm_per_px = 30.0, 5.0
    tv = np.arange(int(t[-1] * fps)) / fps
    y_px = 300.0 - np.interp(tv, t, pos) * 1000.0 / mm_per_px
    starts = 6.0 + np.concatenate([[0.0], np.cumsum([2.0 + a for a in asc])])   # 各レップの下降の始まり
    bottom = lambda i: starts[i] + 1.3                                          # 底（0.2 s 止まる中ほど）
    tracked = ~(((tv >= bottom(2) - 0.15) & (tv < bottom(2) + 0.15))          # 3 回目: 0.3 s
                | ((tv >= bottom(3) - 0.05) & (tv < bottom(3) + 0.05)))       # 4 回目: 0.1 s
    path = pd.DataFrame({"frame": np.arange(len(tv)), "y_px": np.where(tracked, y_px, np.nan),
                         "tracked": tracked})
    summary = {"fps": fps, "mm_per_px": mm_per_px, "plate_diameter_mm": 450.0,
               "scale_source": "hough_radius", "tracked_ratio": float(tracked.mean())}
    side_receiver_folder(tmp_path, t, pos)
    m = pd.read_csv(tmp_path / "markers.csv").iloc[0]
    imu = cs.cut_imu(pd.read_csv(tmp_path / "raw_BAR.csv"), m["start_s"], m["end_s"])
    rows, info = cs.compare_set(path, summary, imu, 450.0, expected_reps=5)
    assert info["n_reps_detected"] == 5 and info["n_lost_skipped"] == 1
    assert [r["rep"] for r in rows] == [1, 2, 4, 5]
    assert abs(rows[2]["lost_s"] - 0.1) < 0.04
    reps = pd.DataFrame(rows)
    assert abs(reps["vid_mcv"].mean() / reps["imu_mcv"].mean() - 1) < 0.05


def test_side_judgement_against_front_ratio():
    """セットごとの倍率の 95 % 信頼区間が 1.00 だけを含めば IMU は正しい、前の動画の 1.09 だけなら IMU が小さい、
    両方なら決められない"""
    import compare_side_imu as cs
    assert cs.judge([1.00, 1.02, 0.99, 1.01, 0.98, 1.00], 1.09)["verdict"].startswith("IMU はほぼ正しい")
    assert cs.judge([1.08, 1.10, 1.09, 1.07, 1.11, 1.09], 1.09)["verdict"].startswith("IMU が小さく")
    assert cs.judge([1.00, 1.09], 1.09)["verdict"].startswith("まだ決められない")
    assert cs.judge([1.05], 1.09)["verdict"].startswith("セットが足りない")


# ---------------------------------------------------------------------------
# IMU の校正（M5 の 6 つの面を順に下にして置いた記録 → 軸ごとのゼロ点のずれと感度）
# ---------------------------------------------------------------------------
FACES = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]   # 上を向く軸（z+ = 画面が上）


def make_calibration_recording(offset, sens, faces=FACES, rounds=2, hold=10.0, move=3.0, pause=0.0,
                               tilt_deg=8.0, fs=100.0, seed=0):
    """
    M5 を机の上で、面を順に下にして置いた受信の記録（raw_BAR.csv と同じ列）を作る。
    センサの読み = 感度 × 本当の加速度 + ゼロ点のずれ + ノイズ（軸ごと。単位 g）。
    置き方はきっちり垂直ではなく、tilt_deg まで斜め。置き換える間（move 秒）は手で持って動かす。
    pause > 0 なら、置き換える途中でその秒数だけ手が止まる（置き方には数えないはずの短い静止）
    """
    rng = np.random.default_rng(seed)

    def tilted(d):
        d = np.asarray(d, float)
        side = np.cross(d, rng.normal(size=3))
        side /= np.linalg.norm(side)
        u = d + np.tan(np.radians(rng.uniform(0, tilt_deg))) * side
        return u / np.linalg.norm(u)

    def moving(n):      # 手で持っている間: 向きが定まらず、加速度もジャイロも大きく揺れる
        u = rng.normal(size=(n, 3))
        return u / np.linalg.norm(u, axis=1, keepdims=True) + rng.normal(0, 0.15, (n, 3)), rng.normal(0, 60, (n, 3))

    def resting(u, n):
        return np.tile(u, (n, 1)), np.zeros((n, 3))

    parts = [moving(int(move * fs))]
    for _ in range(rounds):
        for d in faces:
            parts.append(resting(tilted(d), int(hold * fs)))
            if pause > 0:
                parts += [moving(int(move * fs / 2)), resting(tilted(rng.normal(size=3)), int(pause * fs))]
            parts.append(moving(int(move * fs)))
    acc = np.vstack([p[0] for p in parts])
    gyr = np.vstack([p[1] for p in parts])
    n = len(acc)
    acc = acc * np.asarray(sens, float) + np.asarray(offset, float) + rng.normal(0, 0.008, (n, 3))
    gyr = gyr + np.array([-4.3, -15.4, 17.8]) + rng.normal(0, 0.3, (n, 3))      # M5 のジャイロはオフセットが大きい
    t = np.arange(n) / fs
    return pd.DataFrame({"recv_time_s": t + rng.uniform(0, 0.004, n), "boot_ms": 500000 + np.round(t * 1000).astype(int),
                         "ax_g": acc[:, 0], "ay_g": acc[:, 1], "az_g": acc[:, 2],
                         "gx_dps": gyr[:, 0], "gy_dps": gyr[:, 1], "gz_dps": gyr[:, 2]})


def test_calibration_recovers_zero_point_and_sensitivity():
    """6 つの面を 2 周（置き方は 8° まで斜め）置いた記録から、軸ごとのゼロ点のずれと感度が出る。
    z の感度が 4.5 % 大きいセンサでは、いつもの付け方（画面が上 = z が上）の速度も約 4.5 % 大きく出ると判定する"""
    import calibrate_imu as ci
    offset, sens = (-0.018, 0.012, 0.010), (1.010, 0.985, 1.045)
    res = ci.calibrate(make_calibration_recording(offset, sens))
    assert res["ok"] and res["n_places"] == 12
    assert np.allclose(res["offset_g"], offset, atol=0.003)
    assert np.allclose(res["sensitivity"], sens, atol=0.003)
    assert res["rms_mg"] < 3.0                                   # 補正した後の大きさは、どの置き方でもほぼ 1 g
    assert abs(res["velocity_factor"] - 1.045) < 0.005
    lo, hi = res["velocity_factor_ci"]
    assert lo < 1.045 < hi and hi - lo < 0.01
    assert res["verdict"].startswith("IMU の速度が約 4.") and "大きく" in res["verdict"]


def test_calibration_zero_point_shift_leaves_velocity_right():
    """静止で 1.046 g と出ても、それが z のゼロ点のずれ（+0.046 g）なら感度は 1.00 で、IMU の速度の倍率は正しいと判定する。
    言えるのは倍率（平均して本当の何倍に出るか）までで、レップごとの値まで正しいとは言わない（2026-11-16）"""
    import calibrate_imu as ci
    res = ci.calibrate(make_calibration_recording((-0.018, 0.0, 0.046), (1.0, 1.0, 1.0), seed=1))
    assert res["ok"]
    assert abs(res["velocity_factor"] - 1.0) < 0.005
    assert abs(res["rest_reading_g"] - 1.046) < 0.003            # いつもの付け方で静止したときの読み
    assert res["verdict"].startswith("IMU の速度の倍率は正しい")


def test_calibration_needs_all_six_faces():
    """置いていない面があると計算しない（足りない向きを返す）。ここでは y の 2 面を置いていない"""
    import calibrate_imu as ci
    res = ci.calibrate(make_calibration_recording((0.0, 0.0, 0.046), (1.0, 1.0, 1.0), faces=FACES[:4]))
    assert not res["ok"] and res["n_places"] == 8
    assert res["missing"] == ["y+", "y-"]
    assert "sensitivity" not in res and "velocity_factor" not in res


def test_calibration_ignores_short_stops():
    """置き換える途中で 1 秒だけ手が止まったところは、置き方に数えない（3 秒以上の静止だけ使う）"""
    import calibrate_imu as ci
    res = ci.calibrate(make_calibration_recording((0.0, 0.0, 0.046), (1.0, 1.0, 1.0), pause=1.0, seed=2))
    assert res["ok"] and res["n_places"] == 12
    assert abs(res["velocity_factor"] - 1.0) < 0.005


def test_calibration_single_round_gives_no_error_range():
    """1 周（6 回）だけだと値は出るが、合い具合と幅は出せない（未知数 6 個に置き方 6 回でちょうど）"""
    import calibrate_imu as ci
    res = ci.calibrate(make_calibration_recording((0.0, 0.0, 0.046), (1.0, 1.0, 1.045), rounds=1, seed=3))
    assert res["ok"] and res["n_places"] == 6
    assert abs(res["velocity_factor"] - 1.045) < 0.005
    assert res["rms_mg"] is None and res["velocity_factor_ci"] is None


def test_calibration_session_reads_newest_folder_and_saves_result(tmp_path):
    """受信フォルダ（日本語を含む場所でも）の raw_BAR.csv を読み、結果を同じフォルダの calibration_BAR.json に保存する。
    フォルダを指定しないときは、いちばん新しい受信フォルダを使う"""
    import calibrate_imu as ci
    root = tmp_path / "校正"
    old, new = root / "session_20261115_100000", root / "session_20261116_090000"
    for folder, sens_z in ((old, 1.0), (new, 1.045)):
        folder.mkdir(parents=True)
        make_calibration_recording((0.0, 0.0, 0.046), (1.0, 1.0, sens_z)).to_csv(folder / "raw_BAR.csv", index=False)
    assert ci.find_latest_session(root) == new
    res = ci.calibrate_session(new)
    saved = json.loads((new / "calibration_BAR.json").read_text(encoding="utf-8"))
    assert saved["device"] == "BAR" and saved["ok"]
    assert abs(saved["sensitivity"][2] - 1.045) < 0.003 and saved["verdict"] == res["verdict"]


def test_calibration_session_saves_nothing_without_six_faces(tmp_path):
    """6 つの面がそろわない記録（バーに付けたままのふつうの受信フォルダなど）には、結果のファイルを作らない"""
    import calibrate_imu as ci
    make_calibration_recording((0.0, 0.0, 0.046), (1.0, 1.0, 1.0), faces=FACES[:1]).to_csv(tmp_path / "raw_BAR.csv",
                                                                                         index=False)
    res = ci.calibrate_session(tmp_path)
    assert not res["ok"] and res["missing"] == ["x+", "x-", "y+", "y-", "z-"]
    assert [p.name for p in tmp_path.iterdir()] == ["raw_BAR.csv"]


def test_calibration_reads_past_rest_readings(tmp_path):
    """校正の結果と見比べるために、これまでのセットの静止時の値（特徴量の一覧の imu_gravity_ref_g）を読む。
    一覧が無ければ None。校正を当てはめて作ったセット（imu_calibrated）は別に読む: 当てはめる前は 1.047 g 前後、
    当てはめた後は 1.00 g 前後になるので、混ぜると見比べられない"""
    import calibrate_imu as ci
    path = tmp_path / "features_all.csv"
    assert ci.past_rest_readings(path) is None
    pd.DataFrame({"set_no": [1, 2, 3], "imu_gravity_ref_g": [1.046, np.nan, 1.048]}).to_csv(path, index=False)
    assert ci.past_rest_readings(path).tolist() == [1.046, 1.048]
    assert ci.past_rest_readings(path, calibrated=True) is None          # 校正を当てはめたセットはまだ無い
    pd.DataFrame({"set_no": [1, 2, 3, 4], "imu_gravity_ref_g": [1.046, 1.001, np.nan, 0.999],
                  "imu_calibrated": [False, True, np.nan, True]}).to_csv(path, index=False)   # 3 は動画だけのセット
    assert ci.past_rest_readings(path).tolist() == [1.046]
    assert ci.past_rest_readings(path, calibrated=True).tolist() == [1.001, 0.999]


# ---------------------------------------------------------------------------
# 校正を速度の計算に当てはめる（2026-11-16）
#   calibrate_imu.py --apply が data/_calib/calibration_<デバイス>.json を保存し、extract_features が IMU を読むときに
#   本当の加速度 = (読み − ゼロ点のずれ) / 感度 を当てはめる。ファイルが無ければ、これまでと同じ計算
# ---------------------------------------------------------------------------
SENSOR_OFFSET, SENSOR_SENS = (-0.018, 0.0, 0.049), (0.998, 0.996, 1.045)   # z の感度が 4.5 % 大きいセンサ


def sensor_reading(true_imu, offset=SENSOR_OFFSET, sens=SENSOR_SENS):
    """ずれの無いセンサの IMU（make_imu(g_bias=1.0)）を、感度 sens・ゼロ点のずれ offset のセンサの読みに作り替える"""
    read = true_imu.copy()
    for c, b, s in zip(("ax_g", "ay_g", "az_g"), offset, sens):
        read[c] = true_imu[c] * s + b
    return read


def calibration_folders(root, offset=SENSOR_OFFSET, sens=SENSOR_SENS, n=3, faces_each=None):
    """校正の記録を n 回（受信フォルダ n 個。1 回につき面を 1 周）作る。faces_each で記録ごとに置く面を変えられる"""
    folders = []
    for k in range(n):
        folder = root / f"session_20261115_14{k:02d}00"
        folder.mkdir(parents=True)
        faces = faces_each[k] if faces_each else FACES
        make_calibration_recording(offset, sens, faces=faces, rounds=1, seed=10 + k).to_csv(folder / "raw_BAR.csv",
                                                                                           index=False)
        folders.append(folder)
    return folders


def test_apply_calibration_undoes_zero_point_and_sensitivity():
    """校正を当てはめると、センサの読み（感度 × 本当の加速度 + ゼロ点のずれ）が本当の加速度に戻る。
    ジャイロ・時刻と、元の表は変えない。校正が無ければ（None）表をそのまま返す"""
    true, _, _ = make_imu(n_reps=3, g_bias=1.0)
    read = sensor_reading(true)
    before = read.copy()
    got = ef.apply_calibration(read, {"offset_g": list(SENSOR_OFFSET), "sensitivity": list(SENSOR_SENS)})
    for c in ("ax_g", "ay_g", "az_g"):
        assert np.allclose(got[c], true[c], atol=1e-12)
    for c in ("timestamp_ms", "gx_dps", "gy_dps", "gz_dps"):
        assert got[c].equals(read[c])
    pd.testing.assert_frame_equal(read, before)
    assert ef.apply_calibration(read, None) is read


def test_imu_velocity_is_right_after_calibration_when_sensitivity_is_off():
    """z の感度が 4.5 % 大きいセンサでは、そのままだと速度が約 4.5 % 大きく出る（重力の実測値を引いても感度のずれは残る）。
    校正を当てはめると、ずれの無いセンサと同じ速度になり、静止時の値も 1 g になる"""
    true, _, _ = make_imu(n_reps=5, g_bias=1.0)
    read = sensor_reading(true)
    ref = ef.compute_imu_features(true)
    raw = ef.compute_imu_features(read)
    cal = ef.compute_imu_features(ef.apply_calibration(read, {"offset_g": list(SENSOR_OFFSET),
                                                              "sensitivity": list(SENSOR_SENS)}))
    assert ref["imu_n_reps_detected"] == raw["imu_n_reps_detected"] == cal["imu_n_reps_detected"] == 5
    for key in ("imu_first_rep_mcv_mps", "imu_last_rep_mcv_mps"):
        assert abs(raw[key] / ref[key] - 1.045) < 0.01
        assert abs(cal[key] / ref[key] - 1.0) < 0.002
    assert abs(raw["imu_gravity_ref_g"] - 1.094) < 0.005         # 1.045 + 0.049（z が上）
    assert abs(cal["imu_gravity_ref_g"] - 1.0) < 0.005


def test_load_calibration_reads_the_device_file(tmp_path):
    """解析に使う校正は data/_calib/calibration_<デバイス>.json。無ければ None（当てはめない）。
    値の足りないファイルは、黙って無視せず止める（気づかないまま数字が変わらないように）"""
    assert ef.load_calibration(tmp_path, "BAR") is None
    (tmp_path / "_calib").mkdir()
    assert ef.load_calibration(tmp_path, "BAR") is None
    calib = {"device": "BAR", "offset_g": list(SENSOR_OFFSET), "sensitivity": list(SENSOR_SENS),
             "sources": ["session_20261115_143304"]}
    path = tmp_path / "_calib" / "calibration_BAR.json"
    path.write_text(json.dumps(calib), encoding="utf-8")
    got = ef.load_calibration(tmp_path, "BAR")
    assert got["offset_g"] == calib["offset_g"] and got["sensitivity"] == calib["sensitivity"]
    assert got["sources"] == calib["sources"]
    assert ef.load_calibration(tmp_path, "HIP") is None          # ほかのデバイスの校正は使わない
    path.write_text(json.dumps({"offset_g": [0.0, 0.0, 0.0]}), encoding="utf-8")
    with pytest.raises(ValueError):
        ef.load_calibration(tmp_path, "BAR")


def test_calibration_all_pools_recordings_into_one_file(tmp_path):
    """1 周だけの記録が 3 回あるとき、置き方 18 回をまとめて 1 つの当てはめにし（1 回ずつでは出せない合い具合と幅も出る）、
    解析に使う校正として data/_calib の直下に保存する。記録ごとの倍率も返す（記録どうしが合っているかの確認用）。
    記録のフォルダには何も足さない"""
    import calibrate_imu as ci
    root = tmp_path / "data" / "_calib"
    folders = calibration_folders(root)
    res = ci.calibrate_all(root)
    assert res["ok"] and res["n_places"] == 18 and res["dof"] == 12
    assert res["device"] == "BAR" and res["sources"] == [f.name for f in folders]
    assert np.allclose(res["offset_g"], SENSOR_OFFSET, atol=0.003)
    assert np.allclose(res["sensitivity"], SENSOR_SENS, atol=0.003)
    assert res["rms_mg"] is not None and res["velocity_factor_ci"] is not None
    assert [s["source"] for s in res["per_source"]] == res["sources"]
    assert all(s["n_places"] == 6 and abs(s["velocity_factor"] - res["velocity_factor"]) < 0.01
               for s in res["per_source"])
    saved = json.loads((root / "calibration_BAR.json").read_text(encoding="utf-8"))
    assert saved["offset_g"] == res["offset_g"] and saved["sensitivity"] == res["sensitivity"]
    assert saved["device"] == "BAR" and saved["sources"] == res["sources"]
    assert all([p.name for p in f.iterdir()] == ["raw_BAR.csv"] for f in folders)


def test_calibration_all_completes_six_faces_across_recordings(tmp_path):
    """1 回の記録では面が足りなくても、まとめて 6 つの面がそろえば計算する（その記録だけの倍率は出せないので None）"""
    import calibrate_imu as ci
    root = tmp_path / "_calib"
    calibration_folders(root, n=2, faces_each=[FACES[:4], FACES[2:]])
    res = ci.calibrate_all(root)
    assert res["ok"] and res["n_places"] == 8
    assert np.allclose(res["sensitivity"], SENSOR_SENS, atol=0.003)
    assert [s["velocity_factor"] for s in res["per_source"]] == [None, None]


def test_calibration_all_saves_nothing_without_six_faces(tmp_path):
    """まとめても 6 つの面がそろわなければ保存しない。前に保存した校正はそのまま残る"""
    import calibrate_imu as ci
    root = tmp_path / "_calib"
    calibration_folders(root, n=2, faces_each=[FACES[:4], FACES[:2]])
    (root / "calibration_BAR.json").write_text('{"keep": true}', encoding="utf-8")
    res = ci.calibrate_all(root)
    assert not res["ok"] and res["missing"] == ["y+", "y-"] and res["n_places"] == 6
    assert json.loads((root / "calibration_BAR.json").read_text(encoding="utf-8")) == {"keep": True}


def test_features_apply_saved_calibration_and_record_it(tmp_path):
    """校正の記録 → calibrate_imu.calibrate_all が保存 → extract_features が IMU を読むときに当てはめる、の通し。
    z の感度が 4.5 % 大きいセンサ: 校正のファイルが無ければこれまでと同じ計算（速度は約 4.5 % 大きいまま）、
    あれば本当の速度になり、特徴量に「校正を当てはめた」ことと使った値が残る。IMU の CSV は書き換えない"""
    import calibrate_imu as ci
    data = tmp_path / "data"
    ses = data / "S999" / "SES001"
    (ses / "imu").mkdir(parents=True)
    true, _, _ = make_imu(n_reps=5, g_bias=1.0)
    imu_csv = ses / "imu" / "set01_BAR.csv"
    sensor_reading(true).to_csv(imu_csv, index=False)
    csv_bytes = imu_csv.read_bytes()
    meta = {"session_id": "SES001", "subject_id": "S999", "date": "2026-11-16",
            "sets": [{"set_no": 1, "reps_completed": 5, "rpe": 8, "weight_kg": 100}]}
    (ses / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ref = ef.compute_imu_features(true, expected_reps=5)["imu_first_rep_mcv_mps"]

    f0 = ef.extract_one_trial(imu_csv, None, ses / "meta.json", 1)
    assert f0["imu_calibrated"] is False and f0["_imu_calibration"] is None
    as_before = ef.compute_imu_features(pd.read_csv(imu_csv), expected_reps=5)
    assert f0["imu_first_rep_mcv_mps"] == as_before["imu_first_rep_mcv_mps"]
    assert abs(f0["imu_first_rep_mcv_mps"] / ref - 1.045) < 0.01

    calibration_folders(data / "_calib")
    res = ci.calibrate_all(data / "_calib")
    f1 = ef.extract_one_trial(imu_csv, None, ses / "meta.json", 1)
    assert f1["imu_calibrated"] is True
    assert f1["_imu_calibration"] == {"offset_g": res["offset_g"], "sensitivity": res["sensitivity"],
                                      "sources": res["sources"]}
    assert abs(f1["imu_first_rep_mcv_mps"] / ref - 1.0) < 0.01
    assert abs(f1["imu_gravity_ref_g"] - 1.0) < 0.005
    assert imu_csv.read_bytes() == csv_bytes


def test_compare_side_imu_applies_saved_calibration(tmp_path):
    """横から撮るセットの比較でも、保存した校正を IMU に当てはめる（前の動画との比較と同じ IMU の速度で比べるため）。
    z の感度が 4.5 % 大きいセンサ: 校正を保存すると、IMU の MCV が 4.5 % 小さくなり、横 ÷ IMU がほぼ 1 になる"""
    import compare_side_imu as cs
    folder = tmp_path / "_side" / "session_20261105_180000_60kg_5rep"
    folder.mkdir(parents=True)
    t, pos, _ = side_truth()
    side_receiver_folder(folder, t, pos)
    raw = pd.read_csv(folder / "raw_BAR.csv")
    raw["az_g"] = (raw["az_g"] - 0.05) * 1.045 + 0.05            # 感度 1.045・ゼロ点のずれ +0.05 g のセンサの読みに
    raw.to_csv(folder / "raw_BAR.csv", index=False)
    write_plate_video(folder / "side01.mp4", t, pos)
    ratio = lambda rows: pd.DataFrame(rows)["vid_mcv"].mean() / pd.DataFrame(rows)["imu_mcv"].mean()   # noqa: E731
    before, _ = cs.compare_session(folder, plate_mm=450.0, allow_gui=False)
    (tmp_path / "_calib").mkdir()
    (tmp_path / "_calib" / "calibration_BAR.json").write_text(json.dumps(
        {"device": "BAR", "offset_g": [0.0, 0.0, 0.05], "sensitivity": [1.0, 1.0, 1.045], "sources": ["合成"]}),
        encoding="utf-8")
    after, _ = cs.compare_session(folder, plate_mm=450.0, allow_gui=False)
    assert len(before) == len(after) == 5
    assert abs(ratio(after) / ratio(before) - 1.045) < 0.01
    assert abs(ratio(after) - 1.0) < 0.05
