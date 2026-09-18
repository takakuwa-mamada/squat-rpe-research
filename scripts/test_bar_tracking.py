# -*- coding: utf-8 -*-
"""
=============================================================================
 bar_tracking.py のユニットテスト
 ・合成フレーム（描画した円）で Hough検出・CSRT追跡・再取得を検証
 ・既知の解析解（正弦波）で速度算出・レップ分割を検証

 実行:
   pytest test_bar_tracking.py -v
=============================================================================
"""

import cv2
import numpy as np
import pytest

from bar_tracking import (
    CsrtPlateTracker,
    HoughRefiner,
    PipelineConfig,
    PlateTrackingPipeline,
    detect_plate_circle,
    estimate_mm_per_px,
    interpolate_gaps,
    savgol_window,
    segment_reps,
    smooth_and_differentiate,
)


# ===========================================================================
# 合成フレーム生成
# ===========================================================================
def make_frame(cx: float, cy: float, radius: int = 40,
               size=(480, 640), seed: int = 0,
               draw_plate: bool = True) -> np.ndarray:
    """暗いノイズ背景にプレート状の円を描いた合成フレームを返す。"""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 50, (size[0], size[1], 3)).astype(np.uint8)
    if draw_plate:
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(frame, c, radius, (40, 40, 190), -1)          # プレート面
        cv2.circle(frame, c, radius, (240, 240, 240), 3)         # 外周リム
        cv2.circle(frame, c, max(4, radius // 5), (25, 25, 25), -1)  # ハブ
    return frame


# ===========================================================================
# プレート円の自動検出
# ===========================================================================
def test_detect_plate_circle_finds_plate():
    frame = make_frame(420, 260, radius=50)
    # 紛らわしい構造物: ラック様の矩形と、円弧（部分円 → 支持率不足で除外）
    cv2.rectangle(frame, (40, 60), (160, 420), (170, 170, 170), 2)
    cv2.ellipse(frame, (540, 100), (60, 60), 0, 0, 120, (220, 220, 220), 3)
    result = detect_plate_circle(frame)
    assert result is not None
    cx, cy, r = result
    assert abs(cx - 420) < 5
    assert abs(cy - 260) < 5
    assert abs(r - 50) < 6


def test_detect_plate_circle_none_on_empty():
    frame = make_frame(0, 0, draw_plate=False)
    assert detect_plate_circle(frame) is None


def test_detect_plate_circle_rejects_clipped_circle():
    # フレーム端で切れている円（プレート全体が映っていない）は採用しない
    frame = make_frame(20, 240, radius=50)
    result = detect_plate_circle(frame)
    if result is not None:
        cx, cy, r = result
        assert abs(cx - 20) > 10 or abs(cy - 240) > 10


# ===========================================================================
# HoughRefiner
# ===========================================================================
def test_hough_refiner_finds_circle():
    frame = make_frame(300, 200, radius=45)
    refiner = HoughRefiner()
    # 少しずれた初期推定から精緻化できること
    result = refiner.refine(frame, cx=310, cy=210, radius=40)
    assert result is not None
    fx, fy, fr = result
    assert abs(fx - 300) < 5
    assert abs(fy - 200) < 5
    assert abs(fr - 45) < 6


def test_hough_refiner_returns_none_without_circle():
    frame = make_frame(0, 0, draw_plate=False)
    refiner = HoughRefiner()
    assert refiner.refine(frame, cx=320, cy=240, radius=40) is None


# ===========================================================================
# CSRT追跡
# ===========================================================================
def test_csrt_tracks_moving_circle():
    r = 40
    x, y = 200.0, 240.0
    tracker = CsrtPlateTracker()
    tracker.init(make_frame(x, y, r, seed=0),
                 (int(x - r), int(y - r), 2 * r, 2 * r))
    for i in range(1, 25):
        x += 4.0
        y += 2.0
        res = tracker.update(make_frame(x, y, r, seed=i))
        assert res.ok
    assert abs(res.x - x) < 10
    assert abs(res.y - y) < 10


# ===========================================================================
# パイプライン（追跡 + Hough補正 + 再取得）
# ===========================================================================
def test_pipeline_tracks_and_refines():
    r = 40
    x, y = 200.0, 240.0
    pipe = PlateTrackingPipeline()
    res0 = pipe.start(make_frame(x, y, r, seed=0),
                      (int(x - r - 5), int(y - r - 5), 2 * r + 10, 2 * r + 10))
    # ROIが少し大きめでもHoughで円にスナップされること
    assert res0.ok
    assert abs(res0.x - x) < 5 and abs(res0.y - y) < 5

    for i in range(1, 30):
        x += 3.0
        y -= 4.0
        res = pipe.process(make_frame(x, y, r, seed=i))
        assert res.ok
        assert abs(res.x - x) < 8
        assert abs(res.y - y) < 8


class _GrowingRefiner:
    """常に入力半径を拡大して返す敵対的リファイナ（半径暴走の回帰テスト用）。

    HoughRefinerの探索許容範囲が「直近の（既に肥大化した）半径」を
    基準にしてしまうと、1フレームの過大検出が次フレームの探索窓を
    広げ、さらに大きな誤検出を招くフィードバックループになる。
    探索半径が固定の self._radius0 を基準にしていれば、このリファイナ
    を繋いでも半径は一定値に収束するはずである。
    """

    def __init__(self, growth: float = 1.3) -> None:
        self.growth = growth

    def refine(self, frame, cx, cy, radius):
        return cx, cy, radius * self.growth


def test_pipeline_radius_does_not_runaway_with_adversarial_refiner():
    r0 = 80.0
    pipe = PlateTrackingPipeline(refiner=_GrowingRefiner(growth=1.3))
    frame = make_frame(200, 200, radius=int(r0))
    res0 = pipe.start(frame, (int(200 - r0), int(200 - r0),
                              int(2 * r0), int(2 * r0)))
    radius0 = res0.radius   # start()時点で1回拡大されたものが固定基準になる

    reported = []
    for i in range(1, 60):
        res = pipe.process(make_frame(200, 200, radius=int(r0), seed=i))
        assert res.ok
        reported.append(res.radius)

    # 固定探索半径なら hr は radius0*1.3 近辺に収束し、
    # 決してハード上限（radius0*2.0）に張り付いたまま増え続けない
    assert max(reported) <= radius0 * 2.0 + 1e-6
    assert reported[-1] == pytest.approx(radius0 * 1.3, rel=0.05)
    # 単調に増大し続けていない（収束している）ことを確認
    assert abs(reported[-1] - reported[-10]) < 0.5


def test_pipeline_redetects_after_occlusion():
    r = 40
    pipe = PlateTrackingPipeline(
        config=PipelineConfig(max_unconfirmed=3))
    x, y = 200.0, 240.0
    pipe.start(make_frame(x, y, r, seed=0),
               (int(x - r), int(y - r), 2 * r, 2 * r))
    # 通常追跡
    for i in range(1, 10):
        x += 5.0
        res = pipe.process(make_frame(x, y, r, seed=i))
        assert res.ok
    # オクルージョン（プレート消失）→ ロストすること
    lost_seen = False
    for i in range(10, 18):
        res = pipe.process(make_frame(0, 0, r, seed=i, draw_plate=False))
        lost_seen = lost_seen or (not res.ok)
    assert lost_seen
    # 別の位置に再出現 → 数フレーム以内に再取得できること
    nx, ny = x + 70.0, y + 50.0
    recovered = None
    for i in range(18, 26):
        res = pipe.process(make_frame(nx, ny, r, seed=i))
        if res.ok:
            recovered = res
            break
    assert recovered is not None
    assert abs(recovered.x - nx) < 15
    assert abs(recovered.y - ny) < 15


# ===========================================================================
# キャリブレーション
# ===========================================================================
def test_estimate_mm_per_px_from_hough_radii():
    radii = [44.0, 45.0, 45.5, 44.5, 45.0] * 3   # 15サンプル、中央値45
    mm_per_px, src = estimate_mm_per_px(radii, roi_diameter_px=100,
                                        plate_diameter_mm=450.0)
    assert src == "hough_radius"
    assert mm_per_px == pytest.approx(450.0 / 90.0, rel=1e-6)


def test_estimate_mm_per_px_fallback_to_roi():
    mm_per_px, src = estimate_mm_per_px([44.0, 45.0], roi_diameter_px=90,
                                        plate_diameter_mm=450.0)
    assert src == "roi_size"
    assert mm_per_px == pytest.approx(5.0, rel=1e-6)


# ===========================================================================
# 補間
# ===========================================================================
def test_interpolate_gaps_interior_only():
    v = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, 6.0, np.nan])
    out, mask = interpolate_gaps(v)
    # 内部ギャップは線形補間される
    assert out[2] == pytest.approx(2.0)
    assert out[4] == pytest.approx(4.0)
    assert out[5] == pytest.approx(5.0)
    assert mask.tolist() == [False, False, True, False, True, True,
                             False, False]
    # 先頭・末尾のNaNは残る
    assert np.isnan(out[0]) and np.isnan(out[7])


# ===========================================================================
# 平滑化・微分
# ===========================================================================
def test_savgol_window_is_odd_and_bounded():
    w = savgol_window(fps=30.0, n_samples=1000)
    assert w % 2 == 1
    assert w >= 5
    # データ長より大きくならない
    assert savgol_window(fps=120.0, n_samples=15) <= 15


def test_smooth_and_differentiate_sine():
    fps = 30.0
    t = np.arange(0, 4, 1 / fps)
    amp, freq = 100.0, 0.5                       # 100mm, 0.5Hz
    pos = amp * np.sin(2 * np.pi * freq * t)
    smoothed, vel = smooth_and_differentiate(pos, fps)
    expected_vel = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    # 端の効果を避けて中央部で比較
    mid = slice(15, len(t) - 15)
    peak = np.max(np.abs(expected_vel))
    assert np.nanmax(np.abs(vel[mid] - expected_vel[mid])) < 0.08 * peak
    assert np.nanmax(np.abs(smoothed[mid] - pos[mid])) < 0.05 * amp


def test_smooth_and_differentiate_keeps_edge_nans():
    fps = 30.0
    pos = np.full(100, np.nan)
    pos[10:90] = np.linspace(0, 500, 80)         # 一定速度区間
    smoothed, vel = smooth_and_differentiate(pos, fps)
    assert np.all(np.isnan(vel[:10])) and np.all(np.isnan(vel[90:]))
    # 一定速度 = 500mm / (79/30)s ≒ 189.9 mm/s
    expected = 500.0 / (79 / fps)
    assert np.nanmedian(vel[20:80]) == pytest.approx(expected, rel=0.05)


def test_smooth_and_differentiate_rejects_interior_nan():
    pos = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    with pytest.raises(ValueError):
        smooth_and_differentiate(pos, fps=30.0)


# ===========================================================================
# レップ分割
# ===========================================================================
def _make_squat_signal(n_reps: int = 3, depth_mm: float = 400.0,
                       rep_duration_s: float = 2.0, fps: float = 30.0):
    """cos型のスクワット波形（上向き正、トップ=0）とその速度を作る。"""
    n = int(n_reps * rep_duration_s * fps)
    t = np.arange(n) / fps
    tau = (t % rep_duration_s) / rep_duration_s      # レップ内位相 0..1
    y = -depth_mm * (1 - np.cos(2 * np.pi * tau)) / 2.0
    vy = np.gradient(y, 1 / fps) / 1000.0            # m/s
    return t, y, vy


def test_segment_reps_finds_all_reps():
    fps = 30.0
    t, y, vy = _make_squat_signal(n_reps=3, depth_mm=400.0, fps=fps)
    reps = segment_reps(y, vy, fps)
    assert len(reps) == 3
    for k, r in enumerate(reps):
        assert r.rep == k + 1
        # ボトムはレップ中央（1s, 3s, 5s）付近
        assert r.bottom_time_s == pytest.approx(1.0 + 2.0 * k, abs=0.15)
        assert r.rom_mm == pytest.approx(400.0, rel=0.05)
        assert r.mean_concentric_velocity_mps > 0
        assert (r.peak_concentric_velocity_mps
                >= r.mean_concentric_velocity_mps)
        # 解析解: ピーク挙上速度 = depth*pi/T_con = 0.4*pi/1 /2 ≒ 0.628 m/s
        assert r.peak_concentric_velocity_mps == pytest.approx(0.628,
                                                               rel=0.10)


def test_segment_reps_ignores_small_movement():
    fps = 30.0
    t, y, vy = _make_squat_signal(n_reps=3, depth_mm=50.0, fps=fps)
    assert segment_reps(y, vy, fps, min_rom_mm=150.0) == []


def test_segment_reps_empty_on_all_nan():
    y = np.full(200, np.nan)
    assert segment_reps(y, y, fps=30.0) == []
