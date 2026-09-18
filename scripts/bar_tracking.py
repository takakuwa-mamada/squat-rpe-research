# -*- coding: utf-8 -*-
"""
=============================================================================
 バーベルプレート追跡ライブラリ
 ・横撮り動画からバーベル先端（プレートの円）を検出・追跡するための部品集
 ・CSRTトラッカー + Hough円補正 + テンプレートマッチング再取得を
   PlateTrackingPipeline として統合
 ・座標系列の補間・Savitzky-Golay平滑化・速度算出・レップ分割の
   解析関数も提供
 ・CLIは bar_path_extract.py（本モジュールを利用）

 差し替え方法:
   追跡手法を変える場合は PlateTracker を継承して init/update を実装し、
   PlateTrackingPipeline(tracker=YourTracker()) と渡すだけでよい。

 依存ライブラリ:
   pip install opencv-python numpy scipy   （既存環境に含まれる）

 著者: masaki（大学院修士研究 - RPE推定）
=============================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ===========================================================================
# データ型
# ===========================================================================
@dataclass
class TrackResult:
    """1フレーム分の追跡結果。

    Attributes:
        ok:     追跡に成功したか
        x, y:   プレート中心の画像座標 [px]（失敗時は NaN）
        radius: プレート半径 [px]（失敗時は NaN）
        source: 位置の由来 "roi" | "csrt" | "hough" | "template" | "lost"
    """
    ok: bool
    x: float = float("nan")
    y: float = float("nan")
    radius: float = float("nan")
    source: str = "lost"


@dataclass
class PipelineConfig:
    """PlateTrackingPipeline の動作パラメータ。

    Attributes:
        max_jump_factor:      1フレームで許容する移動量（プレート半径比）。
                              超えたらロストとみなす
        reinit_offset_factor: Hough補正のずれがこの半径比を超えたら
                              CSRTを補正位置で再初期化する（ドリフト防止）
        max_unconfirmed:      Hough円で確認できないフレームがこの数連続したら
                              ロスト扱いにして再取得を試す（0で無効）
        search_grow_per_frame: ロスト1フレームごとに探索窓を広げる量（半径比）
    """
    max_jump_factor: float = 1.5
    reinit_offset_factor: float = 0.35
    max_unconfirmed: int = 30
    search_grow_per_frame: float = 1.0


@dataclass
class RepMetrics:
    """1レップ分の速度指標。

    ボトム（切り返し）からトップまでをコンセントリック局面とする。
    """
    rep: int
    bottom_frame: int
    top_frame: int
    bottom_time_s: float
    top_time_s: float
    rom_mm: float
    mean_concentric_velocity_mps: float
    peak_concentric_velocity_mps: float


# ===========================================================================
# 追跡手法（差し替え可能なインタフェース）
# ===========================================================================
class PlateTracker(ABC):
    """プレート追跡アルゴリズムの共通インタフェース。

    別の追跡手法（KCF・MOSSE・学習ベース等）に差し替える場合は
    このクラスを継承し、PlateTrackingPipeline に渡す。
    """

    @abstractmethod
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """初期フレームとROI (x, y, w, h) で追跡を開始する。"""

    @abstractmethod
    def update(self, frame: np.ndarray) -> TrackResult:
        """次フレームを処理し追跡結果を返す。"""


def _create_csrt_tracker():
    """OpenCVのバージョン差を吸収してCSRTトラッカーを生成する。"""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError(
        "CSRT tracker not available. Install opencv-contrib-python."
    )


class CsrtPlateTracker(PlateTracker):
    """OpenCV CSRT (Discriminative Correlation Filter) による追跡。"""

    def __init__(self) -> None:
        self._tracker = None

    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        self._tracker = _create_csrt_tracker()
        self._tracker.init(frame, tuple(int(round(v)) for v in roi))

    def update(self, frame: np.ndarray) -> TrackResult:
        if self._tracker is None:
            return TrackResult(ok=False)
        ok, box = self._tracker.update(frame)
        if not ok:
            return TrackResult(ok=False)
        x, y, w, h = box
        return TrackResult(True, x + w / 2.0, y + h / 2.0,
                           (w + h) / 4.0, source="csrt")


# ===========================================================================
# Hough円による中心・半径の精緻化
# ===========================================================================
class HoughRefiner:
    """追跡位置の近傍でHough円検出を行い、中心・半径を精緻化する。

    CSRTのボックス中心はドリフトすることがあるため、円としての
    プレート輪郭にスナップさせる。検出半径はピクセル→mm換算の
    スケール推定にも使う。
    """

    def __init__(self, radius_tol: float = 0.3,
                 param1: float = 120.0, param2: float = 30.0) -> None:
        """
        Args:
            radius_tol: 期待半径からの許容変動比（min/maxRadiusに反映）
            param1: Canny上側閾値
            param2: 円中心のアキュムレータ閾値（小さいほど検出されやすい）
        """
        self.radius_tol = radius_tol
        self.param1 = param1
        self.param2 = param2

    def refine(self, frame: np.ndarray, cx: float, cy: float,
               radius: float) -> Optional[Tuple[float, float, float]]:
        """(cx, cy) 近傍の円を検出する。

        Returns:
            (中心x, 中心y, 半径) [px]。見つからなければ None。
        """
        h, w = frame.shape[:2]
        margin = radius * 1.6
        x0 = int(max(0, cx - margin))
        y0 = int(max(0, cy - margin))
        x1 = int(min(w, cx + margin))
        y1 = int(min(h, cy + margin))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        sub = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        min_r = max(1, int(radius * (1.0 - self.radius_tol)))
        max_r = max(min_r + 1, int(radius * (1.0 + self.radius_tol)))
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=radius * 2.0,
            param1=self.param1, param2=self.param2,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            return None
        # 期待中心に最も近い円を採用
        cand = min(circles[0],
                   key=lambda c: (c[0] + x0 - cx) ** 2 + (c[1] + y0 - cy) ** 2)
        fx, fy, fr = float(cand[0]), float(cand[1]), float(cand[2])
        # Houghのアキュムレータ量子化（dp=1.2 → 約1px強のジッタ）は
        # 微分時に速度ノイズとして増幅されるため、サブピクセル精緻化を行う
        refined = self._fit_circle_subpix(gray, fx, fy, fr)
        if refined is not None:
            fx, fy, fr = refined
        fx, fy = fx + x0, fy + y0
        if np.hypot(fx - cx, fy - cy) > radius * 0.8:
            return None
        return fx, fy, fr

    def _fit_circle_subpix(self, gray: np.ndarray, cx: float, cy: float,
                           r: float) -> Optional[Tuple[float, float, float]]:
        """検出円のリム近傍のエッジ点に最小二乗円フィット（Kasa法）を行う。

        Args:
            gray: 検出に使ったグレースケール部分画像
            cx, cy, r: Hough検出の円（部分画像座標系）[px]
        Returns:
            サブピクセル精度の (中心x, 中心y, 半径)。点数不足や
            フィット結果が元の円から大きく外れる場合は None。
        """
        edges = cv2.Canny(gray, self.param1 / 2.0, self.param1)
        ys, xs = np.nonzero(edges)
        if len(xs) == 0:
            return None
        dist = np.hypot(xs - cx, ys - cy)
        sel = np.abs(dist - r) < max(2.0, 0.12 * r)
        if sel.sum() < 20:
            return None
        x, y = xs[sel].astype(float), ys[sel].astype(float)
        # Kasa法: x^2+y^2 + a*x + b*y + c = 0 を最小二乗で解く
        A = np.column_stack([x, y, np.ones(len(x))])
        b = -(x ** 2 + y ** 2)
        try:
            (a1, b1, c1), *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        fx, fy = -a1 / 2.0, -b1 / 2.0
        rr = a1 ** 2 / 4.0 + b1 ** 2 / 4.0 - c1
        if rr <= 0:
            return None
        fr = float(np.sqrt(rr))
        # 元の円から大きく外れたフィットは棄却
        if np.hypot(fx - cx, fy - cy) > 0.3 * r or abs(fr - r) > 0.3 * r:
            return None
        return float(fx), float(fy), fr


# ===========================================================================
# プレート円の自動検出（初期ROIの自動決定）
# ===========================================================================
def detect_plate_circle(frame: np.ndarray,
                        min_radius_frac: float = 0.04,
                        max_radius_frac: float = 0.22,
                        edge_support_thresh: float = 0.4,
                        max_candidates: int = 8,
                        border_margin_frac: float = 0.01
                        ) -> Optional[Tuple[float, float, float]]:
    """フレーム全体からバーベルプレートの円を自動検出する。

    Hough円検出の候補それぞれについて「円周上にCannyエッジが存在する
    割合（エッジ支持率）」を評価し、最もプレートらしい円を選ぶ。
    フレーム端で切れている円（プレート全体が映っていない）は除外する。

    Args:
        frame: BGR画像
        min_radius_frac / max_radius_frac:
            探索する半径の範囲（フレーム短辺に対する比）
        edge_support_thresh: 採用する最低エッジ支持率（0〜1）
        max_candidates: スコア評価するHough候補の最大数
        border_margin_frac: フレーム端からの必要マージン（短辺比）
    Returns:
        (中心x, 中心y, 半径) [px]。見つからなければ None。
    """
    h, w = frame.shape[:2]
    s = min(h, w)
    min_r = max(4, int(s * min_radius_frac))
    max_r = max(min_r + 1, int(s * max_radius_frac))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=float(min_r),
        param1=120.0, param2=30.0,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None

    edges = cv2.Canny(blur, 60, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    margin = s * border_margin_frac
    theta = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)

    best_score = -1.0
    best: Optional[Tuple[float, float, float]] = None
    for cx, cy, r in circles[0][:max_candidates]:
        # プレート全体が映っていない（端で切れる）円は除外
        if (cx - r < margin or cy - r < margin
                or cx + r > w - margin or cy + r > h - margin):
            continue
        xs = np.clip((cx + r * np.cos(theta)).astype(int), 0, w - 1)
        ys = np.clip((cy + r * np.sin(theta)).astype(int), 0, h - 1)
        support = float(np.mean(edges[ys, xs] > 0))
        # 支持率が同程度なら大きい円（プレート）を優先
        score = support + 0.05 * (r / max_r)
        if support >= edge_support_thresh and score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(r))
    return best


# ===========================================================================
# テンプレートマッチングによる再取得
# ===========================================================================
class TemplateRedetector:
    """ロスト時にテンプレートマッチングでプレートを再取得する。

    初期ROIのテンプレートと直近成功時（Hough確認済み）のテンプレートを
    保持し、最後の既知位置周辺の探索窓（ロスト継続で拡大）で
    TM_CCOEFF_NORMED マッチングを行う。
    """

    def __init__(self, score_thresh: float = 0.55,
                 update_interval: int = 5,
                 search_grow_per_frame: float = 1.0) -> None:
        """
        Args:
            score_thresh: 採用する最低マッチングスコア
            update_interval: 直近テンプレートを更新する成功フレーム間隔
            search_grow_per_frame: ロスト1フレームごとの探索窓拡大量（半径比）
        """
        self.score_thresh = score_thresh
        self.update_interval = update_interval
        self.search_grow_per_frame = search_grow_per_frame
        self._initial: Optional[np.ndarray] = None
        self._recent: Optional[np.ndarray] = None
        self._n_confirmed = 0

    @staticmethod
    def _crop_gray(frame: np.ndarray, cx: float, cy: float,
                   half: float) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x0 = int(max(0, cx - half))
        y0 = int(max(0, cy - half))
        x1 = int(min(w, cx + half))
        y1 = int(min(h, cy + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    def set_initial(self, frame: np.ndarray, cx: float, cy: float,
                    radius: float) -> None:
        """初期フレームのプレート近傍をテンプレートとして保存する。"""
        self._initial = self._crop_gray(frame, cx, cy, radius * 1.1)

    def maybe_update(self, frame: np.ndarray, cx: float, cy: float,
                     radius: float, confirmed: bool) -> None:
        """Hough確認済みフレームから一定間隔で直近テンプレートを更新する。"""
        if not confirmed:
            return
        self._n_confirmed += 1
        if self._n_confirmed % self.update_interval != 0:
            return
        crop = self._crop_gray(frame, cx, cy, radius * 1.1)
        if crop is not None:
            self._recent = crop

    def detect(self, frame: np.ndarray, last_x: float, last_y: float,
               radius: float, lost_frames: int
               ) -> Optional[Tuple[float, float]]:
        """最後の既知位置周辺でテンプレートを探索する。

        Args:
            last_x, last_y: 最後に追跡できた中心座標 [px]
            radius: プレート半径 [px]
            lost_frames: ロスト継続フレーム数（探索窓の拡大に使用）
        Returns:
            (中心x, 中心y) [px]。スコアが閾値未満なら None。
        """
        h, w = frame.shape[:2]
        half = radius * (3.0 + lost_frames * self.search_grow_per_frame)
        x0 = int(max(0, last_x - half))
        y0 = int(max(0, last_y - half))
        x1 = int(min(w, last_x + half))
        y1 = int(min(h, last_y + half))
        window = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_center: Optional[Tuple[float, float]] = None
        for tmpl in (self._recent, self._initial):
            if tmpl is None:
                continue
            th, tw = tmpl.shape[:2]
            if th > window.shape[0] or tw > window.shape[1]:
                continue
            res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_center = (x0 + max_loc[0] + tw / 2.0,
                               y0 + max_loc[1] + th / 2.0)
        if best_center is None or best_score < self.score_thresh:
            return None
        return best_center


# ===========================================================================
# 統合パイプライン
# ===========================================================================
class PlateTrackingPipeline:
    """CSRT追跡 + Hough円補正 + テンプレート再取得を束ねた追跡パイプライン。

    使い方:
        pipe = PlateTrackingPipeline()
        res0 = pipe.start(first_frame, roi)      # roi = (x, y, w, h)
        for frame in frames:
            res = pipe.process(frame)            # res.ok / res.x / res.y ...
    """

    def __init__(self,
                 tracker: Optional[PlateTracker] = None,
                 refiner: Optional[HoughRefiner] = None,
                 redetector: Optional[TemplateRedetector] = None,
                 config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.tracker = tracker or CsrtPlateTracker()
        self.refiner = HoughRefiner() if refiner is None else refiner
        self.redetector = (TemplateRedetector(
            search_grow_per_frame=self.config.search_grow_per_frame)
            if redetector is None else redetector)
        self._roi_size: Tuple[int, int] = (0, 0)
        self._radius0 = float("nan")
        self._radius = float("nan")
        self._last = TrackResult(ok=False)
        self._lost = True
        self._lost_frames = 0
        self._unconfirmed = 0

    # -- 内部ヘルパ --------------------------------------------------------
    def _reinit_tracker(self, frame: np.ndarray, cx: float, cy: float) -> None:
        w, h = self._roi_size
        box = (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), w, h)
        self.tracker.init(frame, box)

    def _update_radius(self, r: float) -> None:
        # 半径は指数移動平均で更新し、初期値の0.5〜2.0倍にクランプ
        r = float(np.clip(r, self._radius0 * 0.5, self._radius0 * 2.0))
        self._radius = 0.8 * self._radius + 0.2 * r

    # -- 公開API -----------------------------------------------------------
    def start(self, frame: np.ndarray,
              roi: Tuple[int, int, int, int]) -> TrackResult:
        """初期フレームとユーザー指定ROIで追跡を開始する。

        ROI中心近傍でHough円が見つかれば、中心・半径を円にスナップ
        してから追跡を初期化する（ROI指定の粗さを吸収）。

        Args:
            frame: 初期フレーム (BGR)
            roi:   プレートを囲む矩形 (x, y, w, h) [px]
        Returns:
            初期フレームの TrackResult（source は "roi" か "hough"）
        """
        x, y, w, h = roi
        cx, cy = x + w / 2.0, y + h / 2.0
        r = (w + h) / 4.0
        source = "roi"
        if self.refiner is not None:
            refined = self.refiner.refine(frame, cx, cy, r)
            if refined is not None:
                cx, cy, r = refined
                source = "hough"
        self._roi_size = (int(w), int(h))
        self._radius0 = r
        self._radius = r
        self._reinit_tracker(frame, cx, cy)
        self.redetector.set_initial(frame, cx, cy, r)
        self._last = TrackResult(True, cx, cy, r, source)
        self._lost = False
        self._lost_frames = 0
        self._unconfirmed = 0
        return self._last

    def process(self, frame: np.ndarray) -> TrackResult:
        """1フレームを処理して追跡結果を返す。

        通常はCSRTで追跡し、Hough円で中心を補正する。CSRT失敗・
        急激なジャンプ・Hough未確認の長期化でロストと判定し、
        テンプレートマッチングでの再取得に切り替える。
        """
        cfg = self.config
        if not self._lost:
            tr = self.tracker.update(frame)
            if tr.ok:
                jump = float(np.hypot(tr.x - self._last.x,
                                      tr.y - self._last.y))
                if jump <= self._radius * cfg.max_jump_factor:
                    out = TrackResult(True, tr.x, tr.y, self._radius, "csrt")
                    if self.refiner is not None:
                        # 探索半径は固定の self._radius0 を基準にする。
                        # 直近の self._radius を基準にすると、誤検出で
                        # 半径が一度膨らんだ際に探索範囲も広がり、さらに
                        # 大きな誤検出を招くフィードバックループになる。
                        refined = self.refiner.refine(
                            frame, tr.x, tr.y, self._radius0)
                        if refined is not None:
                            hx, hy, hr = refined
                            self._update_radius(hr)
                            out = TrackResult(True, hx, hy, self._radius,
                                              "hough")
                            self._unconfirmed = 0
                            if (np.hypot(hx - tr.x, hy - tr.y)
                                    > cfg.reinit_offset_factor * self._radius):
                                self._reinit_tracker(frame, hx, hy)
                        else:
                            self._unconfirmed += 1
                    hough_timeout = (self.refiner is not None
                                     and cfg.max_unconfirmed > 0
                                     and self._unconfirmed >= cfg.max_unconfirmed)
                    if not hough_timeout:
                        self.redetector.maybe_update(
                            frame, out.x, out.y, self._radius,
                            confirmed=(out.source == "hough"))
                        self._last = out
                        return out
            # CSRT失敗 / ジャンプ / Hough未確認の長期化 → ロスト
            self._lost = True
            self._lost_frames = 0
            self._unconfirmed = 0

        # --- ロスト中: テンプレートマッチングで再取得 ---
        self._lost_frames += 1
        det = self.redetector.detect(frame, self._last.x, self._last.y,
                                     self._radius, self._lost_frames)
        if det is not None:
            dx, dy = det
            out = TrackResult(True, dx, dy, self._radius, "template")
            if self.refiner is not None:
                refined = self.refiner.refine(frame, dx, dy, self._radius0)
                if refined is not None:
                    self._update_radius(refined[2])
                    out = TrackResult(True, refined[0], refined[1],
                                      self._radius, "hough")
            self._reinit_tracker(frame, out.x, out.y)
            self._lost = False
            self._lost_frames = 0
            self._last = out
            return out
        return TrackResult(ok=False)


# ===========================================================================
# キャリブレーション（ピクセル → mm）
# ===========================================================================
def estimate_mm_per_px(hough_radii_px: Sequence[float],
                       roi_diameter_px: float,
                       plate_diameter_mm: float = 450.0,
                       min_samples: int = 10) -> Tuple[float, str]:
    """既知のプレート直径からピクセル→mm換算係数を推定する。

    Hough円で検出された半径の中央値を優先し、サンプル不足時は
    初期ROIのサイズにフォールバックする。

    Args:
        hough_radii_px: Hough検出された半径のリスト [px]
        roi_diameter_px: 初期ROIの (幅+高さ)/2 [px]
        plate_diameter_mm: プレートの実直径 [mm]（オリンピックプレート=450）
        min_samples: Hough半径を採用する最低サンプル数
    Returns:
        (mm_per_px, 推定方法 "hough_radius" | "roi_size")
    """
    radii = np.asarray([r for r in hough_radii_px
                        if np.isfinite(r) and r > 0], dtype=float)
    if len(radii) >= min_samples:
        diameter_px = 2.0 * float(np.median(radii))
        return plate_diameter_mm / diameter_px, "hough_radius"
    return plate_diameter_mm / float(roi_diameter_px), "roi_size"


# ===========================================================================
# 系列処理（補間・平滑化・微分）
# ===========================================================================
def interpolate_gaps(values: Sequence[float]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """系列内部のNaN区間を線形補間する。

    先頭・末尾の連続NaN（追跡開始前/終了後）は補間せず残す。

    Args:
        values: NaNを含みうる系列
    Returns:
        (補間後の配列, 補間されたインデックスのboolマスク)
    """
    v = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(v)
    mask = np.zeros(len(v), dtype=bool)
    if valid.sum() < 2:
        return v, mask
    idx = np.arange(len(v))
    first, last = idx[valid][0], idx[valid][-1]
    mask = (~valid) & (idx >= first) & (idx <= last)
    v[mask] = np.interp(idx[mask], idx[valid], v[valid])
    return v, mask


def savgol_window(fps: float, n_samples: int, polyorder: int = 3) -> int:
    """Savitzky-Golayフィルタの窓幅（奇数）をfpsから決める。

    約0.5秒（fps/2サンプル）を目安に、多項式次数より大きく
    データ長を超えない奇数を返す。合成動画での検証では、この窓幅で
    ピーク速度の過大評価（追跡ジッタの微分による増幅）が最小だった。
    """
    w = int(round(fps / 2.0))
    w = max(w, polyorder + 2)
    if w % 2 == 0:
        w += 1
    if w > n_samples:
        w = n_samples if n_samples % 2 == 1 else n_samples - 1
    return w


def smooth_and_differentiate(pos: Sequence[float], fps: float,
                             polyorder: int = 3
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """位置系列をSavitzky-Golayで平滑化し、1階微分（速度）を返す。

    内部のNaNは事前に interpolate_gaps() で埋めておくこと。
    先頭・末尾の未追跡区間（NaN）はそのままNaNで返す。

    Args:
        pos: 等間隔サンプリングされた位置系列（単位任意、例 [mm]）
        fps: サンプリング周波数 [Hz]
        polyorder: 多項式次数
    Returns:
        (平滑化位置, 速度 [posの単位/s])
    """
    p = np.asarray(pos, dtype=float)
    out_pos = np.full_like(p, np.nan)
    out_vel = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() < polyorder + 2:
        return out_pos, out_vel
    i0, i1 = np.where(valid)[0][[0, -1]]
    seg = p[i0:i1 + 1]
    if not np.all(np.isfinite(seg)):
        raise ValueError(
            "内部にNaNが残っています。先に interpolate_gaps() を適用してください。")
    w = savgol_window(fps, len(seg), polyorder)
    if w <= polyorder:
        return out_pos, out_vel
    out_pos[i0:i1 + 1] = savgol_filter(seg, w, polyorder)
    out_vel[i0:i1 + 1] = savgol_filter(seg, w, polyorder,
                                       deriv=1, delta=1.0 / fps)
    return out_pos, out_vel


# ===========================================================================
# レップ分割
# ===========================================================================
def segment_reps(y_up_mm: Sequence[float], vy_mps: Sequence[float],
                 fps: float,
                 min_rom_mm: float = 150.0,
                 min_rep_interval_s: float = 0.8) -> List[RepMetrics]:
    """鉛直位置の折り返し（ボトム）からレップを分割し、指標を算出する。

    ボトム = 平滑化した鉛直位置の極小値。ボトムから次のボトム（または
    系列末尾）までの最高点をトップとし、ボトム→トップを
    コンセントリック局面として平均速度・ピーク速度を算出する。

    Args:
        y_up_mm: 上向き正の鉛直位置 [mm]（平滑化済み、NaNは端のみ）
        vy_mps:  上向き正の鉛直速度 [m/s]
        fps:     フレームレート [Hz]
        min_rom_mm: レップとして認める最小可動域 [mm]
        min_rep_interval_s: ボトム間の最小間隔 [s]
    Returns:
        RepMetrics のリスト（レップなしなら空リスト）
    """
    y = np.asarray(y_up_mm, dtype=float)
    vy = np.asarray(vy_mps, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < int(fps):
        return []
    i0, i1 = np.where(valid)[0][[0, -1]]
    seg = y[i0:i1 + 1]
    amp = float(np.max(seg) - np.min(seg))
    if amp < min_rom_mm:
        return []

    prominence = max(0.3 * amp, 0.5 * min_rom_mm)
    distance = max(1, int(fps * min_rep_interval_s))
    bottoms, _ = find_peaks(-seg, prominence=prominence, distance=distance)

    reps: List[RepMetrics] = []
    for k, b in enumerate(bottoms):
        end = bottoms[k + 1] if k + 1 < len(bottoms) else len(seg)
        top_rel = b + int(np.argmax(seg[b:end]))
        rom = float(seg[top_rel] - seg[b])
        if rom < min_rom_mm or top_rel <= b:
            continue
        bf, tf = int(i0 + b), int(i0 + top_rel)
        con = vy[bf:tf + 1]
        con = con[np.isfinite(con)]
        if len(con) == 0:
            continue
        reps.append(RepMetrics(
            rep=len(reps) + 1,
            bottom_frame=bf, top_frame=tf,
            bottom_time_s=float(bf / fps), top_time_s=float(tf / fps),
            rom_mm=rom,
            mean_concentric_velocity_mps=float(np.mean(con)),
            peak_concentric_velocity_mps=float(np.max(con)),
        ))
    return reps
