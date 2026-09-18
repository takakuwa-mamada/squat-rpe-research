# -*- coding: utf-8 -*-
"""
=============================================================================
 バーベル軌道・速度解析スクリプト
 ・横撮りスクワット動画からバーベル先端（プレートの円）を追跡
 ・プレート円を自動検出（Hough円+エッジ支持率）して追跡を開始
   → CSRT追跡 + Hough円補正 + テンプレート再取得
   （追跡ロジックは bar_tracking.py、差し替え可能）
 ・既知のプレート直径（オリンピックプレート=450mm）でピクセル→mm換算
 ・Savitzky-Golay平滑化後に微分して速度 [m/s] を算出
 ・鉛直位置の折り返しからレップを分割し、コンセントリック局面の
   平均速度・ピーク速度を算出
 ・data/ 配下を再帰的に走査して未処理動画を自動処理（pose_extract系と同様）

 使い方:
   python bar_path_extract.py                            # data/配下を全自動処理
   python bar_path_extract.py --video path/to/x.mp4      # 単一ファイル
   python bar_path_extract.py --subject S001             # 被験者指定
   python bar_path_extract.py --force                    # 処理済みも再処理
   python bar_path_extract.py --video x.mp4 --roi 100,200,80,80   # ROI手動指定
   python bar_path_extract.py --select-roi               # GUIでROIをドラッグ指定
   python bar_path_extract.py --plate-diameter 350       # プレート径変更 [mm]
   python bar_path_extract.py --reselect-roi             # 保存済みROIを無視して再検出

 プレート位置の決定（優先順位）:
   1. --roi x,y,w,h の手動指定
   2. 保存済み <stem>_bar_roi.json（--reselect-roi で無視）
   3. 自動検出（デフォルト）: Hough円検出+エッジ支持率でプレート円を探す。
      初回フレームで見つからなければ先頭10秒を0.2秒刻みでスキャンし、
      検出できたフレームから追跡を開始する
   4. 自動検出失敗時のみGUIフォールバック（--select-roi で最初からGUI）
   検出結果は <stem>_bar_roi_preview.png で確認できる。

 出力（data/<S>/<SES>/pose/ 配下）:
   <stem>_bar_path.csv       フレームごとの座標・変位・速度
   <stem>_bar_annotated.mp4  軌跡・速度をオーバーレイした動画
   <stem>_bar_velocity.png   鉛直位置・速度の時系列プロット
   <stem>_bar_path.png       バー軌道の2Dプロット（実寸mm）
   <stem>_bar_summary.json   キャリブレーション情報・レップごと指標
   <stem>_bar_roi.json       採用ROI（再実行時に再利用）
   <stem>_bar_roi_preview.png 検出/指定したプレート位置の確認画像

 依存ライブラリ:
   pip install opencv-python numpy pandas scipy matplotlib tqdm  （既存環境のみ）

 著者: masaki（大学院修士研究 - RPE推定）
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

from bar_tracking import (
    PlateTrackingPipeline,
    RepMetrics,
    detect_plate_circle,
    estimate_mm_per_px,
    interpolate_gaps,
    segment_reps,
    smooth_and_differentiate,
)


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT  = PROJECT_ROOT / "data"

# オリンピックプレートの直径 [mm]（--plate-diameter で変更可）
PLATE_DIAMETER_MM = 450.0

# 自動検出のスキャン設定
AUTO_SCAN_MAX_S    = 10.0   # 先頭何秒まで検出を試みるか
AUTO_SCAN_STRIDE_S = 0.2    # 検出を試みる間隔 [s]

# 検出円からROIを作るときの余白（半径比）
ROI_PAD_FACTOR = 1.15

# レップとして認める最小可動域 [mm]
MIN_ROM_MM = 150.0

# 動画コーデック候補（pose_extract系と同じ）
VIDEO_CODEC_CANDIDATES = ("avc1", "H264", "mp4v", "XVID")

# オーバーレイ配色 (BGR)
TRAIL_COLOR_RECENT = (0, 215, 255)   # 直近2秒の軌跡
TRAIL_COLOR_OLD    = (0, 110, 130)   # それ以前の軌跡
CIRCLE_COLOR       = (0, 255, 0)     # 追跡成功フレームの円
CIRCLE_COLOR_INTERP = (160, 160, 160)  # 補間フレームの円
CENTER_COLOR       = (0, 0, 255)     # 中心点
TEXT_COLOR         = (0, 255, 255)   # 情報テキスト


# ===========================================================================
# 動画ライター（pose_extract系と同じコーデック自動選択）
# ===========================================================================
def make_video_writer(out_path: Path, fps: float, w: int, h: int):
    for codec_name in VIDEO_CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        wr = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if wr.isOpened():
            print(f"  [VIDEO] codec={codec_name}")
            return wr, codec_name
        wr.release()
    print("  [WARN] no working codec; annotated video will be skipped")
    return None, None


# ===========================================================================
# ROIの取得（引数 / 保存済みJSON / GUI）
# ===========================================================================
def parse_roi_arg(s: str) -> Tuple[int, int, int, int]:
    """'x,y,w,h' 形式のROI文字列をパースする。"""
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError(f"invalid ROI: {s} (expected x,y,w,h with w,h > 0)")
    return parts[0], parts[1], parts[2], parts[3]


def select_roi_gui(frame: np.ndarray, title: str
                   ) -> Optional[Tuple[int, int, int, int]]:
    """GUIでプレートのROIをドラッグ指定させる。

    Returns:
        (x, y, w, h)。キャンセル時は None。
    """
    print("  [ROI] プレートの円をちょうど囲むようにドラッグ → ENTER/SPACE で確定")
    print("        （ESC または c でキャンセル）")
    win = f"Select barbell plate - {title}"
    roi = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def circle_to_roi(cx: float, cy: float, r: float,
                  frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """検出円から少し余白を持たせた正方形ROIを作る（フレーム内にクランプ）。"""
    half = r * ROI_PAD_FACTOR
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(frame_w, cx + half))
    y1 = int(min(frame_h, cy + half))
    return x0, y0, x1 - x0, y1 - y0


def auto_detect_roi(cap: cv2.VideoCapture, first_frame: np.ndarray,
                    fps: float, n_frames: int
                    ) -> Optional[Tuple[Tuple[int, int, int, int], int, np.ndarray]]:
    """プレート円を自動検出し、(ROI, 開始フレーム, その画像) を返す。

    初回フレームで見つからなければ AUTO_SCAN_MAX_S 秒まで
    AUTO_SCAN_STRIDE_S 間隔でスキャンする（capは読み進められる）。
    """
    h, w = first_frame.shape[:2]
    idx = 0
    frame = first_frame
    stride = max(1, int(round(fps * AUTO_SCAN_STRIDE_S)))
    scan_limit = min(max(n_frames - 1, 0), int(fps * AUTO_SCAN_MAX_S))
    while True:
        circ = detect_plate_circle(frame)
        if circ is not None:
            cx, cy, r = circ
            roi = circle_to_roi(cx, cy, r, w, h)
            print(f"  [ROI] auto-detected: center=({cx:.0f},{cy:.0f}) "
                  f"r={r:.0f}px at frame {idx}")
            return roi, idx, frame
        if idx + stride > scan_limit:
            return None
        for _ in range(stride):
            ok, nxt = cap.read()
            if not ok:
                return None
            idx += 1
        frame = nxt


def resolve_plate_roi(cap: cv2.VideoCapture, video_path: Path,
                      first_frame: np.ndarray, fps: float, n_frames: int,
                      stem: str, roi_json: Path,
                      roi_arg: Optional[str], reselect: bool,
                      force_gui: bool
                      ) -> Optional[Tuple[Tuple[int, int, int, int], int,
                                          np.ndarray, str, cv2.VideoCapture]]:
    """プレートROIと追跡開始フレームを決める。

    優先順位: --roi 引数 > 保存済みJSON > 自動検出 > GUIフォールバック。
    自動検出はcapを読み進めるため、GUIフォールバック時はcapを開き直す。

    Returns:
        (roi, start_frame, start_frameの画像, 決定方法, cap)。中断時 None。
        capは start_frame まで読み終えた状態で返る。
    """
    if roi_arg:
        roi = parse_roi_arg(roi_arg)
        print(f"  [ROI] from --roi: {roi}")
        return roi, 0, first_frame, "arg", cap

    if roi_json.exists() and not reselect:
        try:
            saved = json.loads(roi_json.read_text(encoding="utf-8"))
            roi = tuple(int(v) for v in saved["roi"])
            start = int(saved.get("start_frame", 0))
            frame = first_frame
            for _ in range(start):          # 保存時の開始フレームまで進める
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"cannot seek to frame {start}")
            print(f"  [ROI] reuse saved: {roi} (start_frame={start})")
            return roi, start, frame, "saved", cap  # type: ignore[return-value]
        except Exception as e:
            print(f"  [WARN] broken roi json ({e}); redetecting")

    if not force_gui:
        found = auto_detect_roi(cap, first_frame, fps, n_frames)
        if found is not None:
            roi, start, frame = found
            return roi, start, frame, "auto", cap
        print("  [WARN] auto-detection failed; falling back to GUI")
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        ok, first_frame = cap.read()
        if not ok:
            return None

    roi = select_roi_gui(first_frame, stem)
    if roi is None:
        return None
    return roi, 0, first_frame, "gui", cap


def save_roi_preview(frame: np.ndarray, roi: Tuple[int, int, int, int],
                     method: str, out_png: Path) -> None:
    """採用したプレート位置を描いた確認画像を保存する。"""
    img = frame.copy()
    x, y, w, h = roi
    c = (x + w // 2, y + h // 2)
    r = (w + h) // 4
    cv2.circle(img, c, r, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 215, 255), 1)
    cv2.putText(img, f"plate ROI ({method})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_png), img)


# ===========================================================================
# 追跡（パス1）
# ===========================================================================
def track_video(cap: cv2.VideoCapture, start_frame_img: np.ndarray,
                roi: Tuple[int, int, int, int], fps: float,
                n_frames: int, stem: str,
                start_frame: int = 0) -> pd.DataFrame:
    """全フレームを追跡し、フレームごとの生座標を返す。

    start_frame より前（プレート検出前にスキャンで消費した区間）は
    未追跡（lost）行として記録する。capは start_frame まで
    読み終えた状態で渡すこと。

    Returns:
        columns = [frame, time_s, x_px, y_px, radius_px, source, tracked]
    """
    from bar_tracking import TrackResult

    pipe = PlateTrackingPipeline()
    rows: List[Dict] = []

    def add_row(idx: int, r) -> None:
        rows.append({
            "frame": idx,
            "time_s": round(idx / fps, 4),
            "x_px": r.x, "y_px": r.y,
            "radius_px": r.radius,
            "source": r.source,
            "tracked": bool(r.ok),
        })

    pbar = tqdm(total=n_frames, desc=f"  {stem}", unit="frame", leave=False)
    for idx in range(start_frame):
        add_row(idx, TrackResult(ok=False))
        pbar.update(1)

    add_row(start_frame, pipe.start(start_frame_img, roi))
    pbar.update(1)
    frame_idx = start_frame + 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        add_row(frame_idx, pipe.process(frame))
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# ===========================================================================
# 解析（パス1の座標 → 補間・換算・平滑化・速度・レップ）
# ===========================================================================
def analyze_trajectory(df: pd.DataFrame, fps: float,
                       roi: Tuple[int, int, int, int],
                       plate_diameter_mm: float
                       ) -> Tuple[pd.DataFrame, float, str, List[RepMetrics]]:
    """生座標CSVに換算座標・変位・速度・レップ情報を付加する。

    座標系: x_mm は右向き正、y_mm は上向き正（画像座標を反転）。
    原点は最初に追跡できたフレームのプレート中心。
    x_mm / y_mm は平滑化後の値。

    Returns:
        (拡張DataFrame, mm_per_px, スケール推定方法, レップリスト)
    """
    hough_radii = df.loc[df["source"] == "hough", "radius_px"].to_numpy()
    roi_diameter_px = (roi[2] + roi[3]) / 2.0
    mm_per_px, scale_src = estimate_mm_per_px(
        hough_radii, roi_diameter_px, plate_diameter_mm)

    x_i, x_interp = interpolate_gaps(df["x_px"].to_numpy())
    y_i, y_interp = interpolate_gaps(df["y_px"].to_numpy())
    interp_mask = x_interp | y_interp

    valid = np.isfinite(x_i) & np.isfinite(y_i)
    if valid.sum() < 5:
        raise RuntimeError("追跡できたフレームが少なすぎます（<5）")
    first = int(np.where(valid)[0][0])
    x0, y0 = x_i[first], y_i[first]

    x_rel_mm = (x_i - x0) * mm_per_px          # 右向き正
    y_up_mm  = (y0 - y_i) * mm_per_px          # 上向き正

    xs_mm, vx_mm_s = smooth_and_differentiate(x_rel_mm, fps)
    ys_mm, vy_mm_s = smooth_and_differentiate(y_up_mm, fps)

    vx_mps = vx_mm_s / 1000.0
    vy_mps = vy_mm_s / 1000.0
    speed_mps = np.hypot(vx_mps, vy_mps)

    disp_mm = np.full(len(df), np.nan)
    disp_mm[1:] = np.hypot(np.diff(xs_mm), np.diff(ys_mm))
    if np.isfinite(xs_mm[first]):
        disp_mm[first] = 0.0
    path_mm = np.where(np.isfinite(disp_mm),
                       np.nancumsum(np.nan_to_num(disp_mm)), np.nan)

    out = df.copy()
    out["interpolated"] = interp_mask
    out["x_mm"] = xs_mm
    out["y_mm"] = ys_mm
    out["disp_mm"] = disp_mm
    out["path_mm"] = path_mm
    out["vx_mps"] = vx_mps
    out["vy_mps"] = vy_mps
    out["speed_mps"] = speed_mps

    reps = segment_reps(ys_mm, vy_mps, fps, min_rom_mm=MIN_ROM_MM)
    return out, mm_per_px, scale_src, reps


def build_summary(video_path: Path, fps: float, df: pd.DataFrame,
                  roi: Tuple[int, int, int, int], mm_per_px: float,
                  scale_src: str, plate_diameter_mm: float,
                  reps: List[RepMetrics]) -> Dict:
    """サマリJSONの中身を組み立てる。"""
    speed = df["speed_mps"].to_numpy()
    vy = df["vy_mps"].to_numpy()
    n = len(df)
    n_tracked = int(df["tracked"].sum())
    n_interp = int(df["interpolated"].sum())
    return {
        "video": str(video_path),
        "fps": fps,
        "n_frames": n,
        "n_tracked": n_tracked,
        "n_interpolated": n_interp,
        "tracked_ratio": round(n_tracked / max(n, 1), 4),
        "roi": list(roi),
        "plate_diameter_mm": plate_diameter_mm,
        "mm_per_px": round(mm_per_px, 5),
        "scale_source": scale_src,
        "overall": {
            "mean_speed_mps": _nanround(np.nanmean(speed)),
            "max_speed_mps": _nanround(np.nanmax(speed)),
            "peak_concentric_velocity_mps": _nanround(np.nanmax(vy)),
            "peak_eccentric_velocity_mps": _nanround(np.nanmin(vy)),
            "total_path_mm": _nanround(np.nanmax(df["path_mm"].to_numpy())),
        },
        "n_reps": len(reps),
        "reps": [asdict(r) for r in reps],
    }


def _nanround(v: float, nd: int = 4) -> Optional[float]:
    return None if not np.isfinite(v) else round(float(v), nd)


# ===========================================================================
# プロット
# ===========================================================================
def make_velocity_plot(df: pd.DataFrame, reps: List[RepMetrics],
                       out_path: Path, title: str = "") -> None:
    """鉛直位置と鉛直速度の時系列プロットを保存する。

    コンセントリック局面（ボトム→トップ）を薄く塗り、レップごとの
    平均/ピーク速度を注釈する。
    """
    t = df["time_s"].to_numpy()
    y = df["y_mm"].to_numpy()
    vy = df["vy_mps"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, y, lw=1.5, color="#1f77b4")
    axes[0].set_ylabel("bar height [mm]")
    axes[0].set_title(f"Bar Path & Velocity - {title}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, vy, lw=1.5, color="#1f77b4")
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_ylabel("vertical velocity [m/s]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(alpha=0.3)

    for r in reps:
        for ax in axes:
            ax.axvspan(r.bottom_time_s, r.top_time_s,
                       color="#1f77b4", alpha=0.10, lw=0)
        axes[0].plot(r.bottom_time_s, y[r.bottom_frame], marker="v",
                     color="#d62728", ms=7,
                     label="bottom" if r.rep == 1 else None)
        axes[0].plot(r.top_time_s, y[r.top_frame], marker="^",
                     color="#2ca02c", ms=7,
                     label="top" if r.rep == 1 else None)
        axes[1].annotate(
            f"Rep {r.rep}\nmean {r.mean_concentric_velocity_mps:.2f}\n"
            f"peak {r.peak_concentric_velocity_mps:.2f}",
            xy=((r.bottom_time_s + r.top_time_s) / 2, 0),
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="#333333",
        )
    if reps:
        axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def make_path_plot(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    """バー軌道の2Dプロット（実寸mm、時間で色付け）を保存する。"""
    x = df["x_mm"].to_numpy()
    y = df["y_mm"].to_numpy()
    t = df["time_s"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 8))
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    good = np.all(np.isfinite(segs.reshape(len(segs), -1)), axis=1)
    if good.any():
        lc = LineCollection(segs[good], cmap="viridis",
                            array=t[:-1][good], linewidth=2)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, shrink=0.8)
        cbar.set_label("time [s]")
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.any():
        i0 = int(np.where(valid)[0][0])
        ax.plot(x[i0], y[i0], marker="o", color="#2ca02c", ms=8,
                label="start")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(np.nanmin(x) - 30, np.nanmax(x) + 30)
        ax.set_ylim(np.nanmin(y) - 30, np.nanmax(y) + 30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("horizontal [mm]")
    ax.set_ylabel("vertical [mm]")
    ax.set_title(f"Bar Path - {title}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ===========================================================================
# 注釈付き動画（パス2）
# ===========================================================================
def write_annotated_video(video_path: Path, out_vid: Path,
                          df: pd.DataFrame, reps: List[RepMetrics],
                          fps: float, w: int, h: int,
                          mm_per_px: float) -> Optional[str]:
    """軌跡・現在円・速度をオーバーレイした動画を書き出す。

    Returns:
        使用したコーデック名。ライター生成失敗時は None。
    """
    writer, codec = make_video_writer(out_vid, fps, w, h)
    if writer is None:
        return None

    # 平滑化mm座標をピクセルに戻して軌跡に使う（描画のガタつき防止）
    valid = df["tracked"].to_numpy() | df["interpolated"].to_numpy()
    x_i, _ = interpolate_gaps(df["x_px"].to_numpy())
    y_i, _ = interpolate_gaps(df["y_px"].to_numpy())
    first_idx = np.where(np.isfinite(x_i) & np.isfinite(y_i))[0]
    x0 = x_i[first_idx[0]] if len(first_idx) else float("nan")
    y0 = y_i[first_idx[0]] if len(first_idx) else float("nan")
    xs_px = x0 + df["x_mm"].to_numpy() / mm_per_px
    ys_px = y0 - df["y_mm"].to_numpy() / mm_per_px

    radius_med = float(np.nanmedian(df["radius_px"].to_numpy()))
    vy = df["vy_mps"].to_numpy()
    trail_recent = int(fps * 2)

    # フレーム→コンセントリック中のレップ番号
    rep_of_frame = np.zeros(len(df), dtype=int)
    for r in reps:
        rep_of_frame[r.bottom_frame:r.top_frame + 1] = r.rep

    cap = cv2.VideoCapture(str(video_path))
    pbar = tqdm(total=len(df), desc="  overlay", unit="frame", leave=False)
    i = 0
    while i < len(df):
        ok, frame = cap.read()
        if not ok:
            break

        # 軌跡（古い区間は暗色、直近2秒は明色）
        pts = np.stack([xs_px[:i + 1], ys_px[:i + 1]], axis=1)
        finite = np.all(np.isfinite(pts), axis=1)
        split = max(0, i + 1 - trail_recent)
        for lo, hi, color in ((0, split + 1, TRAIL_COLOR_OLD),
                              (split, i + 1, TRAIL_COLOR_RECENT)):
            seg = pts[lo:hi][finite[lo:hi]]
            if len(seg) >= 2:
                cv2.polylines(frame, [seg.astype(np.int32)], False, color, 2,
                              lineType=cv2.LINE_AA)

        # 現在のプレート円と中心
        if valid[i] and np.isfinite(xs_px[i]) and np.isfinite(ys_px[i]):
            c = (int(round(xs_px[i])), int(round(ys_px[i])))
            r_px = df["radius_px"].iloc[i]
            r_draw = int(round(r_px if np.isfinite(r_px) else radius_med))
            color = (CIRCLE_COLOR if df["tracked"].iloc[i]
                     else CIRCLE_COLOR_INTERP)
            cv2.circle(frame, c, r_draw, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, c, 3, CENTER_COLOR, -1, lineType=cv2.LINE_AA)
            vy_txt = f"{vy[i]:+.2f}" if np.isfinite(vy[i]) else "--"
            cv2.putText(frame, f"vy={vy_txt} m/s",
                        (c[0] + r_draw + 6, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)

        # ヘッダ情報
        t_s = df["time_s"].iloc[i]
        cv2.putText(frame, f"BAR f={i} t={t_s:.2f}s",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        if rep_of_frame[i] > 0:
            cv2.putText(frame, f"Rep {rep_of_frame[i]} concentric",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        writer.write(frame)
        i += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    return codec


# ===========================================================================
# 単一動画の処理
# ===========================================================================
def process_one_video(video_path: Path, out_dir: Path,
                      roi_arg: Optional[str] = None,
                      plate_diameter_mm: float = PLATE_DIAMETER_MM,
                      save_annotated: bool = True,
                      save_plot: bool = True,
                      reselect_roi: bool = False,
                      force_gui: bool = False) -> dict:
    """1本の動画に対して追跡・解析・出力を行う。

    Returns:
        処理結果のサマリ辞書（statusキーで成否判定）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_csv  = out_dir / f"{stem}_bar_path.csv"
    out_vid  = out_dir / f"{stem}_bar_annotated.mp4"
    out_vplt = out_dir / f"{stem}_bar_velocity.png"
    out_pplt = out_dir / f"{stem}_bar_path.png"
    out_sum  = out_dir / f"{stem}_bar_summary.json"
    out_roi  = out_dir / f"{stem}_bar_roi.json"
    out_prev = out_dir / f"{stem}_bar_roi_preview.png"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "reason": "cannot open video"}
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        return {"status": "error", "reason": "cannot read first frame"}

    resolved = resolve_plate_roi(cap, video_path, first_frame, fps, n_frm,
                                 stem, out_roi, roi_arg, reselect_roi,
                                 force_gui)
    if resolved is None:
        cap.release()
        return {"status": "skipped",
                "reason": "plate not found / ROI selection cancelled"}
    roi, start_frame, start_img, roi_method, cap = resolved
    out_roi.write_text(json.dumps({
        "video": video_path.name, "roi": list(roi),
        "start_frame": start_frame, "method": roi_method,
        "plate_diameter_mm": plate_diameter_mm,
    }, indent=2), encoding="utf-8")
    save_roi_preview(start_img, roi, roi_method, out_prev)

    # パス1: 追跡
    df_raw = track_video(cap, start_img, roi, fps, n_frm, stem,
                         start_frame=start_frame)
    cap.release()

    # 解析
    df, mm_per_px, scale_src, reps = analyze_trajectory(
        df_raw, fps, roi, plate_diameter_mm)
    df.to_csv(out_csv, index=False)

    summary = build_summary(video_path, fps, df, roi, mm_per_px,
                            scale_src, plate_diameter_mm, reps)
    out_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    if save_plot:
        make_velocity_plot(df, reps, out_vplt, title=stem)
        make_path_plot(df, out_pplt, title=stem)

    codec = None
    if save_annotated:
        codec = write_annotated_video(video_path, out_vid, df, reps,
                                      fps, w, h, mm_per_px)

    return {
        "status": "ok",
        "video": str(video_path),
        "n_frames": len(df),
        "roi_method": roi_method,
        "start_frame": start_frame,
        "tracked_ratio": summary["tracked_ratio"],
        "mm_per_px": summary["mm_per_px"],
        "scale_source": scale_src,
        "n_reps": len(reps),
        "reps": [
            {"rep": r.rep,
             "mean_con_v": round(r.mean_concentric_velocity_mps, 3),
             "peak_con_v": round(r.peak_concentric_velocity_mps, 3)}
            for r in reps
        ],
        "codec": codec,
        "out_csv": str(out_csv),
        "out_summary": str(out_sum),
        "out_video": str(out_vid) if save_annotated and codec else None,
        "out_velocity_plot": str(out_vplt) if save_plot else None,
        "out_path_plot": str(out_pplt) if save_plot else None,
    }


# ===========================================================================
# data/ を走査（pose_extract系と同じ規約）
# ===========================================================================
def find_videos(filter_subject: Optional[str] = None):
    """
    data/ 配下から処理対象の動画を探す。

    以下のどちらの置き方にも対応する:
      (A) data/<subject>/<session>/videos/xxx.mp4  → 出力先 .../<session>/pose/
      (B) data/<何か>/<何か>/xxx.mp4               → 出力先 その動画と同じ階層の pose/
          例: data/_sessions/session_20260918_004509/130.mp4
              → data/_sessions/session_20260918_004509/pose/

    自分が生成したオーバーレイ動画（pose/ 配下、*_annotated.mp4）は除外する。
    """
    if not DATA_ROOT.exists():
        return []
    results = []
    for ext in ("mp4", "mov", "avi", "MP4", "MOV", "AVI"):
        for video in DATA_ROOT.rglob(f"*.{ext}"):
            try:
                parts = video.relative_to(DATA_ROOT).parts
            except ValueError:
                continue
            if len(parts) < 2:
                continue                      # data/ 直下の動画は対象外
            # 自身の出力を再処理しない
            if video.parent.name == "pose" or "_annotated" in video.stem:
                continue
            if filter_subject and parts[0] != filter_subject:
                continue
            # 出力先: videos/ の中なら1つ上、そうでなければ同じ階層に pose/ を作る
            if video.parent.name == "videos":
                out_dir = video.parent.parent / "pose"
            else:
                out_dir = video.parent / "pose"
            results.append((video, out_dir))

    seen, uniq = set(), []
    for v, o in results:
        key = str(v).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((v, o))
    return uniq


def is_already_processed(video: Path, out_dir: Path) -> bool:
    stem = video.stem
    needed = [
        out_dir / f"{stem}_bar_path.csv",
        out_dir / f"{stem}_bar_summary.json",
    ]
    return all(p.exists() for p in needed)


# ===========================================================================
# メイン
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Barbell plate tracking: bar path & velocity from side-view video")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--roi", type=str, default=None,
                    help="ROI as x,y,w,h (single-video mode only)")
    ap.add_argument("--plate-diameter", type=float, default=PLATE_DIAMETER_MM,
                    help=f"plate diameter in mm (default: {PLATE_DIAMETER_MM})")
    ap.add_argument("--reselect-roi", action="store_true",
                    help="ignore saved ROI json and redetect")
    ap.add_argument("--select-roi", action="store_true",
                    help="skip auto-detection and select ROI on GUI")
    args = ap.parse_args()

    save_annotated = not args.no_video
    save_plot      = not args.no_plot

    if args.video:
        video = Path(args.video).expanduser().resolve()
        if not video.exists():
            sys.exit(f"[ERROR] video not found: {video}")
        out_dir = Path(args.out).resolve() if args.out else (video.parent / "pose")
        print(f"[SINGLE] {video} -> {out_dir}")
        result = process_one_video(video, out_dir,
                                   roi_arg=args.roi,
                                   plate_diameter_mm=args.plate_diameter,
                                   save_annotated=save_annotated,
                                   save_plot=save_plot,
                                   reselect_roi=args.reselect_roi,
                                   force_gui=args.select_roi)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.roi:
        print("[WARN] --roi is only used in single-video mode (--video); ignored")

    print(f"[BATCH] scanning {DATA_ROOT}")
    tasks = find_videos(filter_subject=args.subject)
    if not tasks:
        print("[INFO] no videos found")
        print(f"  動画の置き場所（どちらでも可）:")
        print(f"    {DATA_ROOT}/<subject>/<session>/videos/*.mp4")
        print(f"    {DATA_ROOT}/<任意フォルダ>/<任意フォルダ>/*.mp4")
        return

    pending = []
    skipped = 0
    for video, out_dir in tasks:
        if (not args.force) and is_already_processed(video, out_dir):
            skipped += 1
            continue
        pending.append((video, out_dir))

    print(f"[INFO] To process: {len(pending)} videos (skipped: {skipped})")
    for video, out_dir in pending:
        rel = video.relative_to(DATA_ROOT)
        print(f"\n[RUN] {rel}")
        try:
            res = process_one_video(video, out_dir,
                                    plate_diameter_mm=args.plate_diameter,
                                    save_annotated=save_annotated,
                                    save_plot=save_plot,
                                    reselect_roi=args.reselect_roi,
                                    force_gui=args.select_roi)
            if res["status"] == "ok":
                rep_txt = ", ".join(
                    f"rep{r['rep']}: mean {r['mean_con_v']} / peak {r['peak_con_v']} m/s"
                    for r in res["reps"]) or "no reps detected"
                print(f"  -> frames={res['n_frames']}, "
                      f"roi={res['roi_method']}, "
                      f"tracked={res['tracked_ratio']:.1%}, "
                      f"scale={res['mm_per_px']} mm/px ({res['scale_source']})")
                print(f"     {rep_txt}")
            else:
                print(f"  -> {res['status']}: {res.get('reason')}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
