# -*- coding: utf-8 -*-
"""
=============================================================================
 スクワット動画 骨格抽出スクリプト (YOLO26 Pose 版)
 ・Ultralytics YOLO26 Pose で動画から17点のCOCOキーポイントを抽出
 ・遮蔽下でのロバスト性が MediaPipe より高い
 ・pose_extract.py (MediaPipe版) と同じ出力形式のCSVを生成
 ・data/ 配下を再帰的に走査して未処理動画を自動処理

 出力ファイル命名: setNN_yolo_*.csv  (MediaPipe版と区別するため)

 使い方:
   python pose_extract_yolo.py                              # 全自動処理
   python pose_extract_yolo.py --subject S001
   python pose_extract_yolo.py --video path/to/x.mp4
   python pose_extract_yolo.py --force --model yolo26m-pose.pt
   python pose_extract_yolo.py --device cuda                # GPU使用 (デフォルト自動)
   python pose_extract_yolo.py --device cpu                 # CPU強制

 依存ライブラリ:
   pip install ultralytics opencv-python numpy pandas matplotlib tqdm

 著者: masaki（大学院修士研究 - RPE推定）
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ultralytics (YOLO26)
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT  = PROJECT_ROOT / "data"

# モデル選択: n(nano) < s(small) < m(medium) < l(large) < x(xlarge)
# 精度と速度のバランスで m を既定
_WEIGHT = PROJECT_ROOT / "weights" / "yolo26m-pose.pt"
DEFAULT_MODEL = str(_WEIGHT) if _WEIGHT.exists() else "yolo26m-pose.pt"

# 推論時の閾値
CONF_THRESH = 0.25     # 検出信頼度
IOU_THRESH  = 0.45     # NMS閾値 (YOLO26はNMS-freeだが互換のため)

# COCO 17 キーポイント
COCO_KEYPOINTS = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
assert len(COCO_KEYPOINTS) == 17
KP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# 動画コーデック候補
VIDEO_CODEC_CANDIDATES = ("avc1", "H264", "mp4v", "XVID")


# ===========================================================================
# 数学ユーティリティ
# ===========================================================================
def angle_3pt(p1, p2, p3) -> float:
    v1 = p1 - p2
    v3 = p3 - p2
    n1 = np.linalg.norm(v1)
    n3 = np.linalg.norm(v3)
    if n1 < 1e-9 or n3 < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(v1, v3) / (n1 * n3), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_to_vertical(p_top, p_bottom) -> float:
    v = p_bottom - p_top
    return float(np.degrees(np.arctan2(v[0], v[1])))


# ===========================================================================
# 特徴量計算
# COCO 17 キーポイントには foot_index がないため、足関節角度は計算不可。
# 代替: 体の他の特徴量に集中する
# ===========================================================================
FEATURE_KEYS = [
    "knee_angle_left", "knee_angle_right",
    "hip_angle_left", "hip_angle_right",
    "trunk_lean_deg", "hip_y", "knee_forward_x",
    "asymmetry_knee", "key_visibility",
    # ankle角度は計算不可（foot_indexがCOCO17にない）
]


def compute_features_per_frame(kp: np.ndarray) -> dict:
    """
    kp: shape (17, 3)  [x, y, confidence]
    """
    xy = kp[:, :2]
    conf = kp[:, 2]

    def pt(name):
        return xy[KP[name]]

    knee_l = angle_3pt(pt("left_hip"),  pt("left_knee"),  pt("left_ankle"))
    knee_r = angle_3pt(pt("right_hip"), pt("right_knee"), pt("right_ankle"))
    hip_l  = angle_3pt(pt("left_shoulder"),  pt("left_hip"),  pt("left_knee"))
    hip_r  = angle_3pt(pt("right_shoulder"), pt("right_hip"), pt("right_knee"))

    shoulder_mid = (pt("left_shoulder") + pt("right_shoulder")) / 2.0
    hip_mid      = (pt("left_hip")      + pt("right_hip"))      / 2.0
    trunk_lean = angle_to_vertical(shoulder_mid, hip_mid)
    hip_y = float(hip_mid[1])

    ankle_mid = (pt("left_ankle") + pt("right_ankle")) / 2.0
    knee_mid  = (pt("left_knee")  + pt("right_knee"))  / 2.0
    knee_forward = float(knee_mid[0] - ankle_mid[0])

    if np.isnan(knee_l) or np.isnan(knee_r):
        asymmetry_knee = float("nan")
    else:
        asymmetry_knee = abs(knee_l - knee_r)

    key_names = ["left_hip", "right_hip", "left_knee", "right_knee",
                 "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
    vis_mean = float(np.mean([conf[KP[n]] for n in key_names]))

    return {
        "knee_angle_left":  knee_l,
        "knee_angle_right": knee_r,
        "hip_angle_left":   hip_l,
        "hip_angle_right":  hip_r,
        "trunk_lean_deg":   trunk_lean,
        "hip_y":            hip_y,
        "knee_forward_x":   knee_forward,
        "asymmetry_knee":   asymmetry_knee,
        "key_visibility":   vis_mean,
    }


# ===========================================================================
# 動画ライター
# ===========================================================================
def make_video_writer(out_path: Path, fps: float, w: int, h: int):
    for codec_name in VIDEO_CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        wr = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if wr.isOpened():
            print(f"  [VIDEO] codec={codec_name}")
            return wr, codec_name
        wr.release()
    return None, None


# ===========================================================================
# YOLO骨格描画（オーバーレイ）
# ===========================================================================
COCO_SKELETON_LINES = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 6),            # shoulders
    (5, 11), (6, 12),  # torso
    (11, 12),          # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def select_person(r, prev_center):
    """
    複数人が検出されたときに挙上者1人を選ぶ。
      初回      : 画面上で最も大きく映っている人（カメラに一番近い＝挙上者）
      2回目以降 : 前フレームで選んだ人にボックス中心が最も近い人（追跡）
    戻り値: (選んだ人の番号, そのボックス中心 or None)
    """
    boxes = getattr(r, "boxes", None)
    n = len(r.keypoints.xyn)
    if boxes is None or len(boxes) != n:
        return 0, None
    xywhn = boxes.xywhn.cpu().numpy()          # (n, 4): cx, cy, w, h
    if prev_center is None:
        best = int(np.argmax(xywhn[:, 2] * xywhn[:, 3]))
    else:
        d = np.hypot(xywhn[:, 0] - prev_center[0], xywhn[:, 1] - prev_center[1])
        best = int(np.argmin(d))
    return best, (float(xywhn[best, 0]), float(xywhn[best, 1]))


def draw_skeleton(frame, kp_xy, kp_conf, conf_thresh=0.3):
    h, w = frame.shape[:2]
    pts_px = (kp_xy * np.array([w, h])).astype(int)
    # 骨格線
    for i, j in COCO_SKELETON_LINES:
        if kp_conf[i] > conf_thresh and kp_conf[j] > conf_thresh:
            cv2.line(frame, tuple(pts_px[i]), tuple(pts_px[j]), (0, 255, 0), 2)
    # キーポイント
    for i in range(len(kp_xy)):
        if kp_conf[i] > conf_thresh:
            cv2.circle(frame, tuple(pts_px[i]), 4, (0, 0, 255), -1)


# ===========================================================================
# 単一動画の処理
# ===========================================================================
def process_one_video(model, video_path: Path, out_dir: Path,
                      save_annotated=True, save_plot=True,
                      device="auto") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_kp   = out_dir / f"{stem}_yolo_keypoints.csv"
    out_ft   = out_dir / f"{stem}_yolo_features.csv"
    out_vid  = out_dir / f"{stem}_yolo_annotated.mp4"
    out_plot = out_dir / f"{stem}_yolo_plots.png"
    out_info = out_dir / f"{stem}_yolo_info.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "reason": "cannot open video"}

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    used_codec = None
    if save_annotated:
        writer, used_codec = make_video_writer(out_vid, fps, w, h)

    keypoint_rows = []
    feature_rows  = []
    pbar = tqdm(total=n_frm, desc=f"  {stem}", unit="frame", leave=False)

    frame_idx = 0
    prev_center = None          # 前フレームで選んだ人物のボックス中心（正規化座標）
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_idx / fps

        # YOLO推論
        results = model.predict(
            source=frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            verbose=False,
            device=device,
        )

        # 挙上者1人を選ぶ（ジムで他の人が映り込んでも追跡し続ける）
        kp_xy = None
        kp_conf = None
        if len(results) > 0 and results[0].keypoints is not None:
            r = results[0]
            if r.keypoints.xyn is not None and len(r.keypoints.xyn) > 0:
                best, center = select_person(r, prev_center)
                if center is not None:
                    prev_center = center
                kp_xy   = r.keypoints.xyn[best].cpu().numpy()   # (17, 2) normalized
                kp_conf = r.keypoints.conf[best].cpu().numpy()  # (17,)

        # 行の作成
        row = {"frame": frame_idx, "time_s": round(time_s, 4)}
        if kp_xy is not None:
            kp_full = np.concatenate([kp_xy, kp_conf[:, None]], axis=1)
            for i, name in enumerate(COCO_KEYPOINTS):
                row[f"{name}_x"]    = float(kp_full[i, 0])
                row[f"{name}_y"]    = float(kp_full[i, 1])
                row[f"{name}_conf"] = float(kp_full[i, 2])
            feat = compute_features_per_frame(kp_full)
        else:
            for name in COCO_KEYPOINTS:
                row[f"{name}_x"]    = float("nan")
                row[f"{name}_y"]    = float("nan")
                row[f"{name}_conf"] = 0.0
            feat = {k: float("nan") for k in FEATURE_KEYS}

        keypoint_rows.append(row)
        feat_row = {"frame": frame_idx, "time_s": round(time_s, 4)}
        feat_row.update(feat)
        feature_rows.append(feat_row)

        # オーバーレイ動画
        if writer is not None:
            if kp_xy is not None:
                draw_skeleton(frame, kp_xy, kp_conf)
            cv2.putText(frame, f"YOLO26 f={frame_idx} t={time_s:.2f}s",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            writer.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    # CSV保存
    df_kp = pd.DataFrame(keypoint_rows)
    df_ft = pd.DataFrame(feature_rows)
    df_kp.to_csv(out_kp, index=False)
    df_ft.to_csv(out_ft, index=False)

    # 動画の縦横サイズ（keypoints は 0〜1 正規化なので、角度・距離を正しく出すのに必要）
    info = {"video": video_path.name, "width": w, "height": h,
            "fps": fps, "n_frames": frame_idx, "model": str(getattr(model, "ckpt_path", "") or "")}
    out_info.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    if save_plot and len(df_ft) > 0:
        make_feature_plot(df_ft, out_plot, title=stem)

    valid = df_ft["key_visibility"].notna().sum()
    return {
        "status": "ok",
        "video": str(video_path),
        "n_frames": frame_idx,
        "n_valid_frames": int(valid),
        "nan_ratio": round(1.0 - valid / max(frame_idx, 1), 4),
        "fps": fps,
        "codec": used_codec,
        "out_keypoints": str(out_kp),
        "out_features":  str(out_ft),
        "out_video":     str(out_vid) if save_annotated and writer is not None else None,
        "out_plot":      str(out_plot) if save_plot else None,
    }


# ===========================================================================
# 特徴量プロット
# ===========================================================================
def make_feature_plot(df: pd.DataFrame, out_path: Path, title: str = ""):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = df["time_s"].to_numpy()

    axes[0].plot(t, df["knee_angle_left"],  lw=1.2, label="left knee",  color="C0")
    axes[0].plot(t, df["knee_angle_right"], lw=1.2, label="right knee", color="C1")
    axes[0].axhline(90, color="gray", lw=0.5, linestyle="--")
    axes[0].set_ylabel("knee angle [deg]")
    axes[0].set_title(f"Squat Pose Features (YOLO26) - {title}")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, df["hip_angle_left"],  lw=1.2, label="left hip",  color="C2")
    axes[1].plot(t, df["hip_angle_right"], lw=1.2, label="right hip", color="C3")
    axes[1].set_ylabel("hip angle [deg]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, df["trunk_lean_deg"], lw=1.2, color="C4")
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("trunk lean [deg]")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, df["hip_y"], lw=1.2, color="C5")
    axes[3].set_ylabel("hip_y (image coord)")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(alpha=0.3)
    axes[3].invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ===========================================================================
# data/ を走査
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
            # 協力者の受け取り置き場は ingest_remote.py で取り込んだ後に処理する
            if parts[0] == "_remote" and filter_subject != "_remote":
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
        out_dir / f"{stem}_yolo_keypoints.csv",
        out_dir / f"{stem}_yolo_features.csv",
    ]
    return all(p.exists() for p in needed)


# ===========================================================================
# メイン
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="YOLO26 pose extraction for squat videos")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help=f"YOLO model file (default: {DEFAULT_MODEL})")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda, cpu, or auto (default: auto)")
    args = ap.parse_args()

    save_annotated = not args.no_video
    save_plot      = not args.no_plot
    device = args.device  # None=auto

    print(f"[YOLO] loading model: {args.model}")
    model = YOLO(args.model)
    if device is None:
        # ultralyticsは未指定なら自動でGPU/CPUを選ぶ
        device_msg = "auto"
    else:
        device_msg = device
    print(f"[YOLO] device: {device_msg}")

    if args.video:
        video = Path(args.video).expanduser().resolve()
        if not video.exists():
            sys.exit(f"[ERROR] video not found: {video}")
        out_dir = Path(args.out).resolve() if args.out else (video.parent / "pose")
        print(f"[SINGLE] {video} -> {out_dir}")
        result = process_one_video(model, video, out_dir,
                                   save_annotated=save_annotated,
                                   save_plot=save_plot,
                                   device=device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

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
            res = process_one_video(model, video, out_dir,
                                    save_annotated=save_annotated,
                                    save_plot=save_plot,
                                    device=device)
            print(f"  -> frames={res.get('n_frames')}, "
                  f"valid={res.get('n_valid_frames')}, "
                  f"nan_ratio={res.get('nan_ratio')}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
