# -*- coding: utf-8 -*-
"""
=============================================================================
 スクワット動画 骨格抽出スクリプト
 ・MediaPipe Pose を使って動画から33点の関節座標を抽出
 ・スクワット解析に必要な特徴量（関節角度、深さ、体幹傾き等）を算出
 ・骨格オーバーレイ動画と特徴量プロットを保存
 ・data/ フォルダ配下を再帰的に走査して、未処理の動画を自動処理
 ・既に処理済みのファイルはスキップ（途中再開可能）

 使い方:
   python pose_extract.py                          # data/ 配下を全自動処理
   python pose_extract.py --video path/to/x.mp4    # 単一ファイルを処理
   python pose_extract.py --subject S001           # 特定被験者のみ処理
   python pose_extract.py --force                  # 処理済みも再処理
   python pose_extract.py --no-video --no-plot     # 高速モード

 依存ライブラリ:
   pip install mediapipe opencv-python numpy pandas matplotlib tqdm

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

# MediaPipe: バージョンによって solutions のパスが異なるので複数試す
mp_pose = None
mp_draw = None
mp_styles = None
_mediapipe_import_error = None
for _import_strategy in ("direct_submodule", "python_submodule", "from_solutions"):
    try:
        if _import_strategy == "direct_submodule":
            import mediapipe.solutions.pose as _pose
            import mediapipe.solutions.drawing_utils as _draw
            import mediapipe.solutions.drawing_styles as _styles
        elif _import_strategy == "python_submodule":
            import mediapipe.python.solutions.pose as _pose
            import mediapipe.python.solutions.drawing_utils as _draw
            import mediapipe.python.solutions.drawing_styles as _styles
        else:
            from mediapipe import solutions as _sol
            _pose = _sol.pose
            _draw = _sol.drawing_utils
            _styles = _sol.drawing_styles
        mp_pose, mp_draw, mp_styles = _pose, _draw, _styles
        break
    except Exception as e:
        _mediapipe_import_error = e
        continue
if mp_pose is None:
    raise ImportError(
        f"Failed to import mediapipe.solutions.pose. Last error: {_mediapipe_import_error}\n"
        "Try: pip uninstall mediapipe -y && pip install mediapipe==0.10.14"
    )


# ===========================================================================
# 設定
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_ROOT  = PROJECT_ROOT / "data"

POSE_MODEL_COMPLEXITY = 2     # 0=Lite, 1=Full, 2=Heavy(深いポーズに強い)
POSE_MIN_DETECTION_CONFIDENCE = 0.3   # 下げて深いボトムでも検出を試みる
POSE_MIN_TRACKING_CONFIDENCE  = 0.3

# 出力動画のコーデック候補（左から優先）
# avc1/H264 はWindows標準プレイヤーで再生可、mp4v はVLC等で再生可
VIDEO_CODEC_CANDIDATES = ("avc1", "H264", "mp4v", "XVID")

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
assert len(LANDMARK_NAMES) == 33
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}


# ===========================================================================
# 数学ユーティリティ
# ===========================================================================
def angle_3pt(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v3 = p3 - p2
    n1 = np.linalg.norm(v1)
    n3 = np.linalg.norm(v3)
    if n1 < 1e-9 or n3 < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(v1, v3) / (n1 * n3), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_to_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
    v = p_bottom - p_top
    return float(np.degrees(np.arctan2(v[0], v[1])))


# ===========================================================================
# 特徴量計算
# ===========================================================================
FEATURE_KEYS = [
    "knee_angle_left", "knee_angle_right",
    "hip_angle_left", "hip_angle_right",
    "ankle_angle_left", "ankle_angle_right",
    "trunk_lean_deg", "hip_y", "knee_forward_x",
    "asymmetry_knee", "key_visibility",
]


def compute_features_per_frame(kp: np.ndarray) -> dict:
    xy = kp[:, :2]
    v  = kp[:, 3]

    def pt(name):
        return xy[LM[name]]

    knee_l = angle_3pt(pt("left_hip"),  pt("left_knee"),  pt("left_ankle"))
    knee_r = angle_3pt(pt("right_hip"), pt("right_knee"), pt("right_ankle"))
    hip_l = angle_3pt(pt("left_shoulder"),  pt("left_hip"),  pt("left_knee"))
    hip_r = angle_3pt(pt("right_shoulder"), pt("right_hip"), pt("right_knee"))
    ankle_l = angle_3pt(pt("left_knee"),  pt("left_ankle"),  pt("left_foot_index"))
    ankle_r = angle_3pt(pt("right_knee"), pt("right_ankle"), pt("right_foot_index"))

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

    key_lm = ["left_hip", "right_hip", "left_knee", "right_knee",
              "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
    vis_mean = float(np.mean([v[LM[n]] for n in key_lm]))

    return {
        "knee_angle_left":  knee_l,
        "knee_angle_right": knee_r,
        "hip_angle_left":   hip_l,
        "hip_angle_right":  hip_r,
        "ankle_angle_left":  ankle_l,
        "ankle_angle_right": ankle_r,
        "trunk_lean_deg":   trunk_lean,
        "hip_y":            hip_y,
        "knee_forward_x":   knee_forward,
        "asymmetry_knee":   asymmetry_knee,
        "key_visibility":   vis_mean,
    }


# ===========================================================================
# 動画ライターを作成（コーデック自動選択）
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
# 単一動画の処理
# ===========================================================================
def process_one_video(video_path: Path, out_dir: Path,
                      save_annotated: bool = True,
                      save_plot: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_kp   = out_dir / f"{stem}_keypoints.csv"
    out_ft   = out_dir / f"{stem}_features.csv"
    out_vid  = out_dir / f"{stem}_annotated.mp4"
    out_plot = out_dir / f"{stem}_plots.png"

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

    pose = mp_pose.Pose(
        model_complexity=POSE_MODEL_COMPLEXITY,
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
        smooth_landmarks=True,
    )

    keypoint_rows = []
    feature_rows  = []
    pbar = tqdm(total=n_frm, desc=f"  {stem}", unit="frame", leave=False)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks is not None:
            lms = res.pose_landmarks.landmark
            kp = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms])

            row = {"frame": frame_idx, "time_s": round(time_s, 4)}
            for i, name in enumerate(LANDMARK_NAMES):
                row[f"{name}_x"]   = kp[i, 0]
                row[f"{name}_y"]   = kp[i, 1]
                row[f"{name}_z"]   = kp[i, 2]
                row[f"{name}_vis"] = kp[i, 3]
            keypoint_rows.append(row)

            feat = compute_features_per_frame(kp)
            feat_row = {"frame": frame_idx, "time_s": round(time_s, 4)}
            feat_row.update(feat)
            feature_rows.append(feat_row)

            if writer is not None:
                mp_draw.draw_landmarks(
                    frame, res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )
        else:
            row = {"frame": frame_idx, "time_s": round(time_s, 4)}
            for name in LANDMARK_NAMES:
                row[f"{name}_x"]   = float("nan")
                row[f"{name}_y"]   = float("nan")
                row[f"{name}_z"]   = float("nan")
                row[f"{name}_vis"] = 0.0
            keypoint_rows.append(row)
            feature_rows.append({
                "frame": frame_idx, "time_s": round(time_s, 4),
                **{k: float("nan") for k in FEATURE_KEYS}
            })

        if writer is not None:
            cv2.putText(frame, f"f={frame_idx}  t={time_s:.2f}s",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            writer.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    pose.close()

    df_kp = pd.DataFrame(keypoint_rows)
    df_ft = pd.DataFrame(feature_rows)
    df_kp.to_csv(out_kp, index=False)
    df_ft.to_csv(out_ft, index=False)

    if save_plot and len(df_ft) > 0:
        make_feature_plot(df_ft, out_plot, title=stem)

    return {
        "status":   "ok",
        "video":    str(video_path),
        "n_frames": frame_idx,
        "fps":      fps,
        "codec":    used_codec,
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
    axes[0].set_title(f"Squat Pose Features - {title}")
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
# data/ 配下を走査して動画リストを返す
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
        out_dir / f"{stem}_keypoints.csv",
        out_dir / f"{stem}_features.csv",
    ]
    return all(p.exists() for p in needed)


# ===========================================================================
# メイン
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Squat video pose extraction")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
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
                                   save_annotated=save_annotated,
                                   save_plot=save_plot)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"[BATCH] scanning {DATA_ROOT}")
    tasks = find_videos(filter_subject=args.subject)
    if not tasks:
        print("[INFO] No videos found.")
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
                                    save_annotated=save_annotated,
                                    save_plot=save_plot)
            print(f"  -> frames={res.get('n_frames')}, codec={res.get('codec')}, "
                  f"out={res.get('out_features')}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n[DONE] all videos processed.")


if __name__ == "__main__":
    main()
