# -*- coding: utf-8 -*-
"""
=============================================================================
 1 セットの品質チェック（run_pipeline.py の品質レポートと train_rpe_model.py の除外で共通に使う）

 features_all.csv の 1 行（= 1 セット）を見て、要確認事項を返す。
 そのうち「データが欠けている・壊れていて学習に使えない」ものは exclude=True。
 train_rpe_model.py は、IMU を使う条件からは IMU が使えないセットを、
 動画を使う条件からは動画が使えないセットを外す（context の条件には残る）。

 学習から外すもの（exclude=True）
   動画: 骨格なし／人物を検出できないフレームが 20 % 超
         ／足元が画面の下端で切れている（体の大きさ BL が正しく出ず、速度がずれる）
         ／正面から撮れていない（正面用の特徴が出ない）／骨格のレップ数 ≠ 申告の回数
         ／速度低下が −10 % 未満（最後のレップが 1 回目より 10 % 以上速い = レップの取り違え）
   IMU:  IMU のレップ数 ≠ 申告の回数（0 は送信が途中で止まったなど）／IMU のレップを検出できない
         ／受信レートが 80 Hz 未満（受信が途切れている）
         ／速度低下が −10 % 未満（ラックインをレップと数えたなど。2026-10-20 set02 で発生）
 要確認として出すだけ（学習には使う）
   キーポイントの信頼度が低い、M5 の向き（速度は 3 軸の合成加速度から出すので向きに依らない）、
   動画と IMU の時刻合わせが不確か（IMU だけでレップを数えた。回数・速度低下の確認は上のとおり）、
   IMU に静止区間がない（重力・ドリフトの補正の足場がなく、速度・変位がずれやすい。2026-09-18 S002 set02 で
   上昇の変位が同じ人のほかのセットより 11〜12 cm 小さく出た。外すかどうかはまだ決めない: 2026-10-31 ユーザー判断）、
   骨格の追跡が別の人に移ったコマがあった（カメラの前を人が横切ったなど。そのコマは特徴量に使っていない）
 ラベル（RPE・重量・回数）の未入力は学習スクリプト側で扱う
   （RPE が無いセットは使わない。重量・回数が無いセットは context の条件から外れる）

 使い方（ほかのスクリプトから）:
   from quality_checks import check_set, exclusion_reasons
   issues = check_set(row)              # [Issue(part, text, exclude), ...]
   reasons = exclusion_reasons(row)     # {"imu": [...], "video": [...]}（空なら使える）
=============================================================================
"""

from typing import NamedTuple, Optional

import pandas as pd


POSE_NAN_MAX = 0.2      # 人物を検出できないフレームの割合の上限
POSE_VIS_MIN = 0.6      # キーポイントの信頼度（平均）の下限。下回っても学習には使う
ANKLE_Y_MAX = 0.97      # 足首の y（画面の上 0〜下 1）の最大がこれを超えたら、足元が切れている
SHOULDER_W_MIN = 0.12   # 肩幅 / 体の高さ がこれ未満なら、正面から撮れていない
IMU_AZ_MIN = 0.8        # 静止時の az の中央値 [g]。下回っても学習には使う
IMU_FS_MIN = 80.0       # 受信レート [Hz]。下回ると受信が途切れている
VL_MIN_PCT = -10.0      # 速度低下 [%] がこれ未満 = 最後のレップが 1 回目より 10 % 以上速い（レップの取り違え）


class Issue(NamedTuple):
    part: str        # "label" / "video" / "imu"
    text: str
    exclude: bool    # True: そのデータ（動画 / IMU）は学習に使えない


def _num(row, key) -> Optional[float]:
    v = row.get(key)
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def _flag(v) -> Optional[bool]:
    """True / False（CSV では文字列のこともある）。分からなければ None"""
    if isinstance(v, str):
        s = v.strip().lower()
        return True if s in ("true", "1") else False if s in ("false", "0") else None
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return bool(v)


def has_pose(row) -> bool:
    """骨格（動画）の結果があるか。pose_available の列が無ければ、骨格の速度があるかで判断する"""
    flag = _flag(row.get("pose_available"))
    return flag if flag is not None else _num(row, "pose_first_rep_mcv_bl") is not None


def has_imu(row) -> bool:
    return row.get("modality") == "imu+video" or _num(row, "imu_n_samples") is not None


def check_set(row) -> list:
    """1 セットぶんの要確認事項: [Issue(part, text, exclude), ...]"""
    out = []
    if _num(row, "rpe") is None:
        out.append(Issue("label", "RPE 未入力", False))
    if _num(row, "weight_kg") is None:
        out.append(Issue("label", "重量 未入力", False))
    reps = _num(row, "reps_completed")
    if reps is None:
        out.append(Issue("label", "回数 未入力", False))

    # 骨格（動画）
    if not has_pose(row):
        out.append(Issue("video", f"骨格なし（{row.get('pose_error') or '動画/骨格推定を確認'}）", True))
    else:
        nan_ratio = _num(row, "pose_nan_ratio")
        vis = _num(row, "pose_key_visibility_mean")
        ankle = _num(row, "pose_ankle_y_max_norm")
        shw = _num(row, "pose_shoulder_width_bl")
        pr = _num(row, "pose_n_reps_detected")
        vl = _num(row, "pose_velocity_loss_pct")
        if nan_ratio is not None and nan_ratio > POSE_NAN_MAX:
            out.append(Issue("video", f"人物を検出できないフレームが {nan_ratio:.0%}", True))
        other = _num(row, "pose_other_person_frames")
        if other:
            out.append(Issue("video", f"骨格の追跡が別の人に移ったコマ {int(other)} を使わなかった（カメラの前を人が横切ったなど）",
                             False))
        if vis is not None and vis < POSE_VIS_MIN:
            out.append(Issue("video", f"キーポイントの信頼度が低い（{vis:.2f}）", False))
        if ankle is not None and ankle > ANKLE_Y_MAX:
            out.append(Issue("video", "足元が画面の下端で切れている", True))
        if shw is not None and shw < SHOULDER_W_MIN:
            out.append(Issue("video", f"正面から撮れていない可能性（肩幅/体高 {shw:.2f}）", True))
        if reps is not None and pr is not None and pr != reps:
            out.append(Issue("video", f"骨格のレップ検出 {int(pr)} ≠ 申告 {int(reps)}", True))
        elif _num(row, "pose_first_rep_mcv_bl") is None:
            out.append(Issue("video", "骨格のレップを検出できない", True))
        if vl is not None and vl < VL_MIN_PCT:
            out.append(Issue("video", f"骨格の速度低下 {vl:.0f} %（最後のレップが速すぎる。レップの取り違えの可能性）",
                             True))

    # IMU（ある場合）
    if has_imu(row):
        az = _num(row, "imu_az_median_g")
        fs = _num(row, "imu_fs_hz")
        ir = _num(row, "imu_n_reps_detected")
        vl = _num(row, "imu_velocity_loss_mcv_pct")
        if az is not None and az < IMU_AZ_MIN:
            out.append(Issue("imu", f"M5 の向きが違う可能性（az 中央値 {az:.2f} g。画面を真上に）", False))
        sr, sm = _num(row, "sync_r"), _num(row, "sync_margin")
        if row.get("imu_rep_source") == "imu" and sr is not None and has_pose(row):
            out.append(Issue("imu", f"動画と IMU の時刻合わせが不確か（r {sr:.2f}・別のずれとの差 "
                                    f"{sm if sm is not None else float('nan'):.2f}）。IMU だけでレップを数えた", False))
        if _num(row, "imu_n_still") == 0:
            out.append(Issue("imu", "IMU に静止区間がない（担いでいる間ずっと揺れていた）。速度・変位がずれやすい", False))
        if fs is not None and fs < IMU_FS_MIN:
            out.append(Issue("imu", f"IMU の受信レートが低い（{fs:.0f} Hz）", True))
        if reps is not None and ir is not None and ir != reps:
            out.append(Issue("imu", f"IMU のレップ検出 {int(ir)} ≠ 申告 {int(reps)}", True))
        elif _num(row, "imu_first_rep_mcv_mps") is None:
            out.append(Issue("imu", "IMU のレップを検出できない（送信の途切れなど）", True))
        if vl is not None and vl < VL_MIN_PCT:
            out.append(Issue("imu", f"IMU の速度低下 {vl:.0f} %（最後のレップが速すぎる。"
                                    "ラックインなどをレップと数えた可能性）", True))
    return out


def exclusion_reasons(row) -> dict:
    """学習から外す理由をデータごとに: {"imu": [...], "video": [...]}（空リスト = 使える）"""
    issues = check_set(row)
    return {part: [i.text for i in issues if i.part == part and i.exclude] for part in ("imu", "video")}
