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


def test_imu_single_rep_has_no_velocity_loss():
    imu, _, _ = make_imu(n_reps=1)
    f = ef.compute_imu_features(imu, expected_reps=1)
    assert f["imu_n_reps_detected"] == 1
    assert f["imu_velocity_loss_mcv_pct"] is None


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
    assert (out / "report.md").exists() and (out / "best.pkl").exists()
    # 同じセッションが学習とテストにまたがっていない（A: フォールド = 被験者/セッション）
    pred = pd.read_csv(out / "predictions.csv")
    a = pred[pred["scheme"] == "A_within"]
    assert (a["fold"] == a["subject_id"] + "/" + a["session_id"]).all()
    # RPE 5 未満が無いデータでは「全 RPE」だけで学習。RPE 6 未満も無ければ評価は全セットだけ
    feats = synthetic_features()
    assert set(res["train_set"]) == {"all"}
    assert set(res["eval_set"]) == ({"all", "rpe>=6"} if (feats["rpe"] < 6).any() else {"all"})


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
    r = subprocess.run([sys.executable, str(SCRIPT_DIR / "train_rpe_model.py"), "--input", str(csv),
                        "--output-dir", str(tmp_path / "models"), "--kmax", "2"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    assert "学習データの範囲の比較" in (out / "report.md").read_text(encoding="utf-8")


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
