# -*- coding: utf-8 -*-
"""
=============================================================================
 IMU（M5StickC Plus）の加速度センサの校正
 （基準にしている IMU の速度の目盛りが正しいかの確認。2026-11-15）

 なぜ要るか:
   バーに付けた M5 は、静止しているとき 1.04〜1.05 g と出る（本来は 1.00 g）。原因は 2 通りありうる。
     感度のずれ（1 g を 1.045 g と読む）      → 加速度も速度も 4.5 % 大きく出る
     ゼロ点のずれ（いつも +0.045 g 足される） → 重力の実測値を引くと消えるので、速度は正しい
   extract_features.imu_vertical_velocity は「3 軸の合成加速度 − 静止時の合成加速度」を積分するので、
   ゼロ点のずれは消えるが、感度のずれはそのまま速度に乗る。バーに付けたままの記録（傾き最大 41°）では
   どちらとも決められなかった（2026-11-15。両方の当てはめが同じくらい合う）ので、
   M5 の 6 つの面を順に下にして置いた記録から、軸ごとに分けて求める。

 置き方（チートシート「IMU の校正」）:
   受信は python scripts\\multi_imu_receiver.py --outdir data\\_calib → data/_calib/session_…/raw_BAR.csv
   M5 を机に置き、6 つの面を順に下にして、1 面につき手を離して 10 秒。2 周（計 12 回）。SPACE は押さない
   data/_ で始まるフォルダなので、run_pipeline.py の取り込み・学習には入らない

 計算:
   1. 静止している区間を見つける（加速度の揺れが小さく、ジャイロも小さい状態。前後 0.5 秒を除いて 3 秒以上）
      = 1 回の置き方。区間の中の平均を、その置き方の読みとする
   2. 6 つの面（x・y・z の各軸が上向き・下向き）がそろっているか確かめる。足りなければ計算しない
   3. センサの読み = 感度 × 本当の加速度 + ゼロ点のずれ（軸ごと）として、補正した後の大きさが
      どの置き方でも 1 g になるように、ゼロ点のずれ 3 個と感度 3 個を最小二乗で求める
      （面をきっちり垂直に置かなくてよい。置き方が 7 回以上あれば、合い具合と 95 % の幅も出る）
   4. いつもの付け方（画面が上 = z 軸が上）で、今の速度の計算が本当の何倍を出すかを計算する（velocity_factor）
 判定: その倍率が 1.00 ± 0.01 なら IMU の速度の倍率は正しい。外れていれば、その分だけ大きい・小さい
       （言えるのは倍率 = 平均して本当の何倍に出るか、まで。レップごとの値は、ゼロ点のずれを当てはめるかどうかでも
         数 % 動く: 速度の計算が入力のわずかな違いに敏感なため。2026-11-16 の確認）
 出力:
   記録を確かめるだけ（既定）: data/_calib/session_…/calibration_<デバイス>.json（6 つの面がそろったときだけ）。
                             解析の数字は変わらない
   解析に使う（--apply）      : data/_calib の記録を全部まとめて当てはめ、data/_calib/calibration_<デバイス>.json に保存する。
                             このファイルがあると、extract_features.py などが IMU を読むときに
                             本当の加速度 = (読み − ゼロ点のずれ) / 感度 を当てはめてから速度を計算する（2026-11-16）。
                             → 解析の数字が変わるので、保存したら python scripts\\run_pipeline.py --no-split --force-features
                             やめるとき・使わない記録があるとき: そのファイル・フォルダを data/_calib/_archive/ へ移して作り直す
                             （元の記録 raw_*.csv と imu/setNN_*.csv は書き換えない）

 使い方:
   python scripts\\calibrate_imu.py                               # data/_calib のいちばん新しい記録を確かめる
   python scripts\\calibrate_imu.py --src data\\_calib\\session_…   # 記録を指定して確かめる
   python scripts\\calibrate_imu.py --apply                       # 全部の記録をまとめて、解析に使う校正として保存

 依存: numpy pandas scipy
=============================================================================
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
CALIB_ROOT = PROJECT_ROOT / "data" / "_calib"
FEATURES_CSV = PROJECT_ROOT / "outputs" / "features_all.csv"

AXES = ("x", "y", "z")
ACC_COLS = ["ax_g", "ay_g", "az_g"]
GYR_COLS = ["gx_dps", "gy_dps", "gz_dps"]
MOUNT_UP = (0.0, 0.0, 1.0)   # いつもの付け方で上を向く軸（画面が真上 = z+。CLAUDE.md §9）
FACE_NAME = {"z+": "画面が上", "z-": "画面が下"}

STILL_WIN_S = 0.5        # 静止の判定に使う窓 [s]
STILL_ACC_SD_G = 0.02    # 窓の中の加速度の SD（3 軸とも）がこれ未満 [g]（机の上の M5 は 0.008 前後）
STILL_GYRO_DPS = 5.0     # ジャイロ（記録全体の中央値を引いたもの）の窓平均の大きさがこれ未満 [dps]
EDGE_S = 0.5             # 置いた直後・持ち上げる直前として、静止の区間の前後から除く [s]
MIN_STILL_S = 3.0        # 前後を除いた後にこれ以上続く静止を、1 回の置き方として使う [s]
FACE_COS = 0.8           # ある軸と鉛直のなす角が約 37° 以内なら、その軸の面を置いたと数える
WEIGHT_CAP_S = 10.0      # 当てはめの重み = 置いた時間の平方根（これより長く置いても重みは増やさない）
VEL_OK = 0.01            # 速度の倍率が 1.00 ± これ以内なら「IMU の速度の倍率は正しい」
REST_OFF_G = 0.01        # 静止時の読みが 1 g からこれより外れていたら、判定にその理由を添える


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def find_sessions(root: Path = CALIB_ROOT) -> list:
    """raw_*.csv のある受信フォルダ（session_…）を、名前（= 受信を始めた日時）の順で。_archive などの中は見ない"""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("raw_*.csv")))


def find_latest_session(root: Path = CALIB_ROOT):
    """raw_*.csv のある受信フォルダ（session_…）のうち、名前（= 受信を始めた日時）がいちばん新しいもの。無ければ None"""
    folders = find_sessions(root)
    return folders[-1] if folders else None


def load_raw(folder: Path, device=None):
    """raw_<デバイス>.csv を読む。device が無ければ BAR、それも無ければ名前の順で最初。戻り値: (デバイス, 表) か (None, None)"""
    raws = sorted(folder.glob("raw_*.csv"))
    if device:
        raws = [p for p in raws if p.stem == f"raw_{device}"]
    if not raws:
        return None, None
    pick = next((p for p in raws if p.stem == "raw_BAR"), raws[0])
    df = pd.read_csv(pick)
    if not set(ACC_COLS + GYR_COLS + ["boot_ms"]).issubset(df.columns):
        return None, None
    return pick.stem.replace("raw_", ""), df


def past_rest_readings(path: Path = FEATURES_CSV, calibrated: bool = False):
    """
    これまでのセットの静止時の合成加速度 [g]（特徴量の一覧の imu_gravity_ref_g）。校正の結果と見比べる。無ければ None。
    calibrated=False: 校正を当てはめずに作ったセット（センサの読みのまま。校正から計算した「静止したときの読み」と比べる）
    calibrated=True : 校正を当てはめて作ったセット（imu_calibrated。1.00 g に近ければ、その校正がジムの記録にも合っている）
    """
    try:
        df = pd.read_csv(path)
        g = df["imu_gravity_ref_g"]
    except Exception:
        return None
    if "imu_calibrated" in df.columns:
        done = df["imu_calibrated"].astype(str).str.lower() == "true"
    else:
        done = pd.Series(False, index=df.index)
    g = g[done if calibrated else ~done].dropna()
    return g if len(g) else None


# ---------------------------------------------------------------------------
# 置き方（静止している区間）
# ---------------------------------------------------------------------------
def face_of(mean) -> str:
    """読みの向きから、上を向いている軸（"z+" など）。どの軸からも約 37° より離れていれば ""（斜めに置いた）"""
    u = np.asarray(mean, float) / np.linalg.norm(mean)
    i = int(np.argmax(np.abs(u)))
    return f"{AXES[i]}{'+' if u[i] > 0 else '-'}" if abs(u[i]) >= FACE_COS else ""


def still_places(df: pd.DataFrame) -> list:
    """
    静止している区間（= 1 回の置き方）を見つけ、区間ごとの 3 軸の読みの平均を返す。
    時間は M5 の時計（boot_ms）で測る（PC の受信時刻は Wi-Fi で揺れる。CLAUDE.md §7 の 2）。
    戻り値: [{"t0_s": 記録の始まりからの秒, "dur_s", "n", "mean": [x, y, z], "norm": 大きさ, "face"}, ...]
    """
    acc = df[ACC_COLS].to_numpy(float)
    gyr = df[GYR_COLS].to_numpy(float)
    t = df["boot_ms"].to_numpy(float) / 1000.0
    n = len(t)
    if n < 10:
        return []
    dt = float(np.median(np.diff(t)))
    win = max(5, int(round(STILL_WIN_S / dt))) if dt > 0 else 50
    roll = lambda a: pd.DataFrame(a).rolling(win, center=True, min_periods=win)   # noqa: E731
    acc_sd = roll(acc).std().to_numpy().max(axis=1)
    # M5 のジャイロはオフセットが大きい（十数 dps）ので、記録全体の中央値を引いてから見る
    gyr_mean = np.linalg.norm(roll(gyr - np.median(gyr, axis=0)).mean().to_numpy(), axis=1)
    still = (acc_sd < STILL_ACC_SD_G) & (gyr_mean < STILL_GYRO_DPS)      # 端（窓が足りない所）は NaN → 静止としない

    places, i = [], 0
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and still[j + 1]:
            j += 1
        keep = np.arange(i, j + 1)
        keep = keep[(t[keep] >= t[i] + EDGE_S) & (t[keep] <= t[j] - EDGE_S)]
        if len(keep) >= 2 and t[keep[-1]] - t[keep[0]] >= MIN_STILL_S:
            mean = acc[keep].mean(axis=0)
            places.append({"t0_s": round(float(t[keep[0]] - t[0]), 2),
                           "dur_s": round(float(t[keep[-1]] - t[keep[0]]), 2), "n": int(len(keep)),
                           "mean": [float(x) for x in mean], "norm": float(np.linalg.norm(mean)),
                           "face": face_of(mean)})
        i = j + 1
    return places


# ---------------------------------------------------------------------------
# 当てはめ
# ---------------------------------------------------------------------------
def velocity_factor(offset, sens, up=MOUNT_UP):
    """
    いつもの付け方（up の軸が上）で、今の速度の計算（合成加速度 − 静止時の合成加速度）が本当の何倍を出すか。
    本当の加速度が up の向きに (1 + α) g のとき、読みは (1 + α)·S·up + b（S = 感度、b = ゼロ点のずれ）。
    合成加速度を α の 1 次まで開くと、静止時の値 |S·up + b| からの増え方は α × (S·up)·(S·up + b) / |S·up + b|。
    この係数が倍率（ゼロ点のずれだけならほぼ 1、up の軸の感度が 1.045 ならほぼ 1.045）。
    戻り値: (倍率, 静止時の読み [g])
    """
    su = np.asarray(sens, float) * np.asarray(up, float)
    rest = su + np.asarray(offset, float)
    g = float(np.linalg.norm(rest))
    return float(su @ rest / g), g


def fit_places(places: list) -> dict:
    """
    置き方ごとの読みから、軸ごとのゼロ点のずれ b と感度 s を求める: |(読み − b) / s| = 1 g がどの置き方でも成り立つように。
    6 つの面がそろっていなければ計算しない（{"ok": False, "missing": 足りない向き}）。
    戻り値（計算できたとき）: ok, n_places, offset_g, sensitivity, dof（置き方の数 − 6）, rms_mg（補正後の大きさの 1 g からのずれ）,
      offset_ci_half・sensitivity_ci_half・velocity_factor_ci（95 % の幅。この記録の中のばらつきから。dof = 0 なら None）,
      velocity_factor, rest_reading_g。各置き方には norm_after（補正後の大きさ）を足す
    """
    out = {"ok": False, "n_places": len(places)}
    have = {p["face"] for p in places}
    missing = [f"{a}{sg}" for a in AXES for sg in "+-" if f"{a}{sg}" not in have]
    if missing:
        return dict(out, missing=missing)

    A = np.array([p["mean"] for p in places], float)
    w = np.sqrt(np.minimum([p["dur_s"] for p in places], WEIGHT_CAP_S))
    # 初期値: 向かい合う面の読みの平均（ゼロ点のずれ）と、差の半分（感度）
    b0, s0 = np.zeros(3), np.ones(3)
    for i, a in enumerate(AXES):
        up = np.mean([p["mean"][i] for p in places if p["face"] == a + "+"])
        dn = np.mean([p["mean"][i] for p in places if p["face"] == a + "-"])
        b0[i], s0[i] = (up + dn) / 2.0, (up - dn) / 2.0

    def resid(q):
        return (np.linalg.norm((A - q[:3]) / q[3:], axis=1) - 1.0) * w

    sol = least_squares(resid, np.r_[b0, s0])
    q = sol.x
    after = np.linalg.norm((A - q[:3]) / q[3:], axis=1)
    for p, x in zip(places, after):
        p["norm_after"] = float(x)
    dof = len(places) - 6
    factor, rest = velocity_factor(q[:3], q[3:])
    out.update(ok=True, offset_g=[float(x) for x in q[:3]], sensitivity=[float(x) for x in q[3:]], dof=dof,
               rms_mg=None, offset_ci_half=None, sensitivity_ci_half=None,
               velocity_factor=factor, velocity_factor_ci=None, rest_reading_g=rest)
    if dof >= 1:      # 置き方が 7 回以上: 当てはめの残りから、合い具合と 95 % の幅（t 分布）を出す
        out["rms_mg"] = float(1000.0 * np.sqrt(np.mean((after - 1.0) ** 2)))
        cov = float(np.sum(sol.fun ** 2) / dof) * np.linalg.inv(sol.jac.T @ sol.jac)
        tq = float(stats.t.ppf(0.975, dof))
        half = tq * np.sqrt(np.diag(cov))
        out["offset_ci_half"] = [float(x) for x in half[:3]]
        out["sensitivity_ci_half"] = [float(x) for x in half[3:]]
        eps = 1e-6    # 倍率の幅: 6 個の値を少しずつ動かしたときの倍率の変わり方から
        grad = np.array([(velocity_factor(*np.split(q + e, 2))[0] - factor) / eps for e in np.eye(6) * eps])
        kh = tq * float(np.sqrt(grad @ cov @ grad))
        out["velocity_factor_ci"] = [factor - kh, factor + kh]
    return out


def judge(factor: float, rest_g: float) -> str:
    """
    速度の倍率（velocity_factor）と静止時の読みから、IMU の速度の倍率が正しいかを文にする。
    言えるのは倍率（平均して本当の何倍に出るか）まで。レップごとの値は、ゼロ点のずれを当てはめるかどうかでも数 % 動く
    （速度の計算が入力のわずかな違いに敏感なため。2026-11-16 の確認）ので、「速度は正しい」とは書かない
    """
    if abs(factor - 1.0) <= VEL_OK:
        why = f"。静止で {rest_g:.3f} g と出るのは、ゼロ点のずれのため" if abs(rest_g - 1.0) > REST_OFF_G else ""
        return f"IMU の速度の倍率は正しい（平均して本当の何倍に出るか、のずれは {100 * VEL_OK:.0f} % 以内）{why}"
    pct = 100.0 * (factor - 1.0)
    return (f"IMU の速度が約 {abs(pct):.1f} % {'大きく' if pct > 0 else '小さく'}出ている（センサの感度のずれ）。"
            "校正を解析に使うと直る（python scripts\\calibrate_imu.py --apply）")


def calibrate(df: pd.DataFrame) -> dict:
    """受信の記録（raw_<デバイス>.csv の表）→ 校正の結果。places に置き方の一覧を入れる"""
    places = still_places(df)
    res = fit_places(places)
    res["places"] = places
    if res["ok"]:
        res["verdict"] = judge(res["velocity_factor"], res["rest_reading_g"])
    return res


def calibrate_session(folder: Path, device=None) -> dict:
    """
    受信フォルダの raw_<デバイス>.csv を校正し、結果を同じフォルダの calibration_<デバイス>.json に保存する。
    6 つの面がそろわず計算できなかったときは保存しない（ふつうの受信フォルダを指しても、ファイルを足さない）
    """
    folder = Path(folder)
    dev, raw = load_raw(folder, device)
    if raw is None:
        return {"ok": False, "n_places": 0, "places": [], "error": "raw_*.csv が無いか、列が足りない"}
    res = dict(calibrate(raw), device=dev, source=folder.name)
    if res["ok"]:
        res["saved_to"] = str(folder / f"calibration_{dev}.json")
        Path(res["saved_to"]).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def calibrate_all(root: Path = CALIB_ROOT, device=None) -> dict:
    """
    root（data/_calib）の中の校正の記録（受信フォルダ）を全部まとめて 1 つの当てはめにし、解析に使う校正として
    root/calibration_<デバイス>.json に保存する（extract_features.load_calibration が読む。前のファイルは上書きする）。
    1 周だけの記録でも、何回分かをまとめれば合い具合と 95 % の幅が出る。記録のフォルダには何も足さない。
    6 つの面がそろわなければ保存しない（前に保存した校正はそのまま残る）。
    戻り値は fit_places の値に次を足したもの: device、sources（使った記録の名前）、
      per_source（記録ごとの置き方の数と、その記録だけで当てはめた倍率・静止時の読み。面が足りない記録は None。
                  記録どうしが合っているかの確認用）、places（source つき）、verdict、saved_to
    """
    root = Path(root)
    dev, places, per_source = device, [], []
    for folder in find_sessions(root):
        d, raw = load_raw(folder, dev)
        if raw is None:
            continue
        dev = dev or d                  # 最初に見つけたデバイス（BAR があれば BAR）にそろえる
        own = still_places(raw)
        for p in own:
            p["source"] = folder.name
        one = fit_places(own)
        per_source.append({"source": folder.name, "n_places": len(own),
                           "velocity_factor": one.get("velocity_factor"), "rest_reading_g": one.get("rest_reading_g")})
        places += own
    if not per_source:
        return {"ok": False, "n_places": 0, "places": [], "error": "校正の記録（raw_*.csv のある受信フォルダ）が無い"}
    res = fit_places(places)
    res.update(places=places, device=dev, sources=[s["source"] for s in per_source], per_source=per_source)
    if res["ok"]:
        res["verdict"] = judge(res["velocity_factor"], res["rest_reading_g"])
        res["saved_to"] = str(root / f"calibration_{dev}.json")
        Path(res["saved_to"]).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# 表示
# ---------------------------------------------------------------------------
def print_places(res: dict):
    """置き方の一覧（何回目の記録か・上を向いた面・静止した秒数・読み・補正の前後の大きさ）"""
    places = res["places"]
    order = {s: k for k, s in enumerate(res.get("sources") or [], 1)}
    print(f"[置き方] {len(places)} 回（{MIN_STILL_S:.0f} 秒以上の静止。置き換える間は使わない）")
    for k, p in enumerate(places, 1):
        face = p["face"] or "斜め"
        name = f"{face}（{FACE_NAME[face]}）" if face in FACE_NAME else face
        after = f" → 補正後 {p['norm_after']:.4f}" if "norm_after" in p else ""
        src = f"記録 {order[p['source']]}  " if p.get("source") in order else ""
        x, y, z = p["mean"]
        print(f"  {k:2d}  {src}{name:<12} {p['dur_s']:5.1f} 秒  読み ({x:+.3f}, {y:+.3f}, {z:+.3f}) g"
              f"  大きさ {p['norm']:.4f}{after}")


def print_missing(res: dict):
    print("\n[結果] 6 つの面がそろっていないので、計算しません")
    print(f"  足りない向き: {'、'.join(res['missing'])}"
          "（上を向いた軸で数える。z+ = 画面が上、z- = 画面が下。x・y は残りの 4 つの細い面）")
    print("  → 画面が上・画面が下・細い面 4 つを、それぞれ下にして 10 秒ずつ置けているか確かめて、もう一度測る")
    print("    （上の表で「斜め」と出た置き方は、約 37° より傾いていて面として数えていない。"
          f"{MIN_STILL_S:.0f} 秒より短い静止も数えない）")


def print_fit(res: dict):
    """軸ごとの値・合い具合・判定を、用語の意味を添えて表示する"""
    print("\n[校正] 軸ごとの値（ゼロ点のずれ = 加速度が 0 のときの読み、感度 = 1 g を何 g と読むか）")
    for i, a in enumerate(AXES):
        hb = f"（±{res['offset_ci_half'][i]:.4f}）" if res["offset_ci_half"] else ""
        hs = f"（±{res['sensitivity_ci_half'][i]:.4f}）" if res["sensitivity_ci_half"] else ""
        print(f"  {a}: ゼロ点のずれ {res['offset_g'][i]:+.4f} g{hb}、感度 {res['sensitivity'][i]:.4f}{hs}")
    if res["rms_mg"] is None:
        print("  置き方が 6 回ちょうどなので、合い具合と 95 % の幅は出せません（2 周すると出ます）")
    else:
        print(f"  合い具合: 補正した後の大きさの 1 g からのずれは {res['rms_mg']:.1f} mg"
              f"（{res['n_places']} 回の置き方の二乗平均。1 mg = 0.001 g）。± は 95 % の幅")

    ci = res["velocity_factor_ci"]
    width = f"（95 % の幅 {ci[0]:.4f}〜{ci[1]:.4f}）" if ci else ""
    print(f"\n[判定] いつもの付け方（画面が上）で、校正を当てはめない速度の計算が出す値は本当の "
          f"{res['velocity_factor']:.4f} 倍{width}")
    print(f"  → {res['verdict']}")
    span = lambda g: f"中央値 {g.median():.3f} g、{g.min():.3f}〜{g.max():.3f}、{len(g)} セット"   # noqa: E731
    raw_past, cal_past = past_rest_readings(), past_rest_readings(calibrated=True)
    seen = f"（校正を当てはめずに作ったセットの実測: {span(raw_past)}）" if raw_past is not None else ""
    print(f"  いつもの付け方で静止したときの読みは {res['rest_reading_g']:.3f} g になる計算{seen}")
    if cal_past is not None:
        print(f"  解析に使っている校正を当てはめて作ったセットの静止時の値: {span(cal_past)}"
              "（1.00 g に近ければ、その校正がジムの記録にも合っている）")
    print("[注意] " + ("95 % の幅は、この記録の中のばらつきから出したもの。" if ci else "")
          + "静止時の値は日によって 0.5 % ほど動くので、別の日にもう一度測ると確かめられる")


def report(res: dict, folder: Path):
    """1 回の記録の校正の結果を表示する（確かめるだけ。解析の数字は変わらない）"""
    print(f"[記録] {folder}（{res.get('device', '?')}）")
    if res.get("error"):
        print(f"[ERROR] {res['error']}")
        return
    print_places(res)
    if not res["ok"]:
        print_missing(res)
        return
    print_fit(res)
    print(f"[OK] {res['saved_to']}（この記録の結果。解析の数字は変わらない。解析に使うには --apply）")


def report_all(res: dict, root: Path):
    """全部の記録をまとめた校正（解析に使う校正）の結果を表示する"""
    print(f"[記録] {root} の全部（{res.get('device', '?')}）")
    if res.get("error"):
        print(f"[ERROR] {res['error']}")
        return
    for k, s in enumerate(res["per_source"], 1):
        own = (f"この記録だけの倍率 {s['velocity_factor']:.4f}・静止時の読み {s['rest_reading_g']:.3f} g"
               if s["velocity_factor"] is not None else "この記録だけでは 6 つの面がそろわない")
        print(f"  記録 {k}: {s['source']}  置き方 {s['n_places']} 回  {own}")
    print_places(res)
    if not res["ok"]:
        print_missing(res)
        print("[INFO] 解析に使う校正は保存していません（前に保存したものがあれば、そのまま）")
        return
    print_fit(res)
    print(f"[OK] {res['saved_to']}（解析に使う校正）")
    print("  これ以降、IMU を読むときに 本当の加速度 = (読み − ゼロ点のずれ) / 感度 を当てはめてから速度を計算する")
    print("  今ある数字を作り直す: python scripts\\run_pipeline.py --no-split --force-features")
    print("    （そのあと compare_video_imu.py・compare_side_imu.py・train_rpe_model.py も）")
    print("  やめるとき: 上のファイルを data\\_calib\\_archive\\ へ移して、同じように作り直す")


def main():
    ap = argparse.ArgumentParser(description="IMU（M5）の加速度センサの校正: 6 つの面を置いた記録から、"
                                             "軸ごとのゼロ点のずれと感度を求める")
    ap.add_argument("--src", type=str, default=None, help="受信フォルダ（既定: data/_calib のいちばん新しい session_…）")
    ap.add_argument("--device", type=str, default=None, help="デバイス名（既定: BAR、無ければ最初の raw_*.csv）")
    ap.add_argument("--apply", action="store_true",
                    help="data/_calib の記録を全部まとめて当てはめ、解析に使う校正（data/_calib/calibration_<デバイス>.json）"
                         "として保存する（解析の数字が変わる）")
    args = ap.parse_args()

    how = ("  受信: python scripts\\multi_imu_receiver.py --outdir data\\_calib（重量・回数は Enter だけ）\n"
           "  M5 を机に置き、6 つの面を順に下にして、1 面につき手を離して 10 秒。2 周。q で終了")
    if args.apply:
        if args.src:
            print("[ERROR] --apply は data\\_calib の記録を全部使います（--src とは一緒に使えません）。"
                  "使わない記録は data\\_calib\\_archive\\ へ移してください")
            return
        if not find_sessions():
            print(f"[INFO] 校正の記録がありません（{CALIB_ROOT}）\n{how}")
            return
        report_all(calibrate_all(device=args.device), CALIB_ROOT)
        return

    folder = Path(args.src) if args.src else find_latest_session()
    if folder is None or not folder.is_dir():
        print(f"[INFO] 校正の記録がありません（{args.src or CALIB_ROOT}）\n{how}")
        return
    report(calibrate_session(folder, args.device), folder)


if __name__ == "__main__":
    main()
