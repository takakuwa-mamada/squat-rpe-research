# -*- coding: utf-8 -*-
"""
=============================================================================
 RPE 推定モデルの学習・評価スクリプト（中間審査の図表をまとめて出す）

 入力: outputs/features_all.csv（run_pipeline.py / aggregate_features.py の出力）

 評価方式（データ量が足りないものは理由を表示してスキップ）:
   A. 本人専用   被験者ごとに Leave-One-Session-Out（その人の他の日だけで学習）
   B. 汎用       Leave-One-Subject-Out（他の人だけで学習し、初めて見る人を予測）
   C. 汎用＋本人 他の人 ＋ その人の最初の k セッションで学習し、残りのセッションを予測
   D. 学習曲線   その人の最初の k セッションだけで学習し、残りを予測（何試技で ±1 RPE か）
   ※ どの方式でも、同じセッションのセットが学習とテストにまたがることはない
   ※ C・D の曲線は、各条件を線形回帰（無ければリッジ）で描く。少ないセッションでも学習できるモデルにそろえるため
     （HGB は学習が 10 セット未満だと平均を答えるだけになる。2026-11-05 修正）。A・B の表のいちばん良いモデルとは違うことがある

 比較条件（特徴量セット。MCV = 上昇局面の平均速度）:
   mean_rpe        学習データの RPE の平均をいつも答える  … 比べる基準（これより良くなければ特徴が役に立っていない）
   context         %1RM（無ければ重量）＋ 回数           … RPE 早見表に相当
   imu_velocity    IMU の1レップ目 MCV だけ（線形回帰）
   imu_lrv         IMU の最後のレップの MCV だけ（線形回帰） … VBT でよく使う最終レップの速度
   imu_vbt         IMU の速度の指標一式（1レップ目・最後・平均の MCV、速度低下）＋ 回数 … VBT / Stance 相当
   imu_all         IMU のレップ単位の特徴すべて
   video_velocity  動画の1レップ目 MCV だけ（線形回帰）
   video_lrv       動画の最後のレップの MCV だけ（線形回帰）
   video_vbt       動画の速度の指標一式 ＋ 回数            … 動画だけの VBT
   video_all       動画（骨格）の特徴すべて               … 最終目標（動画だけ）
   imu+video       IMU ＋ 動画
   ※ mean_rpe は比べる基準なので、best.pkl の候補にせず、モデルも保存しない（2026-11-05 追加）
   --with-context で、各条件に %1RM・回数を足した版（*+ctx）も出す
   IMU の条件は IMU のある試技（本人）だけで評価する

 学習データの範囲（RPE 5 未満の軽いセットを学習に入れるか）:
   all       全 RPE で学習（図・保存モデルはこちら）
   rpe>=5    RPE 5 以上のセットだけで学習（--train-min-rpe で変更）
   ※ テストに使うセットはどちらも同じなので、MAE をそのまま比べられる
   ※ RPE 5 未満のセットが 1 つも無いときは all だけ
 評価の範囲:
   all       テストの全セット
   rpe>=6    RPE 6 以上のセットだけ（実際にトレーニングで使う範囲。主結果。--eval-min-rpe で変更）
   ※ RPE 5 以下は申告がぶれやすい（限界から遠いほど RIR の見積もりがずれる）ため、主結果から外す
 学習に使えないセット（データの欠け・レップの取り違え）:
   IMU が使えないセットは IMU を使う条件から、動画が使えないセットは動画を使う条件から外す
   （学習にもテストにも使わない。context の条件には残る）。基準は quality_checks.py（品質レポートと同じ）。
   外したセットと理由は report.md の「学習から外したセット」に出す。

 出力: models/<日時>/
   report.md          結果のまとめ（表・図へのリンク・データの内訳・外したセット・学習データの範囲の比較）
   results.csv        方式 × 条件 × モデル × 学習の範囲 × 評価の範囲 の MAE / RMSE / ±0.5・±1.0 以内率
   curves.csv         方式 C・D の k ごとの結果（学習の範囲・評価の範囲つき）
   predictions.csv    全予測（どのフォールドで予測したか付き）
   fig_*.png          比較の棒グラフ、学習曲線、散布図、特徴量重要度
   <条件>__<モデル>.pkl, best.pkl   全データで学習し直したモデル（predict_rpe.py 用）

 使い方:
   python scripts\\train_rpe_model.py
   python scripts\\train_rpe_model.py --with-context
   python scripts\\train_rpe_model.py --input outputs\\features_all.csv --kmax 6
   python scripts\\train_rpe_model.py --train-min-rpe 5 --eval-min-rpe 6   # 既定値

 依存: pip install numpy pandas scikit-learn matplotlib joblib
=============================================================================
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from quality_checks import exclusion_reasons     # 学習に使えないセットの基準（品質レポートと共通）

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "features_all.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "models"

RPE_FLOOR, RPE_MAX = 1.0, 10.0   # RPE の目盛りの下限・上限（予測をこの範囲に収める）
MAIN_TRAIN = "all"               # 図・保存モデルに使う学習データの範囲

# ---------------------------------------------------------------------------
# 特徴量セット（明示的に列挙する。記録長・品質指標・ドリフトを含む列は使わない）
#   使わない例: imu_n_samples / imu_total_time_s / imu_fs_hz / pose_n_frames（SPACE の押し方で変わる）
#              imu_bar_roll/pitch_range_deg（ジャイロ積分のドリフト込み）
#              imu_peak_velocity_up_mps 等のセット全体の統計（ラックの歩き出しを含む）
# ---------------------------------------------------------------------------
IMU_FEATURES = [
    "imu_first_rep_mcv_mps", "imu_last_rep_mcv_mps", "imu_mean_rep_mcv_mps",
    "imu_first_rep_mpv_mps", "imu_mean_rep_mpv_mps", "imu_velocity_loss_mcv_pct",
    "imu_first_rep_peak_v_mps", "imu_last_rep_peak_v_mps", "imu_mean_rep_peak_v_mps",
    "imu_velocity_loss_pct", "imu_first_rep_concentric_s", "imu_mean_rep_concentric_s",
]
VIDEO_FEATURES = [
    "pose_first_rep_mcv_bl", "pose_last_rep_mcv_bl", "pose_mean_rep_mcv_bl",
    "pose_first_rep_peak_v_bl", "pose_mean_rep_peak_v_bl", "pose_velocity_loss_pct",
    "pose_mean_descent_s", "pose_mean_ascent_s", "pose_mean_bottom_pause_s", "pose_ascent_time_ratio",
    "pose_mean_depth_bl", "pose_depth_change_bl", "pose_mean_hip_knee_dy_bl", "pose_mean_sticking_ratio",
    "pose_min_knee_ankle_ratio", "pose_knee_ankle_ratio_change",
    "pose_max_lateral_shift_bl", "pose_lateral_shift_change_bl",
    "pose_max_shoulder_tilt_deg", "pose_shoulder_tilt_change_deg",
    "pose_max_hip_tilt_deg", "pose_max_trunk_side_lean_deg",
]
IMU_KEY = "imu_first_rep_mcv_mps"      # これが無い試技は IMU 条件から外す
VIDEO_KEY = "pose_first_rep_mcv_bl"    # これが無い試技は動画条件から外す
# VBT でふつう使う速度の指標（1レップ目・最後・平均の MCV、速度低下）。回数は条件の側で足す
IMU_LRV, VIDEO_LRV = "imu_last_rep_mcv_mps", "pose_last_rep_mcv_bl"
IMU_VBT = [IMU_KEY, IMU_LRV, "imu_mean_rep_mcv_mps", "imu_velocity_loss_mcv_pct"]
VIDEO_VBT = [VIDEO_KEY, VIDEO_LRV, "pose_mean_rep_mcv_bl", "pose_velocity_loss_pct"]
BASELINE_COND = "mean_rpe"             # 比べる基準（特徴を使わず、学習データの RPE の平均を答える）

# report.md の「条件の説明」に出す文（スクリプトの中の名前を説明なしで使わないため）
COND_DESC = {
    "mean_rpe":       "比べる基準。学習データの RPE の平均をいつも答える（特徴は使わない）",
    "context":        "%1RM（無ければ重量）と回数だけ（RPE 早見表に相当）",
    "imu_velocity":   "IMU の 1 回目のレップの速度（MCV）だけ",
    "imu_lrv":        "IMU の最後のレップの速度だけ（VBT でよく使う最終レップの速度）",
    "imu_vbt":        "IMU の速度の指標一式（1 回目・最後・平均の速度、速度低下）＋ 回数",
    "imu_all":        "IMU のレップ単位の特徴すべて（12 個）",
    "video_velocity": "動画の 1 回目のレップの速度だけ",
    "video_lrv":      "動画の最後のレップの速度だけ",
    "video_vbt":      "動画の速度の指標一式（1 回目・最後・平均の速度、速度低下）＋ 回数",
    "video_all":      "動画（骨格）の特徴すべて（速度・深さ・左右の傾きなど 22 個）",
    "imu+video":      "IMU と動画の特徴すべて",
}


def build_conditions(df: pd.DataFrame, with_context: bool) -> dict:
    """条件名 → {cols, key, models}"""
    ctx_w = "pct_1rm" if ("pct_1rm" in df and df["pct_1rm"].notna().mean() >= 0.8) else "weight_kg"
    context = [ctx_w, "reps_completed"]
    # needs: その条件が使うデータ。学習に使えない（quality_checks.py で外す）セットはその条件から外す
    conds = {
        BASELINE_COND:    {"cols": [], "key": [], "models": ["mean"], "needs": [], "baseline": True},
        "context":        {"cols": context, "key": context, "models": ["linear", "hgb"], "needs": []},
        "imu_velocity":   {"cols": [IMU_KEY], "key": [IMU_KEY], "models": ["linear"], "needs": ["imu"]},
        "imu_lrv":        {"cols": [IMU_LRV], "key": [IMU_KEY], "models": ["linear"], "needs": ["imu"]},
        "imu_vbt":        {"cols": IMU_VBT + ["reps_completed"], "key": [IMU_KEY], "models": ["ridge"],
                           "needs": ["imu"]},
        "imu_all":        {"cols": IMU_FEATURES, "key": [IMU_KEY], "models": ["ridge", "rf", "hgb"],
                           "needs": ["imu"]},
        "video_velocity": {"cols": [VIDEO_KEY], "key": [VIDEO_KEY], "models": ["linear"], "needs": ["video"]},
        "video_lrv":      {"cols": [VIDEO_LRV], "key": [VIDEO_KEY], "models": ["linear"], "needs": ["video"]},
        "video_vbt":      {"cols": VIDEO_VBT + ["reps_completed"], "key": [VIDEO_KEY], "models": ["ridge"],
                           "needs": ["video"]},
        "video_all":      {"cols": VIDEO_FEATURES, "key": [VIDEO_KEY], "models": ["ridge", "rf", "hgb"],
                           "needs": ["video"]},
        "imu+video":      {"cols": IMU_FEATURES + VIDEO_FEATURES, "key": [IMU_KEY, VIDEO_KEY],
                           "models": ["ridge", "rf", "hgb"], "needs": ["imu", "video"]},
    }
    if with_context:
        for name in ["imu_velocity", "imu_all", "video_velocity", "video_all", "imu+video"]:
            c = conds[name]
            conds[f"{name}+ctx"] = {"cols": c["cols"] + context, "key": c["key"] + context,
                                   "models": ["ridge", "hgb"], "needs": c["needs"]}
    return conds


def make_model(name: str):
    if name == "mean":  # 比べる基準: 学習データの RPE の平均をいつも答える
        return DummyRegressor(strategy="mean")
    if name == "linear":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("m", LinearRegression())])
    if name == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("m", RidgeCV(alphas=np.logspace(-2, 3, 20)))])
    if name == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                     random_state=0, n_jobs=-1))])
    if name == "hgb":   # 欠損（1レップのセットの変化量など）をそのまま扱える
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                             min_samples_leaf=5, l2_regularization=1.0, random_state=0)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 評価の部品
# ---------------------------------------------------------------------------
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {"n": 0}
    e = p - y
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "hit05": float(np.mean(np.abs(e) <= 0.5)), "hit10": float(np.mean(np.abs(e) <= 1.0))}


def fit_predict(model_name, cols, train, test, clip_lo=RPE_FLOOR):
    """列が学習データで全部欠損なら落としてから学習する。予測は [clip_lo, 10] に収める。
    model_name == "mean"（比べる基準）は列を使わない（学習データの RPE の平均を答える）"""
    use = [c for c in cols if c in train and train[c].notna().any()]
    if len(train) < 3 or (not use and model_name != "mean"):
        return None, use
    X = (lambda d: d[use].to_numpy(float) if use else np.zeros((len(d), 1)))
    m = make_model(model_name)
    m.fit(X(train), train["rpe"].to_numpy(float))
    pred = np.clip(m.predict(X(test)), clip_lo, RPE_MAX)
    return (pred, m), use


def restrict(train: pd.DataFrame, train_min) -> pd.DataFrame:
    """学習データだけを RPE で絞る（テストは絞らない）。train_min が None なら全部使う"""
    return train if train_min is None else train[train["rpe"] >= train_min]


def clip_low(train_min) -> float:
    """予測の下限: RPE 5 以上だけで学習したモデルは 5 未満を予測しない"""
    return RPE_FLOOR if train_min is None else float(train_min)


def eval_rows(base: dict, te: pd.DataFrame, pred, eval_sets: dict) -> list:
    """1 つのテストの予測を、評価の範囲ごとに集計した行にする"""
    y, p = te["rpe"].to_numpy(float), np.asarray(pred, float)
    out = []
    for ename, emin in eval_sets.items():
        sel = np.ones(len(y), bool) if emin is None else (y >= emin)
        if sel.any():
            out.append(dict(base, eval_set=ename, **metrics(y[sel], p[sel])))
    return out


def label_train(name: str) -> str:
    return "全 RPE" if name == "all" else name.replace("rpe>=", "RPE ") + " 以上"


def label_eval(name: str) -> str:
    return "全セット" if name == "all" else name.replace("rpe>=", "RPE ") + " 以上"


def eligible(df, cond) -> pd.DataFrame:
    """その条件で使えるセット: 要の列と RPE があり、使うデータ（IMU / 動画）が学習から外されていない"""
    d = df.dropna(subset=[c for c in cond["key"] if c in df] + ["rpe"])
    for part in cond.get("needs", []):
        col = f"excl_{part}"
        if col in d:
            d = d[d[col].fillna("") == ""]
    return d


def mark_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """セットごとに、学習から外す理由を excl_imu / excl_video に入れる（空 = 使える）"""
    reasons = [exclusion_reasons(r) for r in df.to_dict("records")]
    df["excl_imu"] = [" / ".join(x["imu"]) for x in reasons]
    df["excl_video"] = [" / ".join(x["video"]) for x in reasons]
    return df


def sessions_in_order(d: pd.DataFrame) -> list:
    s = d.groupby("session_id")["date"].first().reset_index().sort_values(["date", "session_id"])
    return s["session_id"].tolist()


# ---------------------------------------------------------------------------
# 評価方式
# ---------------------------------------------------------------------------
def scheme_within(df, conds, train_sets, min_sessions, log):
    """A. 本人専用: 被験者ごとに Leave-One-Session-Out（学習データの範囲ごと）"""
    rows = []
    for cname, cond in conds.items():
        d_all = eligible(df, cond)
        for sid, d in d_all.groupby("subject_id"):
            sess = sessions_in_order(d)
            if len(sess) < min_sessions:
                log.add(f"A 本人専用 / {cname} / {sid}: セッション {len(sess)} < {min_sessions} のためスキップ")
                continue
            for tname, tmin in train_sets.items():
                for mname in cond["models"]:
                    for s in sess:
                        tr, te = restrict(d[d["session_id"] != s], tmin), d[d["session_id"] == s]
                        res, _ = fit_predict(mname, cond["cols"], tr, te, clip_low(tmin))
                        if res is None:
                            continue
                        rows += [dict(scheme="A_within", condition=cname, model=mname, train_set=tname,
                                      subject_id=sid, session_id=s, set_no=r.set_no, fold=f"{sid}/{s}",
                                      rpe=r.rpe, pred=p)
                                 for r, p in zip(te.itertuples(), res[0])]
    return rows


def scheme_loso(df, conds, train_sets, min_subjects, log):
    """B. 汎用: Leave-One-Subject-Out（学習データの範囲ごと）"""
    rows = []
    for cname, cond in conds.items():
        d = eligible(df, cond)
        subs = sorted(d["subject_id"].unique())
        if len(subs) < min_subjects:
            log.add(f"B 汎用 / {cname}: 被験者 {len(subs)} 人 < {min_subjects} 人のためスキップ")
            continue
        for tname, tmin in train_sets.items():
            for mname in cond["models"]:
                for sid in subs:
                    tr, te = restrict(d[d["subject_id"] != sid], tmin), d[d["subject_id"] == sid]
                    res, _ = fit_predict(mname, cond["cols"], tr, te, clip_low(tmin))
                    if res is None:
                        continue
                    rows += [dict(scheme="B_loso", condition=cname, model=mname, train_set=tname,
                                  subject_id=sid, session_id=r.session_id, set_no=r.set_no, fold=sid,
                                  rpe=r.rpe, pred=p)
                             for r, p in zip(te.itertuples(), res[0])]
    return rows


def curve_model(models: list) -> str:
    """学習曲線（C・D）で条件を代表させるモデル: 線形回帰があれば linear、無ければリッジ、どちらも無ければその条件のモデル。
    少ないセッションでも学習できるモデルにそろえる（2026-11-05 修正）。それまでは HGB がある条件（context・特徴すべて）を
    HGB で描いていたが、HGB は分けた先に 5 セット以上要る（min_samples_leaf=5）ので、学習が 10 セット未満だと分けられず
    平均を答えるだけになり、少ないセッションの曲線が基準（mean_rpe）の線と重なって実際より悪く見えていた"""
    for name in ("linear", "ridge"):
        if name in models:
            return name
    return models[0]


def scheme_curves(df, conds, train_sets, eval_sets, kmax, log):
    """C. 汎用＋本人 k セッション / D. 本人だけの学習曲線（各条件の代表モデル curve_model で、学習・評価の範囲ごと）"""
    rows = []
    for cname, cond in conds.items():
        mname = curve_model(cond["models"])
        d = eligible(df, cond)
        subs = sorted(d["subject_id"].unique())
        for sid in subs:
            me, others = d[d["subject_id"] == sid], d[d["subject_id"] != sid]
            sess = sessions_in_order(me)
            for k in range(0, min(len(sess) - 1, kmax) + 1):
                train_me = me[me["session_id"].isin(sess[:k])]
                te = me[me["session_id"].isin(sess[k:])]
                for tname, tmin in train_sets.items():
                    tr_me = restrict(train_me, tmin)
                    base = dict(condition=cname, model=mname, train_set=tname, subject_id=sid, k=k,
                                n_train_me=len(tr_me))
                    # C: 他の人 ＋ 本人 k セッション（他の人が 2 人以上いるとき）
                    if others["subject_id"].nunique() >= 2:
                        res, _ = fit_predict(mname, cond["cols"], pd.concat([restrict(others, tmin), tr_me]),
                                             te, clip_low(tmin))
                        if res is not None:
                            rows += eval_rows(dict(base, scheme="C_general_plus_k"), te, res[0], eval_sets)
                    # D: 本人だけ（k ≥ 1）
                    if k >= 1 and len(tr_me) >= 3:
                        res, _ = fit_predict(mname, cond["cols"], tr_me, te, clip_low(tmin))
                        if res is not None:
                            rows += eval_rows(dict(base, scheme="D_personal_curve"), te, res[0], eval_sets)
    if not rows:
        log.add("C/D 学習曲線: 2セッション以上ある被験者がいないためスキップ")
    return rows


def summarize(pred: pd.DataFrame, eval_sets: dict) -> pd.DataFrame:
    """予測を 方式 × 条件 × モデル × 学習の範囲 × 評価の範囲 ごとに集計する"""
    out = []
    for ename, emin in eval_sets.items():
        p = pred if emin is None else pred[pred["rpe"] >= emin]
        if p.empty:
            continue
        r = (p.groupby(["scheme", "condition", "model", "train_set"])
             .apply(lambda g: pd.Series({**metrics(g["rpe"], g["pred"]),
                                         "subjects": g["subject_id"].nunique(),
                                         "folds": g["fold"].nunique()}))
             .reset_index())
        r.insert(4, "eval_set", ename)
        out.append(r)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    for c in ("n", "subjects", "folds"):
        res[c] = res[c].astype(int)
    return res.sort_values(["scheme", "train_set", "eval_set", "mae"])


def importance_on_folds(df, cond, scheme, min_sessions):
    """テスト側フォールドでの permutation importance の平均（学習データでの過大評価を避ける）"""
    d = eligible(df, cond)
    imps = []
    if scheme == "B_loso":
        folds = [(d[d["subject_id"] != s], d[d["subject_id"] == s]) for s in d["subject_id"].unique()]
    else:
        folds = []
        for sid, g in d.groupby("subject_id"):
            sess = sessions_in_order(g)
            if len(sess) >= min_sessions:
                folds += [(g[g["session_id"] != s], g[g["session_id"] == s]) for s in sess]
    for tr, te in folds:
        res, use = fit_predict("hgb", cond["cols"], tr, te)
        if res is None or len(te) < 2:
            continue
        pi = permutation_importance(res[1], te[use].to_numpy(float), te["rpe"].to_numpy(float),
                                    n_repeats=10, random_state=0, scoring="neg_mean_absolute_error")
        imps.append(pd.Series(pi.importances_mean, index=use))
    if not imps:
        return None
    return pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def fig_compare(res: pd.DataFrame, scheme: str, out: Path, title: str):
    r = res[res["scheme"] == scheme].copy()
    if r.empty:
        return None
    best = r.sort_values("mae").groupby("condition").head(1).sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(best) + 1.5))
    colors = ["#999999" if c == BASELINE_COND else "#4C72B0" for c in best["condition"]]   # 基準は灰色
    ax.barh(best["condition"] + " (" + best["model"] + ")", best["mae"], color=colors)
    for i, (m, h, n) in enumerate(zip(best["mae"], best["hit10"], best["n"])):
        ax.text(m, i, f"  MAE {m:.2f} / ±1以内 {h:.0%} (n={n})", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE [RPE]（小さいほど良い）")
    ax.set_title(title)
    ax.set_xlim(0, max(best["mae"].max() * 1.8, 1.0))
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


# 学習曲線の線の色: 特徴の種類ごとに決まった色（条件が増えても色が一巡して重ならないように、
# データ（IMU / 動画 / IMU＋動画）ごとに図を分け、1 枚の中は種類で色を分ける）
CURVE_KINDS = [  # (種類, 凡例の説明, 色)
    ("velocity", "1 回目のレップの速度", "#2a78d6"),
    ("lrv", "最後のレップの速度", "#eb6834"),
    ("vbt", "速度の指標一式＋回数", "#1baf7a"),
    ("all", "特徴すべて", "#eda100"),
    ("context", "%1RM・回数", "#e87ba4"),
]
BASELINE_COLOR = "#8a8a8a"
CURVE_SOURCES = {"imu": "IMU", "video": "動画", "imu+video": "IMU＋動画"}
MODEL_JA = {"linear": "線形回帰", "ridge": "リッジ", "hgb": "HGB", "rf": "ランダムフォレスト"}


def curve_parts(cond: str):
    """条件名 → (データ, 種類, %1RM・回数を足した版か)。context と mean_rpe はデータを問わず全部の図に出す"""
    base = cond.removesuffix("+ctx")
    if base in ("context", BASELINE_COND):
        return None, base, False
    if base == "imu+video":
        return "imu+video", "all", cond.endswith("+ctx")
    src, kind = base.split("_", 1)                  # imu_lrv → ("imu", "lrv")
    return src, kind, cond.endswith("+ctx")


def fig_curve(curves: pd.DataFrame, scheme: str, out: Path, title: str):
    c = curves[curves["scheme"] == scheme]
    if c.empty:
        return None
    # 被験者をまたいで n で重み付け平均
    agg = (c.assign(w_mae=c["mae"] * c["n"], w_hit=c["hit10"] * c["n"])
           .groupby(["condition", "k"]).agg(n=("n", "sum"), w_mae=("w_mae", "sum"),
                                            w_hit=("w_hit", "sum"), n_train=("n_train_me", "mean"))
           .reset_index())
    agg["mae"], agg["hit10"] = agg["w_mae"] / agg["n"], agg["w_hit"] / agg["n"]
    parts = {cond: curve_parts(cond) for cond in agg["condition"].unique()}
    srcs = [s for s in CURVE_SOURCES if any(p[0] == s for p in parts.values())] or [None]
    color = {k: col for k, _, col in CURVE_KINDS} | {BASELINE_COND: BASELINE_COLOR}

    def style(kind, ctx):
        return dict(color=color.get(kind, "#444444"), lw=2, marker="o", ms=5,
                    ls="--" if kind == BASELINE_COND else (":" if ctx else "-"),
                    mfc="white" if ctx else color.get(kind, "#444444"))

    fig, axes = plt.subplots(2, len(srcs), figsize=(4.3 * len(srcs) + 0.8, 7.6),
                             sharex=True, sharey="row", squeeze=False)
    for j, src in enumerate(srcs):
        for cond, g in agg.groupby("condition"):
            s, kind, ctx = parts[cond]
            if s in (src, None):
                axes[0, j].plot(g["k"], g["mae"], **style(kind, ctx))
                axes[1, j].plot(g["k"], g["hit10"], **style(kind, ctx))
        axes[0, j].set_title(CURVE_SOURCES.get(src, ""))
        axes[1, j].axhline(0.8, color="#bbbbbb", lw=0.8, ls=":")
        axes[1, j].set_xlabel("学習に使った本人のセッション数 k")
    axes[0, 0].set_ylabel("MAE [RPE]（小さいほど良い）")
    axes[1, 0].set_ylabel("±1 RPE 以内の割合")
    for a in axes.flat:
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        a.grid(alpha=0.3)
    # 凡例は図の下に 1 つだけ（線に重ねない）。種類の名前は条件名の後半（imu_lrv の lrv）。
    # 線を描いたモデルも添える（A・B の表のいちばん良いモデルとは違うことがある）
    model_of = c.groupby("condition")["model"].first()

    def with_models(text: str, match) -> str:
        ms = sorted({MODEL_JA.get(model_of[cd], model_of[cd]) for cd, p in parts.items() if match(p)})
        return f"{text}／{'・'.join(ms)}" if ms else text

    kinds = {p[1] for p in parts.values()}
    handles = [Line2D([], [], label=f"{k}（{with_models(desc, lambda p, k=k: p[1] == k and not p[2])}）",
                      **style(k, False)) for k, desc, _ in CURVE_KINDS if k in kinds]
    if BASELINE_COND in kinds:
        handles.append(Line2D([], [], label=f"{BASELINE_COND}（平均を答える基準）", **style(BASELINE_COND, False)))
    if any(p[2] for p in parts.values()):
        handles.append(Line2D([], [], label="点線・白抜き = " + with_models("＋%1RM・回数", lambda p: p[2]),
                              **style(None, True)))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.035 * ((len(handles) + 2) // 3) + 0.02, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def fig_scatter(pred: pd.DataFrame, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    jitter = np.random.default_rng(0).normal(0, 0.05, len(pred))
    for sid, g in pred.groupby("subject_id"):
        ax.scatter(g["rpe"] + jitter[:len(g)], g["pred"], s=18, alpha=0.7, label=sid)
    lo = float(np.floor(min(pred["rpe"].min(), pred["pred"].min())))
    ax.plot([lo, RPE_MAX], [lo, RPE_MAX], "k--", lw=0.8)
    ax.fill_between([lo, RPE_MAX], [lo - 1, RPE_MAX - 1], [lo + 1, RPE_MAX + 1],
                    color="gray", alpha=0.12, label="±1 RPE")
    ax.set_xlabel("申告 RPE")
    ax.set_ylabel("推定 RPE")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def fig_importance(imp: pd.Series, out: Path, title: str, top=15):
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.2))
    ax.barh(s.index, s.values, color="#55A868")
    ax.set_xlabel("MAE の悪化量（permutation importance）")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
class Log(list):
    def add(self, msg):
        if msg not in self:
            self.append(msg)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="RPE 推定モデルの学習・評価")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--with-context", action="store_true", help="%%1RM・回数を足した条件も出す")
    ap.add_argument("--min-sessions", type=int, default=3, help="本人専用の評価に必要なセッション数")
    ap.add_argument("--min-subjects", type=int, default=3, help="汎用の評価に必要な被験者数")
    ap.add_argument("--kmax", type=int, default=8, help="学習曲線の k の上限")
    ap.add_argument("--train-min-rpe", type=float, default=5.0,
                    help="比較用に「この RPE 以上だけで学習」した版も作る（既定 5）")
    ap.add_argument("--eval-min-rpe", type=float, default=6.0,
                    help="主結果の評価範囲: この RPE 以上のセット（既定 6）")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["rpe"]).copy()
    df["rpe"] = df["rpe"].astype(float)
    if "date" not in df:
        df["date"] = ""
    df["date"] = df["date"].fillna("").astype(str)
    if len(df) < 5:
        sys.exit(f"[ERROR] RPE のある試技が {len(df)} 件しかありません（最低 5 件）")

    # 学習に使えないセット（データの欠け・レップの取り違え）: そのデータを使う条件だけから外す
    df = mark_unusable(df)
    excluded = df[(df["excl_imu"] != "") | (df["excl_video"] != "")]
    if len(excluded):
        print(f"[除外] 学習に使えないデータ: IMU {int((df['excl_imu'] != '').sum())} セット、"
              f"動画 {int((df['excl_video'] != '').sum())} セット（そのデータを使う条件だけから外す）")
        for r in excluded.to_dict("records"):
            for part, name in (("imu", "IMU"), ("video", "動画")):
                if r[f"excl_{part}"]:
                    print(f"  {r['subject_id']}/{r['session_id']} set{int(r['set_no']):02d}"
                          f"（RPE {r['rpe']:g}）{name}: {r[f'excl_{part}']}")

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = build_conditions(df, args.with_context)
    log = Log()

    # 学習データの範囲（全 RPE ／ RPE 5 以上だけ）と評価の範囲（全セット ／ RPE 6 以上）
    train_sets = {"all": None}
    if (df["rpe"] < args.train_min_rpe).any():
        train_sets[f"rpe>={args.train_min_rpe:g}"] = args.train_min_rpe
    else:
        log.add(f"学習データの範囲の比較: RPE {args.train_min_rpe:g} 未満のセットが無いので「全 RPE」だけで学習")
    eval_sets = {"all": None}
    if (df["rpe"] < args.eval_min_rpe).any() and (df["rpe"] >= args.eval_min_rpe).any():
        eval_sets[f"rpe>={args.eval_min_rpe:g}"] = args.eval_min_rpe
    main_eval = list(eval_sets)[-1]          # RPE 6 以上があればそれ、無ければ全セット
    print(f"[範囲] 学習: {', '.join(label_train(t) for t in train_sets)}　"
          f"評価: {', '.join(label_eval(e) for e in eval_sets)}（主結果: {label_eval(main_eval)}）")

    # データの内訳（imu / video = 学習に使えるセット数、*_excluded = データの欠けなどで外したセット数）
    df["use_imu"] = (df[IMU_KEY].notna() if IMU_KEY in df else False) & (df["excl_imu"] == "")
    df["use_video"] = (df[VIDEO_KEY].notna() if VIDEO_KEY in df else False) & (df["excl_video"] == "")
    inv = (df.groupby("subject_id")
           .agg(sessions=("session_id", "nunique"), trials=("rpe", "size"),
                imu=("use_imu", "sum"), video=("use_video", "sum"),
                imu_excluded=("excl_imu", lambda s: int((s != "").sum())),
                video_excluded=("excl_video", lambda s: int((s != "").sum())),
                rpe_min=("rpe", "min"), rpe_max=("rpe", "max"))
           .reset_index())
    print("[DATA]\n" + inv.to_string(index=False))

    pred = pd.DataFrame(scheme_within(df, conds, train_sets, args.min_sessions, log)
                        + scheme_loso(df, conds, train_sets, args.min_subjects, log))
    curves = pd.DataFrame(scheme_curves(df, conds, train_sets, eval_sets, args.kmax, log))

    res = summarize(pred, eval_sets) if not pred.empty else pd.DataFrame()
    if not res.empty:
        res.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
        pred.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    if not curves.empty:
        curves.to_csv(out_dir / "curves.csv", index=False, encoding="utf-8-sig")

    # 主結果（図・保存モデル・表）: 学習 = 全 RPE、評価 = RPE 6 以上（無ければ全セット）
    def main_view(t: pd.DataFrame) -> pd.DataFrame:
        if t.empty:
            return t
        return t[(t["train_set"] == MAIN_TRAIN) & (t["eval_set"] == main_eval)]
    res_main, curves_main = main_view(res), main_view(curves)
    eval_note = f"（評価: {label_eval(main_eval)}）"

    # 図
    figs = {}
    titles = {"A_within": "A. 本人専用（被験者内 Leave-One-Session-Out）",
              "B_loso": "B. 汎用（Leave-One-Subject-Out）"}
    for sch, t in titles.items():
        if not res_main.empty and (f := fig_compare(res_main, sch, out_dir / f"fig_compare_{sch}.png",
                                                    t + eval_note)):
            figs[sch] = f
    if not curves_main.empty:
        if f := fig_curve(curves_main, "D_personal_curve", out_dir / "fig_curve_personal.png",
                          "D. 本人のデータだけで学習（学習曲線）" + eval_note):
            figs["D"] = f
        if f := fig_curve(curves_main, "C_general_plus_k", out_dir / "fig_curve_general_plus_k.png",
                          "C. 汎用モデル ＋ 本人 k セッション" + eval_note):
            figs["C"] = f
    primary = "B_loso" if (not res_main.empty and (res_main["scheme"] == "B_loso").any()) else "A_within"
    best_row = None
    cand = (res_main[(res_main["scheme"] == primary) & (res_main["condition"] != BASELINE_COND)]
            if not res_main.empty else res_main)                 # 比べる基準は最良モデルの候補にしない
    if not cand.empty:
        best_row = cand.sort_values("mae").iloc[0]
        bp = pred[(pred["scheme"] == primary) & (pred["condition"] == best_row["condition"])
                  & (pred["model"] == best_row["model"]) & (pred["train_set"] == MAIN_TRAIN)]
        if eval_sets[main_eval] is not None:
            bp = bp[bp["rpe"] >= eval_sets[main_eval]]
        figs["scatter"] = fig_scatter(bp, out_dir / "fig_scatter_best.png",
                                      f"{titles[primary]}{eval_note}\n{best_row['condition']} ({best_row['model']})")
    # 特徴量重要度: 汎用評価があればそれで、無ければ（IMU は本人だけなど）本人専用の評価で
    for cname in ["video_all", "imu+video"]:
        if cname not in conds or res.empty:
            continue
        done = set(res[res["condition"] == cname]["scheme"])
        sch = "B_loso" if "B_loso" in done else ("A_within" if "A_within" in done else None)
        if sch is None:
            continue
        imp = importance_on_folds(df, conds[cname], sch, args.min_sessions)
        if imp is not None:
            figs[f"imp_{cname}"] = fig_importance(imp, out_dir / f"fig_importance_{cname.replace('+', '_')}.png",
                                                  f"特徴量重要度: {cname}（{sch}）")
            imp.to_csv(out_dir / f"importance_{cname.replace('+', '_')}.csv", encoding="utf-8-sig")

    # 全データで学習し直したモデル（predict_rpe.py 用）
    saved = []
    for cname, cond in conds.items():
        if cond.get("baseline"):          # 比べる基準は予測に使わないので保存しない
            continue
        d = eligible(df, cond)
        for mname in cond["models"]:
            res_fit, use = fit_predict(mname, cond["cols"], d, d.head(1))
            if res_fit is None:
                continue
            bundle = {"model": res_fit[1], "feature_cols": use, "condition": cname, "model_name": mname,
                      "n_train": int(len(d)), "trained_at": datetime.now().isoformat(timespec="seconds")}
            fname = f"{cname.replace('+', '_')}__{mname}.pkl"
            joblib.dump(bundle, out_dir / fname)
            saved.append(fname)
            if best_row is not None and cname == best_row["condition"] and mname == best_row["model"]:
                joblib.dump(bundle, out_dir / "best.pkl")

    # レポート
    lines = [f"# RPE 推定 評価レポート（{datetime.now():%Y-%m-%d %H:%M}）", "",
             f"入力: `{Path(args.input).name}`　試技 {len(df)}　被験者 {df['subject_id'].nunique()} 人", "",
             f"主結果は「学習: {label_train(MAIN_TRAIN)}／評価: {label_eval(main_eval)}」。"
             f"学習データの範囲の比較は後半の表。", "",
             "## データの内訳", "",
             "imu / video は学習に使えるセット数、*_excluded はデータの欠けなどで外したセット数。", "",
             md_table(inv), ""]
    if len(excluded):
        tab = excluded[["subject_id", "session_id", "date", "set_no", "rpe", "excl_imu", "excl_video"]]
        tab = tab.rename(columns={"excl_imu": "IMU を外した理由", "excl_video": "動画を外した理由"})
        lines += ["## 学習から外したセット", "",
                  "データが欠けている・レップを取り違えているセットは、そのデータ（IMU / 動画）を使う条件の"
                  "学習とテストから外した（context の条件には残る）。基準は `quality_checks.py`（品質レポートと同じ）。",
                  "", md_table(tab), ""]
    # 条件の説明（表に出る条件名の意味。MCV = 上昇局面の平均速度）
    desc = pd.DataFrame([{"condition": c, "使うもの": COND_DESC.get(c.removesuffix("+ctx"), "")
                          + ("（＋ %1RM・回数）" if c.endswith("+ctx") else ""),
                          "models": " / ".join(conds[c]["models"])} for c in conds])
    lines += ["## 条件の説明", "",
              f"`{BASELINE_COND}` は比べる基準。これより MAE が小さくなければ、その条件の特徴は RPE を当てるのに"
              "役立っていない。速度は MCV（上昇局面の平均速度）。", "", md_table(desc), ""]
    if not res_main.empty:
        for sch, t in titles.items():
            r = res_main[res_main["scheme"] == sch]
            if r.empty:
                continue
            lines += [f"## {t}{eval_note}", "", md_table(r[["condition", "model", "n", "subjects", "folds",
                                                             "mae", "rmse", "hit05", "hit10"]]), ""]
            if sch in figs:
                lines += [f"![]({figs[sch]})", ""]
    # 学習データの範囲 × 評価の範囲（各条件で、主結果がいちばん良いモデル）
    if not res.empty and (len(train_sets) > 1 or len(eval_sets) > 1):
        lines += ["## 学習データの範囲の比較（RPE 5 未満の軽いセットを学習に入れるか）", "",
                  "セルは MAE / ±1 RPE 以内率（n）。テストに使うセットはどちらの学習でも同じ。", ""]
        for sch, t in titles.items():
            r = res[res["scheme"] == sch]
            if r.empty or res_main.empty:
                continue
            best_m = (res_main[res_main["scheme"] == sch].sort_values("mae")
                      .groupby("condition").head(1)[["condition", "model"]])
            tab = r.merge(best_m, on=["condition", "model"])
            if tab.empty:
                continue
            tab["cell"] = [f"{m:.2f} / {h:.0%}（{int(n)}）" for m, h, n in zip(tab["mae"], tab["hit10"], tab["n"])]
            piv = tab.pivot_table(index=["condition", "model"], columns=["train_set", "eval_set"],
                                  values="cell", aggfunc="first")
            piv = piv.reindex(columns=[(tr, ev) for tr in train_sets for ev in eval_sets]).fillna("—")
            piv.columns = [f"学習 {label_train(tr)} → 評価 {label_eval(ev)}" for tr, ev in piv.columns]
            lines += [f"### {t}", "", md_table(piv.reset_index()), ""]
    curve_note = ("曲線は各条件を線形回帰（無ければリッジ）で描いている（凡例に表示）。少ないセッションでも学習できる"
                  "モデルにそろえるためで、上の表のいちばん良いモデルとは違うことがある。")
    for key, t in [("D", "## D. 本人のデータだけで学習（何セッションで実用水準か）"),
                   ("C", "## C. 汎用モデル ＋ 本人 k セッション（パーソナライズの効果）")]:
        if key in figs:
            lines += [t, "", curve_note, "", f"![]({figs[key]})", ""]
    if "scatter" in figs:
        lines += ["## 最良モデルの予測", "", f"![]({figs['scatter']})", ""]
    for k in [k for k in figs if k.startswith("imp_")]:
        lines += [f"## 特徴量重要度（{k[4:]}）", "", f"![]({figs[k]})", ""]
    lines += ["## 注意", "",
              "- n が小さい結果は参考値。MAE の差が小さいときは結論を出さない。",
              "- IMU の条件は IMU が使える試技（本人）だけ、動画の条件は動画が使える試技だけで評価しているので、"
              "条件ごとに n が違う。",
              "- 同じセッションのセットが学習とテストにまたがることはない。",
              f"- RPE {args.eval_min_rpe:g} 未満は申告がぶれやすい（限界から遠いほど RIR の見積もりがずれる）ので、"
              "主結果の評価から外している。",
              f"- 「RPE {args.train_min_rpe:g} 以上だけで学習」したモデルは {args.train_min_rpe:g} 未満を予測しない"
              f"（予測を {args.train_min_rpe:g}〜10 に収める）。", ""]
    if log:
        lines += ["## スキップした評価", ""] + [f"- {m}" for m in log] + [""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[OK] {out_dir}")
    if not res_main.empty:
        print(f"主結果（学習: {label_train(MAIN_TRAIN)}／評価: {label_eval(main_eval)}）")
        print(res_main[["scheme", "condition", "model", "n", "mae", "hit10"]].to_string(index=False))
    if len(train_sets) > 1:
        print("  学習データの範囲の比較は report.md の後半の表")
    for m in log:
        print(f"  [SKIP] {m}")
    print(f"  レポート: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
