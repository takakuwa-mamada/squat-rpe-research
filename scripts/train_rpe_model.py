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

 比較条件（特徴量セット）:
   context         %1RM（無ければ重量）＋ 回数           … RPE 早見表に相当
   imu_velocity    IMU の1レップ目 MCV だけ（線形回帰）   … VBT / Stance 相当
   imu_all         IMU のレップ単位の特徴すべて
   video_velocity  動画の1レップ目 MCV だけ（線形回帰）   … 動画だけの VBT
   video_all       動画（骨格）の特徴すべて               … 最終目標（動画だけ）
   imu+video       IMU ＋ 動画
   --with-context で、各条件に %1RM・回数を足した版（*+ctx）も出す
   IMU の条件は IMU のある試技（本人）だけで評価する

 出力: models/<日時>/
   report.md          結果のまとめ（表・図へのリンク・データの内訳）
   results.csv        方式 × 条件 × モデル の MAE / RMSE / ±0.5・±1.0 以内率
   curves.csv         方式 C・D の k ごとの結果
   predictions.csv    全予測（どのフォールドで予測したか付き）
   fig_*.png          比較の棒グラフ、学習曲線、散布図、特徴量重要度
   <条件>__<モデル>.pkl, best.pkl   全データで学習し直したモデル（predict_rpe.py 用）

 使い方:
   python scripts\\train_rpe_model.py
   python scripts\\train_rpe_model.py --with-context
   python scripts\\train_rpe_model.py --input outputs\\features_all.csv --kmax 6

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
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.base import clone

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]

SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "features_all.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "models"

RPE_MIN, RPE_MAX = 5.0, 10.0

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


def build_conditions(df: pd.DataFrame, with_context: bool) -> dict:
    """条件名 → {cols, key, models}"""
    ctx_w = "pct_1rm" if ("pct_1rm" in df and df["pct_1rm"].notna().mean() >= 0.8) else "weight_kg"
    context = [ctx_w, "reps_completed"]
    conds = {
        "context":        {"cols": context, "key": context, "models": ["linear", "hgb"]},
        "imu_velocity":   {"cols": [IMU_KEY], "key": [IMU_KEY], "models": ["linear"]},
        "imu_all":        {"cols": IMU_FEATURES, "key": [IMU_KEY], "models": ["ridge", "rf", "hgb"]},
        "video_velocity": {"cols": [VIDEO_KEY], "key": [VIDEO_KEY], "models": ["linear"]},
        "video_all":      {"cols": VIDEO_FEATURES, "key": [VIDEO_KEY], "models": ["ridge", "rf", "hgb"]},
        "imu+video":      {"cols": IMU_FEATURES + VIDEO_FEATURES, "key": [IMU_KEY, VIDEO_KEY],
                           "models": ["ridge", "rf", "hgb"]},
    }
    if with_context:
        for name in ["imu_velocity", "imu_all", "video_velocity", "video_all", "imu+video"]:
            c = conds[name]
            conds[f"{name}+ctx"] = {"cols": c["cols"] + context, "key": c["key"] + context,
                                   "models": ["ridge", "hgb"]}
    return conds


def make_model(name: str):
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


def fit_predict(model_name, cols, train, test):
    """列が学習データで全部欠損なら落としてから学習する"""
    use = [c for c in cols if c in train and train[c].notna().any()]
    if not use or len(train) < 3:
        return None, use
    m = make_model(model_name)
    m.fit(train[use].to_numpy(float), train["rpe"].to_numpy(float))
    pred = np.clip(m.predict(test[use].to_numpy(float)), RPE_MIN, RPE_MAX)
    return (pred, m), use


def eligible(df, cond) -> pd.DataFrame:
    return df.dropna(subset=[c for c in cond["key"] if c in df] + ["rpe"])


def sessions_in_order(d: pd.DataFrame) -> list:
    s = d.groupby("session_id")["date"].first().reset_index().sort_values(["date", "session_id"])
    return s["session_id"].tolist()


# ---------------------------------------------------------------------------
# 評価方式
# ---------------------------------------------------------------------------
def scheme_within(df, conds, min_sessions, log):
    """A. 本人専用: 被験者ごとに Leave-One-Session-Out"""
    rows = []
    for cname, cond in conds.items():
        d_all = eligible(df, cond)
        for sid, d in d_all.groupby("subject_id"):
            sess = sessions_in_order(d)
            if len(sess) < min_sessions:
                log.add(f"A 本人専用 / {cname} / {sid}: セッション {len(sess)} < {min_sessions} のためスキップ")
                continue
            for mname in cond["models"]:
                for s in sess:
                    tr, te = d[d["session_id"] != s], d[d["session_id"] == s]
                    res, _ = fit_predict(mname, cond["cols"], tr, te)
                    if res is None:
                        continue
                    rows += [dict(scheme="A_within", condition=cname, model=mname, subject_id=sid,
                                  session_id=s, set_no=r.set_no, fold=f"{sid}/{s}", rpe=r.rpe, pred=p)
                             for r, p in zip(te.itertuples(), res[0])]
    return rows


def scheme_loso(df, conds, min_subjects, log):
    """B. 汎用: Leave-One-Subject-Out"""
    rows = []
    for cname, cond in conds.items():
        d = eligible(df, cond)
        subs = sorted(d["subject_id"].unique())
        if len(subs) < min_subjects:
            log.add(f"B 汎用 / {cname}: 被験者 {len(subs)} 人 < {min_subjects} 人のためスキップ")
            continue
        for mname in cond["models"]:
            for sid in subs:
                tr, te = d[d["subject_id"] != sid], d[d["subject_id"] == sid]
                res, _ = fit_predict(mname, cond["cols"], tr, te)
                if res is None:
                    continue
                rows += [dict(scheme="B_loso", condition=cname, model=mname, subject_id=sid,
                              session_id=r.session_id, set_no=r.set_no, fold=sid, rpe=r.rpe, pred=p)
                         for r, p in zip(te.itertuples(), res[0])]
    return rows


def scheme_curves(df, conds, kmax, log):
    """C. 汎用＋本人 k セッション / D. 本人だけの学習曲線（各条件の代表モデルで）"""
    rows = []
    for cname, cond in conds.items():
        mname = "linear" if cond["models"] == ["linear"] else "hgb"
        d = eligible(df, cond)
        subs = sorted(d["subject_id"].unique())
        for sid in subs:
            me, others = d[d["subject_id"] == sid], d[d["subject_id"] != sid]
            sess = sessions_in_order(me)
            for k in range(0, min(len(sess) - 1, kmax) + 1):
                train_me = me[me["session_id"].isin(sess[:k])]
                te = me[me["session_id"].isin(sess[k:])]
                # C: 他の人 ＋ 本人 k セッション（他の人が 2 人以上いるとき）
                if others["subject_id"].nunique() >= 2:
                    res, _ = fit_predict(mname, cond["cols"], pd.concat([others, train_me]), te)
                    if res is not None:
                        rows.append(dict(scheme="C_general_plus_k", condition=cname, model=mname,
                                         subject_id=sid, k=k, n_train_me=len(train_me),
                                         **metrics(te["rpe"], res[0])))
                # D: 本人だけ（k ≥ 1）
                if k >= 1 and len(train_me) >= 3:
                    res, _ = fit_predict(mname, cond["cols"], train_me, te)
                    if res is not None:
                        rows.append(dict(scheme="D_personal_curve", condition=cname, model=mname,
                                         subject_id=sid, k=k, n_train_me=len(train_me),
                                         **metrics(te["rpe"], res[0])))
    if not rows:
        log.add("C/D 学習曲線: 2セッション以上ある被験者がいないためスキップ")
    return rows


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
    ax.barh(best["condition"] + " (" + best["model"] + ")", best["mae"], color="#4C72B0")
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cond, g in agg.groupby("condition"):
        axes[0].plot(g["k"], g["mae"], marker="o", label=cond)
        axes[1].plot(g["k"], g["hit10"], marker="o", label=cond)
    axes[0].set_ylabel("MAE [RPE]")
    axes[1].set_ylabel("±1 RPE 以内の割合")
    axes[1].axhline(0.8, color="gray", lw=0.8, ls="--")
    for a in axes:
        a.set_xlabel("学習に使った本人のセッション数 k")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def fig_scatter(pred: pd.DataFrame, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    jitter = np.random.default_rng(0).normal(0, 0.05, len(pred))
    for sid, g in pred.groupby("subject_id"):
        ax.scatter(g["rpe"] + jitter[:len(g)], g["pred"], s=18, alpha=0.7, label=sid)
    ax.plot([RPE_MIN, RPE_MAX], [RPE_MIN, RPE_MAX], "k--", lw=0.8)
    ax.fill_between([RPE_MIN, RPE_MAX], [RPE_MIN - 1, RPE_MAX - 1], [RPE_MIN + 1, RPE_MAX + 1],
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
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["rpe"]).copy()
    df["rpe"] = df["rpe"].astype(float)
    if "date" not in df:
        df["date"] = ""
    df["date"] = df["date"].fillna("").astype(str)
    if len(df) < 5:
        sys.exit(f"[ERROR] RPE のある試技が {len(df)} 件しかありません（最低 5 件）")

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = build_conditions(df, args.with_context)
    log = Log()

    # データの内訳
    inv = (df.groupby("subject_id")
           .agg(sessions=("session_id", "nunique"), trials=("rpe", "size"),
                imu=(IMU_KEY, lambda s: int(s.notna().sum())) if IMU_KEY in df else ("rpe", lambda s: 0),
                video=(VIDEO_KEY, lambda s: int(s.notna().sum())) if VIDEO_KEY in df else ("rpe", lambda s: 0),
                rpe_min=("rpe", "min"), rpe_max=("rpe", "max"))
           .reset_index())
    print("[DATA]\n" + inv.to_string(index=False))

    pred = pd.DataFrame(scheme_within(df, conds, args.min_sessions, log)
                        + scheme_loso(df, conds, args.min_subjects, log))
    curves = pd.DataFrame(scheme_curves(df, conds, args.kmax, log))

    res = pd.DataFrame()
    if not pred.empty:
        res = (pred.groupby(["scheme", "condition", "model"])
               .apply(lambda g: pd.Series({**metrics(g["rpe"], g["pred"]),
                                           "subjects": g["subject_id"].nunique(),
                                           "folds": g["fold"].nunique()}))
               .reset_index().sort_values(["scheme", "mae"]))
        res.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
        pred.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    if not curves.empty:
        curves.to_csv(out_dir / "curves.csv", index=False, encoding="utf-8-sig")

    # 図
    figs = {}
    titles = {"A_within": "A. 本人専用（被験者内 Leave-One-Session-Out）",
              "B_loso": "B. 汎用（Leave-One-Subject-Out）"}
    for sch, t in titles.items():
        if not res.empty and (f := fig_compare(res, sch, out_dir / f"fig_compare_{sch}.png", t)):
            figs[sch] = f
    if not curves.empty:
        if f := fig_curve(curves, "D_personal_curve", out_dir / "fig_curve_personal.png",
                          "D. 本人のデータだけで学習（学習曲線）"):
            figs["D"] = f
        if f := fig_curve(curves, "C_general_plus_k", out_dir / "fig_curve_general_plus_k.png",
                          "C. 汎用モデル ＋ 本人 k セッション"):
            figs["C"] = f
    primary = "B_loso" if (not res.empty and (res["scheme"] == "B_loso").any()) else "A_within"
    best_row = None
    if not res.empty and (res["scheme"] == primary).any():
        best_row = res[res["scheme"] == primary].sort_values("mae").iloc[0]
        bp = pred[(pred["scheme"] == primary) & (pred["condition"] == best_row["condition"])
                  & (pred["model"] == best_row["model"])]
        figs["scatter"] = fig_scatter(bp, out_dir / "fig_scatter_best.png",
                                      f"{titles[primary]}\n{best_row['condition']} ({best_row['model']})")
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
             "## データの内訳", "", md_table(inv), ""]
    if not res.empty:
        for sch, t in titles.items():
            r = res[res["scheme"] == sch]
            if r.empty:
                continue
            lines += [f"## {t}", "", md_table(r[["condition", "model", "n", "subjects", "folds",
                                                  "mae", "rmse", "hit05", "hit10"]]), ""]
            if sch in figs:
                lines += [f"![]({figs[sch]})", ""]
    for key, t in [("D", "## D. 本人のデータだけで学習（何セッションで実用水準か）"),
                   ("C", "## C. 汎用モデル ＋ 本人 k セッション（パーソナライズの効果）")]:
        if key in figs:
            lines += [t, "", f"![]({figs[key]})", ""]
    if "scatter" in figs:
        lines += ["## 最良モデルの予測", "", f"![]({figs['scatter']})", ""]
    for k in [k for k in figs if k.startswith("imp_")]:
        lines += [f"## 特徴量重要度（{k[4:]}）", "", f"![]({figs[k]})", ""]
    lines += ["## 注意", "",
              "- n が小さい結果は参考値。MAE の差が小さいときは結論を出さない。",
              "- IMU の条件は IMU のある試技（本人）だけで評価しているので、動画の条件と n が違う。",
              "- 同じセッションのセットが学習とテストにまたがることはない。", ""]
    if log:
        lines += ["## スキップした評価", ""] + [f"- {m}" for m in log] + [""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[OK] {out_dir}")
    if not res.empty:
        print(res[["scheme", "condition", "model", "n", "mae", "hit10"]].to_string(index=False))
    for m in log:
        print(f"  [SKIP] {m}")
    print(f"  レポート: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
