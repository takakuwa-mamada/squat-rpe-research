# -*- coding: utf-8 -*-
"""
=============================================================================
 RPE推定モデルの学習スクリプト

 入力: features_all.csv (aggregate_features.py の出力)
 出力:
   models/<タイムスタンプ>/
     ├ baseline_linear.pkl       VBT準拠線形回帰 (peak_velocityのみ)
     ├ random_forest.pkl         Random Forest (全特徴量)
     ├ gradient_boosting.pkl     Gradient Boosting (全特徴量)
     ├ feature_columns.json      使った特徴量の一覧
     ├ metrics.json              評価結果 (MAE, RMSE, ±1RPE hit rate)
     ├ predictions.csv           各サンプルの予測値
     ├ feature_importance.png    特徴量重要度プロット
     └ scatter_plot.png          予測 vs 実測の散布図

 評価戦略:
   - 被験者が2名以上: Leave-One-Subject-Out CV (推奨)
   - 被験者が1名:     5-fold CV (応急対応)
   - サンプル数<5:    全データ訓練, 評価はtrain上 (スケルトン確認用)

 使い方:
   python train_rpe_model.py                          # features_all.csv を使用
   python train_rpe_model.py --input my_features.csv
   python train_rpe_model.py --output-dir models/
   python train_rpe_model.py --models rf,gb           # 一部だけ実行

 依存ライブラリ:
   pip install numpy pandas scikit-learn matplotlib joblib
=============================================================================
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI不要
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore", category=UserWarning)


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "features_all.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "models"

# 学習に使わないメタ列
META_COLUMNS = {
    "subject_id", "session_id", "date", "set_no", "exercise",
    "rpe", "set_notes", "_source",
    "reps_planned", "reps_completed", "rest_before_sec",
}
# 学習に使わないPose補助列（ベースラインから除外したい場合）
POSE_AUX = {"pose_available"}


# ===========================================================================
# データ読み込みと整形
# ===========================================================================
def load_and_prepare(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "rpe" not in df.columns:
        sys.exit(f"[ERROR] 'rpe' 列がありません: {input_path}")
    if df["rpe"].isna().all():
        sys.exit("[ERROR] RPE値が全て欠損しています")

    n_before = len(df)
    df = df.dropna(subset=["rpe"]).copy()
    df["rpe"] = df["rpe"].astype(float)
    print(f"[DATA] loaded {n_before} rows, valid RPE: {len(df)} rows")

    return df


def select_features(df: pd.DataFrame, baseline_only: bool = False) -> List[str]:
    """
    数値型の特徴量列を自動選択。メタ列・目的変数・全NaN列を除外。
    baseline_only=True: imu_first_rep_peak_v_mps のみ（VBT準拠）
    """
    if baseline_only:
        if "imu_first_rep_peak_v_mps" in df.columns:
            return ["imu_first_rep_peak_v_mps"]
        # フォールバック
        return ["imu_peak_velocity_up_mps"]

    candidates = []
    for c in df.columns:
        if c in META_COLUMNS or c in POSE_AUX:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        # 全NaN or 全て同じ値の列は除外
        if df[c].isna().all():
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        candidates.append(c)
    return candidates


def make_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NaNを含む行は除外する。
    戻り値: X, y, groups(subject_id)
    """
    sub_df = df[feature_cols + ["rpe", "subject_id"]].dropna(subset=feature_cols + ["rpe"])
    X = sub_df[feature_cols].to_numpy()
    y = sub_df["rpe"].to_numpy()
    groups = sub_df["subject_id"].fillna("UNKNOWN").to_numpy()
    return X, y, groups


# ===========================================================================
# モデルファクトリ
# ===========================================================================
def make_model(name: str):
    if name == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ])
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=8,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )
    if name == "gb":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42,
        )
    raise ValueError(f"unknown model: {name}")


# ===========================================================================
# 評価指標
# ===========================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    abs_err = np.abs(y_true - y_pred)
    hit_05 = float(np.mean(abs_err <= 0.5) * 100.0)
    hit_10 = float(np.mean(abs_err <= 1.0) * 100.0)
    return {
        "n":               int(len(y_true)),
        "mae":             round(mae, 4),
        "rmse":            round(rmse, 4),
        "hit_within_0.5":  round(hit_05, 2),
        "hit_within_1.0":  round(hit_10, 2),
    }


# ===========================================================================
# CV戦略の選択
# ===========================================================================
def choose_cv_strategy(groups: np.ndarray, n_samples: int) -> Tuple[str, object]:
    n_subjects = len(np.unique(groups))
    if n_samples < 5:
        return "train_only", None
    if n_subjects >= 2:
        return "loso", LeaveOneGroupOut()
    # 被験者1名のフォールバック
    n_folds = min(5, n_samples)
    return "kfold", KFold(n_splits=n_folds, shuffle=True, random_state=42)


def cv_predict(model, X, y, groups, cv) -> Tuple[np.ndarray, np.ndarray]:
    """CV予測値を返す（各サンプルがテストとなった時の予測）"""
    y_pred = np.full_like(y, np.nan, dtype=float)
    splits = cv.split(X, y, groups=groups) if isinstance(cv, LeaveOneGroupOut) else cv.split(X, y)
    for tr, te in splits:
        if len(tr) == 0 or len(te) == 0:
            continue
        m = make_model_clone(model)
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    return y, y_pred


def make_model_clone(model):
    """sklearnのmodelを新たに作り直す（fit済みを使い回さないため）"""
    # Pipelineにも対応
    from sklearn.base import clone
    return clone(model)


# ===========================================================================
# 学習＆評価
# ===========================================================================
def train_and_eval(name: str, model, X, y, groups, strategy: str, cv):
    print(f"\n[TRAIN] {name}  (samples={len(y)}, features={X.shape[1]}, strategy={strategy})")

    metrics = {}
    cv_pred = None

    if strategy == "train_only":
        model.fit(X, y)
        y_pred_train = model.predict(X)
        metrics["train"] = compute_metrics(y, y_pred_train)
        print(f"  TRAIN: {metrics['train']}")
    else:
        _, y_pred = cv_predict(model, X, y, groups, cv)
        cv_pred = y_pred
        valid = ~np.isnan(y_pred)
        metrics["cv"] = compute_metrics(y[valid], y_pred[valid])
        print(f"  CV:    {metrics['cv']}")
        # 全データで再学習（最終モデル）
        model.fit(X, y)

    return model, metrics, cv_pred


# ===========================================================================
# 可視化
# ===========================================================================
def plot_scatter(y_true, y_pred, name, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    valid = ~np.isnan(y_pred)
    ax.scatter(y_true[valid], y_pred[valid], alpha=0.6)
    lo = min(np.min(y_true[valid]), np.min(y_pred[valid])) - 0.5
    hi = max(np.max(y_true[valid]), np.max(y_pred[valid])) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
    ax.plot([lo, hi], [lo + 1, hi + 1], "r:", lw=0.5, alpha=0.6)
    ax.plot([lo, hi], [lo - 1, hi - 1], "r:", lw=0.5, alpha=0.6, label="+/- 1 RPE")
    ax.set_xlabel("actual RPE")
    ax.set_ylabel("predicted RPE")
    ax.set_title(f"Predicted vs Actual RPE - {name}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_cols, out_path, name="model"):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_)
    if importances is None:
        return False

    order = np.argsort(importances)[::-1]
    feats = [feature_cols[i] for i in order]
    vals  = importances[order]

    n_show = min(20, len(feats))
    fig, ax = plt.subplots(figsize=(9, max(4, n_show * 0.3)))
    ax.barh(range(n_show)[::-1], vals[:n_show])
    ax.set_yticks(range(n_show)[::-1])
    ax.set_yticklabels(feats[:n_show], fontsize=9)
    ax.set_xlabel("importance")
    ax.set_title(f"Feature importance - {name}")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ===========================================================================
# メイン
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Train RPE estimation models")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--models", type=str, default="linear,rf,gb",
                    help="comma-separated: linear,rf,gb")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"[ERROR] input not found: {in_path}\n"
                 f"        run aggregate_features.py first")

    df = load_and_prepare(in_path)

    # 出力ディレクトリ
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {out_dir}")

    # 全モデル共通: 特徴量と CV 戦略
    full_features = select_features(df, baseline_only=False)
    baseline_features = select_features(df, baseline_only=True)
    print(f"[FEATURES] baseline: {baseline_features}")
    print(f"[FEATURES] full:     {len(full_features)} columns")

    # 学習・評価
    results = {}
    requested = [m.strip() for m in args.models.split(",")]

    for name in requested:
        if name == "linear":
            feats = baseline_features
        else:
            feats = full_features

        X, y, groups = make_xy(df, feats)
        if len(y) == 0:
            print(f"[SKIP] {name}: no valid rows after NaN drop")
            continue

        strategy, cv = choose_cv_strategy(groups, len(y))
        model = make_model(name)
        model, metrics, cv_pred = train_and_eval(name, model, X, y, groups, strategy, cv)

        # モデル保存
        model_path = out_dir / f"{name}.pkl"
        joblib.dump({
            "model":         model,
            "feature_cols":  feats,
            "model_name":    name,
            "trained_at":    ts,
            "n_samples":     int(len(y)),
        }, model_path)

        # 可視化
        if cv_pred is not None:
            scatter_path = out_dir / f"scatter_{name}.png"
            plot_scatter(y, cv_pred, name, scatter_path)
        importance_path = out_dir / f"importance_{name}.png"
        plot_feature_importance(model, feats, importance_path, name=name)

        results[name] = {
            "features":  feats,
            "strategy":  strategy,
            "metrics":   metrics,
            "model_pkl": str(model_path.relative_to(PROJECT_ROOT)),
        }

    # 結果まとめ
    summary = {
        "input":            str(in_path.relative_to(PROJECT_ROOT)) if in_path.is_relative_to(PROJECT_ROOT) else str(in_path),
        "n_total_samples":  int(len(df)),
        "n_subjects":       int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "trained_at":       ts,
        "results":          results,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline_features": baseline_features,
            "full_features":     full_features,
        }, f, indent=2, ensure_ascii=False)

    # 予測値も保存
    pred_rows = []
    for name in requested:
        if name not in results:
            continue
        feats = results[name]["features"]
        X, y, groups = make_xy(df, feats)
        if results[name]["strategy"] == "train_only":
            continue
        strategy, cv = choose_cv_strategy(groups, len(y))
        model = make_model(name)
        _, y_pred = cv_predict(model, X, y, groups, cv)
        for i in range(len(y)):
            pred_rows.append({
                "model":  name,
                "actual_rpe": float(y[i]),
                "predicted_rpe": float(y_pred[i]) if not np.isnan(y_pred[i]) else None,
                "subject_id": groups[i],
            })
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)

    # サマリ表示
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f" Input:    {in_path.name}")
    print(f" Samples:  {len(df)}")
    print(f" Subjects: {summary['n_subjects']}")
    print(f" Output:   {out_dir}")
    print()
    for name, r in results.items():
        m = r["metrics"].get("cv") or r["metrics"].get("train") or {}
        print(f"  {name:10s} ({r['strategy']:10s}): "
              f"MAE={m.get('mae')}, RMSE={m.get('rmse')}, "
              f"hit_1.0={m.get('hit_within_1.0')}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
