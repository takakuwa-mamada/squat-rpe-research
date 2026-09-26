# -*- coding: utf-8 -*-
"""
=============================================================================
 学習済みRPE推定モデルで新規試技のRPEを予測するスクリプト

 入力:
   - 学習済みモデル: models/<タイムスタンプ>/<model_name>.pkl
   - 予測対象: features_all.csv の一部 OR 単一試技のJSON

 出力:
   - 標準出力に予測RPE
   - --out 指定で予測結果CSV

 使い方:
   # 学習済みモデルで features_all.csv 全行を予測
   python predict_rpe.py --model models/20260602_080000/random_forest.pkl

   # 単一試技のJSONを予測
   python predict_rpe.py --model models/.../rf.pkl \
                         --json data/S001/SES001/features/set01_features.json

   # 最新のモデルを自動選択
   python predict_rpe.py --latest

依存ライブラリ:
   pip install numpy pandas scikit-learn joblib
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


SCRIPT_DIR = Path(__file__).resolve().parent
# プロジェクトルート: scripts/ 配下にあれば親、直下にあれば自分
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "features_all.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def find_latest_model(models_dir: Path, prefer: str = "random_forest") -> Path:
    """最新の学習結果フォルダから指定モデルを選ぶ"""
    if not models_dir.exists():
        sys.exit(f"[ERROR] models dir not found: {models_dir}")
    timestamps = sorted([p for p in models_dir.iterdir() if p.is_dir()])
    if not timestamps:
        sys.exit(f"[ERROR] no trained models in {models_dir}")
    latest = timestamps[-1]
    # 優先順位
    for name in ("best", prefer, "rf", "gb", "gradient_boosting", "linear"):
        cand = latest / f"{name}.pkl"
        if cand.exists():
            return cand
    # フォールバック: 何でも1つ
    pkls = list(latest.glob("*.pkl"))
    if not pkls:
        sys.exit(f"[ERROR] no .pkl files in {latest}")
    return pkls[0]


def load_model(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        sys.exit(f"[ERROR] invalid model file: {model_path}")
    return bundle


def predict_dataframe(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] required features missing in input: {missing}")
    # 学習時のモデルは欠損を補完できる（1レップのセットの変化量などは NaN のまま渡す）
    valid = df[df[feature_cols].notna().any(axis=1)]
    if len(valid) == 0:
        sys.exit("[ERROR] all rows have NaN in required features")

    preds = bundle["model"].predict(valid[feature_cols].to_numpy(dtype=float))
    result = valid.copy()
    result["predicted_rpe"] = preds
    if "rpe" in df.columns:
        result["actual_rpe"] = df.loc[valid.index, "rpe"]
        result["error"] = result["predicted_rpe"] - result["actual_rpe"]
    # メタ情報を付与
    for c in ("subject_id", "session_id", "set_no", "weight_kg"):
        if c in df.columns and c not in result.columns:
            result[c] = df.loc[valid.index, c]
    return result


def predict_single(bundle: dict, feat_json: dict) -> float:
    feature_cols = bundle["feature_cols"]
    row = []
    for c in feature_cols:
        v = feat_json.get(c)
        row.append(float("nan") if v is None else float(v))   # 欠損はモデル側で補完する
    pred = bundle["model"].predict(np.array([row]))[0]
    return float(pred)


def main():
    ap = argparse.ArgumentParser(description="Predict RPE using a trained model")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--latest", action="store_true",
                    help="use the latest trained model")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                    help="features CSV (default: features_all.csv)")
    ap.add_argument("--json", type=str, default=None,
                    help="single trial features JSON")
    ap.add_argument("--out", type=str, default=None,
                    help="output CSV for batch predictions")
    args = ap.parse_args()

    # モデル選択
    if args.latest or args.model is None:
        model_path = find_latest_model(DEFAULT_MODELS_DIR)
    else:
        model_path = Path(args.model).resolve()
    if not model_path.exists():
        sys.exit(f"[ERROR] model not found: {model_path}")
    print(f"[MODEL] {model_path}")
    bundle = load_model(model_path)
    print(f"[MODEL] name={bundle.get('model_name')}, "
          f"trained_at={bundle.get('trained_at')}, "
          f"n_samples_trained={bundle.get('n_samples')}")
    print(f"[MODEL] features ({len(bundle['feature_cols'])}): "
          f"{bundle['feature_cols'][:5]}{'...' if len(bundle['feature_cols'])>5 else ''}")

    # 単一JSON予測モード
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            feat = json.load(f)
        pred = predict_single(bundle, feat)
        print(f"\n[PREDICTION] {args.json}")
        print(f"  predicted RPE = {pred:.2f}")
        if feat.get("rpe") is not None:
            print(f"  actual    RPE = {feat['rpe']}")
            print(f"  error         = {pred - float(feat['rpe']):+.2f}")
        return

    # バッチ予測モード
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"[ERROR] input not found: {in_path}")
    df = pd.read_csv(in_path)
    result = predict_dataframe(bundle, df)

    out_cols = ["subject_id", "session_id", "set_no", "weight_kg",
                "actual_rpe", "predicted_rpe", "error"]
    out_cols = [c for c in out_cols if c in result.columns]
    print(f"\n[RESULT] {len(result)} predictions")
    print(result[out_cols].to_string(index=False, max_rows=20))

    if args.out:
        out_path = Path(args.out).resolve()
        result[out_cols].to_csv(out_path, index=False)
        print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
