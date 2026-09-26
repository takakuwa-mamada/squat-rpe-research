---
description: ジムで計測した生データ（data/_sessions）を解析用に変換し、骨格推定・特徴量・集計まで回して品質を報告する
argument-hint: <被験者ID> <セッションID>   例: S001 SES005
---

計測後の処理を行う。引数: $ARGUMENTS （未指定なら被験者ID・セッションIDをユーザーに聞く）

1. `data/_sessions/` のうち、どの `data/*/*/meta.json` の `source_sessions` にもまだ入っていないセッションを一覧にし、
   各セッションの `session_info.json` の mean_hz・セット数、`markers.csv` の RPE 欠損、直下の動画の本数を表で報告する。
   セット数と動画の本数が合わない、mean_hz が 90 未満、RPE が空欄、のどれかがあれば先に進まずユーザーに確認する。
2. `python scripts\split_session.py --subject <S> --session <SES> --src <対象1> --src <対象2> ...` を実行する（対象は手順1のもの）。
   既存のセッションIDは上書きになるので使わない。
3. `meta.json` の `weight_kg` と `reps_completed` が null のセットをユーザーに聞いて埋める（推測で埋めない）。
4. 次を順に実行する:
   - `python scripts\pose_extract_yolo.py --subject <S>`
   - `python scripts\extract_features.py --subject <S> --force`
   - `python scripts\aggregate_features.py`
5. 品質チェックを1つの表で報告する:
   - IMU の mean_hz（90〜100）、`imu_peak_velocity_up_mps`（0.3〜1.5 m/s）
   - `imu_n_reps_detected` と `reps_completed` の一致
   - `pose_key_visibility_mean`（0.7 以上）、`pose_nan_ratio`
   - 累計試技数と RPE の分布（偏っていたら次回狙う負荷帯を提案）
6. 問題がなければ変更をコミットする（`*.mp4` と `weights/*.pt` は .gitignore 済み。`firmware/*.ino` が含まれていないか確認）。
