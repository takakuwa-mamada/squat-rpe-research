---
description: 計測データ（本人の data/_sessions、または協力者の Google Drive）を取り込み、骨格推定・特徴量・集計・品質レポートまで回す
argument-hint: <被験者ID> <セッションID> ／ remote [共有フォルダのパス]   例: S001 SES005 ／ remote "G:\マイドライブ\スクワット研究"
---

計測後の処理を行う。引数: $ARGUMENTS

## 本人（IMU＋動画）: 引数が `<被験者ID> <セッションID>` のとき（未指定なら聞く）

1. `data/_sessions/` のうち、どの `data/*/*/meta.json` の `source_sessions` にもまだ入っていないセッションを一覧にし、
   各セッションの `session_info.json` の mean_hz・セット数、`markers.csv` の RPE 欠損、直下の動画の本数を表で報告する。
   セット数と動画の本数が合わない、mean_hz が 90 未満、RPE が空欄、のどれかがあれば先に進まずユーザーに確認する。
2. 重量・回数をユーザーに聞く（推測で埋めない）。動画のファイル名が `130kg_3rep.mp4` の形なら自動で入るので聞かなくてよい。
3. `python scripts\split_session.py --subject <S> --session <SES> --src <対象1> --src <対象2> ... --weights <...> --reps <...>`
   既存のセッションIDを使うと作り直しになる（手入力した meta の値は残る）。新しい計測には新しい ID を使う。
4. `python scripts\run_pipeline.py --subject <S>`

## 協力者（動画のみ）: 引数が `remote [パス]` のとき

1. `python scripts\ingest_remote.py --dry-run [--src <パス>]` で取り込み内容を表示し、`[要確認]` があれば
   どの人のどのフォルダで何が足りないかをまとめる（本人に聞くための文面も用意する）。
2. 問題のないものを `python scripts\ingest_remote.py [--src <パス>]` で取り込む。
3. `python scripts\run_pipeline.py`

## 共通: 報告

- `run_pipeline.py` の品質レポート（要確認のセットと理由）を1つの表にまとめる。
- 被験者ごとの試技数と RPE の分布。偏っていたら次回狙う負荷帯を提案する（本人なら `docs/スクワット計測メニュー.xlsx` の A/B/C）。
- `data/` と `outputs/` は git に入れない（.gitignore 済み）。コードを直した場合だけコミットし、`firmware/*.ino` が含まれていないか確認する。
