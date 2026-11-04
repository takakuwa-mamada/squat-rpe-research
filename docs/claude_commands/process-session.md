---
description: 計測データ（本人の data/_sessions、または協力者から LINE で受け取って data/_remote に保存した動画）を取り込み、骨格推定・特徴量・集計・品質レポートまで回す
argument-hint: （なし）／ remote
---

計測後の処理を行う。引数: $ARGUMENTS

## 本人（IMU＋動画）: 引数なしのとき

1. `python scripts\split_session.py --dry-run` で、まだ取り込んでいない受信フォルダと、日ごとのまとめ方・振られる SES 番号を確認する。
   各フォルダの `session_info.json` の mean_hz、`markers.csv` の RPE・重量・回数の欠損、直下の動画の本数も表で報告する。
   セット数と動画の本数が合わない、mean_hz が 90 未満、RPE が空欄、[待ち]（動画なし）があればユーザーに確認する。
2. 重量・回数が markers.csv にも動画のファイル名にも無いセットは、ユーザーに聞く（推測で埋めない）。
3. `python scripts\run_pipeline.py`（最初に split_session.py が自動モードで取り込み、SES 番号も自動）。
   重量・回数を後から入れるときは、取り込まれた `data/S001/SESnnn/meta.json` を直すか
   `python scripts\split_session.py --session SESnnn --src … --weights … --reps …` で作り直す（手入力の値は残る）。

## 協力者（動画のみ）: 引数が `remote` のとき

協力者の動画は LINE で受け取り、ユーザーが `data\_remote\<名前>\<撮った日>\01_100kg 5回 RPE8.mp4` の名前で保存している
（先頭の番号 = 撮った順、後ろ = 協力者が動画の次に送ってきた1行）。LINE の動画の中の日時は送った時刻なので、順番には番号だけを使う。

1. `python scripts\ingest_remote.py --dry-run` で取り込み内容を表示し、`[要確認]` があれば
   どの人のどのフォルダで何が足りないかをまとめる（本人に聞くための文面も用意する）。
   `[注意] 番号が 1 から順に続いていません` が出たら、置き忘れた動画がないかユーザーに確かめる。
2. 問題のないものを `python scripts\ingest_remote.py` で取り込む（`data\_remote` 以外に置いたときは `--src <パス>`）。
3. `python scripts\run_pipeline.py`

## 共通: 報告

- `run_pipeline.py` の品質レポート（要確認のセットと理由、学習から外すデータ = `excluded_from`）を1つの表にまとめる。
  外れたセットは、撮り直し・設定の見直しで次から防げるもの（足元切れ、M5 の送信の途切れなど）を一言添える。
- 被験者ごとの試技数と RPE の分布。偏っていたら次回狙う負荷帯を提案する（本人は同じ日の中で回数を固定し、重さで RPE を変える: CLAUDE.md §9 の回数のルール）。
- `data/` と `outputs/` は git に入れない（.gitignore 済み）。コードを直した場合だけコミットし、`firmware/*.ino` が含まれていないか確認する。
