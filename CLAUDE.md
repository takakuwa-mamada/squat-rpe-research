# CLAUDE.md — スクワット RPE 推定研究（修士研究）

Claude Code が毎回読む運用メモ。研究の背景・経緯・判断理由・関連研究・計測メニューの詳細は
`docs/研究コンテキスト.md` にある。設計判断、論文・発表資料、実験計画に関わる作業の前には必ず読むこと。

最終更新: 2026-10-15（9/18 の計測は別人物 → S002 に分離、本人の初データ S001/SES005、IMU レップ検出の修正、受信時に重量・回数を入力、
プロジェクトを OneDrive から `C:\dev\claude\Projects\Research` に移動、data/ を git 履歴から削除）
（前回: 2026-09-27 方針転換: 汎用モデル先行・協力者6人は動画のみ・正面撮影）

---

## 1. ユーザーと進め方

- 高鍬 真輝（masaki）。東京都立大学大学院 システムデザイン研究科 情報科学域 M1、横山研究室
  （指導教員: 横山昌平 先生／データ工学・ソーシャルビッグデータ・可視化）。
- 本人が被験者 S001 を兼ねる（スクワット 1RM 実測済み、週3回以上トレーニング）。
- **2026-10 から 10 か月間、海外に留学**（〜2027-07 頃）。本人の計測は留学先のジムで続ける（IMU は本人だけ）。
  協力者（日本のジム仲間 約6人）とのやり取りはすべてオンライン。
- 次の締切: **2027年2月 修士論文 中間審査・発表**（留学中）。
- 返答は日本語で簡潔に。長い手順書より1画面のチートシートを好む（「そんな長ったらしいのいらない」と言われている）。
- 実装の前にマイルストーン・過程を目的とともに説明してから進める（2026-09-27 の依頼）。
- コマンドは Windows PowerShell 形式で、プロジェクトルートから `python scripts\xxx.py`。何をするコマンドか一言添える。
- コードのコメント・docstring・print は日本語。既存スクリプトの書式（冒頭 docstring に使い方、`PROJECT_ROOT` 自動判定）に合わせる。
- **ファイルは消さずに整理する**（ユーザー指示「削除ではなく、フォルダ内分けをして」）。不要物は `_archive/` へ移動。削除が必要なら先に確認。
- 週報: 毎週 GitHub Issue（YokoyamaLab/Lab_Management の Weekly Research Progress）。**ゼミで週2回発表する**:
  **月曜 = 先週の振り返り＋今週の todo**（Issue の上半分: 持ち越し・今週の目標・成果物・活動・懸念）、
  **金曜 = 今週の成果・反省・来週以降の方針**（Issue の下半分: 完了したこと・完了しなかったこと・変更・知見・Evidence・次週への課題）。
  「成果物・Evidence > Commits」に載せるので、作業は意味のある単位でコミットする。振り返り（感想）欄はユーザーが自分で書く。
  下書きは `/weekly-issue`。
- カスタムコマンド `/process-session`・`/weekly-issue` の定義は `docs/claude_commands/`（`.claude/commands/` にコピー済み。直したら両方）。
  週報は `/weekly-issue` で一から書く（雛形 `docs/claude_commands/weekly_issue_template.md`。ユーザーはテンプレートを貼らなくてよい）。
- 役割分担: コード・データ処理・解析は Claude Code。Word/PowerPoint/Excel の資料作成（研究計画書・ポスター・スライド・計測メニュー）は
  これまで Cowork 側で作ってきた（`*_source.js` は docx-js / pptxgenjs の生成スクリプト）。

## 2. 研究の要約（2026-09-27 更新）

- 研究計画書の題目: 「マルチモーダルセンシングによるバーベルトレーニングの主観的運動強度推定 — 個人適応型モデルの構築に向けた研究計画 —」
  （↓の方針転換で題目・貢献③の見直しが必要。先生と相談）
- 対象種目: **バックスクワットのみ**。
- **最終目標: IMU なし、スマホの単眼動画だけで RPE を推定する**（修論の外ではアプリ化）。
- **モデルの方針（ユーザーの予定）: まず多人数のデータで汎用モデル → 実用性が見えたらパーソナライズ**。
  研究計画書の「最初から個人適応型」から変更。評価は1つの仕組みで次の4つを出す（`train_rpe_model.py`）:
  A 本人専用（被験者内 Leave-One-Session-Out）／ B 汎用（Leave-One-Subject-Out）／
  C 汎用＋本人の最初の k セッション ／ D 本人だけの学習曲線。
- データの構成:
  - **本人 S001: IMU（BAR 1台）＋動画**。IMU は製品ではなく、動画由来のバー速度を検証する基準。
  - **S002: 2026-09-18 に日本で計測した別の人物**（IMU＋動画、1セッション。当初 S001/SES004 と誤登録 → 2026-10-15 に移動）。
    この人が協力者として Google Drive からも動画を送る場合は、`subjects.json` の S002 の source を `_remote/<ドライブのフォルダ名>` にして同じ ID にまとめる（別 ID になると LOSO で同じ人が学習と評価に跨る）。
  - **協力者 S003〜（約6人）: 動画のみ**、遠隔（Google Drive で受け取り、`ingest_remote.py` で取り込み。ID は空いている次の番号）。倫理審査は問題なし（ユーザー確認済み）。
- 撮影: **正面**・**1セット＝動画1本**（録画開始→ラックアウト→挙上→ラックイン→停止）。屈曲角・体幹前傾・プレート追跡は使わない。
- 出力: セット単位の RPE（Zourdos 2016 の RIR ベース。10=限界、9=あと1回、8=あと2回）。
- 研究課題（改訂案）:
  - RQ1: 動画だけの特徴で、速度だけの方法（VBT/Stance 相当）より RPE を正確に推定できるか（本人データでは IMU との比較も）。
  - RQ2: 動画から得たバー速度（肩の上下動）は IMU を基準にどの程度妥当か（関節角は IMU 1台・正面のため検証しない）。
  - RQ3: 汎用モデルは初めて見る人にどこまで当たるか、本人のデータを何セッション足せば実用水準（±1 RPE）に届くか。
- 外向きの説明では「パワーリフティング」を前面に出さず「アスリートがバーベル種目でよく使う指標」として説明する（ユーザー方針）。

## 3. 現在地（2026-09-27）

- **データ収集の準備は完了**。あとは本人（留学先）と協力者がデータを集めるだけ。手順は `docs/チートシート.txt`、協力者向けは `docs/協力者向け撮影ガイド.md`。
- 被験者プロフィール（`data/subjects.json`）: **S001 本人 165 cm・1RM 150 kg**、**S002 173 cm・1RM 170 kg**。身長で動画の速度を m/s に換算、1RM で %1RM を出す。
- `data/S002/SES001`（2026-09-18、日本、BAR 1台、正面、縦長 720×1280、1レップ×3、130/160/170 kg = 76/94/100 %1RM、RPE 7/9/9）。
  IMU MCV 0.52 / 0.33 / 0.30 m/s（一般的な負荷-速度関係どおり）、動画の上昇時間 1.04 / 1.37 / 1.50 s（IMU 1.02 / 1.49 / 1.67 s）。
- `data/S001/SES005`（2026-10-15、留学先、BAR 1台、正面、3レップ×3、80/90/90 kg = 53/60/60 %1RM、RPE 5/7/8）。IMU・骨格ともレップ 3/3/3。
  IMU MCV（1レップ目）0.53 / 0.45 / 0.41 m/s、MCV の速度低下 −3.8 / 6.8 / 16.6 %（骨格 −0.3 / 15.8 / 21.4 %）。同じ 90 kg でも RPE 8 の方が遅く、低下も大きい（参考値）。
  **%1RM の割に遅く RPE も高い**（S002 の 76〜94 % と同程度の速度）。理由は本人談: **1RM 150 kg は 2026-07 頃の記録で、
  その後約2か月まともにスクワットをしていなかった**（休止明けの初回）。この日の実力は 1RM 105 kg 前後と推定（90 kg×3 @RPE 7〜8 と速度から）。
  → 古い 1RM の %1RM は実際の強度を表さない。速度と RPE は一致している（研究の動機そのもの）。
  今後しばらくは筋力が戻る途中で、同じ重量でも日に日に軽くなる（本人のデータは時間とともにずれる）。
  **RPE から逆算した推定 1RM を特徴量（%1RM）に使わない**（ラベルの RPE が特徴量に漏れる）。1RM を測り直したら日付つきで更新する。
- **動画 vs IMU（RQ2 の初期値、2026-10-16、12 レップ・2人、同じ定義で比較、参考値）**: 結果は `outputs/rep_compare_video_imu.csv`
  - 上昇時間はほぼ一致（差 +0.002 s、誤差 3 %、r 0.98）→ タイミングは正しい。
  - MCV は動画が **+8 %（+0.032 m/s、一致の範囲 −0.014〜+0.077 m/s）**、r 0.97。ピーク速度 +8 %、r 0.92。上下の移動距離は動画が +12〜14 %。
    → ずれは「距離の換算（身長 × 0.779）」側で、ほぼ一定の倍率（相関が高い）。
  - 倍率はカメラ位置で変わる（MCV で S001 set3（画角が違う）+15〜18 %、set1・2 −2〜+9 %、S002 +1〜10 %）。IMU も 1 レップ目の ROM が 5〜10 cm 小さい傾向があり、IMU 側の誤差も否定できない。
  - 原因の候補: 肩の点はバーではない（前傾すると肩の方が大きく下がる）、0.779 は肩峰の表の平均（YOLO の肩の点は少し低い）、レンズ・遠近。
  - 同じ人の中の比較（速度低下 % など）は一定倍率なら影響しない。人をまたぐ絶対値（m/s）には補正が要る。速度低下は動画の方がばらつく（S001 set2: IMU 6.8 % / 動画 14.3 %）。
  - 改善案（未着手）: 正面から見えるプレートの縦の長さ（直径 45 cm）を物差しにしてバーそのものを追う／IMU の6面キャリブレーション。
- ファーム v4（PC 自動検出）はローカルで編集済み・未書き込み。WIFI_LIST にテザリングを追加して書き込むのはユーザー（出発前）。
- テスト: `pytest scripts\test_pipeline.py -q`（合成データ 13 件、約 1 分）。

## 4. ディレクトリ

```
C:\dev\claude\Projects\Research\   （2026-10-15 に OneDrive の「大学院研究」から移動）
├─ CLAUDE.md, README.md, requirements.txt, .gitignore
├─ cheat_sheet.txt           ユーザーが自分用に編集したチートシート（上書きしない）
├─ scripts/                  Python 一式（すべてプロジェクトルートから実行）
├─ firmware/
│   ├─ m5stick_multi_sender/  ★現行 v4（DEVICE_ID・常時送信・PC 自動検出）。.ino は git 管理外
│   └─ m5stick_squat_sender/  旧版
├─ weights/yolo26m-pose.pt   YOLO26 Pose の重み（gitignore）
├─ data/                     ★git 管理外（public リポジトリなので個人データは入れない）。data/README.txt だけ追跡
│   ├─ subjects.json                         被験者ID ↔ 名前・プロフィール（身長・体重・1RM・歴）
│   ├─ _sessions/session_YYYYMMDD_HHMMSS/   本人の受信スクリプトの生出力＋その日の動画（直下に置く）
│   ├─ _remote/<名前>/<日付>/                協力者の受け取り置き場（--src で Google Drive から直接でも可）
│   └─ S001/SES005/ …, S002/SES001/ …       解析用（meta.json, imu/, videos/, pose/, features/）。S001 の次は SES006
├─ outputs/                  ★git 管理外。features_all.csv（1行=1セット）, quality_report.csv
├─ models/                   ★git 管理外。train_rpe_model.py の出力（report.md ほか）
├─ docs/                     研究計画書, チートシート, 協力者向け撮影ガイド, 研究コンテキスト.md, 計測メニュー.xlsx
├─ presentations/            poster/, slides/
└─ _archive/                 旧版の資料
```

## 5. パイプライン

```powershell
# 本人（IMU＋動画）: 計測後
python scripts\split_session.py --subject S001 --session SES006     # 重量・回数は受信時に入力済み（直すときだけ --weights/--reps）
python scripts\run_pipeline.py                     # 骨格推定 → 特徴量 → 集計 → 品質レポート
# 協力者（動画のみ）
python scripts\ingest_remote.py --src "G:\マイドライブ\スクワット研究"
python scripts\run_pipeline.py
# 評価
python scripts\train_rpe_model.py                  # models\<日時>\report.md
```

| スクリプト | 役割 |
|---|---|
| `multi_imu_receiver.py` | 複数 M5 から UDP 5005 で受信。**UDP 5006 に PC の居場所の合図（"SQUATPC"）を毎秒ブロードキャスト**（v4 ファームが受け取る。`--no-beacon`）。キー: SPACE=セット開始/終了、5〜9=RPE、0=RPE10、u=取消、w=次の重量・回数（ターミナル入力）、q=終了（**q で終えないと markers.csv が出ない**）。**起動時に「重量 回数」を聞き**（`--weight`/`--reps` で省略可）、markers.csv の `weight_kg, reps` 列とフォルダ名 `_90kg_3rep` に入る |
| `split_session.py` | `_sessions` → `data/S001/SESxxx/`。マーカーで切り出し（前後0.3 s）、直下の動画をファイル名順に setNN へ。`--weights`/`--reps` で一括入力、ファイル名 `130kg_3rep` からも読む。**再実行しても meta.json の手入力値は残る**。`--view`（既定 front） |
| `ingest_remote.py` | 協力者: `<名前>/<日付>/*.mp4|mov` → `data/S00x/SESnnn/`。ラベルはファイル名（`100kg_5rep_RPE8`, `100kgx5@8`）か `記録.txt`（撮影順に「重量 回数 RPE」）。撮影順は ffprobe の撮影日時。`プロフィール.txt` → subjects.json。不備のあるフォルダは理由を出して取り込まない |
| `run_pipeline.py` | pose → features → aggregate → **品質レポート**（ラベル欠損、検出失敗、足元切れ、正面でない、レップ数の不一致、M5 の向き、受信レート）＋被験者ごとの集まり具合。`--force-features` |
| `pose_extract_yolo.py` | YOLO26 Pose で 17 点。**挙上者を追跡**（初回は最大の人、以降は前フレームに最も近い人）。`_yolo_info.json` に動画の縦横サイズ。`data/_remote` は対象外 |
| `extract_features.py` | 1セット → JSON。IMU（レップ単位の MCV/MPV/ピーク/VL/上昇時間）＋骨格（正面用 22 次元）＋ %1RM。動画だけの試技も可。`_imu_reps`/`_pose_reps` にレップ詳細 |
| `aggregate_features.py` | 全 JSON → `outputs/features_all.csv`（`_` で始まるキーは入れない） |
| `train_rpe_model.py` | 評価 A/B/C/D × 条件（context / imu_velocity / imu_all / video_velocity / video_all / imu+video、`--with-context`）。特徴量は明示リスト。report.md、図、best.pkl |
| `predict_rpe.py` | 学習済みモデルで予測（`--latest` は best.pkl 優先） |
| `test_pipeline.py` | 合成データのテスト（複数レップの骨格・IMU、ラベル解析、評価の通し） |
| `bar_path_extract.py` + `bar_tracking.py` | プレート追跡（**側面撮影が前提なので現方針では使わない**） |
| `pose_extract.py`, `compare_pose_models.py`, `squat_udp_receiver.py`, `visualize_csv.py` | 旧版・比較用 |

## 6. データ形式

- UDP: `DEVICE_ID,boot_ms,ax,ay,az,gx,gy,gz`（g・dps）。ビーコン: PC → UDP 5006 に `SQUATPC,<受信ポート>`。
- `raw_<DEV>.csv`: `recv_time_s, boot_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps`
- `markers.csv`: `set_no, start_s, end_s, duration_s, rpe, weight_kg, reps`（後ろ2列は 2026-10-15 以降。split_session は markers → 動画ファイル名 の順に使い、`--weights/--reps` が最優先）
- `imu/setNN_<DEV>.csv`: 先頭に `timestamp_ms` =（recv_time_s − start_s）×1000
- `meta.json`: `session_id, subject_id, date, source_sessions[], devices[], sampling_hz, modality(imu+video|video), camera_view(front), exercise, notes,`
  `sets[{set_no, csv_filename(動画のみは null), video_filename, weight_kg, reps_planned, reps_completed, rpe, rest_before_sec, duration_s, source_session|source_video, recorded_at, notes}]`
- `pose/setNN_yolo_keypoints.csv`: `frame, time_s, <17点>_x/_y/_conf`（0〜1 正規化。**角度・距離は info の縦横サイズでピクセルに戻してから計算**）
- 骨格特徴の距離は **BL（立位の肩〜足首の高さ）** で正規化。m/s 換算は 身長 × 0.779（肩峰高 − 足関節高）。
- 命名: 被験者 `S001`（本人）, `S002〜`（協力者）、セッション `SES001`〜、セット `set01`〜。

## 7. 既知の問題・落とし穴

1. **IMU 速度**: 3軸合成加速度 − 重力の実測値 → 積分 → レップごとに「ロックアウトで速度0・変位0」でドリフトを解く。
   レップは上昇 ROM ≥ 0.15 m で選ぶ（ラックインを拾わない）。合成 8 レップで MCV +2〜4%。
   重いレップは上昇中の速度に山が2つできる（切り返し直後＋スティッキングポイント通過後）→ 同じ上昇区間のピークは1つにまとめ、
   レップ数が多すぎるときは「時刻が連続する N 個で ROM 合計 − 間隔 6 s 超の減点」が最大の組を選ぶ（2026-10-15 修正。
   それまでは同じレップが2回入り、本物が1つ落ちていた。テスト `test_imu_double_hump_ascent_counts_each_rep_once`）。
   0.3 Hz ハイパス（旧実装）は遅い挙上を削って上昇時間を短く出していた。水平方向の加速度が大きいと誤差。
   `imu_bar_roll/pitch_range_deg` はドリフト込みで使わない。セット全体の統計（imu_peak_velocity_up_mps 等）はラックの歩き出しを含むので学習に使わない。
2. サンプル時刻は PC 受信時刻（Wi-Fi ジッタあり）。boot_ms → PC 時刻の線形写像は未対応。
3. **動画と IMU の時刻同期は未実装**（RQ2 のレップ単位の比較に必要）。IMU の速度波形と肩の上下動の相互相関で動画1本ごとに合わせる予定。
   今はセット単位の指標（MCV・上昇時間）どうしで比較できる。
4. 骨格: COCO 17 点に足先が無い。正面なので屈曲角・体幹前傾は使わない（`view != front` だと正面用の特徴は None）。
   ジムの鏡で本人が2人映る場合、追跡が鏡像に移る可能性は未検証。
5. 1レップのセットでは velocity_loss・変化量は null（HGB と補完付きモデルで扱う）。
6. RPE の申告の癖は人によって違う → 汎用モデル（B）は精度が伸びにくい可能性。C でパーソナライズの効果を見る。
7. 協力者の動画は **LINE 経由だと撮影日時が消える** → 撮影順がファイル名順になる（ガイドで Google Drive を指定済み）。
8. バッテリー 120 mAh は Wi-Fi 常時送信で 30〜60 分。
9. `mediapipe==0.10.14` 固定（比較用のみ）。OpenH264 の警告は無害。
10. 2026-10-15 に OneDrive の外（`C:\dev\claude\Projects\Research`、ASCII のパス）へ移動した。OneDrive のロック問題は無くなったが、
    **`data/` は git にも OneDrive にも入っていない（このPCにしか無い）**。計測データと動画は別にバックアップする（外付けドライブ・クラウドなど）。
11. `docs/研究計画書.pdf` はパスワード保護。本文は `docs/研究計画書_source.js`。
12. `docs/計測チェックリスト.docx/.pdf` は 2026-09-27 に正面撮影版へ作り直した（生成: `node docs\計測チェックリスト_source.js`、docx は npm で別途）。旧版は `_archive/計測チェックリスト_旧_側面撮影.*`。
13. Bash ツールのヒアドキュメントで `\n` などのバックスラッシュが潰れることがある → 置換スクリプトはファイルに書いてから実行する。

## 8. 次にやること（優先順）

**出発前（ユーザー）**
1. ファーム v4 の WIFI_LIST にテザリングを追加して書き込む（pc_ip は ""）。受信スクリプトを起動して M5 の画面が「PC:… A」になるのを確認。
2. 協力者 6 人に撮影ガイドと Google Drive の共有フォルダ（`スクワット研究/<名前>/`）を渡す。
3. `data/subjects.json` の S001 に身長・1RM を記入。
4. 先生に方針転換（汎用モデル先行・協力者は動画のみ・RQ2 はバー速度のみ）を共有。

**データ収集中（10〜12月）**
5. 本人: A/B/C メニューで 12 セッション以上（複数レップ、RPE 6〜10 を分散）。協力者: 1人 3〜4 回 × 6〜8 セット。
6. 計測のたびに `/process-session`（品質レポートの要確認を潰す）。

**解析（並行して進められる）**
7. `sync_align.py`: 動画1本ごとに IMU 速度と肩の上下動の相互相関で時刻合わせ → レップ単位の Bland–Altman（RQ2）。
8. 複数レップの実データで骨格・IMU のレップ検出のしきい値を確認（`_pose_reps` / `_imu_reps` を見る）。
9. 12月: `train_rpe_model.py` で A〜D、1月: 誤差分析・図表・発表資料。

## 9. 計測プロトコル（要点）

- M5 はシャフト中央、**画面が真上** → 静止で az ≈ +1.0 g（−1.0 なら裏返す、0 付近なら 90° 回す）。
- Wi-Fi は 2.4 GHz のみ（テザリング可）。v4 ファームは受信スクリプトの合図で PC を自動検出（画面の PC: が緑で末尾 A）。Windows ファイアウォールで Python を許可。
- カメラ: **正面**、2.5〜3 m、腰の高さ、縦向き、頭から足先まで全身、三脚固定。
- 各セット: 録画開始 → SPACE → ラックアウト→挙上→ラックイン → SPACE → **速度表示を見る前に RPE を声に出す（15秒以内、後から変えない）** → 数字キー → 録画停止。
- 終了: q → mean Hz 90〜100、セット数、RPE 未入力なしを確認。
- メニュー: A ヘビー / B ミディアム / C ボリューム（`docs/スクワット計測メニュー.xlsx`）。後半はわざと重量を下げても RPE が上がる構成。
- 協力者: `docs/協力者向け撮影ガイド.md`（正面・1セット1本・ファイル名 `100kg_5rep_RPE8` か `記録.txt`・Google Drive で送る）。
  スマホ用ページ: https://claude.ai/artifact/LHkdc4JZxy8qCGsv7iBVBU（元ファイル `docs/協力者向け撮影ガイド.html`。内容を変えたら .md と両方直して、Artifact の url 指定で再公開）。

## 10. やってはいけないこと

- `firmware/*/*.ino` には実際の Wi-Fi SSID・パスワード・IP が入っている。内容をドキュメント・Issue・コミットメッセージ・チャットに転記しない。コミットしない（.gitignore 済み）。
  編集するときも WIFI_LIST の行は表示しない（grep で除外する）。コミット前に `git ls-files firmware` で追跡されていないことを確認する。
- **`data/`・`outputs/`・`models/` はコミットしない**（.gitignore 済み。リポジトリは public）。動画（`*.mp4`）と重み（`weights/*.pt`）も同様。
  過去のコミットにあった 9/18 の計測データ（S002 のもの）は、2026-10-15 に git filter-repo で `data/` ごと履歴から消して force push 済み
  （このためそれ以前のコミットのハッシュは変わった。今の履歴に残る data/ は `data/README.txt` だけ）。
  GitHub は消したコミットもハッシュ指定でしばらく見えることがあるので、完全に消すなら GitHub サポートに依頼する。
- 協力者の名前・動画・プロフィールを外部（Issue・発表資料など）に出さない。出すときは被験者ID（S002〜）で。
- `data/_sessions/` と協力者の元動画は消さない・上書きしない（取り込みはコピー）。
- RPE ラベルを汚さない: 申告は速度を見る前。評価では同一セッションのデータを train/test に跨がせない。
- 結果の数値を盛らない。n が小さいものは「参考値」と明記する。
- ユーザーのメールアドレスはコミットの作者表記以外に使わない。
