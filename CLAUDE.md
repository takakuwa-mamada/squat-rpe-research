# CLAUDE.md — スクワット RPE 推定研究（修士研究）

Claude Code が毎回読む運用メモ。研究の背景・経緯・判断理由・関連研究・計測メニューの詳細は
`docs/研究コンテキスト.md` にある。設計判断、論文・発表資料、実験計画に関わる作業の前には必ず読むこと。

最終更新: 2026-09-26（Cowork での約4か月分の作業を引継ぎ）

---

## 1. ユーザーと進め方

- 高鍬 真輝（masaki）。東京都立大学大学院 システムデザイン研究科 情報科学域 M1、横山研究室
  （指導教員: 横山昌平 先生／データ工学・ソーシャルビッグデータ・可視化）。
- 本人が被験者 S001 を兼ねる（スクワット 1RM 実測済み、週3回以上トレーニング）。
- 次の締切: **2027年2月 修士論文 中間審査・発表**。最低限示したい主張は **「マルチモーダル（IMU＋骨格）の優位性」**。
- 返答は日本語で簡潔に。長い手順書より1画面のチートシートを好む（「そんな長ったらしいのいらない」と言われている）。
- コマンドは Windows PowerShell 形式で、プロジェクトルートから `python scripts\xxx.py`。何をするコマンドか一言添える。
- コードのコメント・docstring・print は日本語。既存スクリプトの書式（冒頭 docstring に使い方、`PROJECT_ROOT` 自動判定）に合わせる。
- **ファイルは消さずに整理する**（ユーザー指示「削除ではなく、フォルダ内分けをして」）。不要物は `_archive/` へ移動。削除が必要なら先に確認。
- 週報: 毎週 GitHub Issue（YokoyamaLab/Lab_Management の Weekly Research Progress、月曜 Plan／金曜 Result、金曜ゼミで発表）。
  「成果物・Evidence > Commits」に載せるので、作業は意味のある単位でコミットする。振り返り（感想）欄はユーザーが自分で書く。
  下書きは `/weekly-issue`。
- カスタムコマンド `/process-session`・`/weekly-issue` の定義は `docs/claude_commands/` にある。
  `.claude/commands/` が無ければ、最初にそこへコピーする（Cowork からは `.claude/` に書き込めなかったため）。
- 役割分担: コード・データ処理・解析は Claude Code。Word/PowerPoint/Excel の資料作成（研究計画書・ポスター・スライド・計測メニュー）は
  これまで Cowork 側で作ってきた（`*_source.js` は docx-js / pptxgenjs の生成スクリプト）。

## 2. 研究の要約

- 研究計画書の題目: 「マルチモーダルセンシングによるバーベルトレーニングの主観的運動強度推定 — 個人適応型モデルの構築に向けた研究計画 —」
- 対象種目: **バックスクワットのみ**（他種目は妥当性確認後）。
- 入力: バー＋身体に付けた IMU（M5StickC Plus、手持ち5台）＋ スマホ単眼動画の骨格推定（YOLO26 Pose）。
- 出力: セット単位の RPE（Zourdos 2016 の RIR ベース。10=限界、9=あと1回、8=あと2回）。
- 目的: アスリート自身の RPE 申告の補助、疲労管理の精度向上、疲労に伴うフォーム変化の可視化。
  外向きの説明では「パワーリフティング」を前面に出さず「アスリートがバーベル種目でよく使う指標」として説明する（ユーザー方針）。
- モデル: **個人適応型**（本人のデータだけで学習）。評価は同一被験者内 **Leave-One-Session-Out**。
  線形回帰（VBT 準拠ベースライン）/ RandomForest / GradientBoosting を比較。
  アブレーション（IMU のみ ≒ Stance 相当 / 骨格のみ / 両方）、学習曲線（何試技で ±1 RPE の実用水準か）、特徴量重要度。
- 最終目標（修論の外）: **スマホの単眼動画だけ**で骨格推定・バー速度・RPE 推定・フォーム分析をするアプリ（市販の Stance を超える）。
- **IMU は製品に載せるものではなく、動画由来の関節角・バー速度を検証する基準（教師）**。「センサーいらなくね？」への答えがこれ。
- 貢献（研究計画書）: ①マルチIMU＋単眼動画の同期計測系 ②IMU 基準での動画由来特徴の妥当性検証 ③個人適応型 RPE 推定と必要試技数の定量化。

## 3. 現在地（2026-09-26）

- 実データは `data/S001/SES004` のみ（2026-09-18、IMU 1台=BAR、3セット×各1レップ、130/160/170 kg、RPE 7/9/9）。
  受信をセットごとに起動し直したため、3つの `_sessions` を `split_session.py --all` で1セッションに統合した。
- 受信は 100.0 Hz、静止時 az ≈ 1.03 g（向き OK）。YOLO26 の検出失敗 0%。RPE↑ で速度↓ の傾向あり（n=3 なので参考値）。
- 学習を回し始める目安の 30 試技に対して 3 試技。複数レップのセットは未取得。
- `meta.json` の `reps_completed` は 1 を記入済み（2026-09-26）。
- **未反映**: `extract_features.py` の修正（レップ検出の改善、pose ファイル名の対応）後に特徴量を再生成していない。
  `outputs/features_all.csv` と `features/*.json` は古い（n_reps 2/3/2、velocity_loss が負、pose_available=False）。→ §8 P0-1
- Git: ブランチ `main`、remote `origin` = github.com/takakuwa-mamada/squat-rpe-research。コミットは 2026-09-18 20:39 の初回のみ。
  それ以降の修正（split_session の書き直し、レップ検出、pose ファイル名対応など）と 2026-09-26 の引継ぎファイル
  （CLAUDE.md、docs/研究コンテキスト.md、docs/claude_commands/、requirements.txt ほか）は未コミットのはず。`git status` で確認してコミットする。
- 9/21 週の週報の目標（複数レップで累計15試技、IMU 3台同時計測、動画-IMU 同期の実装）は、9/18 以降の新規データが無いので
  未達の可能性が高い。着手前にユーザーに状況を確認する。

## 4. ディレクトリ

```
大学院研究/
├─ CLAUDE.md, README.md, requirements.txt, .gitignore
├─ cheat_sheet.txt           ユーザーが自分用に編集したチートシート（上書きしない）
├─ 計測チェックリスト.docx    docs/ と重複（ロックで移動できず残っている。ユーザーが片付ける）
├─ scripts/                  Python 一式（すべてプロジェクトルートから実行）
├─ firmware/
│   ├─ m5stick_multi_sender/  ★現行 v3（DEVICE_ID 付き・常時送信・複数台）
│   └─ m5stick_squat_sender/  旧版（1台・ボタンで記録 ON/OFF）
├─ weights/yolo26m-pose.pt   YOLO26 Pose の重み（gitignore）
├─ data/
│   ├─ _sessions/session_YYYYMMDD_HHMMSS/   受信スクリプトの生出力。動画はここ直下に置く
│   └─ S001/SES004/                          解析用に整形したセッション
│       ├─ meta.json
│       ├─ imu/setNN_BAR.csv
│       ├─ videos/setNN.mp4
│       ├─ pose/setNN_yolo_{keypoints,features}.csv, _annotated.mp4, _plots.png
│       └─ features/setNN_features.json
├─ outputs/                  features_all.csv（1行=1セット）, pose_comparison_summary.csv
├─ models/                   train_rpe_model.py の出力先（空）
├─ docs/                     研究計画書, 計測チェックリスト, スクワット計測メニュー.xlsx, チートシート, 研究コンテキスト.md
├─ presentations/            poster/, slides/（中間発表ポスター・外部発表スライドの最終版）
└─ _archive/                 旧版の資料
```

## 5. パイプライン

| スクリプト | 役割 |
|---|---|
| `multi_imu_receiver.py` | 【現行】複数 M5 から UDP 5005 で受信し、`data/_sessions/session_*/` に `raw_<DEV>.csv`・`markers.csv`・`session_info.json` を保存。ライブ表示あり。キー: SPACE=セット開始/終了、5〜9=RPE、0=RPE10、u=直前取消、q=終了（**q で終えないと markers.csv が出ない**） |
| `split_session.py` | `_sessions` → `data/<S>/<SES>/`。マーカーでセットを切り出し（前後0.3 s）、直下の動画をファイル名順に `videos/setNN.mp4` へコピー、ファイル名の数字（20〜400）を重量として meta.json に入れる。`--all` / `--src`（複数可）/ `--skip` で複数セッションを統合 |
| `pose_extract_yolo.py` | 【現行】YOLO26 Pose で 17 点 → `pose/*_yolo_*`。処理済みはスキップ（`--force`）。`--subject _sessions` で `_sessions` 直下の動画も処理できる |
| `extract_features.py` | 1セット → `features/setNN_features.json`（IMU 19 個＋骨格の品質指標のみ）。meta を直したら `--force` |
| `aggregate_features.py` | 全 JSON → `outputs/features_all.csv`（RPE 分布を表示） |
| `train_rpe_model.py` | linear（`imu_first_rep_peak_v_mps` のみ）/ rf / gb → `models/<日時>/`（pkl, metrics.json, predictions.csv, 散布図, 重要度）。CV は被験者2名以上で LOSO、1名は 5-fold、5件未満は train のみ |
| `predict_rpe.py` | 学習済みモデルで予測（`--latest`, `--json`） |
| `bar_path_extract.py` + `bar_tracking.py` | 動画のプレート円を追跡（Hough 自動検出 → CSRT → Hough 補正 → テンプレ再検出）、直径 450 mm でスケール換算、速度・レップ分割。**側面撮影が前提**。テスト `pytest scripts/test_bar_tracking.py` |
| `pose_extract.py`, `compare_pose_models.py` | MediaPipe 版と YOLO との比較（MediaPipe は遮蔽に弱く不採用、比較用に残す） |
| `squat_udp_receiver.py`, `visualize_csv.py` | 旧・1台用の受信/可視化 |

```powershell
cd "C:\Users\129ka\OneDrive\ドキュメント\Claude\Projects\大学院研究"
python scripts\multi_imu_receiver.py                          # ジムで。理想はジム1回につき1回だけ起動
python scripts\split_session.py --subject S001 --session SES005 --all --skip <処理済みの部分名>
#   → meta.json の null（weight_kg, reps_completed）を埋める。推測で埋めずユーザーに聞く
python scripts\pose_extract_yolo.py --subject S001
python scripts\extract_features.py --subject S001 --force
python scripts\aggregate_features.py
python scripts\train_rpe_model.py                             # 30試技以上たまってから
```

計測後処理は `/process-session S001 SES005` でまとめて実行できる（§1 のとおり `.claude/commands/` へコピー後）。

## 6. データ形式

- UDP: `DEVICE_ID,boot_ms,ax,ay,az,gx,gy,gz`（g・dps）。DEVICE_ID は BAR / TRUNK / PELVIS / THIGH / SHANK。
- `raw_<DEV>.csv`: `recv_time_s`（セッション開始からの PC 受信時刻）, `boot_ms`（M5 起動からの ms）, `ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps`
- `markers.csv`: `set_no, start_s, end_s, duration_s, rpe`（時間軸は recv_time_s と同じ）
- `imu/setNN_<DEV>.csv`: 上に `timestamp_ms` = (recv_time_s − start_s)×1000 を先頭列で追加（マージン分は負値）
- `meta.json`: `session_id, subject_id, date, source_sessions[], devices[], sampling_hz, exercise, notes,`
  `sets[{set_no, csv_filename, video_filename, exercise, weight_kg, reps_planned, reps_completed, rpe, rest_before_sec, duration_s, source_session, notes}]`
- `pose/setNN_yolo_keypoints.csv`: `frame, time_s, <17点>_x/_y/_conf`（0〜1 正規化、y は下向き、30 fps）
- `pose/setNN_yolo_features.csv`: `knee_angle_left/right, hip_angle_left/right, trunk_lean_deg, hip_y, knee_forward_x, asymmetry_knee, key_visibility`
- 命名: 被験者 `S001`、セッション `SES001`〜、セット `set01`〜。YOLO の出力は `_yolo_` 付き（MediaPipe 版と区別）。

## 7. 既知の問題・落とし穴

1. **骨格特徴が品質指標だけ**（pose_available, n_frames, visibility, nan_ratio）。運動特徴がゼロなので、今のままではマルチモーダルの優位性を示せない。最重要ギャップ。
2. **SES004 の動画は正面撮影**と推定（左右の肩・腰・膝の x が 0.1〜0.25 離れ、両目が高信頼）。研究計画・`bar_path_extract.py`・膝/股関節の屈曲角・体幹前傾は側面撮影が前提で、正面動画では妥当でない（正面の trunk_lean は側方傾斜になる）。撮影方向を meta.json に記録し、特徴量を方向別に扱う必要がある。
3. 側面撮影では奥側の手足が隠れるので L/R の非対称性は意味が薄い。手前側を信頼度で選ぶ処理が必要。COCO 17 点には足先が無く足関節角は出せない。
4. IMU 速度は `(az−1)·g` を LPF 10 Hz → 台形積分 → HPF 0.3 Hz（3 s 超のとき）しただけで**姿勢補正なし**。バーが回ると誤差になる。
   `imu_bar_roll/pitch_range_deg` はジャイロ単純積分でドリフト込み（80〜100° は非現実的）なので使わない。
5. `imu_mean_velocity_up_mps` はセグメント全体の v>0 の平均で、VBT の MCV/MPV ではない。レップ単位の MCV/MPV は未実装。
6. 1レップのセットでは velocity_loss は定義できず null。VL を使うには 3〜8 レップのセットが必要。
7. ラックアウト（歩き出し）をレップと誤検出していた → `detect_reps` を「直前に下降があるピークだけ採用＋meta の reps_completed で上限」に修正済み。複数レップのデータで閾値の再確認が必要。
8. サンプル時刻は PC 受信時刻で、Wi-Fi のジッタで間隔が揺れる（例: 受信間隔 15.3 ms に対し boot_ms は 10 ms。セット単位の推定 fs も 96〜104 Hz にばらつく）。デバイス内のタイミングは boot_ms を使い、PC 時刻へは線形回帰で写像するのが正しい（未対応）。
9. `split_session.py` は meta.json を毎回**上書き**する（手入力した reps_completed 等が消える）。`csv_filename` は `_BAR` 固定で、`extract_features.py` も BAR しか読まない。
10. `train_rpe_model.py`: 被験者1名だと KFold（シャッフル）になり、同じセッションのセットが train/test に跨ってリークする。LOSessionO が必要。
    また `weight_kg` と記録長由来の列（`imu_n_samples`, `imu_total_time_s`, `imu_fs_hz`, `pose_n_frames` など）が特徴量に自動で入る。記録長は SPACE を押すタイミング由来で RPE と偽相関しうるので外す。weight_kg を入れるかは比較条件ごとに明示的に決める。
11. 動画と IMU の時刻同期は未実装（→ `sync_align.py`）。
12. **動画の撮り方が資料間で不一致**: `docs/チートシート.txt` と Cowork での合意は「セッション通して回しっぱなし（同期のため）」、`docs/計測チェックリスト` と `split_session.py` は「セットごとに1本」。現行パイプラインはセットごと1本（ファイル名順に set01, set02… へ割当）でしか動かない。→ §8 P0-3 で決める。
13. バッテリー 120 mAh は Wi-Fi 常時送信で 30〜60 分。長いセッションでは途中で切れる。
14. `mediapipe==0.10.14` 固定（新しい版は `solutions` が無い）。OpenH264 の警告は無害（avc1→H264→mp4v→XVID の順でフォールバック）。
15. プロジェクトは OneDrive 同期フォルダ内。大きいファイルの同期中はロックされることがある（ルートの `計測チェックリスト.docx` が移動できなかった）。
16. `docs/研究計画書.pdf` はパスワード保護されていて読めない。内容は `docs/研究計画書_source.js`（本文テキストを含む）で確認できる。

## 8. 次にやること（優先順）

**P0（今すぐ）**
1. 特徴量の再生成: `extract_features.py --subject S001 --force` → `aggregate_features.py`。
   期待値: `imu_n_reps_detected` が 3 セットとも 1、`imu_velocity_loss_pct` が null、`pose_available` が True。違えば直す。
2. 骨格の運動特徴を `compute_pose_features` に実装（目標 20〜25 次元）: 膝・股関節角の min/max/ROM、体幹傾斜の max/mean/std、
   腰の最下点（深さ）、下降/ボトム/上昇の時間、hip_y によるレップ分割、**1レップ目と最終レップの差分（疲労によるフォーム変化＝本研究の肝）**。
   正面なら膝の内外反（膝 x と股・足首 x の関係）・側方シフト・左右差、側面なら屈曲角・体幹前傾・手前側選択。
3. ユーザーと決める: (a) 動画はセットごとに1本（今のまま動く。各動画にラックアウトとラックインを必ず入れる）か、回しっぱなし（`sync_align.py` 前提）か。
   (b) 撮影方向（研究計画は側面。狭いジムなら斜め45°も候補）。決めたら `docs/チートシート.txt` とチェックリストを揃える。
4. データ収集: `docs/スクワット計測メニュー.xlsx` の A/B/C（複数レップ）で、RPE 6〜10 を分散させて累計 30 試技以上。受信はジム1回につき1回起動。

**P1（10月）**
5. `sync_align.py`: ラックアウト/ラックインの衝撃（IMU az の鋭いピーク 3〜5 g）と動画音声のオンセットを照合してオフセットを決定。
   始点・終点の2点でクロックずれを線形補正。markers と合わせて動画をセット単位に分割。予備手段は IMU 積分位置と `bar_path_extract` 軌跡の相互相関。
6. レップ単位の VBT 指標: コンセントリック区間の検出、MCV・MPV・ピーク速度・VL、boot_ms ベースのタイミング。
7. IMU 3台（BAR/TRUNK/THIGH）→ 5台（+PELVIS/SHANK）: DEVICE_ID だけ変えて書き込み、姿勢推定（相補 or Madgwick）、直立静止キャリブレーション、
   セグメント間の相対角（股=骨盤−大腿、膝=大腿−下腿）。`split_session.py` / `extract_features.py` を複数デバイス対応に。装着治具。
8. `train_rpe_model.py`: 同一被験者内 LOSessionO（session_id でグループ化）、アブレーション3条件、学習曲線、特徴量重要度（SHAP 等）、特徴量の除外リスト（§7-10）。
9. `split_session.py` を再実行しても meta.json の手入力値が残るようにする（既存 meta とマージ）。

**P2（11〜12月）**
10. 動画由来特徴の妥当性検証: IMU 基準で関節角・バー速度を Bland–Altman と相関で評価（バーベルの遮蔽局面の誤差を定量化）。
11. `bar_path_extract` の出力（軌跡の水平変位・速度）を特徴量に統合し、IMU 速度と比較。
12. ファームの Wi-Fi 認証情報を `secrets.h`（gitignore）に分離して `.ino` をコミットできるようにする。
13. バッテリー対策（モバイルバッテリー給電や送信間引き）。

目安のスケジュール（研究計画書）: 10〜11月 データ収集・動画特徴の妥当性検証 → 12月 個人適応型モデルとモダリティ比較 → 1月 追加分析・学習曲線・発表資料 → 2月 中間審査。

## 9. 計測プロトコル（要点。詳細は docs/研究コンテキスト.md と docs/計測チェックリスト）

- M5 はシャフト中央、**画面が真上** → 静止で az ≈ +1.0 g（−1.0 なら裏返す、0 付近なら 90° 回す）。ここを外すと全データが無駄。
- Wi-Fi は 2.4 GHz のみ（テザリング可）。`ipconfig` の PC の IP がファームの `WIFI_LIST` の pc_ip と一致していること。
  場所が変わったら `WIFI_LIST` に追記して書き込み直す（Arduino IDE、ボード M5Stick-C-Plus、COM3、upload 1500000）。Windows ファイアウォールで Python を許可。
- カメラ: 2.5〜3 m、腰の高さ、プレート全体とラック支柱2本を画角に入れる（プレート直径 450 mm がスケール基準）。三脚位置はテープで印。
- 各セット: 重量セット →（録画開始）→ SPACE → 挙上 → SPACE → **速度表示を見る前に RPE を声に出す（15秒以内、後から変えない）** → 数字キー →（録画停止）。
  RPE 8.5 等は後で markers.csv / meta.json を直す。
- 終了: q → mean Hz 90〜100、セット数、RPE 未入力なしを確認。記録ログに睡眠・体重・カフェイン・体調・前回からの間隔。
- メニュー: A ヘビー（85〜94%1RM・1〜3回・6セット）/ B ミディアム（72.5〜87.5%・2〜6回・8セット）/ C ボリューム（65〜75%・5〜8回・8セット）。
  週3回なら A→B→C。後半はわざと重量を下げても RPE が上がる構成（重量と RPE の相関を崩し、モデルに動きの情報を見させるため）。

## 10. やってはいけないこと

- `firmware/*/*.ino` には実際の Wi-Fi SSID・パスワード・IP が入っている。内容をドキュメント・Issue・コミットメッセージ・チャットに転記しない。コミットしない（.gitignore 済み）。
  コミット前に `git ls-files firmware` で追跡されていないことを確認する。
- 動画（`*.mp4`）と重み（`weights/*.pt`）はコミットしない（.gitignore 済み）。
- GitHub リポジトリの公開設定は未確認。`data/` には本人の計測データが入っている。他の被験者のデータを入れる前に private か確認し、同意・倫理審査の扱いを先生に確認する。
- `data/_sessions/` の生データは消さない・上書きしない。
- RPE ラベルを汚さない: 申告は速度を見る前。評価では同一セッションのデータを train/test に跨がせない。
- 結果の数値を盛らない。n が小さいものは「参考値」と明記する。
- ユーザーのメールアドレスはコミットの作者表記以外に使わない。
