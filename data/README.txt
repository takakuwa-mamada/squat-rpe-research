data/ - スクワット計測データ格納フォルダ

構造:
  data/
    <subject_id>/        被験者ID（例: S001, S002, ...）
      <session_id>/      セッションID（例: SES001, SES002, ...）
        videos/          入力: スクワット動画 (set01.mp4, set02.mp4, ...)
        imu/             入力: M5計測CSV (set01.csv, set02.csv, ...)
        pose/            出力: pose_extract.py の処理結果
        meta.json        セッションのメタ情報

命名規則:
  - subject_id: S + 3桁ゼロ埋め番号  (例: S001)
  - session_id: SES + 3桁ゼロ埋め番号 (例: SES001)
  - 試技ファイル名: set + 2桁ゼロ埋め番号 (例: set01.mp4, set01.csv)
  - 動画とIMU CSVは同じファイル名（拡張子のみ違い）で対応付ける

被験者を追加する場合:
  data/S002/SES001/videos/, imu/, pose/ をそれぞれ作成
