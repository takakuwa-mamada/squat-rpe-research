data/ - スクワット計測データ格納フォルダ

構造:
  data/
    _sessions/                         受信スクリプト（multi_imu_receiver.py）の生出力
      session_YYYYMMDD_HHMMSS/
        raw_BAR.csv, raw_TRUNK.csv …   デバイスごとの連続データ
        markers.csv                    セット区切り（set_no, start_s, end_s, duration_s, rpe）
        session_info.json              受信統計（サンプル数・mean_hz）
        130.mp4 など                   動画はここ直下に置く（ファイル名の数字が重量になる）
    <subject_id>/                      被験者ID（例: S001）
      <session_id>/                    セッションID（例: SES004）
        meta.json                      セッション・セットのメタ情報（重量・レップ数・RPE）
        imu/     set01_BAR.csv …       split_session.py がセット単位に切り出したIMU
        videos/  set01.mp4 …           split_session.py が配置した動画
        pose/    set01_yolo_*.csv …    pose_extract_yolo.py の出力
        features/set01_features.json   extract_features.py の出力（1セット=1件）

命名規則:
  - subject_id: S + 3桁（S001）、session_id: SES + 3桁（SES004）
  - セット: set + 2桁（set01）。IMU は set01_<DEVICE_ID>.csv、動画は set01.mp4
  - 骨格推定の出力は YOLO 版が _yolo_ 付き（MediaPipe 版と区別）

注意:
  - _sessions の生データは消さない・上書きしない
  - split_session.py を同じ session_id で再実行すると meta.json が上書きされる
