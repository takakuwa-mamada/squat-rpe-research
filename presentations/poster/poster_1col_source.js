const PptxGenJS = require('pptxgenjs');
const pres = new PptxGenJS();

// A0 縦
const W = 33.11;
const H = 46.81;

pres.defineLayout({ name: "A0P", width: W, height: H });
pres.layout = "A0P";

// カラー
const C_DEEP  = "065A82";  // 深い青（ヘッダ帯）
const C_TEAL  = "1C7293";
const C_DARK  = "21295C";
const C_LIGHT = "F0F4F8";
const C_BORDER= "B0BEC5";
const C_TEXT  = "222222";
const C_MUTED = "555555";
const C_WHITE = "FFFFFF";
const C_ACCENT= "F4A261";
const C_GREEN = "2A9D8F";
const C_LINE  = "CCD5DE";

const slide = pres.addSlide();
slide.background = { color: C_WHITE };

// ================================================================
// タイトルバー（薄いバー、井桁さんのポスターに倣う）
// ================================================================
const TITLE_H = 3.0;
slide.addShape("rect", {
  x: 0.5, y: 0.5, w: W - 1.0, h: TITLE_H,
  fill: { color: C_WHITE }, line: { color: C_DEEP, width: 2 }
});

slide.addText(
  "レジスタンストレーニングにおける単眼動画の骨格推定とバーベル挙上速度を用いた RPE 推定とその妥当性検証",
  {
    x: 0.7, y: 0.6, w: W - 1.4, h: 1.8,
    fontSize: 46, bold: true, color: C_DARK,
    align: "center", valign: "middle",
    fontFace: "Arial", margin: 0,
  }
);

slide.addText(
  "情報科学域   25XXXXXX   M. Mamada       指導教員  横山 昌平",
  {
    x: 0.7, y: 2.35, w: W - 1.4, h: 1.05,
    fontSize: 26, color: C_TEXT, align: "center", valign: "middle",
    fontFace: "Arial", margin: 0,
  }
);

// ================================================================
// セクション描画ヘルパ
// ================================================================
function sectionHeader(x, y, w, title) {
  // 色付き帯（ヘッダ）
  slide.addShape("rect", {
    x: x, y: y, w: w, h: 1.1,
    fill: { color: C_DEEP }, line: { color: C_DEEP },
  });
  slide.addText(title, {
    x: x + 0.4, y: y, w: w - 0.8, h: 1.1,
    fontSize: 30, bold: true, color: C_WHITE,
    align: "left", valign: "middle", fontFace: "Arial", margin: 0,
  });
  return y + 1.1;
}

function sectionBody(x, y, w, h) {
  // コンテンツ枠
  slide.addShape("rect", {
    x: x, y: y, w: w, h: h,
    fill: { color: C_WHITE }, line: { color: C_DEEP, width: 2 },
  });
  return y;
}

// ================================================================
// レイアウト
// ================================================================
const M_LEFT = 0.5;
const M_RIGHT = W - 0.5;
const SEC_W = M_RIGHT - M_LEFT;
const FOOTER_H = 1.0;

let cy = 0.5 + TITLE_H + 0.5;   // タイトルの下から

// ================================================================
// 1. 研究背景
// ================================================================
{
  const secY = cy;
  const secH = 5.6;
  const hy = sectionHeader(M_LEFT, secY, SEC_W, "研究背景");
  sectionBody(M_LEFT, hy, SEC_W, secH - 1.1);

  const bodyY = hy + 0.3;
  const bodyH = secH - 1.4;
  const bodyX = M_LEFT + 0.4;
  const bodyW = SEC_W - 0.8;

  // 背景箇条書き
  slide.addText([
    { text: "・パワーリフティングにおいて、主観的運動強度 RPE (Rating of Perceived Exertion) は「あと何レップ余力があるか (RIR)」で申告する評価法として、Zourdos et al. (2016) 以降広く用いられている。", options: { fontSize: 22, color: C_TEXT, breakLine: true } },
    { text: "・しかし RPE は主観指標のため、個人差・経験差が大きく、客観的な計測が求められている。", options: { fontSize: 22, color: C_TEXT } },
  ], {
    x: bodyX, y: bodyY, w: bodyW, h: 2.0,
    fontFace: "Arial", margin: 8, valign: "top",
  });

  // 課題（強調）
  slide.addText("課題：客観的センサ情報から RPE を推定する手法の確立", {
    x: bodyX, y: bodyY + 2.05, w: bodyW, h: 0.75,
    fontSize: 27, bold: true, color: C_DEEP,
    align: "left", valign: "middle", fontFace: "Arial", margin: 8,
  });

  // 問題点サブボックス
  const subY = bodyY + 2.85;
  const subH = 1.55;
  slide.addShape("rect", {
    x: bodyX, y: subY, w: bodyW, h: subH,
    fill: { color: C_LIGHT }, line: { color: C_DEEP, width: 1.5 },
  });
  slide.addText("問題点", {
    x: bodyX + 0.2, y: subY + 0.05, w: 2.5, h: 0.5,
    fontSize: 22, bold: true, color: C_DEEP,
    fontFace: "Arial", margin: 0,
  });
  slide.addText([
    { text: "・速度ベーストレーニング (VBT) はバー速度から負荷を推定するが、フォーム崩れなどの質的変化を捉えられない。", options: { fontSize: 20, color: C_TEXT, breakLine: true } },
    { text: "・単一モダリティ (IMUのみ or 動画のみ) では情報が不完全。特に動画では、バーベルによる遮蔽で骨格推定が破綻するケースが報告されている。", options: { fontSize: 20, color: C_TEXT } },
  ], {
    x: bodyX + 0.25, y: subY + 0.55, w: bodyW - 0.5, h: subH - 0.6,
    fontFace: "Arial", margin: 5, valign: "top",
  });

  cy = secY + secH + 0.4;
}

// ================================================================
// 2. 提案手法
// ================================================================
{
  const secY = cy;
  const secH = 10.5;
  const hy = sectionHeader(M_LEFT, secY, SEC_W, "提案手法");
  sectionBody(M_LEFT, hy, SEC_W, secH - 1.1);

  const bodyY = hy + 0.3;
  const stepX = M_LEFT + 0.5;
  const stepW = SEC_W - 1.0;

  // 上部説明
  slide.addText([
    { text: "バー装着 IMU による挙上速度と、単眼動画からの骨格推定を統合し、", options: { fontSize: 22, color: C_TEXT } },
    { text: "24次元特徴量ベクトルから RPE を推定するマルチモーダル手法を提案する。", options: { fontSize: 22, color: C_TEXT } },
  ], {
    x: stepX, y: bodyY, w: stepW, h: 0.8,
    fontFace: "Arial", margin: 0, valign: "top",
  });

  let sy = bodyY + 1.0;
  const steps = [
    {
      title: "1.  IMU によるバー速度の計測",
      body: [
        "・M5StickC Plus (MPU6886内蔵) をバーシャフト中央に固定。",
        "・加速度3軸・角速度3軸を100Hzで取得し、Wi-Fi (UDP) でPCへリアルタイム送信。",
        "・鉛直加速度 az を積分して速度を推定し、MPV, ピーク速度, RMS, Jerk, 速度低下率 (Velocity Loss) など全19次元の特徴量を算出。",
      ],
      h: 2.6,
    },
    {
      title: "2.  単眼動画からの骨格推定 (2 手法を比較検証)",
      body: [
        "・被験者側面をスマートフォンで撮影 (60fps, フルHD)。",
        "・① MediaPipe Pose (33点、軽量・高速だが遮蔽に脆弱)",
        "・② YOLO26 Pose (17点、2026年1月リリースの SOTA モデル、遮蔽下で頑健)",
        "・両モデルの検出精度・失敗率・異常値率を定量比較し、有効なモデルを選定。",
      ],
      h: 2.8,
    },
    {
      title: "3.  マルチモーダル特徴量統合と RPE 推定 (実装済み)",
      body: [
        "・IMU 由来 19次元 + 骨格由来 5次元 = 計 24次元の試技特徴量ベクトルを構築。",
        "・メタデータ (重量, レップ数, RPEラベル) と紐付けて data/<被験者>/<セッション>/ 配下に格納。",
        "・線形回帰 (VBT準拠ベースライン) / Random Forest / Gradient Boosting の3モデルで RPE を推定。",
        "・被験者間 Leave-One-Subject-Out (LOSO) クロスバリデーションで汎化性能を評価。",
      ],
      h: 3.0,
    },
  ];

  steps.forEach(s => {
    // タイトル
    slide.addText(s.title, {
      x: stepX, y: sy, w: stepW, h: 0.7,
      fontSize: 25, bold: true, color: C_DEEP,
      align: "left", valign: "middle", fontFace: "Arial", margin: 0,
    });
    // 本文
    slide.addText(
      s.body.map((t, i) => ({
        text: t,
        options: { fontSize: 21, color: C_TEXT, breakLine: (i < s.body.length - 1) }
      })),
      {
        x: stepX + 0.3, y: sy + 0.7, w: stepW - 0.3, h: s.h - 0.7,
        fontFace: "Arial", margin: 5, valign: "top",
      }
    );
    sy += s.h;
  });

  cy = secY + secH + 0.4;
}

// ================================================================
// 3. 実験
// ================================================================
{
  const secY = cy;
  const secH = 5.6;
  const hy = sectionHeader(M_LEFT, secY, SEC_W, "実験");
  sectionBody(M_LEFT, hy, SEC_W, secH - 1.1);

  const bodyX = M_LEFT + 0.5;
  const bodyW = SEC_W - 1.0;
  const bodyY = hy + 0.3;

  slide.addText([
    { text: "・対象種目: バックスクワット (60〜95% 1RM の段階的負荷)", options: { fontSize: 22, color: C_TEXT, breakLine: true } },
    { text: "・計測装置: M5StickC Plus (バー装着) + iPhone (側面視, 60fps, フルHD)", options: { fontSize: 22, color: C_TEXT, breakLine: true } },
    { text: "・プロトコル: ウォームアップ後、各セットを全力挙上し、セット終了後15秒以内にRPEを聴取 (González-Badillo, 2010 に準拠)", options: { fontSize: 22, color: C_TEXT, breakLine: true } },
    { text: "・比較手法: MediaPipe Pose v0.10.14 vs Ultralytics YOLO26 Pose (2026年1月リリース、SOTA)", options: { fontSize: 22, color: C_TEXT, breakLine: true } },
    { text: "・評価指標: MAE, RMSE, ±1 RPE 以内ヒット率 (先行研究比較値: 93%)", options: { fontSize: 22, color: C_TEXT, breakLine: true } },
    { text: "・検証戦略: 被験者間 Leave-One-Subject-Out (LOSO) クロスバリデーション", options: { fontSize: 22, color: C_TEXT } },
  ], {
    x: bodyX, y: bodyY, w: bodyW, h: secH - 1.4,
    fontFace: "Arial", margin: 5, valign: "top",
  });

  cy = secY + secH + 0.4;
}

// ================================================================
// 4. 結果
// ================================================================
{
  const secY = cy;
  const secH = 10.0;
  const hy = sectionHeader(M_LEFT, secY, SEC_W, "結果");
  sectionBody(M_LEFT, hy, SEC_W, secH - 1.1);

  const bodyX = M_LEFT + 0.5;
  const bodyW = SEC_W - 1.0;
  const bodyY = hy + 0.3;

  // 概要
  slide.addText([
    { text: "・骨格推定において、MediaPipe はバーベル・セーフティバーの遮蔽で破綻したが、YOLO26 で頑健性が大幅に向上した。", options: { fontSize: 22, color: C_TEXT, breakLine: true } },
    { text: "・合成データによる予備検証で、Gradient Boosting モデルが被験者間 LOSO CV で MAE=0.18, ±1RPE以内 100% を達成。", options: { fontSize: 22, color: C_TEXT } },
  ], {
    x: bodyX, y: bodyY, w: bodyW, h: 1.6,
    fontFace: "Arial", margin: 5, valign: "top",
  });

  // 表1: 骨格モデル比較（左半分）
  const tblY = bodyY + 1.8;
  const tblW = (bodyW - 0.5) / 2.0;

  // 表1タイトル
  slide.addText("表1: 骨格推定モデルの比較 (側面視動画)", {
    x: bodyX, y: tblY, w: tblW, h: 0.55,
    fontSize: 20, bold: true, color: C_DEEP,
    fontFace: "Arial", margin: 0,
  });
  const t1y = tblY + 0.55;
  // ヘッダ
  const t1Cols = [2.4, 2.5, 2.5, tblW - 2.4 - 2.5 - 2.5];
  slide.addShape("rect", {
    x: bodyX, y: t1y, w: tblW, h: 0.55,
    fill: { color: C_DEEP }, line: { color: C_DEEP }
  });
  let hx1 = bodyX;
  ["指標","MediaPipe","YOLO26","改善"].forEach((h, i) => {
    slide.addText(h, {
      x: hx1, y: t1y, w: t1Cols[i], h: 0.55,
      fontSize: 20, bold: true, color: C_WHITE, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
    });
    hx1 += t1Cols[i];
  });
  const t1Data = [
    ["検出失敗率 (NaN率)","32%","6%","▼26pt"],
    ["体幹前傾の異常値率","32%","3%","▼29pt"],
    ["膝角度の異常値率","19%","2%","▼17pt"],
    ["左右非対称の異常値率","28%","5%","▼23pt"],
  ];
  let t1yy = t1y + 0.55;
  t1Data.forEach((row, i) => {
    const fill = (i % 2 === 0) ? C_WHITE : C_LIGHT;
    slide.addShape("rect", {
      x: bodyX, y: t1yy, w: tblW, h: 0.55,
      fill: { color: fill }, line: { color: C_LINE, width: 1 },
    });
    let cx = bodyX;
    row.forEach((v, j) => {
      const isImp = (j === 3);
      slide.addText(v, {
        x: cx, y: t1yy, w: t1Cols[j], h: 0.55,
        fontSize: 19, bold: isImp || j === 0, color: isImp ? C_GREEN : C_TEXT,
        align: (j === 0) ? "left" : "center", valign: "middle", fontFace: "Arial", margin: (j === 0) ? 10 : 0,
      });
      cx += t1Cols[j];
    });
    t1yy += 0.55;
  });

  // 表2: モデル比較（右半分）
  const rx = bodyX + tblW + 0.5;
  slide.addText("表2: RPE 推定モデル比較 (合成データ N=42, 3subj, LOSO CV)", {
    x: rx, y: tblY, w: tblW, h: 0.55,
    fontSize: 20, bold: true, color: C_DEEP,
    fontFace: "Arial", margin: 0,
  });
  const t2y = tblY + 0.55;
  const t2Cols = [tblW - 1.6 - 1.6 - 2.0, 1.6, 1.6, 2.0];
  slide.addShape("rect", {
    x: rx, y: t2y, w: tblW, h: 0.55,
    fill: { color: C_DEEP }, line: { color: C_DEEP }
  });
  let hx2 = rx;
  ["モデル","MAE","RMSE","±1RPE 以内"].forEach((h, i) => {
    slide.addText(h, {
      x: hx2, y: t2y, w: t2Cols[i], h: 0.55,
      fontSize: 20, bold: true, color: C_WHITE, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
    });
    hx2 += t2Cols[i];
  });
  const t2Data = [
    ["Linear (VBT準拠)","0.47","0.57","95.2%"],
    ["Random Forest","0.23","0.28","100%"],
    ["Gradient Boosting","0.18","0.23","100%"],
  ];
  let t2yy = t2y + 0.55;
  t2Data.forEach((row, i) => {
    const isBest = (i === 2);
    const fill = (i % 2 === 0) ? C_WHITE : C_LIGHT;
    slide.addShape("rect", {
      x: rx, y: t2yy, w: tblW, h: 0.55,
      fill: { color: fill }, line: { color: C_LINE, width: 1 },
    });
    let cx = rx;
    row.forEach((v, j) => {
      slide.addText(v, {
        x: cx, y: t2yy, w: t2Cols[j], h: 0.55,
        fontSize: 19, bold: isBest || j === 0,
        color: (isBest && j > 0) ? C_GREEN : C_TEXT,
        align: (j === 0) ? "left" : "center", valign: "middle", fontFace: "Arial", margin: (j === 0) ? 10 : 0,
      });
      cx += t2Cols[j];
    });
    t2yy += 0.55;
  });

  // 実装済み成果の要点（表の下）
  const boxY = tblY + 3.4;
  slide.addShape("rect", {
    x: bodyX, y: boxY, w: bodyW, h: 2.6,
    fill: { color: C_LIGHT }, line: { color: C_GREEN, width: 2 },
  });
  slide.addText("現時点までの実装済み成果", {
    x: bodyX + 0.2, y: boxY + 0.1, w: bodyW - 0.4, h: 0.6,
    fontSize: 23, bold: true, color: C_GREEN, fontFace: "Arial", margin: 0,
  });
  slide.addText([
    { text: "✓ M5StickC + Wi-Fi(UDP) によるリアルタイム IMU 計測系", options: { fontSize: 20, color: C_TEXT, breakLine: true } },
    { text: "✓ MediaPipe / YOLO26 両骨格推定パイプラインと定量比較スクリプト", options: { fontSize: 20, color: C_TEXT, breakLine: true } },
    { text: "✓ 24 次元特徴量抽出とベースライン RPE 推定モデル (Linear / RF / GB)", options: { fontSize: 20, color: C_TEXT, breakLine: true } },
    { text: "✓ 被験者間 LOSO CV 評価フレームワークとメタデータ管理 (xlsx テンプレート)", options: { fontSize: 20, color: C_TEXT } },
  ], {
    x: bodyX + 0.25, y: boxY + 0.7, w: bodyW - 0.5, h: 1.9,
    fontFace: "Arial", margin: 0, valign: "top",
  });

  cy = secY + secH + 0.4;
}

// ================================================================
// 5. まとめ・今後の課題（横2分割）
// ================================================================
{
  const secY = cy;
  const secH = 6.0;
  const hy = sectionHeader(M_LEFT, secY, SEC_W, "まとめ・今後の課題");
  sectionBody(M_LEFT, hy, SEC_W, secH - 1.1);

  const bodyY = hy + 0.3;
  const halfW = (SEC_W - 1.2) / 2.0;
  const leftX = M_LEFT + 0.5;
  const rightX = leftX + halfW + 0.2;

  // まとめ
  slide.addText("まとめ", {
    x: leftX, y: bodyY, w: halfW, h: 0.65,
    fontSize: 25, bold: true, color: C_DEEP,
    fontFace: "Arial", margin: 0,
  });
  slide.addText([
    { text: "・バー IMU と骨格推定を統合するマルチモーダル RPE 推定システムを設計・実装した。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
    { text: "・単眼側面視動画において、MediaPipe Pose の検出失敗率 32% に対し、YOLO26 Pose 採用で 6% まで低減した (▼26pt)。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
    { text: "・合成データを用いたパイプライン検証で、Gradient Boosting が被験者間 LOSO CV で MAE=0.18, ±1RPE 以内 100% を達成した。", options: { fontSize: 21, color: C_TEXT } },
  ], {
    x: leftX, y: bodyY + 0.7, w: halfW, h: secH - 2.0,
    fontFace: "Arial", margin: 5, valign: "top",
  });

  // 今後の課題
  slide.addText("今後の課題", {
    x: rightX, y: bodyY, w: halfW, h: 0.65,
    fontSize: 25, bold: true, color: C_ACCENT,
    fontFace: "Arial", margin: 0,
  });
  slide.addText([
    { text: "・実被験者データの収集 (N=10-15 を目標、倫理審査申請中)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
    { text: "・撮影角度の最適化 (真横 vs 斜め 45 度の比較検証)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
    { text: "・IMU と動画の高精度時刻同期 (LED フラッシュ法の実装)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
    { text: "・Mean Propulsive Velocity (MPV) 等の追加特徴量による精度改善", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
    { text: "・アブレーション研究 (IMU only vs Pose only vs 統合)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
    { text: "・実運用を想定したリアルタイム推論・エッジ実装への発展", options: { fontSize: 21, color: C_TEXT } },
  ], {
    x: rightX, y: bodyY + 0.7, w: halfW, h: secH - 2.0,
    fontFace: "Arial", margin: 5, valign: "top",
  });

  cy = secY + secH + 0.4;
}

// ================================================================
// フッター (日付・参考文献)
// ================================================================
slide.addText(
  "主要参考文献: González-Badillo & Sánchez-Medina (2010) Int J Sports Med. | Zourdos et al. (2016) J Strength Cond Res. | Bazarevsky et al. (2020) BlazePose, arXiv. | Ultralytics YOLO26 (2026) arXiv:2606.03748.",
  {
    x: 0.5, y: H - 0.9, w: W - 1.0, h: 0.6,
    fontSize: 15, color: C_MUTED, italic: true,
    align: "left", valign: "middle", fontFace: "Arial", margin: 0,
  }
);
slide.addText("2026年7月 中間発表", {
  x: 0.5, y: H - 0.4, w: W - 1.0, h: 0.3,
  fontSize: 14, color: C_MUTED, align: "right", fontFace: "Arial", margin: 0,
});

pres.writeFile({ fileName: "poster_1col.pptx" }).then(fn => {
  console.log("Wrote:", fn);
});
