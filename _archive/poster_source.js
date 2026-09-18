const PptxGenJS = require('pptxgenjs');
const pres = new PptxGenJS();

// A0 サイズ (841mm x 1189mm)
const W = 33.11;
const H = 46.81;

pres.defineLayout({ name: "A0_PORTRAIT", width: W, height: H });
pres.layout = "A0_PORTRAIT";

// カラーパレット
const C_DEEP  = "065A82";
const C_TEAL  = "1C7293";
const C_DARK  = "21295C";
const C_LIGHT = "F0F4F8";
const C_TEXT  = "222222";
const C_MUTED = "555555";
const C_LINE  = "CCD5DE";
const C_WHITE = "FFFFFF";
const C_ACCENT= "F4A261";
const C_GREEN = "2A9D8F";

const slide = pres.addSlide();
slide.background = { color: C_WHITE };

// ================================================================
// タイトルバー
// ================================================================
slide.addShape("rect", {
  x: 0, y: 0, w: W, h: 5.2,
  fill: { color: C_DARK }, line: { color: C_DARK }
});

slide.addText(
  "レジスタンストレーニングにおける単眼動画の骨格推定と\nバーベル挙上速度を用いたRPE推定とその妥当性検証",
  {
    x: 0.5, y: 0.4, w: W - 1.0, h: 3.1,
    fontSize: 54, bold: true, color: C_WHITE,
    align: "center", valign: "middle",
    fontFace: "Arial", margin: 0,
  }
);

slide.addText(
  "東京都立大学 システムデザイン研究科 情報科学域 横山研究室",
  {
    x: 0.5, y: 3.55, w: W - 1.0, h: 0.75,
    fontSize: 30, color: C_WHITE, align: "center", valign: "middle",
    fontFace: "Arial", margin: 0,
  }
);
slide.addText(
  "M. Mamada     (指導教員: Prof. Shohei Yokoyama)",
  {
    x: 0.5, y: 4.3, w: W - 1.0, h: 0.7,
    fontSize: 25, color: "CADCFC", align: "center", valign: "middle",
    italic: true, fontFace: "Arial", margin: 0,
  }
);

// ================================================================
// レイアウト定数
// ================================================================
const BODY_TOP = 5.8;
const BODY_LEFT = 0.7;
const BODY_RIGHT = W - 0.7;
const COL_GAP = 0.5;
const COL_W = (BODY_RIGHT - BODY_LEFT - COL_GAP) / 2.0;
const COL1_X = BODY_LEFT;
const COL2_X = BODY_LEFT + COL_W + COL_GAP;
const FOOTER_TOP = H - 1.5;

function section(x, y, w, num, title) {
  slide.addShape("ellipse", {
    x: x, y: y, w: 1.05, h: 1.05,
    fill: { color: C_DEEP }, line: { color: C_DEEP }
  });
  slide.addText(String(num), {
    x: x, y: y, w: 1.05, h: 1.05,
    fontSize: 40, bold: true, color: C_WHITE,
    align: "center", valign: "middle", fontFace: "Arial", margin: 0,
  });
  slide.addText(title, {
    x: x + 1.2, y: y, w: w - 1.2, h: 1.05,
    fontSize: 36, bold: true, color: C_DARK,
    align: "left", valign: "middle", fontFace: "Arial", margin: 0,
  });
  slide.addShape("line", {
    x: x, y: y + 1.2, w: w, h: 0,
    line: { color: C_DEEP, width: 4 }
  });
}

// ================================================================
// 左カラム: (1) 研究背景 -- y=5.8
// ================================================================
let ly = BODY_TOP;
section(COL1_X, ly, COL_W, 1, "研究背景");
ly += 1.5;

// 研究背景の本文 (高さを実サイズに合わせて 5.8)
slide.addText([
  { text: "パワーリフティングにおける RPE (Rating of Perceived Exertion)", options: { bold: true, fontSize: 24, color: C_DEEP, breakLine: true } },
  { text: "自身の主観的運動強度を「あと何レップ余力があるか (RIR)」で申告する評価法。Zourdos et al. (2016) により提案され、競技現場で広く用いられている。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: " ", options: { fontSize: 12, breakLine: true } },
  { text: "既存手法の限界", options: { bold: true, fontSize: 24, color: C_DEEP, breakLine: true } },
  { text: "① RPEは主観的で個人差・経験差が大きく、初心者ほど精度が低い。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "② 速度ベーストレーニング (VBT) はバーの速度から客観的に負荷を推定するが、フォーム崩れなどの質的変化を捉えられない。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "③ 単一モダリティ (IMU or 動画のみ) では情報が不完全。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: " ", options: { fontSize: 12, breakLine: true } },
  { text: "本研究の狙い", options: { bold: true, fontSize: 24, color: C_DEEP, breakLine: true } },
  { text: "バー装着IMUによる挙上速度と、単眼動画からの骨格推定を統合し、客観的センサ情報からRPEを推定するマルチモーダル手法を提案・妥当性検証する。VBT単独より高い精度、および従来にはないフォーム観点の情報を取り込むことを目指す。", options: { fontSize: 21, color: C_TEXT } },
], {
  x: COL1_X, y: ly, w: COL_W, h: 6.5,
  fontFace: "Arial", margin: 8, valign: "top",
});
ly += 6.7;

// RPEスケール表
slide.addText("表1: パワーリフティング RPE スケール (Zourdos, 2016)", {
  x: COL1_X, y: ly, w: COL_W, h: 0.55,
  fontSize: 19, italic: true, color: C_MUTED,
  align: "center", fontFace: "Arial", margin: 0,
});
ly += 0.6;

// ヘッダ
slide.addShape("rect", {
  x: COL1_X + 0.3, y: ly, w: COL_W - 0.6, h: 0.65,
  fill: { color: C_DEEP }, line: { color: C_DEEP }
});
const rpeColW = [1.8, COL_W - 0.6 - 1.8];
slide.addText("RPE", {
  x: COL1_X + 0.3, y: ly, w: rpeColW[0], h: 0.65,
  fontSize: 22, bold: true, color: C_WHITE, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
});
slide.addText("意味 (RIR: あと何レップ余力があるか)", {
  x: COL1_X + 0.3 + rpeColW[0], y: ly, w: rpeColW[1], h: 0.65,
  fontSize: 22, bold: true, color: C_WHITE, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
});
ly += 0.65;

const rpeRows = [
  ["10",  "1RM。これ以上1レップも不可能 (RIR=0)"],
  ["9.5", "1レップの余力あり、ただしフォーム崩れ"],
  ["9",   "あと1レップの余力"],
  ["8",   "あと2レップの余力"],
  ["7",   "あと3レップの余力"],
  ["6",   "あと4〜6レップの余力"],
];
rpeRows.forEach((row, i) => {
  const rowFill = (i % 2 === 0) ? C_WHITE : C_LIGHT;
  slide.addShape("rect", {
    x: COL1_X + 0.3, y: ly, w: COL_W - 0.6, h: 0.6,
    fill: { color: rowFill }, line: { color: C_LINE, width: 1 }
  });
  slide.addText(row[0], {
    x: COL1_X + 0.3, y: ly, w: rpeColW[0], h: 0.6,
    fontSize: 21, bold: true, color: C_DEEP, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
  });
  slide.addText(row[1], {
    x: COL1_X + 0.3 + rpeColW[0] + 0.15, y: ly, w: rpeColW[1] - 0.15, h: 0.6,
    fontSize: 20, color: C_TEXT, align: "left", valign: "middle", fontFace: "Arial", margin: 0,
  });
  ly += 0.6;
});

// ================================================================
// 左カラム: (2) 提案手法
// ================================================================
ly += 0.8;
section(COL1_X, ly, COL_W, 2, "提案手法");
ly += 1.5;

// システム構成図
slide.addText("システム全体構成", {
  x: COL1_X, y: ly, w: COL_W, h: 0.6,
  fontSize: 26, bold: true, color: C_DEEP, fontFace: "Arial", margin: 0,
});
ly += 0.7;

const boxW = (COL_W - 0.6) / 3.0;
const boxH = 2.5;
const boxes = [
  { x: COL1_X, label: "① バー装着 IMU", detail: "M5StickC Plus\n加速度3軸 + ジャイロ3軸\nWi-Fi (UDP) @ 100Hz" },
  { x: COL1_X + boxW + 0.3, label: "② 側面カメラ", detail: "スマートフォン\n単眼動画 @ 60fps\nフルHD以上" },
  { x: COL1_X + (boxW + 0.3) * 2, label: "③ PC 受信・解析", detail: "リアルタイム受信\n特徴量抽出\nRPE 推定モデル" }
];
boxes.forEach(b => {
  slide.addShape("roundRect", {
    x: b.x, y: ly, w: boxW, h: boxH,
    fill: { color: C_LIGHT }, line: { color: C_DEEP, width: 3 },
    rectRadius: 0.15,
  });
  slide.addText(b.label, {
    x: b.x, y: ly + 0.15, w: boxW, h: 0.65,
    fontSize: 22, bold: true, color: C_DEEP,
    align: "center", fontFace: "Arial", margin: 0,
  });
  slide.addText(b.detail, {
    x: b.x + 0.1, y: ly + 0.85, w: boxW - 0.2, h: boxH - 0.9,
    fontSize: 19, color: C_TEXT,
    align: "center", valign: "top", fontFace: "Arial", margin: 0,
  });
});

slide.addShape("rightTriangle", {
  x: COL1_X + boxW + 0.05, y: ly + boxH/2 - 0.15, w: 0.25, h: 0.3,
  fill: { color: C_DEEP }, line: { color: C_DEEP },
});
slide.addShape("rightTriangle", {
  x: COL1_X + boxW * 2 + 0.35, y: ly + boxH/2 - 0.15, w: 0.25, h: 0.3,
  fill: { color: C_DEEP }, line: { color: C_DEEP },
});

ly += boxH + 0.6;

// 処理パイプライン
slide.addText("処理パイプライン", {
  x: COL1_X, y: ly, w: COL_W, h: 0.6,
  fontSize: 26, bold: true, color: C_DEEP, fontFace: "Arial", margin: 0,
});
ly += 0.7;

const steps = [
  { n: "A", title: "IMU 特徴量抽出", detail: "鉛直加速度 az を積分して速度を推定。MPV, ピーク速度,\nRMS, Jerk, 速度低下率 (Velocity Loss) など全19次元。" },
  { n: "B", title: "骨格推定 (2手法を比較検証)", detail: "① MediaPipe Pose (33点): 高速・軽量、しかし遮蔽に脆弱\n② YOLO26 Pose (17点): 遮蔽下で頑健、SOTA (2026年1月)" },
  { n: "C", title: "特徴量ベクトル構築", detail: "IMU 19次元 + Pose 5次元 = 24次元の試技特徴量ベクトル。\nメタデータ (重量, レップ数, RPEラベル) と紐付けて保存。" },
  { n: "D", title: "RPE 推定モデル", detail: "① Linear (VBT準拠) ② Random Forest ③ Gradient Boosting\n評価: 被験者間 Leave-One-Subject-Out クロスバリデーション" },
];
const stepH = 1.7;
steps.forEach((s, i) => {
  const y = ly + i * (stepH + 0.2);
  slide.addShape("ellipse", {
    x: COL1_X, y: y + 0.3, w: 1.1, h: 1.1,
    fill: { color: C_TEAL }, line: { color: C_TEAL }
  });
  slide.addText(s.n, {
    x: COL1_X, y: y + 0.3, w: 1.1, h: 1.1,
    fontSize: 34, bold: true, color: C_WHITE, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
  });
  slide.addText([
    { text: s.title, options: { bold: true, fontSize: 24, color: C_DARK, breakLine: true } },
    { text: s.detail, options: { fontSize: 20, color: C_TEXT } }
  ], {
    x: COL1_X + 1.3, y: y, w: COL_W - 1.3, h: stepH,
    fontFace: "Arial", margin: 6, valign: "middle",
  });
});

// ================================================================
// 右カラム: (3) 実験
// ================================================================
let ry = BODY_TOP;
section(COL2_X, ry, COL_W, 3, "実験");
ry += 1.5;

slide.addText([
  { text: "実験環境", options: { bold: true, fontSize: 24, color: C_DEEP, breakLine: true } },
  { text: "・種目: バックスクワット (60〜95% 1RM の段階負荷)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "・IMU: M5StickC Plus (MPU6886内蔵、バーシャフト中央固定)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "・カメラ: iPhone (側面視、60fps、フルHD)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "・通信: Wi-Fi (UDP) 100Hz サンプリング", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: " ", options: { fontSize: 12, breakLine: true } },
  { text: "実験プロトコル (先行研究 González-Badillo, 2010 に準拠)", options: { bold: true, fontSize: 24, color: C_DEEP, breakLine: true } },
  { text: "① ウォームアップ後、段階的負荷 (60/70/80/90/95% 1RM)。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "② 各セットを全力挙上、セット後15秒以内にRPEを聴取。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "③ IMU と動画を同期記録、CSV と MP4 で保存。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: " ", options: { fontSize: 12, breakLine: true } },
  { text: "評価指標・検証戦略", options: { bold: true, fontSize: 24, color: C_DEEP, breakLine: true } },
  { text: "・MAE, RMSE, ±1RPE以内ヒット率 (先行研究比較値: 93%)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "・被験者間 Leave-One-Subject-Out (LOSO) クロスバリデーション", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "・アブレーション研究による各モダリティの寄与度定量化", options: { fontSize: 21, color: C_TEXT } },
], {
  x: COL2_X, y: ry, w: COL_W, h: 8.5,
  fontFace: "Arial", margin: 8, valign: "top",
});
ry += 8.7;

// ================================================================
// 右カラム: (4) 結果
// ================================================================
section(COL2_X, ry, COL_W, 4, "結果 (中間報告)");
ry += 1.5;

// 4-1
slide.addText("① 骨格推定モデルの比較 (側面視動画)", {
  x: COL2_X, y: ry, w: COL_W, h: 0.6,
  fontSize: 24, bold: true, color: C_DEEP, fontFace: "Arial", margin: 0,
});
ry += 0.65;

slide.addText("MediaPipeはバーベル・セーフティバーの遮蔽で破綻。YOLO26で頑健性が大幅に向上。", {
  x: COL2_X, y: ry, w: COL_W, h: 0.5,
  fontSize: 19, italic: true, color: C_MUTED, fontFace: "Arial", margin: 0,
});
ry += 0.55;

const compareHeader = ["指標", "MediaPipe", "YOLO26", "改善"];
const compareData = [
  ["検出失敗率 (NaN率)", "32%", "6%", "▼26pt"],
  ["体幹前傾の異常値率", "32%", "3%", "▼29pt"],
  ["膝角度の異常値率", "19%", "2%", "▼17pt"],
  ["左右非対称の異常値率", "28%", "5%", "▼23pt"],
];
const cWidths = [5.5, 3.0, 3.0, (COL_W - 0.6) - 5.5 - 3.0 - 3.0];
slide.addShape("rect", {
  x: COL2_X + 0.3, y: ry, w: COL_W - 0.6, h: 0.65,
  fill: { color: C_DEEP }, line: { color: C_DEEP }
});
let cx = COL2_X + 0.3;
compareHeader.forEach((h, i) => {
  slide.addText(h, {
    x: cx, y: ry, w: cWidths[i], h: 0.65,
    fontSize: 22, bold: true, color: C_WHITE, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
  });
  cx += cWidths[i];
});
ry += 0.65;
compareData.forEach((row, i) => {
  const rowFill = (i % 2 === 0) ? C_WHITE : C_LIGHT;
  slide.addShape("rect", {
    x: COL2_X + 0.3, y: ry, w: COL_W - 0.6, h: 0.6,
    fill: { color: rowFill }, line: { color: C_LINE, width: 1 }
  });
  cx = COL2_X + 0.3;
  row.forEach((v, j) => {
    const isImprove = (j === 3);
    slide.addText(v, {
      x: cx, y: ry, w: cWidths[j], h: 0.6,
      fontSize: 21, bold: isImprove || j === 0, color: isImprove ? C_GREEN : C_TEXT,
      align: (j === 0) ? "left" : "center", valign: "middle", fontFace: "Arial", margin: (j === 0) ? 15 : 0,
    });
    cx += cWidths[j];
  });
  ry += 0.6;
});

ry += 0.7;

// 4-2
slide.addText("② RPE 推定モデルの予備検証 (合成データ N=42, 3subjects)", {
  x: COL2_X, y: ry, w: COL_W, h: 0.6,
  fontSize: 24, bold: true, color: C_DEEP, fontFace: "Arial", margin: 0,
});
ry += 0.65;

slide.addText("被験者間 LOSO CV。パイプライン検証のための合成データによる予備結果。", {
  x: COL2_X, y: ry, w: COL_W, h: 0.5,
  fontSize: 19, italic: true, color: C_MUTED, fontFace: "Arial", margin: 0,
});
ry += 0.55;

const modelHeader = ["モデル", "MAE", "RMSE", "±1RPE 以内"];
const modelData = [
  ["Linear (VBT準拠ベースライン)", "0.47", "0.57", "95.2%"],
  ["Random Forest (24次元)",       "0.23", "0.28", "100%"],
  ["Gradient Boosting (24次元)",   "0.18", "0.23", "100%"],
];
const mWidths = [7.5, 2.2, 2.2, (COL_W - 0.6) - 7.5 - 2.2 - 2.2];
slide.addShape("rect", {
  x: COL2_X + 0.3, y: ry, w: COL_W - 0.6, h: 0.65,
  fill: { color: C_DEEP }, line: { color: C_DEEP }
});
let mx = COL2_X + 0.3;
modelHeader.forEach((h, i) => {
  slide.addText(h, {
    x: mx, y: ry, w: mWidths[i], h: 0.65,
    fontSize: 22, bold: true, color: C_WHITE, align: "center", valign: "middle", fontFace: "Arial", margin: 0,
  });
  mx += mWidths[i];
});
ry += 0.65;
modelData.forEach((row, i) => {
  const isBest = (i === 2);
  const rowFill = (i % 2 === 0) ? C_WHITE : C_LIGHT;
  slide.addShape("rect", {
    x: COL2_X + 0.3, y: ry, w: COL_W - 0.6, h: 0.6,
    fill: { color: rowFill }, line: { color: C_LINE, width: 1 }
  });
  mx = COL2_X + 0.3;
  row.forEach((v, j) => {
    slide.addText(v, {
      x: mx, y: ry, w: mWidths[j], h: 0.6,
      fontSize: 21, bold: isBest || j === 0,
      color: (isBest && j > 0) ? C_GREEN : C_TEXT,
      align: (j === 0) ? "left" : "center", valign: "middle", fontFace: "Arial", margin: (j === 0) ? 15 : 0,
    });
    mx += mWidths[j];
  });
  ry += 0.6;
});

ry += 0.8;

// 実装済み成果ボックス
slide.addShape("roundRect", {
  x: COL2_X, y: ry, w: COL_W, h: 3.5,
  fill: { color: C_LIGHT }, line: { color: C_GREEN, width: 4 },
  rectRadius: 0.2,
});
slide.addText([
  { text: "現時点までの実装済み成果", options: { bold: true, fontSize: 26, color: C_GREEN, breakLine: true } },
  { text: "✓ M5StickC + Wi-Fi(UDP) によるリアルタイム IMU 計測系", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "✓ MediaPipe / YOLO26 両骨格推定パイプラインと比較評価", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "✓ 24次元特徴量抽出とベースライン RPE 推定モデル (3種)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "✓ 被験者間 LOSO CV 評価フレームワークとメタデータ管理", options: { fontSize: 21, color: C_TEXT } },
], {
  x: COL2_X + 0.4, y: ry + 0.2, w: COL_W - 0.8, h: 3.1,
  fontFace: "Arial", margin: 0, valign: "top",
});

ry += 3.7;

// ================================================================
// 右カラム: (5) まとめと今後の課題
// ================================================================
section(COL2_X, ry, COL_W, 5, "まとめと今後の課題");
ry += 1.5;

slide.addText([
  { text: "まとめ", options: { bold: true, fontSize: 24, color: C_DEEP, breakLine: true } },
  { text: "・バー IMU と骨格推定を統合するマルチモーダル RPE 推定システムを設計・実装した。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "・単眼側面視動画において、MediaPipe Pose はバーベル遮蔽下で検出失敗率32%と実用困難だったが、YOLO26 Pose への置換で6%まで低減した (▼26pt)。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "・合成データを用いたパイプライン検証で、Gradient Boosting モデルが MAE=0.18, ±1RPE以内100%を達成 (被験者間 LOSO CV)。", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
], {
  x: COL2_X, y: ry, w: COL_W, h: 4.0,
  fontFace: "Arial", margin: 8, valign: "top",
});
ry += 4.2;

slide.addText([
  { text: "今後の課題", options: { bold: true, fontSize: 24, color: C_ACCENT, breakLine: true } },
  { text: "① 実被験者データの収集 (N=10-15を目標、倫理審査申請中)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "② 撮影角度の最適化 (真横 vs 斜め45度の比較検証)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "③ IMU と動画の高精度時刻同期 (LEDフラッシュ法の実装)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "④ Mean Propulsive Velocity (MPV) 等の追加特徴量による精度改善", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "⑤ アブレーション研究 (IMU only vs Pose only vs 統合)", options: { fontSize: 21, color: C_TEXT, breakLine: true } },
  { text: "⑥ 実運用を想定したリアルタイム推論・エッジ実装への発展", options: { fontSize: 21, color: C_TEXT } },
], {
  x: COL2_X, y: ry, w: COL_W, h: 5.0,
  fontFace: "Arial", margin: 8, valign: "top",
});

// ================================================================
// フッター
// ================================================================
slide.addShape("rect", {
  x: 0, y: FOOTER_TOP, w: W, h: 1.5,
  fill: { color: C_DARK }, line: { color: C_DARK }
});
slide.addText([
  { text: "主要参考文献: ", options: { bold: true, fontSize: 17, color: C_WHITE } },
  { text: "González-Badillo & Sánchez-Medina (2010) Int J Sports Med. | Zourdos et al. (2016) J Strength Cond Res. | ", options: { fontSize: 17, color: "CADCFC", breakLine: true } },
  { text: "Bazarevsky et al. (2020) BlazePose, arXiv. | Ultralytics YOLO26 (2026) arXiv:2606.03748.", options: { fontSize: 17, color: "CADCFC" } },
], {
  x: 0.5, y: FOOTER_TOP + 0.15, w: W - 1.0, h: 1.2,
  fontFace: "Arial", margin: 0, valign: "middle",
});

// === 左カラム末尾: 主要特徴量ボックス ===

ly += (stepH + 0.2) * 4 + 0.3;



// 主要特徴量まとめボックス

slide.addShape("roundRect", {

  x: COL1_X, y: ly, w: COL_W, h: 4.5,

  fill: { color: C_LIGHT }, line: { color: C_TEAL, width: 4 },

  rectRadius: 0.2,

});

slide.addText("抽出される主要特徴量 (計24次元)", {

  x: COL1_X + 0.3, y: ly + 0.2, w: COL_W - 0.6, h: 0.7,

  fontSize: 26, bold: true, color: C_TEAL,

  align: "left", fontFace: "Arial", margin: 0,

});



// 2カラム内訳

const featW = (COL_W - 0.8) / 2.0;

slide.addText([

  { text: "IMU 由来 (19次元)", options: { bold: true, fontSize: 22, color: C_DEEP, breakLine: true } },

  { text: "・Mean Propulsive Velocity", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・ピーク速度 / 平均速度", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・速度低下率 (Velocity Loss)", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・RMS 加速度・ピーク加速度", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・Jerk (最大・平均)", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・コンセントリック時間", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・バーの roll/pitch 変動幅", options: { fontSize: 19, color: C_TEXT } },

], {

  x: COL1_X + 0.35, y: ly + 1.0, w: featW, h: 3.3,

  fontFace: "Arial", margin: 0, valign: "top",

});

slide.addText([

  { text: "Pose 由来 (5次元)", options: { bold: true, fontSize: 22, color: C_DEEP, breakLine: true } },

  { text: "・検出成功率 (n_valid/n_total)", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・key_visibility 平均・最小", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・NaN 率", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: "・n_frames", options: { fontSize: 19, color: C_TEXT, breakLine: true } },

  { text: " ", options: { fontSize: 12, breakLine: true } },

  { text: "※ 深さ・関節角度特徴量は", options: { bold: true, fontSize: 19, color: C_MUTED, breakLine: true } },

  { text: "  今後追加予定", options: { bold: true, fontSize: 19, color: C_MUTED } },

], {

  x: COL1_X + 0.45 + featW, y: ly + 1.0, w: featW, h: 3.3,

  fontFace: "Arial", margin: 0, valign: "top",

});



pres.writeFile({ fileName: "poster_v4.pptx" }).then(fn => {
  console.log("Wrote:", fn);
});
