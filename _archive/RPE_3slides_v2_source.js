const PptxGenJS = require('pptxgenjs');
const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";   // 13.3 x 7.5 inch (16:9)
const W = 13.33, H = 7.5;

const NAVY  = "21295C";
const DEEP  = "065A82";
const TEAL  = "1C7293";
const GREEN = "2A9D8F";
const RED   = "C62828";
const TEXT  = "222222";
const MUTED = "555555";
const LIGHT = "F0F4F8";
const WHITE = "FFFFFF";
const LINE  = "CCD5DE";

// 共通: タイトルバー
function titleBar(slide, num, title) {
  slide.addShape("rect", { x:0, y:0, w:W, h:1.0, fill:{color:NAVY}, line:{color:NAVY} });
  slide.addShape("ellipse", { x:0.45, y:0.2, w:0.6, h:0.6, fill:{color:WHITE}, line:{color:WHITE} });
  slide.addText(String(num), { x:0.45, y:0.2, w:0.6, h:0.6, fontSize:22, bold:true,
    color:NAVY, align:"center", valign:"middle", fontFace:"Arial", margin:0 });
  slide.addText(title, { x:1.2, y:0, w:W-1.6, h:1.0, fontSize:27, bold:true,
    color:WHITE, align:"left", valign:"middle", fontFace:"Arial", margin:0 });
}

// ページ番号
function pageNum(slide, n) {
  slide.addText(`${n} / 3`, { x:W-1.1, y:H-0.45, w:0.8, h:0.3, fontSize:12,
    color:MUTED, align:"right", fontFace:"Arial", margin:0 });
}

// =========================================================
// スライド1: 研究背景と課題
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, 1, "研究背景と課題");

  // メインの一文
  s.addText("筋力トレーニングでは、負荷の決定を「アスリートの主観」に頼っている",
    { x:0.5, y:1.15, w:W-1.0, h:0.55, fontSize:20, bold:true, color:NAVY,
      align:"center", valign:"middle", fontFace:"Arial", margin:0 });

  // 概念図
  s.addImage({ path:"img/fig_concept.png", x:0.55, y:1.8, w:12.2, h:3.15 });

  // 下段: RPE とは / 課題
  const boxY = 5.15, boxH = 1.85;
  const halfW = (W - 1.4) / 2;

  // 左: RPE とは
  s.addShape("roundRect", { x:0.5, y:boxY, w:halfW, h:boxH,
    fill:{color:LIGHT}, line:{color:DEEP, width:2}, rectRadius:0.08 });
  s.addText("RPE（主観的運動強度）とは", { x:0.7, y:boxY+0.1, w:halfW-0.4, h:0.35,
    fontSize:15, bold:true, color:DEEP, fontFace:"Arial", margin:0 });
  s.addText([
    { text:"スクワット等のバーベル種目で広く使われる強度指標。", options:{fontSize:12.5, color:TEXT, breakLine:true} },
    { text:"「あと何回挙げられるか」を 1〜10 の数値で申告する。", options:{fontSize:12.5, color:TEXT, breakLine:true} },
    { text:"RPE 10 = 限界 / 9 = あと1回 / 8 = あと2回 / 7 = あと3回", options:{fontSize:12.5, color:DEEP, bold:true} },
  ], { x:0.7, y:boxY+0.48, w:halfW-0.4, h:boxH-0.6, fontFace:"Arial", margin:0, valign:"top" });

  // 右: 課題
  s.addShape("roundRect", { x:0.5+halfW+0.4, y:boxY, w:halfW, h:boxH,
    fill:{color:LIGHT}, line:{color:RED, width:2}, rectRadius:0.08 });
  s.addText("既存手法の課題", { x:0.7+halfW+0.4, y:boxY+0.1, w:halfW-0.4, h:0.35,
    fontSize:15, bold:true, color:RED, fontFace:"Arial", margin:0 });
  s.addText([
    { text:"① 主観ゆえのばらつき（個人差・経験差が大きい）", options:{fontSize:12.5, color:TEXT, breakLine:true} },
    { text:"② 日々のコンディションにより申告がぶれる", options:{fontSize:12.5, color:TEXT, breakLine:true} },
    { text:"③ 既存のセンサ計測はバー速度のみで、フォームの", options:{fontSize:12.5, color:TEXT, breakLine:true} },
    { text:"　 崩れなど質的な変化を捉えられない", options:{fontSize:12.5, color:TEXT} },
  ], { x:0.7+halfW+0.4, y:boxY+0.48, w:halfW-0.4, h:boxH-0.6, fontFace:"Arial", margin:0, valign:"top" });

  pageNum(s, 1);
}

// =========================================================
// スライド2: 実験方法
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, 2, "実験方法：バー装着センサ ＋ 単眼動画による計測");

  // システム構成図
  s.addImage({ path:"img/fig_system.png", x:0.45, y:1.15, w:12.45, h:3.55 });

  // 下段: 実験条件 (左) / 評価 (右)
  const boxY = 4.95, boxH = 2.05;
  const lw = 7.4, rw = W - 1.0 - lw - 0.4;

  s.addShape("roundRect", { x:0.5, y:boxY, w:lw, h:boxH,
    fill:{color:LIGHT}, line:{color:TEAL, width:2}, rectRadius:0.08 });
  s.addText("実験条件", { x:0.7, y:boxY+0.1, w:lw-0.4, h:0.35,
    fontSize:15, bold:true, color:TEAL, fontFace:"Arial", margin:0 });
  s.addText([
    { text:"・対象種目：バックスクワット（1RM の 60〜95%）", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・計測装置：M5StickC Plus（バー装着）＋ スマートフォン（側面撮影）", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・手順：各セットを挙上 → 終了直後に RPE を申告 → センサデータと紐付け", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・比較手法：MediaPipe Pose / YOLO26 Pose（骨格推定モデル 2 種）", options:{fontSize:13, color:TEXT} },
  ], { x:0.7, y:boxY+0.5, w:lw-0.4, h:boxH-0.6, fontFace:"Arial", margin:0, valign:"top" });

  s.addShape("roundRect", { x:0.5+lw+0.4, y:boxY, w:rw, h:boxH,
    fill:{color:LIGHT}, line:{color:GREEN, width:2}, rectRadius:0.08 });
  s.addText("評価方法", { x:0.7+lw+0.4, y:boxY+0.1, w:rw-0.4, h:0.35,
    fontSize:15, bold:true, color:"00695C", fontFace:"Arial", margin:0 });
  s.addText([
    { text:"・±1 RPE 以内ヒット率", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"　（先行研究の到達値：93%）", options:{fontSize:12, color:MUTED, breakLine:true} },
    { text:"・平均絶対誤差（MAE）", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・被験者間クロスバリデーション", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"　（未知の被験者への汎化性能）", options:{fontSize:12, color:MUTED} },
  ], { x:0.7+lw+0.4, y:boxY+0.5, w:rw-0.4, h:boxH-0.6, fontFace:"Arial", margin:0, valign:"top" });

  pageNum(s, 2);
}

// =========================================================
// スライド3: 現時点での進捗とまとめ
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, 3, "現時点での進捗とまとめ");

  // 課題提起
  s.addText("課題：バーベル・セーフティバーによる「遮蔽」で、従来の骨格推定は破綻する",
    { x:0.5, y:1.08, w:W-1.0, h:0.42, fontSize:16, bold:true, color:RED,
      align:"center", valign:"middle", fontFace:"Arial", margin:0 });

  // 比較グラフ (アスペクト比 1.884)
  const gW = 6.7, gH = gW / 1.884;   // = 3.556
  s.addImage({ path:"img/fig_compare.png", x:0.55, y:1.58, w:gW, h:gH });

  // 右側: 結論ボックス
  const rx = 7.55, rw = W - rx - 0.5;
  s.addShape("roundRect", { x:rx, y:1.62, w:rw, h:3.45,
    fill:{color:"E0F2F1"}, line:{color:GREEN, width:2.5}, rectRadius:0.1 });
  s.addText("骨格推定モデルの選定", { x:rx+0.2, y:1.78, w:rw-0.4, h:0.42,
    fontSize:17, bold:true, color:"00695C", align:"center", fontFace:"Arial", margin:0 });
  s.addText("YOLO26 Pose の採用により\n検出失敗率が大幅に低減",
    { x:rx+0.2, y:2.25, w:rw-0.4, h:0.72, fontSize:14, color:TEXT,
      align:"center", valign:"middle", fontFace:"Arial", margin:0 });
  s.addText("32%  →  6%", { x:rx+0.2, y:3.0, w:rw-0.4, h:0.85,
    fontSize:36, bold:true, color:"00695C", align:"center", valign:"middle",
    fontFace:"Arial", margin:0 });
  s.addText("遮蔽下でも安定した\n骨格特徴の取得が可能に",
    { x:rx+0.2, y:3.95, w:rw-0.4, h:0.85, fontSize:14, color:TEXT,
      align:"center", valign:"top", fontFace:"Arial", margin:0 });

  // 下段: まとめ / 今後
  const boxY = 5.3, boxH = 1.75;
  const halfW = (W - 1.4) / 2;

  s.addShape("roundRect", { x:0.5, y:boxY, w:halfW, h:boxH,
    fill:{color:LIGHT}, line:{color:DEEP, width:2}, rectRadius:0.08 });
  s.addText("現時点の成果", { x:0.7, y:boxY+0.08, w:halfW-0.4, h:0.35,
    fontSize:15, bold:true, color:DEEP, fontFace:"Arial", margin:0 });
  s.addText([
    { text:"・センサ・動画の同時計測システムを構築", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・骨格推定モデルを比較し、遮蔽に強い手法を選定", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・特徴量抽出から RPE 推定までの処理系を実装", options:{fontSize:13, color:TEXT} },
  ], { x:0.7, y:boxY+0.46, w:halfW-0.4, h:boxH-0.55, fontFace:"Arial", margin:0, valign:"top" });

  s.addShape("roundRect", { x:0.5+halfW+0.4, y:boxY, w:halfW, h:boxH,
    fill:{color:LIGHT}, line:{color:"E76F51", width:2}, rectRadius:0.08 });
  s.addText("今後の展開", { x:0.7+halfW+0.4, y:boxY+0.08, w:halfW-0.4, h:0.35,
    fontSize:15, bold:true, color:"D35400", fontFace:"Arial", margin:0 });
  s.addText([
    { text:"・被験者データの収集と推定精度の検証", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・計測環境の最適化（カメラ角度・センサ位置）", options:{fontSize:13, color:TEXT, breakLine:true} },
    { text:"・RPE 申告補助・疲労管理ツールとしての実用化", options:{fontSize:13, bold:true, color:"D35400"} },
  ], { x:0.7+halfW+0.4, y:boxY+0.46, w:halfW-0.4, h:boxH-0.55, fontFace:"Arial", margin:0, valign:"top" });

  pageNum(s, 3);
}

pres.writeFile({ fileName: "RPE_3slides.pptx" }).then(fn => console.log("Wrote:", fn));
