const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, LevelFormat, convertInchesToTwip, TableLayoutType
} = require("docx");
const fs = require("fs");

const FONT = "Yu Mincho";
const FONT_G = "Yu Gothic";
const NAVY = "1F3864";
const DARK = "202020";
const GRAY = "595959";

// ---------- helpers ----------
const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: opts.line ?? 300 },
  keepNext: opts.keepNext || false,
  keepLines: opts.keepNext || false,
  indent: opts.indent || { firstLine: opts.noIndent ? 0 : 210 },
  children: [new TextRun({
    text, font: opts.font || FONT, size: opts.size || 21,
    color: opts.color || DARK, bold: opts.bold || false,
    italics: opts.italics || false,
  })],
});

const Runs = (runs, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: opts.line ?? 300 },
  indent: opts.indent || { firstLine: opts.noIndent ? 0 : 210 },
  children: runs.map(r => new TextRun({
    text: r.t, font: r.font || FONT, size: r.size || 21,
    color: r.color || DARK, bold: r.b || false, italics: r.i || false,
    subScript: r.sub || false, superScript: r.sup || false,
  })),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 180 },
  children: [new TextRun({ text, font: FONT_G, size: 26, bold: true, color: NAVY })],
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 260, after: 140 },
  children: [new TextRun({ text, font: FONT_G, size: 23, bold: true, color: NAVY })],
});

const H3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 200, after: 110 },
  children: [new TextRun({ text, font: FONT_G, size: 21, bold: true, color: "2F5496" })],
});

const BULLET = (text, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { after: 80, line: 290 },
  children: [new TextRun({ text, font: FONT, size: 21, color: DARK })],
});

const CAPTION = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 100, after: 160 },
  children: [new TextRun({ text, font: FONT_G, size: 18, color: GRAY })],
});

const REF = (text) => new Paragraph({
  spacing: { after: 100, line: 280 },
  indent: { left: 340, hanging: 340 },
  children: [new TextRun({ text, font: FONT, size: 19, color: DARK })],
});

// table helper
function makeTable(headers, rows, colW) {
  const total = colW.reduce((a, b) => a + b, 0);
  const cell = (txt, opts = {}) => new TableCell({
    width: { size: opts.w, type: WidthType.DXA },
    shading: opts.head ? { type: ShadingType.CLEAR, fill: "DCE6F1" } :
             (opts.alt ? { type: ShadingType.CLEAR, fill: "F5F8FB" } : undefined),
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      spacing: { after: 0, line: 260 },
      keepNext: true,
      keepLines: true,
      children: [new TextRun({
        text: txt, font: FONT_G, size: 18,
        bold: opts.head || false, color: DARK,
      })],
    })],
  });
  return new Table({
    columnWidths: colW,
    width: { size: total, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    borders: {
      top:    { style: BorderStyle.SINGLE, size: 4, color: "8EA9DB" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "8EA9DB" },
      left:   { style: BorderStyle.SINGLE, size: 2, color: "BFCDE4" },
      right:  { style: BorderStyle.SINGLE, size: 2, color: "BFCDE4" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "BFCDE4" },
      insideVertical:   { style: BorderStyle.SINGLE, size: 2, color: "BFCDE4" },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        cantSplit: true,
        children: headers.map((h, i) => cell(h, { w: colW[i], head: true, center: true })),
      }),
      ...rows.map((r, ri) => new TableRow({
        cantSplit: true,
        children: r.map((c, i) => cell(c, {
          w: colW[i], alt: ri % 2 === 1, center: i > 0,
        })),
      })),
    ],
  });
}

const CW = 9000; // usable width in DXA (A4 with 25mm margins ≈ 9070)

// =====================================================================
const children = [];

// ---------- 表題 ----------
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 300, after: 100 },
  children: [new TextRun({
    text: "マルチモーダルセンシングによる",
    font: FONT_G, size: 32, bold: true, color: NAVY })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 240 },
  children: [new TextRun({
    text: "バーベルトレーニングの主観的運動強度推定",
    font: FONT_G, size: 32, bold: true, color: NAVY })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 320 },
  children: [new TextRun({
    text: "— 個人適応型モデルの構築に向けた研究計画 —",
    font: FONT_G, size: 22, color: GRAY })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 80 },
  children: [new TextRun({ text: "高鍬 真輝", font: FONT_G, size: 23, bold: true })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [new TextRun({
    text: "東京都立大学大学院 システムデザイン研究科 情報科学域",
    font: FONT_G, size: 20, color: GRAY })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [new TextRun({ text: "指導教員：横山 昌平", font: FONT_G, size: 20, color: GRAY })],
}));

// ---------- 概要 ----------
children.push(H1("概要"));
children.push(P("バーベルトレーニングにおいて、アスリートは主観的運動強度（Rating of Perceived Exertion, RPE）を指標として日々の負荷を管理している。RPEは「あと何回挙上できるか」という感覚に基づく簡便な指標である一方、主観に依存するため個人差・経験差が大きく、その日のコンディションによっても申告値が変動するという課題がある。既存の客観的手法である速度ベーストレーニング（Velocity-Based Training, VBT）はバーの挙上速度から負荷強度を推定するが、疲労に伴うフォームの崩れといった質的変化を捉えることができない。", { noIndent: true }));
children.push(P("本研究では、バーおよび身体各部に装着した慣性計測装置（Inertial Measurement Unit, IMU）と、単眼動画からの骨格推定を統合し、客観的センサデータからRPEを推定する手法を提案する。特に、被験者ごとのデータのみを用いて学習する個人適応型モデルを採用することで、RPE申告に含まれる主観バイアスの個人差を回避する。"));
children.push(P("現在までに、バー装着IMUによる無線計測系、単眼動画からの骨格抽出パイプライン、特徴量抽出からRPE推定モデル学習に至る一連の処理系を実装した。また、バーベルによる遮蔽下における骨格推定モデルの比較を予備的に実施し、YOLO26 Poseの適用により検出失敗率が大幅に低減することを確認した。今後は、マルチIMU計測系の構築、IMUを基準とした動画由来特徴量の妥当性検証、および個人適応型RPE推定モデルの構築と評価を行う。"));

// ---------- 1. はじめに ----------
children.push(H1("1. はじめに"));

children.push(H2("1.1 研究背景"));
children.push(P("レジスタンストレーニングにおいて、適切な負荷設定は競技力向上と傷害予防の両面で重要である。従来、負荷は最大挙上重量（1 Repetition Maximum, 1RM）に対する割合として処方されてきたが、1RMは日々のコンディションによって変動するため、固定的な割合処方では過負荷または過小負荷を招く場合がある。", { noIndent: true }));
children.push(P("この問題に対し、Zourdosら [1] は抵抗運動に特化したRPEスケールを提案した。これは「あと何回挙上できるか（Repetitions in Reserve, RIR）」という具体的な認知タスクによってRPEを定義するものであり、RPE 10を限界、RPE 9をあと1回の余力、RPE 8をあと2回の余力とする。この定義により、選手はセット終了直後に自身の疲労状態を数値化でき、翌セット以降の負荷調整に活用できる。RPEは現在、パワーリフティングをはじめとするバーベル種目の現場で広く用いられている。"));

children.push(H2("1.2 既存手法の課題"));
children.push(P("RPEは簡便かつ有効な指標である一方、以下の課題を有する。", { noIndent: true }));
children.push(BULLET("主観性に起因するばらつき：RPEは自己申告値であるため、個人差および経験差が大きい。特にトレーニング経験の浅い者では、実際のRIRとの乖離が大きくなることが報告されている [1]。"));
children.push(BULLET("コンディションによる変動：同一の相対負荷であっても、疲労状態や心理的要因により申告値が変動する。"));
children.push(BULLET("客観的検証手段の不足：申告されたRPEが妥当であったかを事後的に検証する手段が現場には存在しない。"));
children.push(P("客観的な負荷評価手法として、González-Badilloら [2] が確立した速度ベーストレーニング（VBT）がある。VBTでは、バーの平均推進速度（Mean Propulsive Velocity, MPV）と相対負荷（%1RM）の間に強い線形関係が存在することを利用し、速度から負荷強度を推定する。またSánchez-Medinaら [3] は、セット内での速度低下率が血中乳酸値およびアンモニア値と高い相関（r = 0.93）を示すことを報告し、速度低下が神経筋疲労の客観的指標として利用できることを示した。"));
children.push(P("しかしVBTは、バーの並進運動という単一の情報のみに依拠する。疲労が進行すると体幹前傾の増大や膝の内側変位といったフォームの崩れが生じることが知られているが [4]、これらの質的変化は速度指標のみでは捉えることができない。"));

children.push(H2("1.3 本研究の目的"));
children.push(P("本研究の目的は、バーおよび身体各部に装着したIMUと単眼動画からの骨格推定を統合したマルチモーダルセンシングにより、バーベルトレーニングにおけるRPEを客観的に推定する手法を確立することである。これにより、以下の実現を目指す。", { noIndent: true }));
children.push(BULLET("アスリート自身によるRPE申告の補助：主観的な申告値を客観データで裏付ける。"));
children.push(BULLET("疲労管理の精度向上：セット間および日々のコンディション変化を定量的に把握する。"));
children.push(BULLET("フォーム分析：疲労に伴う動作の質的変化を可視化する。"));
children.push(P("特に本研究では、被験者ごとのデータのみを用いて学習する個人適応型モデルを採用する。RPEの申告傾向は個人内では一貫している一方、個人間では大きく異なると考えられるためである。"));

children.push(H2("1.4 本研究の貢献"));
children.push(P("本研究が想定する貢献は以下の三点である。", { noIndent: true }));
children.push(BULLET("マルチIMUと単眼動画による同期計測系の構築と、その実装の公開。"));
children.push(BULLET("IMUを基準とした、単眼動画由来の関節角度およびバー速度の妥当性検証。これは将来的に動画のみによる計測へ移行する際の根拠となる。"));
children.push(BULLET("個人適応型RPE推定モデルの提案と、その学習に必要な試技数の定量化。"));


// ---------- 2. 関連研究 ----------
children.push(H1("2. 関連研究"));

children.push(H2("2.1 主観的運動強度（RPE）"));
children.push(P("RPEの概念はBorg [5] により提唱された。Borgの6–20スケールは、健康な成人において「RPE × 10 ≒ 心拍数」となるよう設計されており、持久系運動における標準的な強度指標として現在も用いられている。その後Borgは、比尺度に基づくCR10スケール [6] を提案し、息切れや筋疲労など多様な感覚への適用を可能とした。", { noIndent: true }));
children.push(P("抵抗運動においては、心肺系よりも神経筋系の限界が支配的であるため、これらの持久系向けスケールをそのまま適用することは適切でない。Zourdosら [1] は、この点に着目し、RIRに基づく抵抗運動特化型のRPEスケールを提案した。競技パワーリフター9名を対象とした検証において、熟練者は±1 RPE以内の精度でRIRを予測できることが示された一方、低強度域および初心者では誤差が大きくなることも報告されている。Helmsら [7] は、このRIRベースRPEの実践的応用について整理している。"));

children.push(H2("2.2 速度ベーストレーニング（VBT）"));
children.push(P("González-Badilloら [2] は、筋力トレーニング経験のある男性120名を対象とし、ベンチプレスにおける平均推進速度（MPV）と相対負荷（%1RM）の間に決定係数 R² = 0.98 という極めて強い線形関係が存在することを示した。さらに、6週間のトレーニング前後で1RMの絶対値は増加する一方、特定の%1RMにおけるMPVは変化しないことを確認し、負荷-速度関係が個人の筋力水準に依存しない不変的性質であることを示唆した。", { noIndent: true }));
children.push(P("Sánchez-Medinaら [3] は、セット内での速度低下率（Velocity Loss）が代謝性疲労指標および神経筋疲労指標と強く相関することを示し、速度低下を疲労の客観的指標として利用する枠組みを確立した。Pareja-Blancoら [8] は、速度低下率を閾値としてセットを終了する介入研究を行い、実践的なトレーニング処方への応用可能性を示している。"));
children.push(P("速度からRPEを推定する試みとしては、Helmsら [7] による最終レップ速度（Last Rep Velocity）法がある。個人の最小速度閾値を事前に測定し、セット最終レップの速度との差からRPEを推定するものであるが、単一の速度指標にのみ依拠する点で情報量に限界がある。"));

children.push(H2("2.3 商用VBT機器"));
children.push(P("現在市販されているVBT機器は、計測原理により大別できる。リニアポジショントランスデューサ（LPT）方式はバーの変位を直接計測するもので、GymAwareやVitruveが代表的である。加速度センサ方式はバーに装着した小型センサから速度を推定するもので、PUSH Band、Vmaxpro、およびパワーリフティングに特化したStance [9] が挙げられる。", { noIndent: true }));
children.push(P("これらの製品はいずれも、バー速度データを提示することでユーザーのRPE判断を支援するに留まり、センサデータからRPEを自動的に推定する機能は有していない。また動画録画機能を備える製品においても、動画は記録・閲覧の用途に限られ、動画データから運動情報を抽出することはない。さらに、これら商用製品について独立した学術的妥当性検証は十分に行われていない。"));

children.push(H2("2.4 マーカーレス骨格推定"));
children.push(P("動画からの人体姿勢推定は、Caoら [10] のOpenPoseにより実用水準に達した。その後、モバイル環境での実時間動作を志向したBlazePose [11] が提案され、MediaPipeフレームワークとして広く利用されている。近年では物体検出ベースのアーキテクチャを採用したYOLO系列の姿勢推定モデルが登場し、Ultralytics社のYOLO26 [12] は遮蔽下でのロバスト性向上を特徴として挙げている。", { noIndent: true }));
children.push(P("バーベル種目への適用においては、バーベルおよびセーフティバーによる身体の遮蔽が固有の課題となる。本研究ではこの点について予備的な比較検証を行った（4.2節）。"));

children.push(H2("2.5 センサデータからのRPE・疲労推定"));
children.push(P("センサデータから機械学習によりRPEを推定する研究は近年活発化している。IMUと表面筋電位（EMG）を併用した研究 [13] では、Random ForestおよびXGBoostによる推定が試みられている。また等速性ベンチプレスの力-時間データを用いた研究 [14] では、Random Forestにより±1 RPE以内の精度93%以上が報告されている。", { noIndent: true }));
children.push(P("しかしこれらの手法は、EMG電極の装着や等速性測定装置の使用を前提としており、トレーニング現場での日常的な運用には適さない。本研究は、バー装着IMUと単眼動画という現場で運用可能な計測手段のみを用いる点で、これらと異なる。"));

children.push(H2("2.6 本研究の位置付け"));
children.push(P("以上より、本研究の位置付けは次のように整理される。第一に、商用VBT機器がバー速度という単一モダリティに依拠するのに対し、本研究は身体運動情報を統合したマルチモーダル手法を採る。第二に、既存の学術研究がEMG等の装着負担の大きいセンサを前提とするのに対し、本研究は現場運用可能な計測系のみを用いる。第三に、被験者間の汎化を目指す従来の枠組みに対し、個人適応型モデルという異なる戦略を採用する。", { noIndent: true }));


// ---------- 3. 提案手法 ----------
children.push(H1("3. 提案手法"));

children.push(H2("3.1 システム全体構成"));
children.push(P("提案システムは、(a) マルチIMUによる計測部、(b) 単眼動画からの骨格推定部、(c) 特徴量統合およびRPE推定部の三つから構成される。IMUおよびカメラで取得したデータは同一のPCに集約され、時刻同期された後に試技単位の特徴量ベクトルへと変換される。", { noIndent: true }));

children.push(H2("3.2 マルチIMUによる計測"));
children.push(P("本研究では、M5StickC Plus（MPU6886内蔵）を用いる。各デバイスは加速度3軸および角速度3軸を100 Hzで取得し、Wi-Fi（UDP）によりPCへ無線送信する。装着位置は、動画から推定される主要な関節角度を検証可能とするよう、以下のとおり設計する。", { noIndent: true }));
children.push(new Paragraph({ spacing: { before: 100, after: 60 } }));
children.push(makeTable(
  ["装着位置", "取得される情報", "検証対象となる動画特徴量"],
  [
    ["バー中央",     "挙上速度・バー軌道",       "動画由来のバー速度推定"],
    ["体幹（胸椎部）", "上体の傾斜角",             "体幹前傾角"],
    ["骨盤（仙骨部）", "骨盤の傾斜・腰の高さ",     "腰位置・深さ指標"],
    ["大腿（外側）",   "大腿の傾斜角",             "股関節屈曲角の算出"],
    ["下腿（脛骨前面）", "下腿の傾斜角",           "膝関節屈曲角の算出"],
  ],
  [2100, 3100, 3800]
));
children.push(CAPTION("表1　IMUの装着位置と取得情報"));
children.push(P("関節角度は隣接するセグメント間の相対角として算出する。すなわち、股関節屈曲角は骨盤と大腿の相対角、膝関節屈曲角は大腿と下腿の相対角として求める。IMUからの姿勢推定には、加速度計と角速度計の情報を統合する相補フィルタまたはMadgwickフィルタ [15] を用いる。装着方向の個体差を吸収するため、計測開始前の直立静止姿勢を基準姿勢としてキャリブレーションを行う。"));
children.push(P("複数デバイスからのデータは同一ポートへ送信し、各パケットに含まれるデバイス識別子により振り分ける。時刻同期はPC側での受信時刻を基準とする方式を採用する。"));

children.push(H2("3.3 単眼動画からの骨格推定"));
children.push(P("被験者の側面よりスマートフォンで動画を撮影し、骨格推定モデルにより各フレームの関節位置を検出する。バーベルおよびセーフティバーによる遮蔽への耐性を考慮し、YOLO26 Pose [12] を採用する。検出された関節座標から、膝関節角度、股関節角度、体幹前傾角、腰の高さ等を算出する。", { noIndent: true }));

children.push(H2("3.4 特徴量設計"));
children.push(P("試技（1セット）を単位として特徴量ベクトルを構成する。IMU由来の特徴量としては、平均推進速度、ピーク速度、速度低下率、RMS加速度、ジャーク、コンセントリック相およびエキセントリック相の所要時間、バー軌道の水平変位等を算出する。骨格由来の特徴量としては、各関節角度の最小値・最大値・可動域、体幹前傾角の推移、左右非対称性、および最初のレップと最終レップの間でのフォーム変化量を算出する。", { noIndent: true }));
children.push(P("最後のフォーム変化量は、疲労の進行を直接的に反映する特徴量として重要である。速度指標のみでは捉えられない情報であり、マルチモーダル化の意義を担保する。"));

children.push(H2("3.5 個人適応型RPE推定モデル"));
children.push(P("本研究では、被験者ごとに独立したモデルを学習する個人適応型のアプローチを採用する。この選択には以下の根拠がある。", { noIndent: true }));
children.push(BULLET("RPEの申告傾向は個人内では一貫している一方、個人間では大きく異なる。個人適応型モデルはこの個人差を回避できる。"));
children.push(BULLET("実運用においても、ユーザーごとに蓄積されたデータで学習する形態が自然である。"));
children.push(BULLET("被験者間の汎化を主張する必要がないため、限られた被験者数でも有意な結果を得られる。"));
children.push(P("推定モデルには、線形回帰（VBT準拠のベースライン）、Random Forest、および勾配ブースティングを用い、性能を比較する。評価は同一被験者内での未使用セッションを対象とし、Leave-One-Session-Out交差検証により行う。"));
children.push(P("さらに本研究では、個人適応型モデルが実用水準に達するために必要な試技数を明らかにする。学習データ数を段階的に増加させた際の推定精度の変化（学習曲線）を測定することで、実運用における「使用開始から何セッションで有用となるか」という問いに定量的な回答を与える。"));


// ---------- 4. 現在までの進捗 ----------
children.push(H1("4. 現在までの進捗"));

children.push(H2("4.1 計測システムの実装"));
children.push(P("バー装着IMUによる無線計測系を実装した。M5StickC Plus上のファームウェアは、MPU6886から加速度および角速度を100 Hzで取得し、CSV形式でUDP送信する。本体ボタンにより計測の開始・停止を制御し、計測中は画面表示およびLEDにより状態を提示する。複数のWi-Fi環境に対応するため、起動時にスキャンを行い登録済みネットワークへ自動接続する機構を実装した。", { noIndent: true }));
children.push(P("PC側では、UDP受信と並行してリアルタイムでの波形表示、挙上速度およびRMS加速度の逐次算出、CSVへの自動保存を行う。加えて、記録済みデータから加速度・速度・位置・姿勢角を可視化する事後解析スクリプトを実装した。"));
children.push(P("動画処理については、MediaPipe PoseおよびYOLO26 Poseの双方に対応した骨格抽出パイプラインを実装し、フレームごとの関節座標および関節角度をCSVとして出力する。さらに、両モデルの結果を定量比較するスクリプトを整備した。"));
children.push(P("特徴量抽出からモデル学習に至る処理系も実装済みである。試技単位でIMUおよび骨格データを集約して特徴量ベクトルを生成し、複数の回帰モデルを学習・評価する一連の流れが自動化されている。"));

children.push(H2("4.2 骨格推定モデルの比較（予備検証）"));
children.push(P("バーベル種目に特有の遮蔽が骨格推定に与える影響を評価するため、側面から撮影したスクワット動画に対しMediaPipe PoseおよびYOLO26 Poseを適用し、比較を行った。結果を表2に示す。", { noIndent: true, keepNext: true }));
children.push(new Paragraph({ spacing: { before: 100, after: 60 } }));
children.push(makeTable(
  ["評価指標", "MediaPipe Pose", "YOLO26 Pose"],
  [
    ["検出失敗率",         "32 %", "6 %"],
    ["体幹角度の異常値率", "32 %", "3 %"],
    ["膝関節角度の異常値率", "19 %", "2 %"],
    ["左右非対称の異常値率", "28 %", "5 %"],
  ],
  [3600, 2700, 2700]
));
children.push(CAPTION("表2　遮蔽下における骨格推定モデルの比較（予備検証）"));
children.push(P("MediaPipe Poseでは、スクワット最深部においてセーフティバーおよびバーベルによる遮蔽が生じる区間で検出が破綻し、体幹前傾角が物理的に不可能な値（80度以上）を示すなどの異常が高頻度で観測された。一方YOLO26 Poseでは、同一の動画に対して検出失敗率が大幅に低減し、関節角度も生理学的に妥当な範囲に収まった。"));
children.push(Runs([
  { t: "ただし、本比較は単一試技の動画に対する予備的な観察であり、統計的な検証には至っていない。今後、複数試技・複数被験者のデータを用いた定量評価が必要である。", i: true, color: GRAY, size: 20 },
], { noIndent: true }));

children.push(H2("4.3 RPE推定パイプラインの動作検証"));
children.push(P("特徴量抽出からモデル学習・評価に至る処理系の動作を確認するため、RPEと特徴量の間に現実的な相関関係を持たせた合成データ（3名分、計42試技）を生成し、被験者間Leave-One-Subject-Out交差検証による評価を実施した。線形回帰、Random Forest、勾配ブースティングの三モデルについて、いずれも正常に学習・評価が完了することを確認した。", { noIndent: true }));
children.push(Runs([
  { t: "なお、この結果は処理系の動作確認を目的とした合成データによるものであり、実際の推定性能を示すものではない。", i: true, color: GRAY, size: 20 },
], { noIndent: true }));


// ---------- 5. 今後の研究計画 ----------
children.push(H1("5. 今後の研究計画"));

children.push(H2("5.1 フェーズ1：マルチIMU計測系の構築"));
children.push(P("現在は単一のIMUをバーに装着する構成であるが、これを身体各部を含む複数台構成へ拡張する。実装課題は、デバイス識別機構の追加、複数デバイス間の時刻同期、IMUからの姿勢推定処理、および装着方向のキャリブレーション処理である。", { noIndent: true }));
children.push(P("段階的な導入を計画しており、まずバー・体幹・大腿の3台構成で運用を確立した後、骨盤および下腿を追加して5台構成へ移行する。装着治具についても並行して検討する。"));

children.push(H2("5.2 フェーズ2：動画由来特徴量の妥当性検証"));
children.push(P("マルチIMUにより得られる関節角度を基準として、単眼動画から推定した関節角度の妥当性を検証する。具体的には、体幹前傾角、股関節屈曲角、膝関節屈曲角について、IMU由来の値と動画由来の値の一致度をBland–Altman分析および相関分析により評価する。", { noIndent: true }));
children.push(P("特に、バーベルによる遮蔽が生じる動作局面において誤差がどの程度増大するかを定量化することが重要である。この検証は、将来的に動画のみによる計測へ移行する際の根拠となる。"));
children.push(P("また、動画からバー位置を追跡することによる挙上速度の推定についても検討し、IMU計測値との比較により精度を評価する。"));

children.push(H2("5.3 フェーズ3：個人適応型RPE推定モデルの構築と評価"));
children.push(P("蓄積したデータを用いて個人適応型のRPE推定モデルを構築し、以下の観点から評価する。", { noIndent: true }));
children.push(BULLET("推定精度：平均絶対誤差（MAE）、二乗平均平方根誤差（RMSE）、±1 RPE以内ヒット率。"));
children.push(BULLET("モダリティ別の寄与：IMU単独、骨格単独、統合の三条件を比較し、各モダリティの寄与を定量化する。"));
children.push(BULLET("学習に必要な試技数：学習データ数を変化させた際の精度の推移を測定し、実用水準に達するまでに必要なデータ量を明らかにする。"));
children.push(BULLET("特徴量の重要度：どの特徴量がRPE推定に寄与しているかを分析し、疲労の身体的発現との対応を考察する。"));

children.push(H2("5.4 実験計画"));
children.push(P("対象種目はバックスクワットとする。他種目への拡張は、本手法の妥当性が確認された後の課題とする。実験プロトコルは、ウォームアップ後に相対負荷を段階的に増加させ（60〜95 %1RM）、各セット終了直後にRPEを聴取するものとする。聴取はセット終了から15秒以内に行い、記憶の変容を最小化する。", { noIndent: true }));
children.push(P("個人適応型モデルを前提とするため、被験者数よりも一被験者あたりの試技数を優先する。まず研究者自身を対象として複数セッションにわたるデータを蓄積し、その後被験者を追加する。"));

children.push(H2("5.5 スケジュール"));
children.push(new Paragraph({ spacing: { before: 100, after: 60 } }));
children.push(makeTable(
  ["時期", "実施内容"],
  [
    ["2026年8月〜9月",   "マルチIMU計測系の構築、実データによるパイプライン検証"],
    ["2026年10月〜11月", "データ収集、動画由来特徴量の妥当性検証"],
    ["2026年12月",       "個人適応型モデルの構築、モダリティ別比較"],
    ["2027年1月",        "追加分析、学習曲線の測定、発表資料作成"],
    ["2027年2月",        "修士論文中間発表"],
  ],
  [2400, 6600]
));
children.push(CAPTION("表3　研究スケジュール"));


// ---------- 6. 想定される限界と将来展望 ----------
children.push(H1("6. 想定される限界と将来展望"));

children.push(H2("6.1 限界"));
children.push(P("本研究には以下の限界が想定される。第一に、個人適応型モデルは新規ユーザーに対して即座には機能しない。一定量のデータが蓄積されるまでは推定が行えないという、いわゆるコールドスタート問題を有する。第二に、正解ラベルとして用いるRPE自体が主観指標であるため、モデルは「被験者が申告するであろうRPE」を推定するに留まり、真の生理学的負荷を推定するものではない。第三に、対象をバックスクワットに限定しているため、他種目への一般化可能性は本研究の範囲外である。", { noIndent: true }));

children.push(H2("6.2 将来展望"));
children.push(P("本研究の最終的な目標は、単眼動画のみによるRPE推定およびフォーム分析の実現である。専用センサを必要とせず、スマートフォン一台で完結する形態は、トレーニング現場への普及可能性という点で大きな意義を持つ。", { noIndent: true }));
children.push(P("この目標に対し、本研究におけるマルチIMU計測系は、動画由来の推定値の妥当性を検証するための基準として位置付けられる。すなわちIMUは最終的な運用形態において使用されるものではなく、動画のみによる計測へ移行するための検証装置である。フェーズ2における妥当性検証は、この移行の可否を判断する根拠を与えるものである。"));
children.push(P("また、種目の拡張についても、スクワットで確立した枠組みをベンチプレスおよびデッドリフトへ適用することを想定している。この際、肘関節へのIMU追加など、種目に応じた計測系の再設計が必要となる。"));

// ---------- 参考文献 ----------
children.push(H1("参考文献"));
const refs = [
  "[1] Zourdos, M. C., Klemp, A., Dolan, C., et al.: Novel Resistance Training-Specific Rating of Perceived Exertion Scale Measuring Repetitions in Reserve. Journal of Strength and Conditioning Research, Vol. 30, No. 1, pp. 267–275, 2016.",
  "[2] González-Badillo, J. J., Sánchez-Medina, L.: Movement Velocity as a Measure of Loading Intensity in Resistance Training. International Journal of Sports Medicine, Vol. 31, No. 5, pp. 347–352, 2010.",
  "[3] Sánchez-Medina, L., González-Badillo, J. J.: Velocity Loss as an Indicator of Neuromuscular Fatigue during Resistance Training. Medicine & Science in Sports & Exercise, Vol. 43, No. 9, pp. 1725–1734, 2011.",
  "[4] Effects of Intensity and Fatigue on the Kinetics and Kinematics of the Barbell Squat, Bench Press, and Deadlift in Experienced Lifters: A Systematic Review. Sports Medicine – Open, 2025.",
  "[5] Borg, G.: Perceived Exertion as an Indicator of Somatic Stress. Scandinavian Journal of Rehabilitation Medicine, Vol. 2, No. 2, pp. 92–98, 1970.",
  "[6] Borg, G.: Psychophysical Bases of Perceived Exertion. Medicine & Science in Sports & Exercise, Vol. 14, No. 5, pp. 377–381, 1982.",
  "[7] Helms, E. R., Cronin, J., Storey, A., Zourdos, M. C.: Application of the Repetitions in Reserve-Based Rating of Perceived Exertion Scale for Resistance Training. Strength and Conditioning Journal, Vol. 38, No. 4, pp. 42–49, 2016.",
  "[8] Pareja-Blanco, F., Rodríguez-Rosell, D., Sánchez-Medina, L., et al.: Effects of Velocity Loss during Resistance Training on Athletic Performance, Strength Gains and Muscle Adaptations. Scandinavian Journal of Medicine & Science in Sports, Vol. 27, No. 7, pp. 724–735, 2017.",
  "[9] Stance Fitness: STANCE Bar-speed Tracker. https://stancefitness.co/ （2026年8月閲覧）",
  "[10] Cao, Z., Simon, T., Wei, S.-E., Sheikh, Y.: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.",
  "[11] Bazarevsky, V., Grishchenko, I., Raveendran, K., et al.: BlazePose: On-device Real-time Body Pose Tracking. arXiv:2006.10204, 2020.",
  "[12] Ultralytics: YOLO26 – Unified Real-Time End-to-End Vision Models. arXiv:2606.03748, 2026.",
  "[13] Estimation of Resistance Training RPE using Inertial Sensors and Electromyography. arXiv:2510.03197, 2025.",
  "[14] Machine Learning-Driven Muscle Fatigue Estimation in Resistance Training with Assistive Robotics. Sensors (MDPI), Vol. 25, No. 21, 6588, 2025.",
  "[15] Madgwick, S. O. H., Harrison, A. J. L., Vaidyanathan, R.: Estimation of IMU and MARG Orientation Using a Gradient Descent Algorithm. Proceedings of the IEEE International Conference on Rehabilitation Robotics (ICORR), 2011.",
];
refs.forEach(r => children.push(REF(r)));

// =====================================================================
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "・", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 400, hanging: 200 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "－", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 760, hanging: 200 } } } },
      ],
    }],
  },
  styles: {
    default: {
      document: { run: { font: FONT, size: 21 } },
    },
  },
  sections: [{
    properties: {
      page: {
        margin: { top: 1420, right: 1420, bottom: 1420, left: 1420 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("研究計画書.docx", buf);
  console.log("Wrote 研究計画書.docx");
});
