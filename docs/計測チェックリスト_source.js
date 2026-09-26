// 計測チェックリスト（正面撮影・1セット＝動画1本・ファーム v4）の生成スクリプト
// 使い方: node docs\計測チェックリスト_source.js   → docs\計測チェックリスト.docx
//   docx が無ければ: npm install docx（プロジェクト外に入れて NODE_PATH で指定してもよい）
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, LineRuleType,
} = require("docx");

const FONT = { ascii: "Yu Gothic", eastAsia: "Yu Gothic", hAnsi: "Yu Gothic" };
const MONO = { ascii: "Consolas", eastAsia: "Yu Gothic", hAnsi: "Consolas" };
const NAVY = "1F3864";
const RED = "C00000";
const GREEN = "2E6B30";
const GREY = "595959";
const PAGE_W = 11906, MARGIN = 850;             // A4、余白 1.5 cm
const CONTENT_W = PAGE_W - MARGIN * 2;          // 10206

const SECTION_FILL = { blue: "DCE6F0", yellow: "FFF2CC", red: "F8E1E1", green: "E2EFDA" };

function run(text, opt = {}) {
  return new TextRun({ text, font: opt.mono ? MONO : FONT, size: opt.size || 18, bold: opt.bold,
                       italics: opt.italics, color: opt.color });
}
function para(children, opt = {}) {
  return new Paragraph({ children: Array.isArray(children) ? children : [children],
                         // 游ゴシックは既定の行の高さが大きいので固定値にする（A4 1枚に収める）
                         spacing: { before: opt.before || 0, after: opt.after || 20,
                                    line: opt.line || 250, lineRule: LineRuleType.EXACT },
                         indent: opt.indent ? { left: opt.indent } : undefined,
                         alignment: opt.align, border: opt.border, shading: opt.shading });
}
function heading(text, fill) {
  return para(run(text, { bold: true, size: 21, color: NAVY }),
              { before: 80, after: 40, line: 290, shading: { type: ShadingType.CLEAR, color: "auto", fill } });
}
function check(text, opt = {}) {
  return para([run("□ ", { color: GREY }), run(text, { bold: opt.bold, color: opt.color })], { indent: 120 });
}
function note(text, opt = {}) {
  return para(run(text, { size: 15, italics: true, color: opt.color || GREY }), { indent: 360, after: 20, line: 220 });
}
function cmd(text) {
  return para(run(text, { mono: true, size: 18, color: GREEN }), { indent: 240, after: 10 });
}

const cellBorder = (color) => ({ style: BorderStyle.SINGLE, size: 6, color });
function table(rows, widths, borderColor, fill) {
  const b = cellBorder(borderColor);
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    borders: { top: b, bottom: b, left: b, right: b, insideHorizontal: b, insideVertical: b },
    rows: rows.map((cells) => new TableRow({
      children: cells.map((c, i) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill: i === 0 ? fill[0] : fill[1] } : undefined,
        margins: { top: 10, bottom: 10, left: 100, right: 100 },
        children: [para(c, { after: 0 })],
      })),
    })),
  });
}

// ---- 各セットの手順 ----
const steps = [
  ["重量をセット"],
  ["スマホの録画を開始"],
  ["PC で SPACE（セット開始）"],
  ["ラックアウト → 挙上 → ラックイン"],
  ["PC で SPACE（セット終了）"],
  ["RPE を声に出して言う　← 画面を見る前に", true],
  ["数字キーを押す　6〜9 ／ 0＝RPE10 ／ 5＝RPE5", true],
  ["スマホの録画を停止"],
];
const stepRows = steps.map(([t, red], i) => [
  run(String(i + 1), { bold: true, color: "B45F06" }),
  run(t, { bold: !!red, color: red ? RED : undefined }),
]);

// ---- トラブル ----
const trouble = [
  ["M5 の PC: が黄色のまま", "受信スクリプトが動いているか／Windows ファイアウォールで Python を許可"],
  ["az が 1.0 にならない", "M5 の向き。画面が真上を向いているか"],
  ["mean Hz が 50 未満", "Wi-Fi が弱い。PC をラックに近づける"],
  ["足元が画面から切れる", "三脚を離す、または低くする"],
].map(([a, b]) => [run(a, { bold: true, color: RED, size: 17 }), run(b, { size: 17 })]);

const children = [
  para(run("スクワット計測　当日チェックリスト", { bold: true, size: 32, color: NAVY }), { after: 20, line: 420 }),
  para(run("正面撮影 ・ 1セット＝動画1本 ・ M5 ファーム v4", { size: 18, color: GREY }), { after: 60 }),
  para(run("被験者 S______　セッション SES______　日付 ____ / ____　タイプ： A ・ B ・ C", { size: 19 }),
       { after: 60, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: NAVY, space: 4 } } }),

  heading("1. 持ち物", SECTION_FILL.blue),
  check("M5StickC Plus（充電済み・ファーム v4）＋ 固定用バンド"),
  check("ノート PC（充電済み）"),
  check("スマホ ＋ 三脚"),
  check("テザリング（2.4GHz）"),
  check("養生テープ（三脚の位置の目印）"),

  heading("2. セットアップ", SECTION_FILL.blue),
  check("テザリング ON → PC を接続"),
  check("python scripts\\multi_imu_receiver.py を起動"),
  check("M5 の電源 ON → 画面が「PC:xx.xx.xx.xx A」（緑）になる"),
  check("M5 をシャフト中央に固定（画面が真上を向く向き）"),
  check("バーを水平にして az ≒ 1.0g を確認", { bold: true, color: RED }),
  note("−1.0 なら裏返す／0 付近なら 90 度回す。ここを外すと全データが無駄になる。", { color: RED }),
  check("三脚：バーベルの真正面、2.5〜3m、高さ＝腰、スマホは縦向き"),
  check("頭から足先まで全身が入る（しゃがんだときも、ラックから出るときも）"),
  note("正面の骨格から膝の開き・横ブレ・肩の傾きを出す。足先が切れると使えない。"),
  check("三脚の脚の位置に養生テープで目印"),

  heading("3. 各セットの手順（記録セットのみ・繰り返し）", SECTION_FILL.yellow),
  table(stepRows, [600, CONTENT_W - 600], "E69138", ["FCE5CD", "FFF8E7"]),
  note("1セット＝動画1本。撮影した順番がそのままセット番号になる。"),
  note("RPE は必ず PC の速度表示を見る前に決める。見てから決めるとラベルが汚染される。", { color: RED }),

  heading("4. 終了後（その場で確認）", SECTION_FILL.blue),
  check("q キーで終了（q で終えないと markers.csv が出ない）"),
  check("mean Hz が 90〜100 の範囲か", { bold: true }),
  check("セット数と動画の本数が同じか"),
  check("RPE 未入力のセットがないか"),
  check("各セットの重量・回数をメモ（帰宅後 --weights / --reps に入れる）"),
  check("記録ログに記入（睡眠時間・体重・体調・前回からの間隔）"),

  heading("5. うまくいかないとき", SECTION_FILL.red),
  table(trouble, [3000, CONTENT_W - 3000], "E6B8B8"),

  heading("6. 帰宅後のデータ処理", SECTION_FILL.green),
  para(run("動画をスマホから data\\_sessions\\session_…\\ の直下へコピー（撮影順＝セット順）", { size: 18 }), { indent: 120 }),
  cmd("python scripts\\split_session.py --subject S001 --session SES005 --weights 100,110,120 --reps 5,3,2"),
  cmd("python scripts\\run_pipeline.py"),
  para(run("→ 最後の品質レポートで「要確認」が出たらそこを直す", { size: 18 }), { indent: 240, after: 80 }),

  para(run("初回は 3 セット程度で一度通し、品質レポートが OK になるのを確認してから本数を増やす。",
           { size: 17, italics: true, color: NAVY }),
       { before: 60, border: { top: { style: BorderStyle.SINGLE, size: 6, color: "A6A6A6", space: 4 } } }),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 19 } } } },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 },
                          margin: { top: 800, bottom: 700, left: MARGIN, right: MARGIN } } },
    children,
  }],
});

const out = path.join(__dirname, "計測チェックリスト.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("saved:", out); });
