---
description: 研究室の週報（GitHub Issue: Weekly Research Progress）の下書きを作る
argument-hint: [result | plan | both]  省略時は both（今週の Result と来週の Plan）
---

週報の下書きを作る。対象: $ARGUMENTS

材料を集める:
- `git log --since="last saturday" --stat`（今週のコミット。Evidence の Commits 欄に短縮ハッシュ＋1行説明で載せる）
- `CLAUDE.md` の §3 現在地・§8 次にやること、`docs/研究コンテキスト.md` §10 の過去の目標
- `data/` に今週追加されたセッション、`outputs/features_all.csv` の累計試技数と RPE 分布

書き方:
- 見出しは `docs/研究コンテキスト.md` §10 のテンプレートをそのまま使う。
- 事実だけを書く。やっていないことを書かない。数値はファイルから取る。
- 今週の目標は3つ、金曜に確認できる形（数と成果物）で書く。
- 「ゼミ出欠」「欠席時の連絡・相談」「振り返り（うまくいったこと／難しかったこと／改善したいこと）」はユーザーが書くので空欄で残す。
- 今週の Result と来週の Plan を、それぞれコピペしやすい1つのコードブロックで出す。
