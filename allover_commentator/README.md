# Allover Commentator MVP

既存の画面認識器を統合し、録画動画から死亡イベント向けの短文コメントを出すMVPです。

## 使い方

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate

python -m allover_commentator.detect_and_comment "../videos/2026-02-06 22-39-39.mkv" \
  --start 30 --end 90 \
  --sample-interval 0.25 \
  --ally-side right \
  --state-jsonl allover_commentator/output/state.jsonl \
  --comments-jsonl allover_commentator/output/comments.jsonl \
  --debug-dir allover_commentator/output/debug
```

Geminiを使わず、テンプレートコメントだけで検証する場合:

```bash
python -m allover_commentator.detect_and_comment "../videos/2026-02-06 22-39-39.mkv" \
  --start 30 --end 90 \
  --sample-interval 0.25 \
  --no-gemini
```

## Gemini設定

Geminiを使う場合は環境変数を設定します。未設定ならテンプレート出力にフォールバックします。

```bash
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-2.5-flash-lite"
```

死亡イベント時だけ、状態JSONと死亡前後のJPEGをGeminiに渡します。毎フレーム画像を送る設計ではありません。

## 使っている既存認識器

- `match_time_ocr`: 残り時間
- `rule_detect`: 開幕ルール名
- `squid_lamp_detect`: イカランプ人数差
- `special_gauge_detect`: 自スペシャル状態
- `ranked_objective_tracker`: ガチルールのカウント、ヤグラ/ホコ/アサリの目的物状態
- `start_detect`: 試合開始検知
- `death_detect`: 死亡検知

途中から動画を解析する場合、開幕ルール名が読めないため `--rule tower_control` のようにルールを指定してください。指定するとガチヤグラでは `ranked_objective.screenPosition` と `request.objective` にヤグラ位置/進行情報が入ります。

## 出力

- stdout: `0:30 また落ちた。今の判断、だいぶ雑。`
- `state-jsonl`: サンプルごとの認識状態
- `comments-jsonl`: コメントが発生したイベントとGemini/テンプレート出力
- `debug-dir`: 死亡時にGeminiへ渡したJPEG
