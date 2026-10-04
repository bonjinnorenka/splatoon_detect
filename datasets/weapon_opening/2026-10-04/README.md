# 開始HUD収集データ — 2026-10-04

Windowsで収集・人力確認したSplatoon 3の開始HUD画像と、既存OBS動画ラベルのスナップショットです。元の保存先からコピーし、元データは変更していません。ライブPNGは再圧縮・リサイズ・cropをしていない元画像そのものです。動画由来cropは元OBS動画をデコードして切り出し、PNGで保存しています。

```text
live_session/matches/<match_id>/
  capture.json       収集条件・timestamp・frame index・画像SHA-256
  annotation.json    人力ラベル・status・crop補正・reference・revision
  frames/00.png      元の1920×1080開始frame
  frames/01.png ...  同じ試合の後続frame（合計5枚）
obs_session/
  videos.json        元OBS動画の識別子・metadata（動画自体は含まない）
  matches/*.json    動画の試合単位人力ラベル
obs_opening_crops/
  dataset.json       原動画識別子・使用ラベル・開始検出記録・抽出条件
  samples.jsonl     cropとmatch/timestamp/frame/slot/state/正解ラベルの対応
  crops/<match_id>/*.png   動画由来の開始HUD crop（2,080枚）
  frames/<match_id>/*.jpg  確認用context（265枚、JPEG）
wepons.json          この時点の武器一覧（174武器）
```

## 収録内容

- ライブ160試合・800 PNG。156試合が人力確定済み、4試合は対象外。対象外データも元statusを維持して保存しています。
- 確定ライブの1,248スロット中、labeled 1,246、Skip 1、Occluded 1。
- 既存動画56試合のラベル。うち53試合が確定済み。
- 動画由来のラベル付きcropは52試合×8スロット×5フレーム＝2,080枚。8スロットすべてOccludedの残り1確定試合はcropから除外し、contextは保存しています。
- 合計209確定試合、174武器中173武器に人力ラベルがあります。168武器は3試合以上、169武器は2試合以上です。
- 未収集: オーダーシェルター レプリカ。
- 1試合のみ: オーダーストリンガー・オーダーチャージャー・オーダーブラスター・オーダーマニューバーの各レプリカ。
- 2試合: オーダーワイパー レプリカ。

800 PNGの読み込み・解像度・SHA-256と武器名を検証しています。これは保存の整合性確認であり、全ラベルの目視再確認や認識精度の保証ではありません。1試合に複数frameがあっても武器ごとのsupportは試合単位で数えてください。評価splitも原則match単位で分離します。

履歴・resume・ロック・プレビュー・推論候補bundle・誤判定出力は含めません。元動画は別途必要です。既存動画metadataの元pathは来歴として保持し、Windowsで表示するときはレビューアプリの `--video-root` を指定してください。ライブPNGの参照pathはcapture.json内の相対pathなので、clone先やWindows/WSLに依存せず利用できます。

## 収集数の再集計

リポジトリrootで実行します。WSLのvirtualenvは `source ../bin/activate`、Windowsは用意したvirtualenvのpython.exeを使ってください。

```bash
python -m live_weapon_collect.coverage \
  --session-dir datasets/weapon_opening/2026-10-04/live_session \
  --coverage-dir datasets/weapon_opening/2026-10-04/obs_session \
  --catalog datasets/weapon_opening/2026-10-04/wepons.json \
  --target 3
```

## ライブ画像から評価用cropを展開

```bash
python -m live_weapon_collect.export_dataset \
  datasets/weapon_opening/2026-10-04/live_session \
  --catalog datasets/weapon_opening/2026-10-04/wepons.json \
  --output live_weapon_collect/data/export_snapshot_20261004 \
  --states alive down unknown
```

出力先は未作成のdirectoryを使ってください。生成結果はGit対象外です。確定済みの対象試合のみを既存評価形式へ展開し、alive/down/unknownは別集計できます。元の手動statusと自動HUD stateは別の情報です。

## 動画由来crop

`obs_opening_crops/crops/` と `samples.jsonl` は元動画なしでも直接閲覧・読み込みできます。cropは各試合の手動geometryを使い、既存の開始HUD検出から最初の5秒だけを1秒間隔で抽出しています。reference時刻そのものからの抽出ではなく、開始HUD検出で決めた基準時刻を `opening_timestamp` に記録しています。試合途中のframeで補充はしません。

自動stateはalive 1,716、down 136、unknown 228です。state判定が外れる可能性もあるため、すべて保存しています。正解武器は人力ラベルであり、自動推論で確定していません。試合ごとの出力件数・検出基準時刻・state・正解武器・PNG読み込みを検証しています。context JPEGは確認用で、武器照合に使うcrop PNGとは別です。

再生成には元OBS動画が必要です。既存評価CLIの原動画照合を使う場合も、`videos.json` の元pathに動画を用意してください。作業用の新しい出力先に生成する例:

```bash
python -m weapon_lamp_detect.build_dataset \
  datasets/weapon_opening/2026-10-04/obs_session \
  --output weapon_lamp_detect/data/poc_video_seed_reexport \
  --sample-interval 1 --include-down --max-frames-per-match 1
python -m weapon_lamp_detect.build_opening_dataset \
  weapon_lamp_detect/data/poc_video_seed_reexport \
  --output weapon_lamp_detect/data/poc_video_opening_reexport \
  --window 5 --sample-interval 1 --workers 4
```

最初のdatasetは開始抽出に渡すmetadata用です。開始datasetの件数・時刻・抽出条件は `dataset.json` のトップレベルに、元datasetの設定は `source_dataset_config` に分けて保存しています。

スナップショットのラベルを確認・修正する場合は作業用コピーを作り、`weapon_label_review` の入力元に指定してください。通常の追加収集はWindows側の既存sessionへ続けます。このdirectoryへ自動で追記・同期はされません。
