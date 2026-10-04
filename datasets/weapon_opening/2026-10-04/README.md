# 開始HUD収集データ — 2026-10-04

Windowsで収集・人力確認したSplatoon 3の開始HUD画像と、既存OBS動画ラベルのスナップショットです。元の保存先からコピーし、元データは変更していません。PNGは再圧縮・リサイズ・cropをしていない元画像そのものです。

```text
live_session/matches/<match_id>/
  capture.json       収集条件・timestamp・frame index・画像SHA-256
  annotation.json    人力ラベル・status・crop補正・reference・revision
  frames/00.png      元の1920×1080開始frame
  frames/01.png ...  同じ試合の後続frame（合計5枚）
obs_session/
  videos.json        元OBS動画の識別子・metadata（動画自体は含まない）
  matches/*.json    動画の試合単位人力ラベル
wepons.json          この時点の武器一覧（174武器）
```

## 収録内容

- ライブ160試合・800 PNG。156試合が人力確定済み、4試合は対象外。対象外データも元statusを維持して保存しています。
- 確定ライブの1,248スロット中、labeled 1,246、Skip 1、Occluded 1。
- 既存動画56試合のラベル。うち53試合が確定済み。
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

スナップショットのラベルを確認・修正する場合は作業用コピーを作り、`weapon_label_review` の入力元に指定してください。通常の追加収集はWindows側の既存sessionへ続けます。このdirectoryへ自動で追記・同期はされません。
