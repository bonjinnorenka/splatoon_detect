# Splatoon 3 イカランプ・インク色検出

`../videos` の試合動画から、上部HUDのイカランプ人数と左右チームのインク色を読み取り、実況補助に渡しやすいJSON/CSVを出すための初期実装。

## 使い方

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
python squid_lamp_detect/detect_video.py "../videos/2026-02-01 21-18-24.mp4" \
  --start 20 --end 80 --sample-interval 1 \
  --jsonl squid_lamp_detect/output/sample.jsonl \
  --csv squid_lamp_detect/output/sample.csv \
  --debug-dir squid_lamp_detect/output/debug
```

標準出力にJSONLを出す場合:

```bash
python squid_lamp_detect/detect_video.py "../videos/2026-02-06 22-39-39.mkv" \
  --start 20 --end 40 --sample-interval 1
```

デフォルトは `--ally-side right`。現在の録画では画面右側HUDが味方、左側HUDが敵として扱われる。

## 出力

主なフィールド:

```json
{
  "ally_alive": 3,
  "enemy_alive": 2,
  "ally_down": 1,
  "enemy_down": 2,
  "ally_unknown": 0,
  "enemy_unknown": 0,
  "ally_special_ready": [false, null, false, false],
  "enemy_special_ready": [false, null, null, true],
  "ally_ink_color": {
    "rgb": [120, 210, 40],
    "hsv": [45, 220, 210],
    "confidence": 0.92,
    "source": "smoothed"
  },
  "enemy_ink_color": {
    "rgb": [190, 60, 220],
    "hsv": [150, 210, 220],
    "confidence": 0.90,
    "source": "smoothed"
  },
  "confidence": 0.88,
  "timestamp_ms": 123456
}
```

`hud_state` が `non_match` の場合、試合HUDではない可能性が高いため人数・色は `null` または低confidenceになる。
敵スペシャルは `enemy_special_ready` にスロット順の `true` / `false` / `null` として出る。`null` は倒されている、または十分に読めない状態。

## 実装方針

- 人数: 16:9録画の上部HUDに固定ROIを置き、各スロットのXオーバーレイを画像特徴で判定する。
- 試合外判定: `match_time_ocr` のタイマーOCRを優先し、試合時間が読めないフレームは原則 `non_match` に落とす。OCRが数字分割で失敗する一部フレームは、上中央のタイマー形状で補助判定する。
- インク色: HUD周辺の細いライン、カウント表示、イカランプから高彩度ピクセルの支配色を取る。
- 敵スペシャル: スロット下部に出る小さなスペシャルアイコン候補を、チーム色以外の高彩度・白色成分から判定する。試合開始直後は特殊ゲージが現実的に溜まらないため false 側に抑制する。
- 平滑化: `SquidLampTracker` が直近フレームでスロット状態を投票し、インク色は試合中に急に別色へ飛ばないようロックする。
- 失敗時: HUD自体の信頼度が低い場合は `non_match` にする。

## サンプル評価データ

`data/eval_samples/annotations.jsonl` に20フレーム分の正解ラベルがある。画像は `data/eval_samples/images/` に保存済み。

評価:

```bash
python squid_lamp_detect/evaluate_samples.py \
  --debug-dir /tmp/squid_lamp_eval \
  --json /tmp/squid_lamp_eval/results.json
```

現在のサンプル評価:

- HUD試合/試合外: 20/20
- イカランプスロット: 72/80
- 敵スペシャルready: 14/14

スロット誤りは、リード表示や大きなオブジェクトがスロットに重なるフレームに残っている。追加改善する場合は、固定ROI内のルール判定よりスロット単位の教師あり分類器へ移すのが次の有効打。

## 学習データ作成

動画からHUDクロップと自動推定メタデータを作る:

```bash
python squid_lamp_detect/extract_training_frames.py ../videos/*.mp4 ../videos/*.mkv \
  --out-dir squid_lamp_detect/data/hud_crops \
  --sample-interval 2 \
  --with-overlay
```

`metadata.jsonl` の `label.left/right` を `alive/down/unknown` に直すと、後続で分類器学習用の教師データとして使える。

## 注意

これは初期のルールベース検出器。ブキアイコン、リード表示、背景色、スペシャル演出が強く重なる場面ではconfidenceが落ちる。高精度化する場合は `extract_training_frames.py` で作ったクロップにラベルを付け、スロット単位分類器へ置き換える。

## 動作確認メモ

構文チェック:

```bash
python -m py_compile squid_lamp_detect/squid_lamp.py \
  squid_lamp_detect/detect_video.py \
  squid_lamp_detect/extract_training_frames.py \
  squid_lamp_detect/evaluate_samples.py
```

代表フレーム:

```bash
python -m squid_lamp_detect.detect_video "../videos/2026-02-06 22-39-39.mkv" \
  --start 30 --end 30 --sample-interval 1 --no-smooth
```

このフレームでは `ally_alive=3`, `enemy_alive=2`, `ally_down=1`, `enemy_down=2`、味方色は黄緑、敵色はマゼンタとして出力される。
