# Splatoon match time OCR

Splatoon の試合画面上部にある残り時間を読み取るための軽量 OCR です。

- 上中央のタイマー領域だけを切り出します。
- `videos` 配下のキャプチャから数字テンプレートを生成します。
- 数字はテンプレート照合、見失った短い区間は時系列補完で埋めます。
- `延長中!` はオレンジ色の延長表示として別扱いで検出します。

## セットアップ

`private` で既存の仮想環境を有効化して使います。

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
```

## 学習

テンプレートは `data/training_segments.json` に書いた動画区間から生成します。
初期設定では `videos/2026-02-01 21-18-24.mp4` と `videos/2026-02-06 22-39-39.mkv` の代表試合を使います。

```bash
python match_time_ocr/train_templates.py
```

生成物は `match_time_ocr/templates/digits/*.png` と `metadata.json` です。

## 検出

CSV に出す例です。

```bash
python match_time_ocr/detect_video.py "../videos/2026-02-01 21-18-24.mp4" \
  --sample-interval 1 \
  --csv match_time_ocr/output/video1_timer.csv
```

標準出力に出したい場合は `--csv` を省略します。主な列は以下です。

- `kind`: `time` または `overtime`
- `text`: `4:53` や `延長中!`
- `seconds`: 通常時間の秒数。延長中は空です。
- `confidence`: テンプレート照合または延長表示検出の信頼度
- `inferred`: 短い欠落を前後の流れから補完した行なら `True`

処理負荷を下げたい場合は `--sample-interval 5` のように間隔を広げます。フルフレーム OCR はせず、固定 ROI だけを見るため軽量です。

## 検証

代表フレームの通常時間と延長中を確認します。

```bash
python match_time_ocr/validate_samples.py
```
