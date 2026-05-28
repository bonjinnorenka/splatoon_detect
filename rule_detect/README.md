# Splatoon 3 rule detector

試合開始演出のルール名表示を1回だけ読み、`match.rule` に固定保存するためのMVP実装です。

## 対象ルール

- `turf_war`: ナワバリバトル
- `splat_zones`: ガチエリア
- `tower_control`: ガチヤグラ
- `rainmaker`: ガチホコバトル
- `clam_blitz`: ガチアサリ
- `unknown`: 不明

## セットアップ

`private` で既存の仮想環境を有効化して使います。

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
```

## テンプレート生成

`data/training_segments.json` のラベル付き開始演出フレームから、前処理済みROIテンプレートを生成します。

```bash
python rule_detect/train_templates.py
```

生成物は `rule_detect/templates/jp_1080p/*.png` です。

## 検出

開始演出の時刻範囲を指定して読みます。

```bash
python rule_detect/detect_video.py "../videos/2026-02-06 22-39-39.mkv" \
  --start 0 --end 2 --sample-interval 0.5
```

出力例:

```json
{
  "rule": "tower_control",
  "confidence": 0.93,
  "source": "template",
  "locked": true,
  "frame_time_ms": 0
}
```

`confidence < 0.78` の場合は `unknown` にします。誤分類を避けるため、試合中の再認識はしません。

## 検証

代表サンプルを確認します。

```bash
python rule_detect/validate_samples.py
```

学習済みテンプレート全体、leave-one-out、非開始フレームをまとめて確認する場合:

```bash
python rule_detect/evaluate_samples.py
```

## 実装メモ

- 固定ROI: `(0.36, 0.22, 0.64, 0.62)`
- 前処理: ROI切り出し、幅320pxへリサイズ、グレースケール化、正規化、軽いぼかし
- 照合: `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`
- 戻り値は `rule`, `confidence`, `source`, `locked`, `frame_time_ms` を含むJSON
