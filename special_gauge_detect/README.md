# Splatoon 3 スペシャルゲージ認識MVP

自プレイヤーの右上スペシャルゲージだけを読む軽量OpenCV実装です。MVPでは残量%を追わず、実況側が使う `ready / not_ready / using / unknown` とイベント検出を優先します。

## セットアップ

`private` で既存の仮想環境を有効化します。

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
```

## 使い方

動画からJSONLを出す例:

```bash
python -m special_gauge_detect.detect_video "../videos/2026-02-01 21-18-24.mp4" \
  --start 230 --end 250 --sample-interval 0.25 \
  --jsonl special_gauge_detect/output/sample.jsonl \
  --csv special_gauge_detect/output/sample.csv \
  --debug-dir special_gauge_detect/output/debug
```

標準出力に出す場合は `--jsonl` と `--csv` を省略します。

## 出力

各フレームの主な形式:

```json
{
  "special_gauge": {
    "state": "ready",
    "confidence": 0.93,
    "roi_visible": true,
    "timestamp_ms": 184230
  },
  "events": [
    {
      "type": "special_ready",
      "confidence": 0.91
    }
  ]
}
```

`state` は `unknown`, `not_ready`, `near_ready`, `ready`, `using`。`near_ready` は補助扱いで、実況の強い断定には `ready` のみを使う前提です。

## 死亡検知との連携

Python APIでは `SpecialGaugeTracker.handle_external_event()` に死亡イベントを渡します。

```python
events = tracker.handle_external_event({
    "type": "player_death",
    "timestamp_ms": 185000,
})
```

CLIではJSONまたはJSONLの死亡イベントを渡せます。

```bash
python -m special_gauge_detect.detect_video "../videos/2026-02-01 21-18-24.mp4" \
  --death-events death_events.jsonl
```

死亡時点または直前2秒以内に `ready` 履歴があれば `death_with_special_ready` を出します。

## 実装方針

- 右上ROIを解像度比率で切り出す。
- ROI内で円形ゲージの黒い中心と外周UIを探す。
- ready時の白/黄色系の全周発光と外周リングを強めに評価する。
- 直近5フレーム中4フレームを基本に平滑化し、単発フレームではイベントを確定しない。
- `SpecialGaugeTracker` が状態履歴、`special_ready`, `special_used`, `death_with_special_ready`, `special_ready_unused_too_long` を管理する。

## 検証

代表サンプルと状態遷移イベントの簡易検証:

```bash
python -m special_gauge_detect.validate_samples
```

構文チェック:

```bash
python -m py_compile special_gauge_detect/special_gauge.py \
  special_gauge_detect/detect_video.py \
  special_gauge_detect/validate_samples.py
```
