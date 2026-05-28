# Splatoon 3 Ranked Objective Tracker MVP

ガチマッチ4ルールの上部HUDカウント、最低限の目的物状態、実況向けイベントを読む軽量OpenCV実装です。

## セットアップ

`private` で既存の仮想環境を有効化して使います。

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
```

## 検出

```bash
python -m ranked_objective_tracker.detect_video "../videos/2026-02-06 22-39-39.mkv" \
  --rule tower_control \
  --start 20 --end 40 \
  --sample-interval 0.2 \
  --jsonl ranked_objective_tracker/output/sample.jsonl \
  --csv ranked_objective_tracker/output/sample.csv \
  --debug-dir ranked_objective_tracker/output/debug
```

標準出力にJSONを出す場合は `--jsonl` と `--csv` を省略します。

## 出力

主な形式:

```json
{
  "timestampMs": 84200,
  "rule": "tower_control",
  "counter": {
    "ourCount": 68,
    "enemyCount": 42,
    "leader": "enemy",
    "progressDirection": "enemy_count_decreasing",
    "recentDelta": {
      "ourCountDelta5s": 0,
      "enemyCountDelta5s": -9
    },
    "freshness": "fresh"
  },
  "objective": {
    "kind": "tower_control",
    "towerOwner": "enemy",
    "towerProgress": {
      "side": "our_side",
      "progressPercent": 58
    },
    "screenPosition": {
      "visible": false,
      "x": null,
      "y": null,
      "distanceClass": "offscreen"
    }
  },
  "events": [],
  "quality": {
    "overallConfidence": 0.78
  }
}
```

## 実装方針

- カウントは固定ROI、数字テンプレート照合、時系列平滑化で読む。
- 初期MVPでは `match_time_ocr/templates/digits` を数字テンプレートとして再利用する。ガチカウント専用テンプレートを作った場合は `--templates-dir` で差し替える。
- ヤグラ/ホコ/ガチアサリの画面内マーカーは保守的に検出し、曖昧なら `offscreen` にする。
- ガチエリアは位置を返さず、カウント進行から支配側だけを推定する。
- ガチアサリは全アサリ位置を扱わず、ゴール状態と見えているパワーアサリ候補だけを返す。

## 評価データ作成

```bash
python -m ranked_objective_tracker.extract_eval_frames ../videos/*.mkv \
  --rule tower_control \
  --out-dir ranked_objective_tracker/data/eval_samples \
  --sample-interval 10 \
  --with-overlay
```

生成される `annotations.draft.jsonl` を `annotations.jsonl` にコピーし、`expected` を目視で修正して評価に使います。

## 検証

```bash
python -m ranked_objective_tracker.validate_samples
```

`data/eval_samples/annotations.jsonl` があれば画像ラベル評価を実行します。未作成でも、平滑化とイベント基礎の合成チェックは実行されます。

4ルールと主要な時系列objective状態が揃っていることまでゲートする場合:

```bash
python -m ranked_objective_tracker.validate_samples --split all --require-full-coverage
```

4ルール分の初期教師データと測定結果は `data/eval_samples/` にあります。

現在の既定設定では、注釈済み29フレーム/52 side-countラベルで以下です。

- eval side-count: 30/30、100.00%
- train side-count: 22/22、100.00%
- all side-count: 52/52、100.00%
- all leader: 26/26、100.00%
- static objective state: 17/17、100.00%
- static goal state: 9/9、100.00%
- static marker/position: 35/35、100.00%
- sequence progress direction: 14/14、100.00%
- sequence objective state: 14/14、100.00%
- sequence smoothed count: 44/44、100.00%
- 720p相当の検出平均時間: 19.51ms/frame

改善前baselineは eval side-count 13/18、72.22%でした。詳細は `data/eval_samples/RESULTS.md` を参照してください。

静止フレーム評価ではカウント/リードに加えて、`zoneControl` / `towerOwner` / `rainmakerState` / `goalState` / marker visible/offscreen と、visible時の正規化 `x`/`y` を検証します。

`sequence` 評価では、同じ動画/ルール/味方側の注釈を時系列に並べ、カウントリセットを区切りとして分割します。教師カウントの5秒差分から `progressDirection` と rule別 objective state を検証します。

検証JSONには `coverage` / `per_rule_annotation_result` / `per_rule_sequence_result` も含めています。4ルールそれぞれについて、教師データ数、side-count/count/leader、static objective/goal/marker、sequence progress/count の内訳を確認できます。`sequenceCoverage` では、カウント差分から教師化した `our_control` / `enemy_carrier` などの時系列objective状態の分布も見られます。

`coverage_requirements` は、評価対象splitに4ルール分の教師データと時系列評価が入っているかを示します。`--require-full-coverage` を付けた場合は、主要な時系列objective状態が欠けていれば終了コードも失敗になります。

`static_coverage_gaps` は、静止フレーム側でまだ薄い教師データを列挙します。現在は時系列では主要状態を測れていますが、静止ラベルではヤグラ/ホコのvisible marker陽性や、エリア/ヤグラ/ホコの非unknown目的状態が不足しています。

カウント専用テンプレートを作る場合:

```bash
python -m ranked_objective_tracker.train_counter_templates --splits train
python -m ranked_objective_tracker.validate_samples \
  --templates-dir ranked_objective_tracker/templates/counter_digits \
  --split eval
```

現時点では、専用テンプレートは all side-count 47/52、90.38%です。通常検証では `--templates-dir` を付けない既定設定の方が高精度です。

構文チェック:

```bash
python -m py_compile ranked_objective_tracker/ranked_objective.py \
  ranked_objective_tracker/detect_video.py \
  ranked_objective_tracker/validate_samples.py \
  ranked_objective_tracker/extract_eval_frames.py \
  ranked_objective_tracker/train_counter_templates.py
```
