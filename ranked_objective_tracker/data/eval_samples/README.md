# Ranked objective eval samples

`extract_eval_frames.py` で作った `annotations.draft.jsonl` を `annotations.jsonl` にコピーし、`expected` を目視で修正して使います。

`validate_samples.py` は静止フレームのカウント/リード/objective/marker評価に加えて、同一動画・同一ルールの注釈を時系列に並べた sequence 評価も実行します。sequence 評価では、教師カウントの5秒差分から `progressDirection` とルール別 objective state を導出して検証します。

`expected.counter` を省略したobjective専用注釈も使えます。その場合はカウント/sequence評価からは除外され、objective/goal/marker評価だけに使われます。

検証結果には `coverage` とルール別結果も出力されます。4ルールすべての教師データ件数と、各ルールの正答率が総合値とは別に確認できます。

`--require-full-coverage` を付けると、4ルールすべてに加えて主要な時系列objective状態のcoverageもゲートできます。

`static_coverage_gaps` は、静止フレームだけではまだ薄いobjective/marker教師データを示します。これは失敗条件ではなく、次に追加する教師データ候補のリストです。

最小形式:

```json
{"id":"sample_001","image":"images/sample_001.jpg","rule":"tower_control","ally_side":"right","expected":{"counter":{"ourCount":68,"enemyCount":42,"leader":"enemy"},"objective":{"kind":"tower_control","towerOwner":"neutral","screenPosition":{"visible":false,"x":null,"y":null,"distanceClass":"offscreen"}}}}
```
