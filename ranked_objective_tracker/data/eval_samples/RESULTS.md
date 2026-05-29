# Ranked objective tracker evaluation

2026-05-28時点の評価セットです。

## Dataset

- 合計30フレーム
- `train`: 11フレーム、22 side-countラベル
- `eval`: 19フレーム、32 side-countラベル
- 4ルールを含む:
  - `splat_zones`
  - `tower_control`
  - `rainmaker`
  - `clam_blitz`
- カウントなしのobjective専用注釈を含む:
  - ガチアサリの画面内パワーアサリ陽性
  - ガチアサリの背景誤検出抑制用陰性
  - ガチヤグラの背景誤検出抑制用陰性
  - eval splitのガチエリア/ガチヤグラ時系列評価用フレーム
  - ガチヤグラの敵push、ガチアサリの敵ゴールopen評価用フレーム
  - ガチヤグラの左カウント `81` がROI端で `8` に落ちる回帰サンプル

## Baseline

初期実装の結果:

コマンド:

```bash
python -m ranked_objective_tracker.validate_samples --split eval
```

結果:

- side count: 13/18 = 72.22%
- our/enemy count: 13/18 = 72.22%
- leader: 7/9 = 77.78%

主な失敗:

- ガチエリア/ホコの左カウントで `5/6` を `8/1` に誤分類する。
- ガチヤグラは通常カウントと表示位置が違い、左側カウントROIがまだ弱い。
- カウント専用テンプレート学習は追加済みだが、現在の自動glyph抽出はノイズが多く、eval精度はbaselineより悪い。

## Improved

2026-05-28の改善後結果:

```bash
python -m ranked_objective_tracker.validate_samples --split eval
python -m ranked_objective_tracker.validate_samples --split train
python -m ranked_objective_tracker.validate_samples --split all
```

- eval side count: 32/32 = 100.00%
- eval our/enemy count: 32/32 = 100.00%
- eval leader: 16/16 = 100.00%
- train side count: 22/22 = 100.00%
- all side count: 54/54 = 100.00%
- all leader: 27/27 = 100.00%
- static objective state: 18/18 = 100.00%
- static goal state: 9/9 = 100.00%
- static marker/position: 36/36 = 100.00%
- sequence progress direction: 15/15 = 100.00%
- sequence objective state: 15/15 = 100.00%
- sequence smoothed count: 46/46 = 100.00%
- full coverage gate: pass
- 720p相当の検出平均時間: 19.51ms/frame

ルール別カバレッジ:

| rule | annotations | counter samples | side-count labels | static objective labels | goal labels | marker labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `splat_zones` | 6 | 6 | 12 | 6 | 0 | 6 |
| `tower_control` | 8 | 7 | 14 | 6 | 0 | 6 |
| `rainmaker` | 6 | 6 | 12 | 6 | 0 | 6 |
| `clam_blitz` | 10 | 8 | 16 | 0 | 9 | 18 |

ルール別正答率:

| rule | side-count | count | leader | static objective/goal | marker | sequence progress | sequence count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `splat_zones` | 12/12 | 12/12 | 6/6 | 6/6 | 6/6 | 4/4 | 12/12 |
| `tower_control` | 14/14 | 14/14 | 7/7 | 6/6 | 6/6 | 4/4 | 14/14 |
| `rainmaker` | 12/12 | 12/12 | 6/6 | 6/6 | 6/6 | 3/3 | 8/8 |
| `clam_blitz` | 16/16 | 16/16 | 8/8 | 9/9 | 18/18 | 4/4 | 12/12 |

時系列objective状態の測定対象:

- `splat_zones`: `our_control` 2件、`enemy_control` 1件、`unknown` 1件
- `tower_control`: `our` 1件、`enemy` 2件、`neutral` 1件
- `rainmaker`: `our_carrier` 1件、`enemy_carrier` 1件、`unknown` 1件
- `clam_blitz`: `ourGoal=open/enemyGoal=closed` 1件、`ourGoal=closed/enemyGoal=open` 2件、`closed/closed` 1件

保存結果:

- `improved_eval_result.json`
- `improved_train_result.json`
- `improved_all_result.json`

主な改善:

- カウンタ用マスクを白系/高輝度文字に絞り、黄色や色付きのカウントプレートを数字として拾わないようにした。
- `00` や `018` のような先頭ゼロ候補を無効化した。
- 非ヤグラではヤグラ用ROIをフォールバック扱いにし、通常カウント候補と同列比較しないようにした。
- 時系列評価では、カウントがリセットされた区間を分割し、教師カウントの5秒差分から `progressDirection` とルール別 objective state を検証するようにした。
- 評価JSONに `coverage` / `per_rule_annotation_result` / `per_rule_sequence_result` を追加し、4ルールそれぞれの教師データ件数と正答率を確認できるようにした。
- 時系列評価JSONに `sequenceCoverage` を追加し、`our_control` / `enemy_carrier` など、カウント差分から教師化したobjective状態の分布を確認できるようにした。
- `--require-full-coverage` を追加し、4ルールと主要な時系列objective状態のcoverage不足を検証ゲートにできるようにした。
- `static_coverage_gaps` を追加し、静止フレーム側で不足しているobjective/marker教師データを次の作業候補として見えるようにした。
- eval splitでも4ルールすべての時系列評価が走るよう、ガチエリアとガチヤグラに連続フレーム注釈を追加した。
- 平滑化の急減チェックを、候補フレームの古さではなく前回安定カウントからの経過時間で判定するようにした。
- ガチヤグラのカウントがROI端に近い単独桁として読まれた場合だけ、端方向へ小さく拡張した候補を追加し、`81` が `8` に落ちるケースを回帰テストに入れた。
- 静止フレームにも rule別 objective ラベルを追加し、`zoneControl` / `towerOwner` / `rainmakerState` / `goalState` / marker visible/offscreen を測定対象にした。
- ガチアサリで背景の縦長成分をパワーアサリ候補として拾わないよう、マーカー候補のアスペクト比条件を締めた。
- objective search ROIの端に接する候補を棄却し、画面端の看板/壁をヤグラやパワーアサリとして拾う誤検出を抑制した。
- 画面内パワーアサリの陽性注釈を追加し、visible時の `x`/`y` 評価も測定対象に入れた。
- ガチアサリのスコアプレートハイライトから、単独フレームでも `enemyGoal=open` を補助判定できるようにした。
- ガチアサリの時系列objective期待値は、明示的なopen静止ラベルがある場合にそれを優先し、開放中だがカウント差分がないフレームを正しく評価するようにした。
- evalラベル `zones_135_2500` の右カウントを、拡大目視確認に基づき `14` から `74` に修正した。

## Trained Counter Templates

`train` split から `ranked_objective_tracker/templates/counter_digits` を生成した場合の結果:

- eval side count: 29/32 = 90.62%
- train side count: 20/22 = 90.91%
- all side count: 49/54 = 90.74%
- all leader: 25/27 = 92.59%
- all static objective/goal/marker: 63/63 = 100.00%
- sequence smoothed count: 41/46 = 89.13%
- sequence progress direction: 12/12 = 100.00%

保存結果:

- `trained_template_eval_result.json`
- `trained_template_train_result.json`
- `trained_template_all_result.json`

現時点では、専用テンプレートよりも既存 `match_time_ocr/templates/digits` を使う既定設定の方が、この小規模評価セットでは高精度。

## Next Accuracy Work

- `static_coverage_gaps` に残っている、ヤグラ/ホコの画面内マーカー陽性、エリア/ヤグラ/ホコの非unknown静止objectiveラベルを増やす。
- ガチアサリは `enemyGoal=open` 以外のopen/breaking静止ラベルも別動画で増やす。
- 専用テンプレート学習はmedian 1枚ではなく、複数テンプレートまたはHOG+SVMに切り替える。
