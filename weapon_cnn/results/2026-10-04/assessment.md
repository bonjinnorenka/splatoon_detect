# CPU CNN：開始HUDの武器識別・実測評価

学習135試合 / validation 21試合 / test 52試合。
candidate 173武器。testは169武器、416 match-slot、2080crop。

同一matchの全slot・全frameを同じsplitに固定。checkpoint選択はvalidationだけ。testは最終評価のみ。
単一frameは先頭frameのmatch-slot accuracy。raw全frameのaccuracyとは分けて記載する。

| 集約 | top-1 | top-3 | top-5 | class | macro top-1 | support |
|---|---:|---:|---:|---:|---:|---:|
| cnn_raw_frame | 94.95% | 96.11% | 96.78% | 96.20% | 93.79% | 2080 |
| cnn_raw_frame_mean_1frame | 98.80% | 99.52% | 100.00% | 99.28% | 97.44% | 416 |
| cnn_raw_frame_mean_3frame | 98.80% | 99.04% | 100.00% | 99.28% | 97.44% | 416 |
| cnn_raw_frame_mean_5frame | 98.80% | 99.28% | 100.00% | 99.28% | 97.44% | 416 |
| cnn_raw_frame_quality_1frame | 98.80% | 99.52% | 100.00% | 99.28% | 97.44% | 416 |
| cnn_raw_frame_quality_3frame | 98.80% | 99.28% | 100.00% | 99.28% | 97.44% | 416 |
| cnn_raw_frame_quality_5frame | 98.80% | 99.28% | 100.00% | 99.28% | 97.44% | 416 |

主指標のmatch bootstrap 95%区間: 97.84%–99.76%。
8武器すべて正解だった試合: 47/52 = 90.38%（不完全な0試合は除外）。slot単位の正解率とは異なる。

## 混同（主指標）

- スプラチャージャーコラボ → .96ガロン爪: 1件
- スプラスコープコラボ → スプラチャージャーコラボ: 1件
- オーダーブラシ レプリカ → オーダーワイパー レプリカ: 1件
- オーダーワイパー レプリカ → オーダーブラシ レプリカ: 1件
- スプラチャージャー → スプラスコープ: 1件

## 評価範囲・制限

- 1試合だけの武器はtrain専用。別試合での精度は未測定。
- 未収集武器はモデルのcandidateに含めない。未知武器の検出は未評価。
- cross-match評価であり、別録画日・別session・別動画への汎化を保証しない。過去に確認したOBS/ライブ記録の固定分割で、完全未観測の新録画ではない。
- 自動alive/down/unknownで主評価を除外しない。shapeをdownと誤認する場合がある。state別frame指標はraw reportに保存する。
- softmax / marginは記録するが、confidenceの校正・Unknown閾値は未実施。
- 過去の58武器template評価とは候補数・splitが違うため、精度の直接比較はできない。

train-only singleton: オーダーブラスター レプリカ, オーダーチャージャー レプリカ, オーダーマニューバー レプリカ, オーダーストリンガー レプリカ

未学習catalog武器: オーダーシェルター レプリカ

checkpoint epoch: 58 / validation選択。
8 cropのCPU推論: median 2.87ms（crop入力・前処理済みtensor、warm実行）。
torch 2.14.1+cpu / threads 8。
