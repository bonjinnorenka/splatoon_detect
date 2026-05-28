# Splatoon 3 イカランプ武器推定 お試し版

`squid_lamp_detect` とは別ディレクトリの試作。上部HUDのイカランプ枠を切り出し、`sample_data/Main Weapons` の武器画像テンプレートと照合して、スロットごとの武器名と武器カテゴリを返す。

## 使い方

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate

python weapon_lamp_detect/detect_video.py "../videos/2026-02-06 22-39-39.mkv" \
  --start 30 --end 60 --sample-interval 5 \
  --jsonl weapon_lamp_detect/output/video_predictions.jsonl \
  --csv weapon_lamp_detect/output/video_predictions.csv \
  --debug-dir weapon_lamp_detect/output/debug
```

出力は各フレーム8スロット分の候補を含むJSONL。`weapon` は個別武器名、`weapon_class` は `shooter` / `roller` / `charger` などのカテゴリ。

## 精度評価

現状の既存評価データには alive/down のラベルしかなく、実フレームの武器名ラベルはない。そのため、実動画での真の武器名精度はまだ測れない。

代わりに、`sample_data/Main Weapons` の181武器をイカランプ風の背景・位置ずれ・回転・圧縮ノイズ・一部X重畳に合成し、正解が分かる疑似データで評価する。

```bash
python weapon_lamp_detect/evaluate_synthetic.py \
  --samples-per-weapon 2 \
  --down-fraction 0.25 \
  --json weapon_lamp_detect/output/synthetic_eval.json \
  --debug-dir weapon_lamp_detect/output/synthetic_mistakes
```

この評価は「sample_data由来テンプレート照合が、HUDっぽい劣化にどれだけ耐えるか」の確認で、実動画の保証値ではない。実動画精度を出すには、既存 `annotations.jsonl` に `left_weapons` / `right_weapons` のような武器名ラベルを追加する必要がある。

今回の測定値:

- 疑似データ: 181件 (`samples-per-weapon=1`, `down-fraction=0.25`, seed=7)
- 個別武器名 top-1: 6/181 = 3.31%
- 個別武器名 top-5: 18/181 = 9.94%
- 武器カテゴリ top-1: 30/181 = 16.57%
- 武器カテゴリ top-5: 89/181 = 49.17%
- aliveのみのカテゴリ top-1: 26/135 = 19.26%
- downのみのカテゴリ top-1: 4/46 = 8.70%

この数値から見ると、現状は候補提示の試作としては動くが、武器特定器としてはまだ実用精度ではない。

## 実装メモ

- `sample_data/Main Weapons/*.png` のアルファをテンプレートマスクとして使う。
- テンプレートを複数スケール・少量回転で展開し、イカランプクロップ上のエッジ一致とグレースケール相関を組み合わせてスコア化する。
- スロット位置は既存イカランプ検出と同じ固定HUD座標を使うが、`squid_lamp_detect` 側のファイルは変更しない。
- `detect_video.py` は可能なら `squid_lamp_detect` を読み込み、alive/down/unknown のスロット状態も併記する。

## 注意

テンプレート一致型なので、実フレームでのブラー、配信圧縮、X重畳、隣接アイコンの重なり、スペシャルバッジ重畳には弱い。特に個別武器名は似た派生武器が多く、まずは `weapon_class` の方を信用し、実用化する場合は実フレームの武器名ラベルを作って評価・調整するのが必要。
