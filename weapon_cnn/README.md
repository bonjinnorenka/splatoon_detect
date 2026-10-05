# 開始HUDの武器識別 CNN（CPU）

既存template PoCとは別の追加実験です。`live_weapon_collect` と動画由来cropを再利用し、試合開始時の8武器を識別する小型residual CNNを学習します。GPU・torchvision・事前学習モデルは使いません。追加の主依存はCPU版PyTorchだけで、既存detector・ラベラー・top-5候補の方式は変更しません。

## 今回の実測結果

学習済みモデル: [opening_cnn_20261004.pt](models/opening_cnn_20261004.pt)（約1.7MB）。validationで選択したepoch 58をそのまま保存し、testを含めた再学習はしていません。

**学習に使っていない52試合・169武器・416スロット**で測定。候補は173武器です。

| 開始時に使うframe | exact top-1 | top-3 | top-5 |
|---|---:|---:|---:|
| 先頭1frame | 98.80% | 99.52% | 100.00% |
| 先頭3frame・quality平均 | 98.80% | 99.28% | 100.00% |
| 先頭5frame・quality平均（主指標） | 98.80%（411/416） | 99.28% | 100.00% |

8武器すべて正解だった試合は **47/52 = 90.38%**。武器別macro top-1は97.44%、weapon class accuracyは99.28%。slot単位主指標のmatch bootstrap 95%区間は97.84%–99.76%です。先頭1frameから5frameに増やしても今回のtop-1は改善しませんでした。5frameそれぞれを別標本として数えるraw frame accuracyは94.95%で、先頭1frameの98.80%とは異なります。

残った5件はチャージャー／スコープ系3件、Orderブラシ／ワイパー系2件です。高confidenceの誤りもあるため、scoreを正解保証として扱わないでください。

- [詳細評価・制限](results/2026-10-04/assessment.md)
- [全169武器のaccuracyと試合数CSV](results/2026-10-04/per_weapon.csv)
- [誤判定5件の画像とcontext](results/2026-10-04/errors.md)
- [固定分割](results/2026-10-04/protocol.json) / [学習設定](results/2026-10-04/settings.json) / [学習履歴](results/2026-10-04/history.json)

4種類のOrder武器は1試合だけなのでtrain専用・精度未評価。オーダーシェルター レプリカは未学習です。同一収集session/動画の別試合を使うcross-match評価であり、新しい録画sessionへの精度を保証するものではありません。

## 環境

リポジトリrootで実行します。WSL:

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
python -m pip install -r weapon_cnn/requirements.txt
```

Windows PowerShellは、UNC作業directoryを失うpyenvのbatではなく実際のpython.exeを使ってください:

```powershell
Set-Location "\\wsl.localhost\Ubuntu\home\ryokuryu\splat_2\private"
$cnnPython = "$env:LOCALAPPDATA\splatoon-live-venv\Scripts\python.exe"
& $cnnPython -m pip install -r weapon_cnn/requirements.txt
```

依存は [PyTorch公式CPU wheel](https://pytorch.org/get-started/locally/) を指定しています。実測に使ったversionは `2.14.1+cpu`。以降の `python` はWindowsでは `& $cnnPython` に置き換えてください。CNNはカメラを開きません。

## 1. dataset生成

```bash
python -m weapon_cnn.data prepare datasets/weapon_opening/2026-10-04 \
  --output weapon_cnn/data/dataset_20261004
```

既存 `export_dataset`、人力status/name検証、source PNGのSHA-256、`calibrated_crop`、動画の開始cropを再利用します。8310crop・208試合・173武器。ライブの156確定試合と動画の52ラベル付き試合を使用し、対象外・未確定・非labeled slotは学習しません。全slotがOccludedの1動画試合は正解武器がないため対象外です。

開始の5frameだけを保存します。alive/down/unknownは自動stateとして記録し、shapeの誤分類で正解データを捨てません。crop PNGはコピー、contextは確認用JPEG。大きなライブcontextの一時コピーは準備完了後に削除し、元データは変更しません。元動画は不要です。

出力先が存在すれば上書きしません。再実行は新しいdirectoryを指定してください。`weapon_cnn/data/` はGit対象外です。

## 2. 試合単位の分割を固定

```bash
python -m weapon_cnn.data split weapon_cnn/data/dataset_20261004 \
  --output weapon_cnn/data/protocol_20261004.json --seed 20261004
```

train 135試合、validation 21試合、test 52試合。同一試合の全slot・全frameは同一splitに置きます。同一元frameが別matchへ重複していたら、手動確認を要求して分割を拒否します。クラスのtest coverageを分割時に優先し、全173candidateがtrainに残ることを検証します。分割の選択にモデルの精度は使いません。

- test: 169武器・416match-slot・2080crop。
- singletonの4種類のOrder武器はmatch全体をtrainに固定。cross-match精度として報告しません。
- オーダーシェルター レプリカは未収集なのでcandidateに含めません。
- 同一録画・収集sessionの別試合が複数splitに入るため、これはcross-matchでありcross-session/videoではありません。

## 3. CPU学習

```bash
python -m weapon_cnn.train weapon_cnn/data/dataset_20261004 \
  --protocol weapon_cnn/data/protocol_20261004.json \
  --output weapon_cnn/data/run_20261004 \
  --epochs 60 --batch-size 96 --threads 8 --seed 20261004
```

414,669パラメータ。既存の134×108 canonical cropと武器ROIを使い、RGB 64×96＋局所輝度の4channelへ変換します。RGBの色を一律に消さず、形状と派生の色の両方を利用します。

AdamW、軽い位置/scale/回転/反転・色/blur/白化augmentation、label smoothing。augmentationはtrainのみです。samplingは武器→試合→slotを均等にし、同一slotの5frameでは既存のlabel-blind画質weightを使用します。頻出武器や同一試合の連続frameでrare武器が埋もれるのを防ぎます。

`best.pt` はvalidationの先頭5frame・quality平均のmacro accuracy、micro accuracy、log lossの順で選択します。trainerはtest画像を読みません。`history.json` に進捗・選択epochを保存し、checkpointは原子的に更新します。CUDAは使用しません。

## 4. 最終testと精度

```bash
python -m weapon_cnn.evaluate weapon_cnn/data/dataset_20261004 \
  --protocol weapon_cnn/data/protocol_20261004.json \
  --checkpoint weapon_cnn/data/run_20261004/best.pt \
  --output weapon_cnn/data/evaluation_20261004 --threads 8
```

checkpointとdataset/protocolのhashを照合します。モデル固定後のtestだけを測定します。モデル修正や閾値調整にtestを使う場合は独立した最終評価とは呼べません。

hashには生成時metadataも含まれるため、再生成datasetに過去の固定protocolを流用しません。再現実験では新しくprepare → split → train → evaluateします。配布checkpointの単体推論はdatasetの有無や保存場所に依存しません。

- `assessment.md`: 1/3/5frameのtop-1/3/5、class、macro、混同・制限。
- `report.json`: 武器別support・accuracy、state別、混同行列、score・margin分布、match単位bootstrap 95%区間、CPU時間。
- `per_weapon.csv`: 日本語武器別accuracyとtrain/validation/test試合数。
- `predictions.jsonl`: crop/context、expected/predicted、top-k、margin、使用frame。

主指標は先頭5frameのquality-weighted probability平均です。先頭1/3/5frameの単純平均も比較します。後から見やすいframeを探して補充しません。raw全frame accuracyと先頭1frameのmatch-slot accuracyは別指標です。

softmaxは未校正のprobabilityで、Unknown閾値は設定していません。173武器を学習したことと173武器の汎化精度を保証できることは別です。既存の58武器template報告とはcandidate・split・supportが違うため直接比較できません。

実測のCPU forwardは8cropでmedian 2.87ms（8 threads）。画像decode・crop抽出・OpenCV ROI/resizeを除く値で、end-to-endの所要時間ではありません。

必要なら、固定checkpointと結果をGit対象の小さな成果物へコピーできます。学習・再推論・元ラベル変更は行いません。既存出力先は上書きしません:

```bash
python -m weapon_cnn.publish weapon_cnn/data/evaluation_20261004 \
  --run weapon_cnn/data/run_20261004 \
  --output weapon_cnn/results/my_experiment \
  --model-output weapon_cnn/models/my_experiment.pt
```

`results/2026-10-04/` は今回の固定結果です。`predictions.jsonl` は全行のtop-5を残し、巨大な173候補全scoreを省いたViewer用形式です。`artifacts.json` にmodel・protocol・出力ファイルのSHA-256を保存しています。

## 5. 誤判定Viewer

既存Viewerをそのまま使えます:

```bash
python -m weapon_lamp_detect.view_errors weapon_cnn/results/2026-10-04 --port 8784
```

<http://127.0.0.1:8784/>。初期表示は5frame quality集約の誤判定です。crop・context・使用frame・正解/予測・top-k・score・marginを確認できます。source/contextはdataset内にexport済みなので元動画は不要です。Viewerの分析保存は元ground truthを変更しません。

ブラウザViewerは`report.json`に記録したprepared datasetの保存場所が存在する環境で使用します。モデル・評価結果だけを別PCへコピーした場合も、[誤判定画像一覧](results/2026-10-04/errors.md)はdatasetなしで確認できます。

## 6. 推論

配布した学習済みモデルだけでcropを認識できます。datasetや元動画は推論時に不要です:

```bash
python -m weapon_cnn.predict "path/to/weapon_crop.png" \
  --checkpoint weapon_cnn/models/opening_cnn_20261004.pt --top-k 5 --threads 4
```

JSONで英語canonical名・日本語名・class・probability・marginを返します。Python APIは `WeaponCNNMatcher.predict_crop()` / `predict_crops()`。動画/cameraからの固定cropは既存 `calibrated_crop` を使ってください。収集アプリへの自動組み込みやground truthの自動変更は行っていません。

## テスト

```bash
python -m unittest weapon_cnn.test_cnn -v
```

一時的な合成fixtureだけで分割・リーク拒否・画像hash・CPU学習/推論・Viewer形式などを検証します。合成fixtureの精度は実データの精度として扱いません。
