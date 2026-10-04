# 開始HUDのリアルタイム収集アプリ

既存の動画ラベラー `weapon_lamp_detect/` とは別のアプリです。カメラとして認識されるキャプチャボード、またはOBS仮想カメラを入力にします。目的は**開始直後の武器アイコンを少数保存し、試合終了後に人間が8武器を入力すること**です。試合中の継続推論・CNN・GPU・全動画録画は行いません。

## 1. 環境

- Python 3.10以上、OpenCV、NumPy、scikit-learn。依存は既存PoCと同じです。
- このディレクトリだけではなく、既存 `start_detect/`、`squid_lamp_detect/`、`match_time_ocr/`、`weapon_lamp_detect/` が含まれたリポジトリが必要です。
- `wepons.json` は既存の探索規則（リポジトリの親）を使用し、なければ既存の同梱catalogを使用します。独自配置は `--catalog "path/to/wepons.json"` で指定できます。
- top-5候補は既定で最新のOBS hybrid方式です。下記の見本準備が必要です。見本がなくても収集・人力入力は動き、公式方式へ黙って切り替わることはありません。旧公式方式を明示する場合のみ `--candidate-matcher official --template-dir "path/to/Main Weapons"` を指定します。
- カメラ入力はまずWindowsのPythonで試してください。WSL側の `/dev/video*` は別途利用可能な環境に限ります。このアプリはUSB転送・ドライバ設定を変更しません。

以降、コマンドはリポジトリのルート（`live_weapon_collect/` と `weapon_lamp_detect/` が見える場所）から実行します。

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r live_weapon_collect/requirements.txt
```

PowerShellでactivateが禁止される場合は、実行ポリシーを変更せず、各コマンドの `python` を `.\.venv\Scripts\python.exe` に置き換えられます。

### WSLのソースをWindows Pythonから実行する場合

`python` がpyenvのbatラッパーだと、UNCの作業ディレクトリからCMDを起動して別の場所へ移動し、`ModuleNotFoundError` になることがあります。Linuxのvirtualenvは使わず、Windowsの **python.exe を直接** 実行します。venvと保存先もWindowsのローカルディスクに置きます。

```powershell
Set-Location "\\wsl.localhost\Ubuntu\home\ryokuryu\splat_2\private"
$basePython = "C:\Users\ryo\.pyenv\pyenv-win\versions\3.13.2\python.exe"
$liveVenv = "$env:LOCALAPPDATA\splatoon-live-venv"
if (-not (Test-Path "$liveVenv\Scripts\python.exe")) { & $basePython -m venv $liveVenv }
$livePython = "$liveVenv\Scripts\python.exe"
& $livePython -m pip install -r live_weapon_collect/requirements.txt
$liveDataDir = "$env:LOCALAPPDATA\splatoon-live-data\session"
& $livePython -m live_weapon_collect.app --source 0 --backend dshow --fps 30 --session-dir $liveDataDir --port 8780
```

Pythonの配置が異なる場合は `$basePython` を変更してください。再起動時は同じ `$liveDataDir` を指定すると保存済み画像・入力を引き継ぎます。コード変更の反映には古いアプリの停止・再起動とブラウザの再読み込みが必要です。UNC上のロックエラーを避けるためにも、保存先には上記のWindowsローカルディレクトリを推奨します。

このワークスペースのWSLで既存環境を使う場合:

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
```

## 2. カメラ入力で起動

Windows:

```powershell
python -m live_weapon_collect.app --source 0 --backend dshow --session-dir "live_weapon_collect/data/session" --port 8780
```

ブラウザで <http://127.0.0.1:8780/> を開きます。既存ラベラーの8765番とは独立しています。

- `--source 0` はカメラ番号です。別のデバイスが映ったら1、2等へ変えて再起動してください。
- DSHOWで開けなければ `--backend msmf` または `--backend auto` を試してください。
- 1920×1080 / 60fpsを要求しますが、設定できる形式はドライバ依存です。UIの「入力デバイス・保存先」で実際の解像度・報告fps・backendを確認してください。60fpsの武器照合をする意味ではありません。
- 必要な機器だけ `--fourcc MJPG` / `--fourcc YUY2` 等を指定します。PNG保存は追加の非可逆圧縮をしませんが、入力段階のMJPEG・スケーリング・色変換を取り消すことはできません。
- 入力解像度に枠が正しく対応しているか、最初の試合で必ず確認してください。左右反転・黒帯・配信オーバーレイは避けてください。
- ポートが使用中なら `--port 8781` 等へ変更します。同じsession-dirで同時に2収集プロセスを動かすことは禁止しています。

Linuxでカメラが利用可能な場合:

```bash
python -m live_weapon_collect.app --source /dev/video0 --backend v4l2 --session-dir "live_weapon_collect/data/session"
```

### OBSと同時使用

キャプチャボードの直接入力がOBSと競合する場合は、OBSの「仮想カメラ開始」を使用し、そのカメラ番号を `--source` に指定してください。OBS側の出力はゲーム画面だけのSourceまたはSceneにするとcropを合わせやすくなります。ボードの同時接続可否は機器依存で、二重接続を保証していません。

形式設定がbackendごとに異なる点は [OpenCV公式説明](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html)、OBS出力の設定は [OBS仮想カメラガイド](https://obsproject.com/kb/virtual-camera-guide) を参照してください。

## 3. 自動収集と試合終了後の入力

1. `start_detect` の既存HOG+SVM JSONでステージ紹介の開始候補を探します（既定0.5秒間隔）。
2. 候補の8秒後以降にHUD・開始残り時間（5:00または3:00付近）を確認します。alive人数は補助情報で、保存の必須条件ではありません。候補を検出した時点のステージ紹介画像は収集しません。
3. 2回連続で確認し、表示の落ち着きを待って、**1秒ごとに5枚**の全画面PNGを保存します。HUDやOCRを確認できなくても、既定で**開始候補の20秒後から5枚を「開始画像か要確認」として保存**します。どちらも最初の5秒の窓を超えて不足分を補いません。推定失敗だけで画像を捨てず、人間が確認して確定/対象外を決めます。
4. 「未確定」一覧に保存されます。**入力は試合終了後でも、何試合かまとめて後でも構いません。** 画像はその場でディスクに保存されているので、ラベル待ちで次の試合が上書きされることはありません。
5. 保存済み試合を選び、left0〜left3 / right0〜right3を一度だけ指定します。武器名は `wepons.json` の候補だけ保存できます。入力候補・保存済みラベルの表示・top-k候補・武器別一覧は日本語名です。既存評価との互換性のためJSONの内部武器名は従来のcanonical名（多くは英語）を維持します。「シューター」等のカテゴリ、未登録名、誤字はサーバー側でも拒否し、警告します。
6. 必要なら保存frameを切り替え、referenceを選び、各slotのX/Y/幅/高さを調整します。枠の調整はその試合の5枚へ適用します。
7. 「開始直後のHUD・cropを人力で確認済み」と8スロットを確認して確定します。未入力と人力で選んだUnknownは区別します。Unknown / Occluded / Skipも人力判断として保存できます。

入力変更は650ms後に下書き自動保存します。無効な名前があれば**その保存全体を拒否**し、以前の有効な保存は残します。新しい収集があっても編集中の試合へ自動切り替えしません。別タブの競合もrevisionで拒否します。ブラウザを閉じても端末が動いていれば収集は続きます。

「top-5候補」は人が明示的にクリックした場合だけラベルに採用します。候補を取得しただけではラベルや確認フラグを書きません。既定は最新の `HybridOpeningMatcher` / `blend_soft`：従来の背景除去照合75%＋局所コントラスト照合25%、逆sideの見本には0.03のペナルティを使います。左右を反転せず、表示中slotのsideを渡します。

### hybrid候補の見本準備（WSLで一度実行）

最新評価の固定設定と既存OBS見本を再利用し、元動画やLinux絶対pathがなくても推論できる小さなcrop bundleを作ります。今回の作業環境では準備済みです。別の環境で再現する場合は元の評価データが必要です。

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
python -m live_weapon_collect.hybrid_candidates prepare \
  --experiment weapon_lamp_detect/data/poc_opening_hybrid_20261002 \
  --kind full_training \
  --output live_weapon_collect/data/hybrid_candidates
```

- このsnapshotは**58武器 / 1,125枚の見本crop**です。未収集・見本なし武器は候補対象外ですが、`wepons.json` の全武器を手入力できます。UI上部に候補方式・見本数を表示します。
- 既存評価の `full_training/templates.json` に含まれる見本だけを書き出します。最近のライブ収集のラベルを勝手に取り込む処理、再学習、GTの変更は行いません。評価用のデータ分割・レポートも変更しません。
- 候補は**表示中の1枚**を照合します。評価報告の品質加重5枚集約とは別なので、その精度の数字を候補表示の精度と同一視しないでください。初回は見本の読み込みがあり、候補取得中も別プロセスで収集を継続します。
- `data/` はGit対象外です。別checkoutへ移る場合はbundle全体をコピーするか上記コマンドで準備します。中断した準備は完了扱いにせず、読み込み時にPNG・metadataのSHA-256を確認します。
- outputを上書きしません。更新する場合は新しい名前で作り、`--hybrid-dir "path/to/new bundle"` で指定してアプリを再起動します。

Windows側は従来の起動コマンドのままでhybridを使います。見本の場所だけ変更する場合:

```powershell
& $livePython -m live_weapon_collect.app --source 0 --backend dshow --fps 30 --session-dir $liveDataDir --hybrid-dir "D:\データ\hybrid_candidates" --port 8780
```

見本がない・破損した場合は候補取得時に警告します。画像収集と人力保存は止めません。保存済みラベルの内部英語名は従来どおりで、候補・入力欄の表示は日本語です。confidenceは正解確率ではなく、best-second marginとともに候補ボタンの補足に表示します。

### 自動検出が外れた場合

- 誤検出した試合は「対象外」で残してください（削除しません）。
- HUDを見つけられない候補も、既定で候補+20秒から画像を保存します。開始画像でなければ「対象外」にします。紹介画面自体を検出できなかった場合は、**開始直後に**「今の画面から手動収集」を押してください。過去の画像を遡る機能はありません。
- 手動収集は押した時点からの画像です。終了後に押して開始画像を遡って復元する機能はありません。今回、長い動画バッファ・全試合録画は実装していません。
- 自動のalive/down推定は完全ではありません。収集後の5枚をaliveだけに選別し直すことはせず、stateを保存し、人間が画像を確認できます。
- HUDで確認できれば早めに保存し、未確認の場合だけ時間による保存を行います。遷移に応じて `--fallback-offset 15` / `--fallback-offset 20` で変更できます（`hud-delay-min ≤ fallback-offset < hud-timeout`）。入力処理が遅れて35秒を超えた場合は後半画像で穴埋めせずタイムアウトします。
- 確認できた自動収集後は60秒待機します。未確認・手動・タイムアウト後は2秒だけ待機し、新しい紹介画面を検出できます。同じ紹介が表示され続けても重複保存しません。次の候補検出はHUDのnon-match判定だけには依存せず、紹介の途切れと再出現も使います。試合終了検出器は不要です。
- `capture.json` に収集直前の最大80回の検出履歴（紹介score、HUD、タイマー、alive、OCRエラー等）を保存します。UIの「入力デバイス・保存先」には現在の診断、上部には未確認時の保存までの秒数を表示します。ラベルの自動確定は行いません。

### キーボード

- 入力欄の外: `1〜8` slot選択、`←/→` 保存frame切替、`N/P` 次/前試合、`S` 保存、`U/O/K` Unknown/Occluded/Skip。
- 武器入力中: `Enter` 候補名を確認して次slot、`Tab` フォーカス移動。`Ctrl+S` は入力欄の中でも保存。
- 「直前の保存を取り消す」で履歴から復元できます。確定済み試合も「全件」「確定済み」の一覧で修正できます。
- 動画seekや再生ではなく、保存した5枚の切り替えです。

## 4. 停止・再開

収集だけ停止するならUIの「収集停止」。アプリ全体は端末の `Ctrl+C` で終了します。停止済みの収集を再開するには同じsession-dirでコマンドを再実行してください。未入力・下書き・確定済みデータは残り、選択中の試合はresumeされます。

保存済み入力だけ行う場合（カメラ不要）:

```bash
python -m live_weapon_collect.app --label-only --session-dir "live_weapon_collect/data/session" --port 8780
```

クラッシュ時に保存途中だった試合は、次回起動時に `interrupted` として残します。保存できた画像だけ確認できます。確定済みのラベルは上書きしません。電源断直前の未保存画像・まだ入力中の無効な名前の復旧は保証しません。

### 武器別試合数・足りない武器一覧

画面上部の「武器別の収集状況・足りない武器」で、`wepons.json` にある全武器の**合計試合数・動画ラベル分・ライブ収集分・不足試合数**を確認できます。既定は3試合目安の「目安未満」表示です。2試合に変更でき、未収集（0試合）・全武器の表示、日本語/英語検索、カテゴリ絞り込み、不足一覧CSVの保存に対応しています。ラベル確定・修正後も更新されます。集計だけ見たいときも `--label-only` で起動できます。

- 人力確定済み・対象外でない試合の `reviewed=true` / `status=labeled` だけ数えます。未入力・下書き・推論候補・Unknown/Occluded/Skipは数えません。
- 同じ武器を1試合で2人以上使っていても、5枚のframeに出ていても、**1武器1試合**として数えます。
- 同じmatch idが複数directoryにコピーされていても重複計上せず、最新revisionを使います。同revisionで異なるラベルがあれば警告してその試合を除外します。
- 中断試合でも保存画像があり、人間が確認・確定した場合は数え、一覧の行に中断試合数を補足します。
- これはラベル収集量であり、実際にtemplateへ採用された試合数・学習済み武器数・精度保証ではありません。動画の所在・画質の検証も別途必要です。

現在のlive sessionに加え、既定では `weapon_lamp_detect/data/obs_session` の既存動画ラベルを読み取り専用で集計します。元のラベル・動画・評価データは変更しません。集計元が見つからなければ画面に警告します。Windowsで別の場所に既存ラベルがある場合や別live sessionも数えたい場合:

```powershell
python -m live_weapon_collect.app --source 0 --backend dshow --session-dir "live_weapon_collect/data/session" --coverage-dir "D:\データ\既存動画ラベル" --coverage-dir "D:\データ\前のlive session"
```

`--coverage-dir` はsession directory（`matches/` がある場所）を複数指定できます。指定した場合はその一覧が追加集計元となり、既定のobs_sessionは使いません。現在のlive sessionは常に含まれます。現在分だけなら `--no-existing` を指定します。

ブラウザなしでも一覧を確認できます:

```bash
python -m live_weapon_collect.coverage --session-dir "live_weapon_collect/data/session" --target 3 --missing-only
python -m live_weapon_collect.coverage --session-dir "live_weapon_collect/data/session" --target 2 --format json
```

CLIも同じ `--coverage-dir` / `--no-existing` / `--catalog` に対応しています。ブラウザのCSVはUTF-8 BOM付きです。

## 5. 保存形式

```text
session/
  collector_status.json       収集状態（推論・作業状況。ground truthではない）
  preview.jpg                 現在の入力の縮小表示
  resume.json                 最後に選択した試合
  matches/<session UUID + frame counter + UUID>/
    capture.json              入力元、backend、要求/報告形式、開始時刻、frame情報、SHA256、state、警告
    frames/00.png ... 04.png   入力解像度の開始画像（元動画全体は複製しない）
    annotation.json           人力の8武器・status・reference・geometry・日時・revision
    history/*.json            ラベル保存履歴
```

収集プロセスはcapture、UIはannotationだけを更新するため、取得と人力入力が互いを上書きしません。ファイルは読みやすいJSON、原子的な置き換えを使います。PNGは先に保存・fsyncしてからmanifestへ登録します。取得側はUIとは別のspawnプロセス、その中の専用threadでカメラを継続的に読みます。検出やPNG圧縮でカメラ読みを待たせず、最新画像を疎に処理します。

カメラの `timestamp` は入力を開いてからのmonotonic経過秒、`frame_index` は成功した読み込みのカウンタです。ハードウェアが実際に取得した全frameの番号や絶対撮影時刻とは区別し、その定義をmetadataに保存します。再起動時には新しいsession UUIDになります。

## 6. crop dataset生成・既存評価へ接続

人力確定した試合だけ、保存PNGから8slotのcropを展開します。全frame・連続何千枚を生成せず、同一match由来情報を保持します。既定はalive/down/unknown全てであり、評価はstate別に確認してください。

```bash
python -m live_weapon_collect.export_dataset "live_weapon_collect/data/session" --output "live_weapon_collect/data/dataset_01"
```

aliveだけなら `--states alive`。中断試合を人力で確認・確定した場合に限って含めたいときは `--include-interrupted`。出力directoryは新しいものにします。cropとcontextをexportするため、既存Viewerも使えます。元の保存PNGは残ります。

既存CLIに接続する例（十分な別試合を登録してから）:

```bash
python -m weapon_lamp_detect.build_obs_templates "live_weapon_collect/data/dataset_01" --output "live_weapon_collect/data/templates_01" --max-weapons 174
python -m weapon_lamp_detect.evaluate_poc "live_weapon_collect/data/dataset_01" --output "live_weapon_collect/data/evaluation_01" --method compare --obs-templates "live_weapon_collect/data/templates_01" --max-weapons 174 --workers 4
python -m weapon_lamp_detect.view_errors "live_weapon_collect/data/evaluation_01" --port 8782
```

このコマンドは既存の公式/OBS median方式の比較です。hybridの既存データ評価を自動更新・再学習するものではありません。候補数174は上限であり、未収集武器を自動的にtemplateへ追加しません。武器1種類2〜3試合を初期目安にし、試合単位のtemplate/evaluation分離と武器別supportを確認してください。同一match内の別timestampを許可する場合は既存CLIの `--allow-within-match` を明示し、cross-match精度と混ぜないでください。

PNG sourceにはSHA256を付け、既存評価のsource検証で確認します。動画が存在しないライブ入力について、架空のvideo pathや動画fpsを作りません。datasetの `videos` は空、`source_kind=image_sequences` と `image_sources` に保存PNGを記録します。

## 7. カメラなしの動作確認

元録画を等速でカメラ相当に入力し、同じUI・収集処理を試せます:

```bash
python -m live_weapon_collect.app --replay --source "../videos/2026-02-01 21-18-24.mp4" --session-dir "live_weapon_collect/data/replay" --port 8780
```

短区間だけ、待受なしで既存の実detector・PNG保存を検証する場合:

```bash
python -m live_weapon_collect.check_replay "../videos/2026-02-01 21-18-24.mp4" --session-dir "live_weapon_collect/data/replay_check_new" --start 0 --duration 35
```

後者はsource時刻を0.5秒ごとに送り、カメラの実時間性能を測るものではありません。検証専用の新しいsession-dirが必要です。いずれも正解ラベルは自動で作りません。

テスト:

```bash
python -m unittest live_weapon_collect.test_live -v
python -m unittest live_weapon_collect.test_coverage -v
python -m unittest live_weapon_collect.test_hybrid_candidates -v
python -m unittest discover -s weapon_lamp_detect -p "test_*.py"
```

開発用に既にNode/jsdomがある環境では `node live_weapon_collect/test_ui.cjs` でも入力・保存競合を検証できます。別配置のjsdomは `SPLATOON_JSDOM_PATH` で指定します。これは開発テストだけで、アプリ実行時にNodeやjsdomを導入する必要はありません。

カメラの実機接続・Windowsでのデバイス選択・実ブラウザの動作は使用PCで確認が必要です。検出漏れや画質差があるので、まず数試合で保存枚数・HUD枠・時刻・Unknown状態を確認してから大量収集してください。
