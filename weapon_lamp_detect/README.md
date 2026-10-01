# Splatoon 3 OBS HUD武器識別 PoC

試合単位の人力ラベリング → 疎なcrop dataset → 公式画像方式(A) / 実OBS median方式(B) → 状態別・試合別評価 → 誤判定レビューのCPU用ツール。
CNN・GPU・新しいWebフレームワークは使用しない。既存の `WeaponIconMatcher`、`crop_slot`、動画iterator、`start_detect` HOG extractor / SVM JSON、`SquidLampDetector` を再利用している。他detectorは変更しない。

## 1. 必要環境・virtualenv

Python 3.10以上、OpenCV、NumPy、scikit-learn（既存start detectorの依存）。UIはPython標準のHTTP server + HTML/JavaScript。すべて以降のコマンドはリポジトリルート `private` から実行する。

既存のこの作業環境では:

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
python -m pip install -r weapon_lamp_detect/requirements.txt
```

別環境では:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r weapon_lamp_detect/requirements.txt
```

Windows PowerShellのactivateは `.\.venv\Scripts\Activate.ps1`。WSLでは `"/mnt/d/OBS録画/動画 01.mkv"` のようにパスを引用する。Windowsの `D:\OBS録画\動画 01.mkv` 形式もWSLで `/mnt/d/...` に変換する。動画はコピーしない。

公式画像は既存配置の `../sample_data/Main Weapons/*.png` を使用する。別配置なら `--template-dir` を指定。武器catalogは `../wepons.json` を優先し、別checkoutでは同ファイル由来の同梱 `weapon_catalog.json` にfallbackする。UIの検索候補はこのJSONの `weapons[].name` の日本語武器名のみ（現在174件）で、シューター等の `category` は武器ラベルにしない。使用中のJSONパス・件数をUIに表示する。別ファイルを使う場合はラベラーの `--catalog "日本語や空白を含むパス/wepons.json"` を指定する。`weapon_aliases.json` で日本語名をテンプレートのcanonical英語名に対応付けて保存し、再表示時は日本語名に戻す。対応していない名前を別武器に推測変換しない。UIには「公式画像なし」と表示し、評価では欠損を明示する。JSON未登録の公式テンプレート候補は参照表示のみで選択できない。

無効な武器名や入力途中の名前は欄を赤くして警告し、自動保存・保存・確定を拒否する。正しい候補を選ぶか Unknown / Occluded / Skip に変更するまで試合移動も行わない。保存API側も同じJSON一覧で検証し、無効な入力にはHTTP 400を返して既存ファイル・履歴を変更しない。未入力内容のブラウザ下書きは復元用であり、正解ラベルではない。

## 2. 試合候補検出

動画1本またはフォルダを指定（フォルダは再帰検索）。既存 `start_detect/cropsize.md` の400×400領域を解像度に比例してcropし、既存HOG+SVMのdecision thresholdで判定する。検出時刻はステージ紹介の画面であり、実プレイやHUD表示の開始ではない。紹介時刻を保持し、referenceの初期値は候補＋20秒にする。重複した紹介画面を30秒の間隔でまとめる。pickleのsklearnバージョン依存を避けるため、同じ既存モデルのJSON weightsを使う。再学習しない。

```bash
python weapon_lamp_detect/detect_matches.py "../videos" \
  --session-dir weapon_lamp_detect/data/obs_session --scan-interval 0.5
```

最初は短い区間でも確認できる:

```bash
python weapon_lamp_detect/detect_matches.py "../videos/2026-02-01 21-18-24.mp4" \
  --session-dir weapon_lamp_detect/data/obs_session --start 0 --end 90
```

CLIに `--cooldown 30`、`--hud-offset 15`（初期値20秒）を指定できる。検出intervalを長くすると短い紹介画面を見逃す。長い動画の解析は時間がかかるため、UIからバックグラウンド検出も可能。検出済み同じ候補を再登録しても既存の人力修正は上書きしない。

## 3. 試合単位ラベラー起動

```bash
python weapon_lamp_detect/label_weapon_samples.py --match-mode "../videos" \
  --session-dir weapon_lamp_detect/data/obs_session
```

[http://127.0.0.1:8765/](http://127.0.0.1:8765/) を開く。`label_matches.py` を直接実行しても同じ。`--scan` で起動時に全登録動画を自動検出できる。動画を省略して同じsessionで起動するとresumeする。

```bash
python weapon_lamp_detect/label_matches.py \
  --session-dir weapon_lamp_detect/data/obs_session
```

操作:

1. 動画一覧から選択。「開始候補を自動検出」で候補を作る（進捗を表示）。検出漏れはseekして「現在位置を開始とする試合を追加」、誤検出は「誤候補を除外」。
2. 前/次試合で移動し、開始・終了を手動修正。紹介時刻を区間開始に保持してよい。「HUD待ち(s)」は初期値20秒で、15秒などに変更して「開始＋HUD待ちへ移動」でHUD表示を確認できる。開始を実プレイ位置に移した場合はHUD待ちを0にする。終了の初期値は次の紹介候補か動画末尾であり、ロビーや別試合を含み得るため、そのまま確定しない。
3. 見やすいHUDへseekして「現在位置→reference」。フレーム全体、HUD拡大、left0〜left3 / right0〜right3 cropを確認する。標準枠は実OBSで確認した中心と約100×100px（1080p）を使い、以前の外側へずれた約134px幅の枠を補正している。武器がcropから切れていればスロットのX/Y/幅/高さ（0〜1の比率）で補正する。HUDの黄色枠で確認でき、「前試合のcrop補正を引継ぎ」で繰り返し入力を省ける。「修正済み標準cropへ戻す」で標準枠に戻せる。補正は試合JSONとdatasetのrectに保持し、crop再生成・推論・state判定にも反映する。他の既存detectorの固定座標は変更しない。
4. 各スロットに武器を1回入力（日本語/英語検索）。必要に応じてUnknown / Occluded / Skipを人力で選択。「top-5候補を表示」は参考情報だけ。候補をクリックした場合のみ人力入力になる。
5. 味方sideと区間を確認し、確認checkboxを入れる。「保存・人力確定」で8武器を保存し、次の試合へ。

編集は650ms後に下書きとして自動保存。未入力のスロットはUnknownと同義ではなく `reviewed:false` を保持する。確定には8スロットの人力入力と区間確認が必要。確定後の編集も再び下書きになり、再確定するまでdatasetへ展開しない。保存失敗は画面に表示し、ブラウザlocalStorageの下書きから復元できる。Undoは直前の保存revisionを復元し、前の試合も再編集できる。

キーボード（画面にも一覧あり）:

| キー | 操作 |
|---|---|
| ← / → | ±1frame |
| Shift + ← / → | ±5秒 |
| Space | 確認用プレビュー再生/停止（疎なフレーム、音声なし） |
| 1〜8 | スロットの入力欄を選択 |
| Enter / Tab / Shift+Tab | 次/前のスロット |
| S / N / P | 保存・試合確定 / 次試合 / 前試合 |
| U / O / X | Unknown / Occluded / Skip |
| R / Esc | referenceに指定 / 入力欄から離れる |

文字入力中はEnter/Tabのみショートカットになる。他のキーはEscで入力欄を離れてから使う。

保存は `session/matches/<match_id>.json`（試合ごとのindent付きJSON）。video path / content識別子 / size / duration / fps / resolution / match id / 開始・終了 / reference時刻とframe index / ally side / 8武器とstatus / created_at / updated_at / revisionを保持する。動画識別子はsize＋先頭・中間・末尾64KiBのSHA-256で、ファイル名には依存しない。完全hashではないため厳密な全内容一致を保証するものではない。`videos.json` に動画一覧、`candidates/` に検出結果、`history/` に履歴、`resume.json` に最終試合を保存する。GT JSONはGitで管理可能だが、元動画、個人pathを含む作業session、大量cropを不用意にcommitしない。

## 4. Dataset生成

人力確定した試合だけを展開。非試合HUDは除外し、stateは既存 `squid_lamp_detect` の自動推定を記録する。主評価はalive。down・unknownは別に収集・集計できる。ラベルstatus（Unknown/Occluded/Skip）とフレームのalive/down/unknownは別の情報であり、武器名が未確定のstatusは評価対象にしない。

```bash
python weapon_lamp_detect/build_dataset.py weapon_lamp_detect/data/obs_session \
  --output weapon_lamp_detect/data/poc_dataset \
  --sample-interval 5 --end-offset 5 \
  --include-down --max-frames-per-match 20
```

aliveのみなら `--include-down` を外す。抽出開始は各試合のHUD待ち時間（初期値20秒）後。`--start-offset 20` で全試合共通の値を明示して上書きできる。`--states alive down` など個別指定も可能。`--sample-interval 1` / `5` / `10` を選べる。`--max-samples 400` はcrop総数、`--max-frames-per-match 20` は1試合の上限。0で無制限だが、60fpsの連続画像を独立した大量データとして扱わない。

既存sessionの旧初期値（＋5秒のreference・未補正の枠）は、ラベラー再起動時に未編集のrevision 1だけ自動移行する。ラベル・手動reference・手動crop・編集済みrevisionは上書きしない。移行前のJSONもhistoryに保持する。`hud_offset_seconds`、`start_event`、`reference_source`、`crop_profile` を試合JSONに、実際の抽出offsetをdataset metadataに記録する。

通常は `dataset.json` と `samples.jsonl` の参照情報のみ保存。評価時に元動画＋frame indexから再生成する。各cropにsource video / video id / match id / timestamp / frame index / side / slot index / state / weapon label / class / label revision / rect / blur variance / special scoreを記録する。元動画を移動した場合はdataset.jsonのvideo pathも更新する。評価前に動画content識別子を再確認する。

crop/contextをexportする場合は別のoutputへ:

```bash
python weapon_lamp_detect/build_dataset.py weapon_lamp_detect/data/obs_session \
  --output weapon_lamp_detect/data/poc_export \
  --sample-interval 5 --include-down --export-crops
```

cropはPNG、contextはJPEG。両方式とも評価時のcropを縦横比を保って134×108の枠内に正規化し、余白を黒で埋める（狭いcropの武器を横に引き伸ばさない）。元のrectは保持。比較条件を揃えるための正規化であり、解像度耐性の保証ではない。既存outputを誤って上書きすることを防ぐため、新しいoutput名を使う。

## 5. まず公式画像方式Aを評価

```bash
python weapon_lamp_detect/evaluate_poc.py weapon_lamp_detect/data/poc_dataset \
  --method official --output weapon_lamp_detect/data/poc_official
```

候補集合はdatasetで人力確認した武器（closed-set）。初期の安全上限は20種で、超えると停止する。`--weapons "スプラシューター" "シャープマーカー" ...` で絞り込むか、収集済み範囲を明示的に拡張する場合だけ `--max-weapons 58` などをtemplate生成・評価の両方に指定する。人力収集していない武器を候補にはできない。181種への全武器識別精度として解釈しない。公式画像欠損は `missing_templates` と誤判定supportに残す。CPU並列化は `--workers 4`（初期値1）。並列・直列の出力一致を機能testで検証する。

## 6. OBS由来template生成

```bash
python weapon_lamp_detect/build_obs_templates.py weapon_lamp_detect/data/poc_dataset \
  --output weapon_lamp_detect/data/poc_templates --max-per-weapon 3 --seed 7
```

試合idをseedで決定的に並べ、約1/3をtemplate専用、残りを評価専用にする。alive cropを武器・side別に最大3timestampからpixel medianで合成する簡単な方式。背景やインク色への依存を含むため、まずこのbaselineを実測する。候補武器はA/B共通にし、template不足の武器を評価集合から都合よく消さない。武器ごとの別試合を追加するとcoverageが改善する。

別試合を用意できない武器を暫定評価する場合だけ:

```bash
python weapon_lamp_detect/build_obs_templates.py weapon_lamp_detect/data/poc_dataset \
  --output weapon_lamp_detect/data/poc_templates_within --allow-within-match
```

不足武器の最初のalive timestampをtemplateに予約する。その元フレームの**全スロット**を評価から除外。同一武器・同一試合での評価は `within_match_other_timestamp` とし、`cross_match` と必ず別集計する。template参照sample id / 元frame / match / 分割 / dataset SHA-256を `templates.json` に保持。評価時はsampleの重複・元frameの重複・scope不整合・dataset変更を拒否する。

## 7. 同じ評価集合でA/B・複数frameを比較

```bash
python weapon_lamp_detect/evaluate_poc.py weapon_lamp_detect/data/poc_dataset \
  --method compare --obs-templates weapon_lamp_detect/data/poc_templates \
  --output weapon_lamp_detect/data/poc_comparison
```

`--allow-within-match` の場合は上のtemplateディレクトリを `poc_templates_within` に替える。A/Bの評価sampleは共通で、template専用sampleをAにも評価させない。

出力:

- `report.md`: 概要とレビュー手順。
- `report.json`: exact top-1/3/5、class top-1、武器別accuracy/support、試合数・match-slot数、alive/down/unknown別、cross-match/within-match別、confusion matrix/pairs、正解/誤判定margin・score分布。
- `predictions.jsonl`: 全評価sampleのexpected/predicted/top-k/best/second/margin/confidenceとsource情報。
- `multi_frame`: score平均・majority・confidence-weighted voteで1/3/5/10frameを比較。同じmatch-slot-state-scopeの早い順Ntimestampを使い、不足groupは除外してsupportを明示。
- `common_cohort`: 10frame以上ある同じgroupだけで1/3/5/10を比較。Nにより対象groupが変わったことによる見かけの改善を避けるため、こちらも見る。

同点は武器名順に決定する。投票方式のtop-kは実際に票を得た武器だけを含め、無得票の武器は含めない。投票方式のscore/marginは得票比であり、single-frame相関scoreと同じ尺度ではない。confidenceは既存式を使うheuristicで、校正された確率ではない。現時点でUnknown閾値を固定しない。

## 8. 誤判定確認

```bash
python weapon_lamp_detect/view_errors.py weapon_lamp_detect/data/poc_comparison
```

[http://127.0.0.1:8766/](http://127.0.0.1:8766/) を開く。方式・expected・predicted・state・scopeで絞り込み、crop/context/top-k/score/margin/confidenceをページごとに確認する。crop未exportなら元動画から再生成。正解例も表示できる。

派生武器、近い形、Order、X重畳、ブラー、HUDずれ、インク色、スペシャル重畳、state誤り、正解ラベル誤りのタグとメモを `error_reviews.json` に保存できる。自動stateの誤りを武器matcherの誤りと混同しない。タグは人力レビュー情報で、GTを自動変更しない。GT誤りはラベラーで修正・再確定し、新dataset/template/reportを生成する。

判断はaliveの**別試合**の武器別supportと混同ペアを優先し、共通cohortで複数frameによる改善を確認する。必要精度を少数の安定frameで満たせればtemplate方式を継続する。系統的な混同が残る場合、どの武器・state・marginで、集約後も残るかを根拠にCNNを検討する。必要精度・sample数は運用目的に依存するため、このツールは未設定の合格閾値でCNN導入を自動判断しない。

以下の旧synthetic評価や機能testの人工fixtureを実OBS精度として扱わない。実OBS評価には人力確定した試合ラベルが必要。圧縮・720p・blur試験は、まずOBS評価を終えてからdataset cropの変換として追加できる。

### 入力済み14試合・58武器の再現コマンド（2026-10-01）

収集済み58武器を候補にして5秒ごとに最大20frame/試合。既存outputの上書き防止のため、再実行時は別の名前に替える。dataset約199MBは確認用crop/contextだけで、16GBの元動画は複製しない。ラベル・区間・crop補正はdataset.jsonにsnapshot保存する。

```bash
source ../bin/activate
python weapon_lamp_detect/build_dataset.py weapon_lamp_detect/data/obs_session \
  --output weapon_lamp_detect/data/poc_obs_20261001_dataset \
  --sample-interval 5 --end-offset 5 --include-down \
  --max-frames-per-match 20 --export-crops
python weapon_lamp_detect/build_obs_templates.py weapon_lamp_detect/data/poc_obs_20261001_dataset \
  --output weapon_lamp_detect/data/poc_obs_20261001_templates \
  --max-per-weapon 3 --seed 7 --max-weapons 58 --allow-within-match
python weapon_lamp_detect/evaluate_poc.py weapon_lamp_detect/data/poc_obs_20261001_dataset \
  --method compare --obs-templates weapon_lamp_detect/data/poc_obs_20261001_templates \
  --output weapon_lamp_detect/data/poc_obs_20261001_comparison --max-weapons 58 --workers 4
python weapon_lamp_detect/view_errors.py weapon_lamp_detect/data/poc_obs_20261001_comparison
```

4試合をtemplate専用に分離し、残る10試合のうちtemplate予約フレームを除外した1,512cropでA/Bを比較する。候補は58種だが、held-out側で正解として出現するのは46種。残る12種はtemplate専用試合にしかなく、別試合での精度は未測定。別試合aliveと同一試合別timestampのaliveを混ぜて成功値としない。2,240cropは14試合・112match-slot由来であり、2,240個の独立試合や独立プレイヤーではない。

初回測定ではalive/cross-match 707cropのtop-1が公式画像51.91%、単純OBS median16.55%。同一46match-slotのscore平均で3frameにすると公式画像65.22%、10frameのOBS medianは39.13%。現状のまま自動確定に使うには不足する。系統的な混同と色・背景依存があり、単純medianの失敗だけでOBS template方式全体やCNN必須と断定はしない。[詳細な初回評価・武器別結果](data/poc_obs_20261001_comparison/assessment.md) に条件、support、混同、margin、未測定範囲をまとめた。ローカルのdataset/reportはGit除外なので、別checkoutでは上のコマンドで生成する。

### 背景除去・照合領域の比較

既存matcherを変更せず、`region_matcher.py` に比較用方式を追加した。実OBS画像は既存template manifestの**同じ元sampleだけ**を前処理してからmedian合成する。評価crop・正解・state・scope・候補集合は前回と同一。中央の106×72pxを使い、インク色は既存squid detectorの色推定を再利用して灰色化する。OBSの位置補正は±4px。公式画像側の変更はscale・masked ZNCC・スコア再構成も含むので、crop単独の効果とは扱わない。

最初にtemplate専用試合だけで設定の動作確認をする。設定JSONの作成後は上書きしない。評価用sampleが混ざると停止する。今回の固定設定では63cropを確認した。評価結果を見ながらこのJSONを変更しない。

```bash
source ../bin/activate
python weapon_lamp_detect/check_region_training.py weapon_lamp_detect/data/poc_obs_20261001_dataset \
  --obs-templates weapon_lamp_detect/data/poc_obs_20261001_templates \
  --output weapon_lamp_detect/data/poc_obs_20261001_regions --workers 4
python weapon_lamp_detect/evaluate_poc.py weapon_lamp_detect/data/poc_obs_20261001_dataset \
  --method regions --obs-templates weapon_lamp_detect/data/poc_obs_20261001_templates \
  --output weapon_lamp_detect/data/poc_obs_20261001_regions \
  --max-weapons 58 --workers 8 --executor process
python weapon_lamp_detect/compare_runs.py \
  weapon_lamp_detect/data/poc_obs_20261001_comparison \
  weapon_lamp_detect/data/poc_obs_20261001_regions \
  --output weapon_lamp_detect/data/poc_obs_20261001_improvement
python weapon_lamp_detect/view_errors.py weapon_lamp_detect/data/poc_obs_20261001_improvement
```

`--executor process` はCPUの独立workerをWindows/WSL共通のspawnで起動する。GPU不要。`--workers 4` に減らしてもよい。通常の評価は従来のthreadが初期値。直列／thread／processの予測一致をtestで確認している。

`compare_runs.py` は前回の予測を再計算せず再利用する。sampleごとのsource・正解・state・scope・元OBS templateが一致しないrunの改善率比較は拒否する。全指標に加え、同じcropの誤り→正解／正解→誤りの件数を出す。summaryのみの再生成は、同じ2入力に限り `--refresh-summary` で可能で、予測・レビューを変更しない。

測定結果（alive/cross-match 707crop）：前回OBS16.55%→領域・位置補正55.30%→背景除去追加66.90%。合計+50.35ポイント、背景除去だけの追加差は+11.60ポイント。同じ46match-slotの3frame平均は新OBSで82.61%（38/46）、top-3は93.48%。公式画像側の変更は23.20%／13.30%に悪化したので既存の51.91%方式をデフォルトから置換していない。

[改善実験の結論](data/poc_obs_20261001_improvement/assessment.md)、[全方式・武器別比較](data/poc_obs_20261001_improvement/comparison.md) に未測定範囲・残る混同・marginを記録。これは過去に見た評価集合での改善実験で、新しい動画での独立した最終評価ではない。Viewerは新OBS方式のalive/cross-matchを初期表示し、「照合領域（前処理後）」で背景除去の様子を見られる。CNNは実装していない。

### 追加動画のラベルで改善したかを測定

追加データを使う前に、試合単位の分割を固定する。前回のcrop・正解・評価集合は保持し、追加試合の1/3だけをtemplate用にする。照合アルゴリズム・ROI・インク除去・scoreは前回の固定設定のまま。新しいtemplateは各追加training試合/武器/sideの最初の最大3枚のalive cropのmedianとして追加し、元templateを残す。追加held-out試合の画像を、未収集武器の救済templateとして使わない。

以下は今回の保存先。再実行時は `poc_obs_20261001_added` を新しい名前に置き換える（既存artifactの上書きは拒否）。準備時にGTのsnapshotを作るので、実行中のUIラベル更新で分割や正解が変化しない。

```bash
source ../bin/activate
python weapon_lamp_detect/evaluate_added_data.py prepare \
  --session-dir weapon_lamp_detect/data/obs_session \
  --baseline-dataset weapon_lamp_detect/data/poc_obs_20261001_dataset \
  --baseline-templates weapon_lamp_detect/data/poc_obs_20261001_templates \
  --baseline-configuration weapon_lamp_detect/data/poc_obs_20261001_regions/configuration_frozen.json \
  --output weapon_lamp_detect/data/poc_obs_20261001_added --seed 7
python weapon_lamp_detect/build_dataset.py \
  weapon_lamp_detect/data/poc_obs_20261001_added/snapshot_session \
  --output weapon_lamp_detect/data/poc_obs_20261001_added/new_dataset \
  --sample-interval 5 --end-offset 5 --include-down \
  --max-frames-per-match 20 --export-crops
python weapon_lamp_detect/evaluate_added_data.py run \
  weapon_lamp_detect/data/poc_obs_20261001_added --workers 8
python weapon_lamp_detect/view_errors.py \
  weapon_lamp_detect/data/poc_obs_20261001_added/comparison --port 8767
```

`run` は完了済みの個別runを検証してresumeできる。元動画はコピーせず、前回と追加datasetのcrop/contextのみhardlinkで統合する（別filesystemでは小画像のみcopy）。個別の前処理方式は `evaluate_poc.py --method obs_foreground` / `--method obs_region` で、4方式全部を再計算せず評価できる。

比較は次の3方式。元の58武器候補での改善と、追加収集された武器まで候補を広げた結果を混ぜない。

- `baseline_obs_foreground`：前回のOBS templateのみ、58候補。
- `obs_foreground`：追加training試合のtemplateを追加、同じ58候補・同じ評価crop。主にこの2方式の差を改善量として読む。
- `expanded_obs_foreground`：人力収集済みの全武器を候補に拡張。templateがない武器も失敗としてsupportに残す。既知58武器のcropだけに絞った指標も別収録し、候補増加の影響を確認できる。

`comparison/comparison.md` に前回の固定holdout／追加動画の固定holdout別のtop-1/3/5と1/3/5/10frame平均を出す。`comparison/report.json` の `paired_comparisons` に同一cropの正誤遷移・武器別accuracy・confusion・marginを保存する。105候補等の全武器評価は58候補との直接改善率として扱わない。追加された全ラベルをtemplateに入れた自己評価も行わない。GT・ラベラーの既存sessionは変更しない。

追加ラベルの**全量投入**も測る場合は、追加39試合全部をtemplate専用にし、前回の固定holdoutのみで評価する。追加動画上の自己評価はしない。13/26分割の独立評価とは区別する。

```bash
python weapon_lamp_detect/evaluate_added_data.py full-training \
  weapon_lamp_detect/data/poc_obs_20261001_added --workers 8
python weapon_lamp_detect/view_errors.py \
  weapon_lamp_detect/data/poc_obs_20261001_added/full_additional_training/comparison --port 8768
```

全量runは `full_additional_training/comparison/` に別保存し、前の13/26評価を変更しない。[今回の結論](data/poc_obs_20261001_added/comparison/assessment.md) に両方の条件・改善と悪化をまとめる。

## 検証

```bash
python -m unittest weapon_lamp_detect.test_poc -v
python -m unittest weapon_lamp_detect.test_added_data -v
```

人工動画で保存/resume、疎sampling、state別収集、A/B比較、同一match分離、リーク拒否、multi-frame集計を検証する。既存pickleとのscore一致も検証するが、pickle保存時のsklearnバージョンと異なる環境ではwarningが出る。通常の開始検出はJSON経由なのでこのpickleをロードしない。

## 既存crop単位ラベラー・合成評価（互換用）

以下は従来のcrop単位CLI。今回の試合単位PoCには上のフローを使う。

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

枠位置の確認用に `--debug-dir` を付けると、実フレーム上のクロップ枠と推定ラベルを書き出す。現状の固定枠は大きく外れてはいないが、スロット内の武器位置は左右・alive/downでずれるため、照合側では枠内をマスク付きテンプレート探索している。

## 精度評価

従来の評価データには alive/down のラベルしかなかった。以下は当時の合成評価であり、今回の人力ラベル済み実OBS評価とは別の結果である。

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
- 個別武器名 top-1: 44/181 = 24.31%
- 個別武器名 top-5: 71/181 = 39.23%
- 武器カテゴリ top-1: 57/181 = 31.49%
- 武器カテゴリ top-5: 110/181 = 60.77%
- aliveのみの個別武器名 top-1: 40/135 = 29.63%
- aliveのみのカテゴリ top-1: 52/135 = 38.52%
- downのみの個別武器名 top-1: 4/46 = 8.70%
- downのみのカテゴリ top-1: 5/46 = 10.87%

この数値から見ると、枠内の探索と色スコア追加で改善はしたが、武器特定器としてはまだ実フレームの教師ラベルで調整する必要がある。特にdown時はX重畳で武器が隠れるため、画像だけからの武器名推定はかなり厳しい。

## 実フレームのラベル収集

実動画での武器名精度を測るためのローカルラベラーを用意している。動画からイカランプのスロット画像を切り出し、ブラウザで武器名を選んで保存する。

```bash
python weapon_lamp_detect/label_weapon_samples.py "../videos/2026-02-06 22-39-39.mkv" \
  --start 30 --end 180 --sample-interval 10 \
  --max-slots 96
```

起動後に表示される `http://127.0.0.1:8765/` を開く。ラベルは `weapon_lamp_detect/data/labeling_sessions/session_*/labels_current.json` と `labels.jsonl` に保存される。down状態も含めたい場合だけ `--include-down` を付ける。

ラベル済みセッションは次で評価する。

```bash
python weapon_lamp_detect/evaluate_labeled.py \
  weapon_lamp_detect/data/labeling_sessions/session_YYYYMMDD_HHMMSS
```

## 実装メモ

- `sample_data/Main Weapons/*.png` のアルファをテンプレートマスクとして使う。
- テンプレートを複数スケール・少量回転で展開し、イカランプクロップ上でマスク付き `matchTemplate` を使って枠内を探索する。
- グレースケール一致だけだと無彩色のOrder系テンプレートに吸われやすいため、探索位置のLab色相関とエッジ一致も組み合わせてスコア化する。
- スロット位置は既存イカランプ検出と同じ固定HUD座標を使うが、`squid_lamp_detect` 側のファイルは変更しない。
- `detect_video.py` は可能なら `squid_lamp_detect` を読み込み、alive/down/unknown のスロット状態も併記する。

## 注意

テンプレート一致型なので、実フレームでのブラー、配信圧縮、X重畳、隣接アイコンの重なり、スペシャルバッジ重畳には弱い。特に個別武器名は似た派生武器が多く、まずは `weapon_class` の方を信用し、実用化する場合は実フレームの武器名ラベルを作って評価・調整するのが必要。
