# 武器別ラベル確認・修正 Web アプリ

収集アプリ `live_weapon_collect/` や試合単位ラベラーとは別のアプリです。カメラ・推論・再学習を行いません。既存の**人力ラベルを武器ごとに一覧表示し、画像を見て確認・修正**します。Python標準HTTP server＋既存OpenCV構成で、Reactなどの追加環境は不要です。

## 起動（Windows PowerShell）

収集アプリと同じWindows virtualenvを使えます。WSL側のvirtualenvではありません。ソースはリポジトリ全体が必要です。

```powershell
Set-Location "\\wsl.localhost\Ubuntu\home\ryokuryu\splat_2\private"
$reviewPython = "$env:LOCALAPPDATA\splatoon-live-venv\Scripts\python.exe"
& $reviewPython -m weapon_label_review.app --live-dir "$env:LOCALAPPDATA\splatoon-live-data\session" --data-dir "$env:LOCALAPPDATA\splatoon-weapon-review" --port 8783
```

<http://127.0.0.1:8783/> を開きます。収集アプリの8780番とは独立です。pyenvのbatラッパー経由の `python` はUNCの作業ディレクトリを失うことがあるため、上記の `python.exe` 直接実行を使ってください。

既存動画ラベルも一緒に確認する場合:

```powershell
& $reviewPython -m weapon_label_review.app --live-dir "$env:LOCALAPPDATA\splatoon-live-data\session" --video-dir "weapon_lamp_detect/data/obs_session" --video-root "\\wsl.localhost\Ubuntu\home\ryokuryu\splat_2\videos" --data-dir "$env:LOCALAPPDATA\splatoon-weapon-review" --port 8783
```

`--video-root` は動画フォルダです。元ラベルのLinux pathをWindowsで開けない場合に、そのフォルダの同名ファイルを利用します。**動画識別子を照合し、同名の別動画なら拒否**します。元ラベルのpathは書き換えません。UNCからの動画読み込みが環境依存で失敗する場合は、下記のWSL起動で元のLinux pathから読んでください。

新しいWindows環境に必要ライブラリを入れる場合:

```powershell
& $reviewPython -m pip install -r weapon_label_review/requirements.txt
```

## 起動（WSL）

```bash
cd /home/ryokuryu/splat_2/private
source ../bin/activate
python -m weapon_label_review.app \
  --live-dir "/mnt/c/Users/ryo/AppData/Local/splatoon-live-data/session" \
  --video-dir "weapon_lamp_detect/data/obs_session" \
  --data-dir "weapon_label_review/data" \
  --port 8783
```

データ入力元の `--live-dir` / `--video-dir` は複数指定できます。存在しない入力元は作成せずエラーにします。確認履歴の `--data-dir` は元ラベルのsession保存先と別にしてください。WindowsではUNCのロックの問題を避けるため、上記のWindowsローカル保存先を推奨します。

## 確認の流れ

1. 左の日本語武器一覧で武器を選びます。試合数・スロット数・未確認数が表示されます。
2. 同じ武器としてラベルされたcropをまとめて確認します。**1試合の1スロットを1件**とし、5frameを5件として水増ししません。同じ武器が同じ試合の別スロットにいれば別々に確認します。
3. cardを選ぶと、拡大crop・対象枠付き上部HUD・context全画面を表示します。ライブでは保存PNG、動画ではreference時刻とその後の最大4秒を表示します。frame切替は確認用で、reference・crop座標・試合区間は変更しません。
4. 正しければ「✓ 正しい」。**元GTを書き換えず**、別の確認履歴へ記録して次の未確認スロットを選びます。
5. 間違っていれば武器名・statusを直して「修正保存」。`wepons.json` の名前だけ許可し、カテゴリ名・誤字は保存しません。Unknown / Occluded / Skipへの修正・その逆も可能です。
6. 同武器の未確認がなくなったら `N` または「次の武器」で次の未確認武器へ移ります。検索・ライブ/動画filterを尊重します。「確認済みも表示」で再確認できます。

既定は人力確定済み・対象外でない試合のみです。下書きも見たい場合は `--include-drafts`。下書きの1スロットを直しただけで試合を自動確定しません。未ラベル・Skip等は専用グループに表示します。

画像を読めない場合は警告します。確認・修正の際もreference画像を読み直し、ライブ画像のSHA-256を検証します。壊れた画像や別動画を見て正しいと記録することを防ぎます。予測によるラベル変更はありません。

## キーボード

- `N` / `P`：次 / 前の未確認武器。
- `J` / `K`：同じ武器の次 / 前のスロット。
- `Enter`：現在のラベルを正しいとして確認し、次へ。
- `←` / `→`：保存frameまたはreference後のframeを切替。
- `S`：修正保存して次へ。
- `Ctrl+Z`：直前のラベル修正を取消（入力欄では通常の文字入力undo）。

武器入力欄の `Enter` は修正保存です。未保存の修正を残して移動する場合は破棄確認を出します。修正の自動保存はせず、元GTを書き換える操作は明示的な保存のみです。

## 保存と安全性

```text
data-dir/
  reviews/<item_id>.json        元ラベルとは別の確認済み記録
  changes/<operation>.json     修正前の試合全体バックアップ＋操作履歴
  resume.json                  最後に選んだ武器とスロット
```

- 元画像・動画は複製・変更しません。
- ラベル修正だけを入力元の `annotation.json`（live）/ `matches/<match_id>.json`（video）へ反映します。対象スロット以外の7スロット・confirmedフラグ・試合区間・geometryは維持します。
- 元ラベラーの保存・検証・履歴機能を再利用し、別途このアプリの修正前バックアップを**元GT保存前に**原子的に保存します。
- revisionが変わっていたら保存を拒否します。ライブ元ラベラーとは同じ `annotation.lock` を使います。動画元ラベルはsession内の `annotation.review.lock/` の原子的な作成で確認アプリ同士を排他し、WindowsからWSLのUNC path経由で保存する際もWindowsのbyte-range lockを使いません。旧動画ラベラーはこのロックに非対応なので、**動画の同じsessionを別アプリで同時に修正しないでください**。ライブ収集自体との同時利用は可能です。
- undoはその修正後に元GTの追加更新がない場合だけ許可します。他の更新を巻き戻して消すことはありません。ブラウザ再読み込み後も直前の修正IDを保持します。
- 元スロットの武器・status・reference・cropが外部で変更された場合は確認済みを無効にします。無関係な別スロットの修正だけでは、このスロットの確認済みを無効にしません。
- 同じ `data-dir` と同じ入力元で再起動すれば確認履歴・resumeを引き継ぎます。Windows版とWSL版で入力元のpath表記が異なる場合は別の確認IDになるので、継続作業は同じ実行環境を使用してください。
- 既に生成したdataset/template/reportは自動更新しません。ラベル修正後、必要なものは元GTから新しい出力先へ再生成してください。

### 更新・ロックの復旧

コード更新後は確認アプリを `Ctrl+C` で終了し、同じ起動コマンド・同じ `--data-dir` で再起動してください。以前の `annotation.lock` ファイルは、残っていてもロック取得中という意味ではありません。この動画保存方式では利用しないので、削除不要です。

動画保存の競合はHTTP 409で、所有者のhost・pid・開始時刻とロックpathを表示します。アクセス権限などの作成失敗は別のエラーになります。稼働中の保存は通常すぐに終わるので、少し待って再試行してください。

強制終了などで `annotation.review.lock/` が残った場合は、勝手に削除せず保存を拒否します。**その動画sessionを使うすべての確認アプリ（Windows・WSLとも）を終了したことを確認した後に限り**、表示されたロックの所有者を確認し、その小さなマーカーだけを手動で削除できます。既存のmatches・history・画像は削除しません。既定の動画sessionの場合のPowerShell例:

```powershell
$videoReviewLock = Join-Path (Resolve-Path "weapon_lamp_detect/data/obs_session").Path "annotation.review.lock"
Get-Content -LiteralPath (Join-Path $videoReviewLock "owner.json")
# 全確認アプリの終了確認後にだけ実行。-Recurse は付けない。
Remove-Item -LiteralPath (Join-Path $videoReviewLock "owner.json")
Remove-Item -LiteralPath $videoReviewLock
```

owner.jsonがない未完成マーカーなら最後の空directory削除だけで十分です。削除に失敗したら再帰削除せず、権限と稼働中のプロセスを確認してください。

画像切替などでブラウザが通信を中止した場合、`ConnectionAbortedError / WinError 10053` を通常の切断として扱い、切れた接続にエラーを送り直しません。修正保存はHTTP応答の送信前に完了します。保存応答を受け取れなかった場合は一覧を再読み込みしてラベル・revisionを確認してください（同じ古いrevisionでの再保存は上書きせず拒否します）。

## 検証

```bash
source ../bin/activate
python -m unittest weapon_label_review.test_locking weapon_label_review.test_review -v
```

テストは一時ファイルのみを使い、実際のユーザー画像・ラベルを変更しません。既存のNode/jsdomがある開発環境では `node weapon_label_review/test_ui.cjs` でUIの移動・修正・取消を検証できます。別配置のjsdomは `SPLATOON_JSDOM_PATH` で指定します。アプリ利用時にNode/jsdomは不要です。
