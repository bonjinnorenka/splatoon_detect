const fs = require('fs').promises;
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const LogisticImagePredictor = require('./logistic_regression_predictor.js');

// --- 設定 ---
const VIDEO_PATH = 'target.mkv';
const FRAME_RATE = 10;
const CROP_SETTINGS = '400:400:760:260'; // width:height:x:y
const TEMP_FRAME_DIR = path.join(__dirname, 'frames');
const RESULT_DIR = path.join(__dirname, 'result');
const MODEL_PATH = path.join(__dirname, 'logistic_model.json');

async function main() {
    // 1. ディレクトリ準備
    console.log('出力ディレクトリを準備しています...');
    const startDir = path.join(RESULT_DIR, 'start');
    const notDir = path.join(RESULT_DIR, 'not');
    await fs.mkdir(TEMP_FRAME_DIR, { recursive: true });
    await fs.mkdir(startDir, { recursive: true });
    await fs.mkdir(notDir, { recursive: true });

    // 2. フレーム抽出とクロップ
    console.log(`動画からフレームを抽出・クロップしています... (fps=${FRAME_RATE})`);
    await new Promise((resolve, reject) => {
        ffmpeg(VIDEO_PATH)
            .on('error', (err) => reject(new Error(`FFmpegエラー: ${err.message}`)))
            .on('end', () => resolve())
            .outputOptions([
                `-vf`, `fps=${FRAME_RATE},crop=${CROP_SETTINGS}`
            ])
            .output(path.join(TEMP_FRAME_DIR, 'frame_%05d.png'))
            .run();
    });
    console.log('フレームの抽出が完了しました。');

    // 3. 推論モデルのロード
    console.log('推論モデルをロードしています...');
    const predictor = new LogisticImagePredictor();
    try {
        predictor.loadModel(MODEL_PATH);
    } catch (error) {
        console.error(`モデルファイル (${MODEL_PATH}) の読み込みに失敗しました。`);
        console.error('学習スクリプトを先に実行して、モデルを生成してください。');
        return;
    }
    console.log('モデルのロードが完了しました。');

    // 4. 各フレームを推論・分類
    console.log('各フレームの推論を開始します...');
    const frameFiles = (await fs.readdir(TEMP_FRAME_DIR)).filter(f => f.endsWith('.png'));

    let startCount = 0;
    let notCount = 0;

    for (const frameFile of frameFiles) {
        const framePath = path.join(TEMP_FRAME_DIR, frameFile);
        try {
            const result = await predictor.predict(framePath);
            const destDir = result.class === 'start' ? startDir : notDir;
            const destPath = path.join(destDir, frameFile);
            await fs.rename(framePath, destPath);
            
            if (result.class === 'start') {
                startCount++;
            } else {
                notCount++;
            }
            process.stdout.write(`\r処理中: ${frameFile} -> ${result.class} (信頼度: ${result.probability.toFixed(2)})`);

        } catch (error) {
            console.error(`\n${frameFile}の処理中にエラーが発生しました:`, error);
        }
    }
    console.log(`\nすべてのフレームの処理が完了しました。`);
    console.log(` - start: ${startCount}枚`);
    console.log(` - not: ${notCount}枚`);


    // 5. 後処理
    try {
        await fs.rm(TEMP_FRAME_DIR, { recursive: true, force: true });
        console.log('一時ディレクトリを削除しました。');
    } catch (error) {
        console.error('一時ディレクトリの削除に失敗しました:', error);
    }
    
    console.log(`結果は ${RESULT_DIR} に保存されました。`);
}

main().catch(error => {
    console.error('\n処理全体でエラーが発生しました:', error.message);
}); 