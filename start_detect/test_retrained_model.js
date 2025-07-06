const LogisticImagePredictor = require('./logistic_regression_predictor.js');
const path = require('path');
const fs = require('fs');

async function testRetrainedModel() {
    const predictor = new LogisticImagePredictor();
    
    // 学習済みモデルを読み込み
    predictor.loadModel('logistic_model.json');
    console.log('学習済みモデルを読み込みました。');
    
    // テスト用のサンプル画像
    const testSamples = [
        // 'not'クラスのサンプル
        { path: 'data/not/output_0001.png', expected: 'not' },
        { path: 'data/not/output_0100.png', expected: 'not' },
        { path: 'data/not/output_0200.png', expected: 'not' },
        
        // 'start'クラスのサンプル
        { path: 'balanced_data/start/output_0056.png', expected: 'start' },
        { path: 'balanced_data/start/output_0306.png', expected: 'start' },
        { path: 'balanced_data/start/output_0721.png', expected: 'start' },
        
        // 偽陽性だったサンプル（'not'として正しく分類されるべき）
        { path: 'result/wrong/frame_00456.png', expected: 'not' },
        { path: 'result/wrong/frame_00591.png', expected: 'not' },
        { path: 'result/wrong/frame_01151.png', expected: 'not' }
    ];
    
    console.log('\nテスト結果:');
    console.log('=' .repeat(50));
    
    let correct = 0;
    let total = 0;
    
    for (const sample of testSamples) {
        const fullPath = path.join(__dirname, sample.path);
        
        if (fs.existsSync(fullPath)) {
            try {
                const result = await predictor.predict(fullPath);
                const isCorrect = result.class === sample.expected;
                
                console.log(`${sample.path}`);
                console.log(`  期待値: ${sample.expected}`);
                console.log(`  予測値: ${result.class} (確率: ${result.probability.toFixed(4)})`);
                console.log(`  結果: ${isCorrect ? '✓ 正解' : '✗ 不正解'}`);
                console.log('');
                
                if (isCorrect) correct++;
                total++;
            } catch (error) {
                console.log(`エラー: ${sample.path} - ${error.message}`);
            }
        } else {
            console.log(`ファイルが見つかりません: ${sample.path}`);
        }
    }
    
    console.log('=' .repeat(50));
    console.log(`テスト精度: ${correct}/${total} (${(correct/total*100).toFixed(2)}%)`);
    
    // 偽陽性データの分類精度をチェック
    console.log('\n偽陽性データの分類テスト:');
    const wrongDir = path.join(__dirname, 'result/wrong');
    const wrongFiles = fs.readdirSync(wrongDir).filter(f => f.endsWith('.png')).slice(0, 10);
    
    let fpCorrect = 0;
    for (const file of wrongFiles) {
        try {
            const result = await predictor.predict(path.join(wrongDir, file));
            const isCorrect = result.class === 'not';
            console.log(`${file}: ${result.class} (${result.probability.toFixed(4)}) ${isCorrect ? '✓' : '✗'}`);
            if (isCorrect) fpCorrect++;
        } catch (error) {
            console.log(`エラー: ${file} - ${error.message}`);
        }
    }
    
    console.log(`偽陽性データ精度: ${fpCorrect}/${wrongFiles.length} (${(fpCorrect/wrongFiles.length*100).toFixed(2)}%)`);
}

testRetrainedModel().catch(console.error);