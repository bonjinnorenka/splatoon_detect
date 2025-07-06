const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

class LogisticImagePredictor {
    constructor() {
        this.weights = null;
        this.bias = null;
        this.scaler = null;
    }

    // 画像から特徴量を抽出
    extractFeatures(imageData) {
        const pixels = imageData.data;
        const grayPixels = [];
        
        // RGBAからグレースケールに変換
        for (let i = 0; i < pixels.length; i += 4) {
            const gray = Math.round(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]);
            grayPixels.push(gray);
        }

        // 基本統計量
        const mean = grayPixels.reduce((a, b) => a + b, 0) / grayPixels.length;
        const variance = grayPixels.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / grayPixels.length;
        const std = Math.sqrt(variance);
        const min = Math.min(...grayPixels);
        const max = Math.max(...grayPixels);

        // エッジ密度（簡単なソーベルフィルター）
        const width = Math.sqrt(grayPixels.length);
        const height = width;
        let edgeCount = 0;
        
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                const gx = grayPixels[idx - 1] - grayPixels[idx + 1];
                const gy = grayPixels[idx - width] - grayPixels[idx + width];
                const gradient = Math.sqrt(gx * gx + gy * gy);
                if (gradient > 50) edgeCount++;
            }
        }
        const edgeDensity = edgeCount / (width * height);

        // ヒストグラム (8bins)
        const hist = new Array(8).fill(0);
        for (const pixel of grayPixels) {
            const bin = Math.min(Math.floor(pixel / 32), 7);
            hist[bin]++;
        }
        const histNormalized = hist.map(h => h / grayPixels.length);

        return [mean, std, min, max, edgeDensity, ...histNormalized];
    }

    // 画像を32x32にリサイズ
    async resizeImage(imagePath, size = 32) {
        const img = await loadImage(imagePath);
        const canvas = createCanvas(size, size);
        const ctx = canvas.getContext('2d');
        
        ctx.drawImage(img, 0, 0, size, size);
        return ctx.getImageData(0, 0, size, size);
    }

    // データを読み込み
    async loadData(dataDir, maxSamples = 100, includeFalsePositives = true) {
        const images = [];
        const labels = [];
        
        for (const [classIdx, className] of ['not', 'start'].entries()) {
            const classDir = path.join(dataDir, className);
            const files = fs.readdirSync(classDir).filter(f => f.endsWith('.png'));
            
            for (const file of files.slice(0, maxSamples)) {
                const imagePath = path.join(classDir, file);
                const imageData = await this.resizeImage(imagePath);
                const features = this.extractFeatures(imageData);
                
                images.push(features);
                labels.push(classIdx);
            }
        }
        
        // 偽陽性データを追加（これらは'not'クラスとして扱う）
        if (includeFalsePositives) {
            const wrongDir = path.join(path.dirname(dataDir), 'result', 'wrong');
            if (fs.existsSync(wrongDir)) {
                console.log('偽陽性データを読み込んでいます...');
                const wrongFiles = fs.readdirSync(wrongDir).filter(f => f.endsWith('.png'));
                
                for (const file of wrongFiles) {
                    try {
                        const imagePath = path.join(wrongDir, file);
                        const imageData = await this.resizeImage(imagePath);
                        const features = this.extractFeatures(imageData);
                        
                        images.push(features);
                        labels.push(0); // 偽陽性は'not'クラス（0）として扱う
                    } catch (error) {
                        console.warn(`偽陽性データの読み込みでエラー: ${file}`, error.message);
                    }
                }
                
                console.log(`${wrongFiles.length}個の偽陽性データを追加しました`);
            }
        }
        
        return { images, labels };
    }

    // 特徴量をスケーリング
    scaleFeatures(features, fit = false) {
        if (fit) {
            // 平均と標準偏差を計算
            const n = features.length;
            const means = new Array(features[0].length).fill(0);
            const stds = new Array(features[0].length).fill(0);
            
            // 平均を計算
            for (const feature of features) {
                for (let i = 0; i < feature.length; i++) {
                    means[i] += feature[i];
                }
            }
            means.forEach((mean, i) => means[i] = mean / n);
            
            // 標準偏差を計算
            for (const feature of features) {
                for (let i = 0; i < feature.length; i++) {
                    stds[i] += Math.pow(feature[i] - means[i], 2);
                }
            }
            stds.forEach((std, i) => stds[i] = Math.sqrt(std / n));
            
            this.scaler = { means, stds };
        }
        
        // スケーリング適用
        return features.map(feature => 
            feature.map((val, i) => 
                this.scaler.stds[i] === 0 ? 0 : (val - this.scaler.means[i]) / this.scaler.stds[i]
            )
        );
    }

    // シグモイド関数
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // 学習
    async train(dataDir, learningRate = 0.01, maxIter = 1000) {
        // データ読み込み
        const { images, labels } = await this.loadData(dataDir);
        
        // 特徴量スケーリング
        const scaledFeatures = this.scaleFeatures(images, true);
        
        // 重みとバイアスを初期化
        const featureCount = scaledFeatures[0].length;
        this.weights = new Array(featureCount).fill(0);
        this.bias = 0;
        
        // 学習
        for (let iter = 0; iter < maxIter; iter++) {
            let totalLoss = 0;
            
            for (let i = 0; i < scaledFeatures.length; i++) {
                const x = scaledFeatures[i];
                const y = labels[i];
                
                // 予測
                const z = this.weights.reduce((sum, w, j) => sum + w * x[j], 0) + this.bias;
                const prediction = this.sigmoid(z);
                
                // 損失
                const loss = -y * Math.log(prediction + 1e-15) - (1 - y) * Math.log(1 - prediction + 1e-15);
                totalLoss += loss;
                
                // 勾配計算
                const error = prediction - y;
                
                // 重みとバイアスの更新
                for (let j = 0; j < this.weights.length; j++) {
                    this.weights[j] -= learningRate * error * x[j];
                }
                this.bias -= learningRate * error;
            }
            
            if (iter % 100 === 0) {
                console.log(`Iteration ${iter}, Loss: ${totalLoss / scaledFeatures.length}`);
            }
        }
        
        // テスト
        await this.evaluate(scaledFeatures, labels);
    }

    // 評価
    async evaluate(features, labels) {
        let correct = 0;
        
        for (let i = 0; i < features.length; i++) {
            const x = features[i];
            const y = labels[i];
            
            const z = this.weights.reduce((sum, w, j) => sum + w * x[j], 0) + this.bias;
            const prediction = this.sigmoid(z) > 0.5 ? 1 : 0;
            
            if (prediction === y) correct++;
        }
        
        const accuracy = correct / features.length;
        console.log(`精度: ${accuracy.toFixed(4)}`);
        
        return accuracy;
    }

    // 予測
    async predict(imagePath) {
        const imageData = await this.resizeImage(imagePath);
        const features = this.extractFeatures(imageData);
        const scaledFeatures = this.scaleFeatures([features], false)[0];
        
        const z = this.weights.reduce((sum, w, j) => sum + w * scaledFeatures[j], 0) + this.bias;
        const probability = this.sigmoid(z);
        const prediction = probability > 0.5 ? 1 : 0;
        
        const classNames = ['not', 'start'];
        return {
            class: classNames[prediction],
            probability: prediction === 1 ? probability : 1 - probability
        };
    }

    // モデルを保存
    saveModel(filepath) {
        const modelData = {
            weights: this.weights,
            bias: this.bias,
            scaler: this.scaler
        };
        
        fs.writeFileSync(filepath, JSON.stringify(modelData, null, 2));
    }

    // モデルを読み込み
    loadModel(filepath) {
        const modelData = JSON.parse(fs.readFileSync(filepath, 'utf8'));
        this.weights = modelData.weights;
        this.bias = modelData.bias;
        this.scaler = modelData.scaler;
    }
}

// メイン関数
async function main() {
    const predictor = new LogisticImagePredictor();
    
    // 学習
    const dataDir = '/home/ryokuryu/splat_2/private/start_detect/data';
    await predictor.train(dataDir);
    
    // モデル保存
    predictor.saveModel('logistic_model.json');
    
    console.log('\nモデルの学習が完了しました。');
    console.log('モデルは logistic_model.json に保存されました。');
}

// 実行
if (require.main === module) {
    main().catch(console.error);
}

module.exports = LogisticImagePredictor;