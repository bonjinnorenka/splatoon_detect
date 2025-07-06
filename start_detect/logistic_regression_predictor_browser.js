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

    // モデルを読み込み (ブラウザ版)
    async loadModel(filepath) {
        try {
            const response = await fetch(filepath);
            const modelData = await response.json();
            this.weights = modelData.weights;
            this.bias = modelData.bias;
            this.scaler = modelData.scaler;
            return true;
        } catch (error) {
            console.error('モデルの読み込みに失敗:', error);
            return false;
        }
    }

    // 画像を32x32にリサイズ
    resizeImageData(imageData, targetSize = 32) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // 元の画像をcanvasに描画
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        ctx.putImageData(imageData, 0, 0);
        
        // リサイズ用の新しいcanvasを作成
        const resizedCanvas = document.createElement('canvas');
        const resizedCtx = resizedCanvas.getContext('2d');
        resizedCanvas.width = targetSize;
        resizedCanvas.height = targetSize;
        
        // リサイズして描画
        resizedCtx.drawImage(canvas, 0, 0, targetSize, targetSize);
        
        return resizedCtx.getImageData(0, 0, targetSize, targetSize);
    }

    // 予測 (ImageDataを直接使用)
    predict(imageData) {
        if (!this.weights || !this.bias || !this.scaler) {
            throw new Error('モデルが読み込まれていません');
        }

        // 32x32にリサイズ
        const resizedImageData = this.resizeImageData(imageData, 32);
        const features = this.extractFeatures(resizedImageData);
        const scaledFeatures = this.scaleFeatures([features], false)[0];
        
        const z = this.weights.reduce((sum, w, j) => sum + w * scaledFeatures[j], 0) + this.bias;
        const probability = this.sigmoid(z);
        const prediction = probability > 0.5 ? 1 : 0;
        
        const classNames = ['not', 'start'];
        return {
            class: classNames[prediction],
            probability: prediction === 1 ? probability : 1 - probability,
            prediction: prediction
        };
    }
}