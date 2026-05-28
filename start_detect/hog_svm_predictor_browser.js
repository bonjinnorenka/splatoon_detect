class HOGSVMImagePredictor {
    constructor() {
        this.weights = null;
        this.bias = 0;
        this.decisionThreshold = 0;
        this.scaler = null;
        this.calibration = null;
        this.recommendedProbabilityThreshold = 0.5;
        this.hogConfig = {
            image_size: 64,
            orientations: 9,
            pixels_per_cell: 8,
            cells_per_block: 2,
            block_norm_epsilon: 1e-6,
            block_clip: 0.2
        };
        this.classNames = ['not', 'start'];
    }

    async loadModel(filepath) {
        try {
            const response = await fetch(filepath);
            const modelData = await response.json();
            this.weights = modelData.weights;
            this.bias = modelData.bias;
            this.decisionThreshold = modelData.decision_threshold || 0;
            this.scaler = modelData.scaler;
            this.calibration = modelData.calibration || { coef: 1, intercept: 0 };
            this.recommendedProbabilityThreshold = modelData.recommended_probability_threshold || 0.5;
            this.hogConfig = modelData.hog_config || this.hogConfig;
            this.classNames = modelData.class_names || this.classNames;
            return true;
        } catch (error) {
            console.error('HOG+SVM model load failed:', error);
            return false;
        }
    }

    resizeImageData(imageData, targetSize) {
        const sourceCanvas = document.createElement('canvas');
        const sourceCtx = sourceCanvas.getContext('2d');
        sourceCanvas.width = imageData.width;
        sourceCanvas.height = imageData.height;
        sourceCtx.putImageData(imageData, 0, 0);

        const resizedCanvas = document.createElement('canvas');
        const resizedCtx = resizedCanvas.getContext('2d');
        resizedCanvas.width = targetSize;
        resizedCanvas.height = targetSize;
        resizedCtx.imageSmoothingEnabled = true;
        resizedCtx.drawImage(sourceCanvas, 0, 0, targetSize, targetSize);

        return resizedCtx.getImageData(0, 0, targetSize, targetSize);
    }

    imageDataToGray(imageData) {
        const pixels = imageData.data;
        const gray = new Float32Array(imageData.width * imageData.height);
        for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
            gray[j] = Math.round(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]);
        }
        return gray;
    }

    extractFeatures(imageData) {
        const config = this.hogConfig;
        const targetSize = config.image_size;
        const resized = imageData.width === targetSize && imageData.height === targetSize
            ? imageData
            : this.resizeImageData(imageData, targetSize);
        const gray = this.imageDataToGray(resized);
        return this.extractHOGFromGray(gray, targetSize, targetSize, config);
    }

    extractHOGFromGray(gray, width, height, config) {
        const orientations = config.orientations;
        const pixelsPerCell = config.pixels_per_cell;
        const cellsPerBlock = config.cells_per_block;
        const epsilon = config.block_norm_epsilon;
        const clipValue = config.block_clip;
        const binWidth = 180 / orientations;
        const pixelCount = width * height;
        const gx = new Float32Array(pixelCount);
        const gy = new Float32Array(pixelCount);
        const magnitude = new Float32Array(pixelCount);
        const lowerBin = new Int32Array(pixelCount);
        const upperBin = new Int32Array(pixelCount);
        const lowerWeight = new Float32Array(pixelCount);
        const upperWeight = new Float32Array(pixelCount);

        for (let y = 0; y < height; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                gx[idx] = gray[idx + 1] - gray[idx - 1];
            }
        }

        for (let y = 1; y < height - 1; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                gy[idx] = gray[idx + width] - gray[idx - width];
            }
        }

        for (let i = 0; i < pixelCount; i++) {
            const gradX = gx[i];
            const gradY = gy[i];
            magnitude[i] = Math.sqrt(gradX * gradX + gradY * gradY);
            let angle = Math.atan2(gradY, gradX) * 180 / Math.PI;
            if (angle < 0) {
                angle += 180;
            }
            if (angle >= 180) {
                angle -= 180;
            }

            const binPosition = angle / binWidth;
            const low = Math.floor(binPosition) % orientations;
            const frac = binPosition - Math.floor(binPosition);
            lowerBin[i] = low;
            upperBin[i] = (low + 1) % orientations;
            upperWeight[i] = frac;
            lowerWeight[i] = 1 - frac;
        }

        const cellsY = Math.floor(height / pixelsPerCell);
        const cellsX = Math.floor(width / pixelsPerCell);
        const cellHist = new Float32Array(cellsY * cellsX * orientations);

        for (let y = 0; y < cellsY * pixelsPerCell; y++) {
            const cellY = Math.floor(y / pixelsPerCell);
            for (let x = 0; x < cellsX * pixelsPerCell; x++) {
                const cellX = Math.floor(x / pixelsPerCell);
                const pixelIdx = y * width + x;
                const histBase = (cellY * cellsX + cellX) * orientations;
                const mag = magnitude[pixelIdx];
                cellHist[histBase + lowerBin[pixelIdx]] += mag * lowerWeight[pixelIdx];
                cellHist[histBase + upperBin[pixelIdx]] += mag * upperWeight[pixelIdx];
            }
        }

        const blockFeatures = [];
        for (let y = 0; y <= cellsY - cellsPerBlock; y++) {
            for (let x = 0; x <= cellsX - cellsPerBlock; x++) {
                const block = [];
                for (let by = 0; by < cellsPerBlock; by++) {
                    for (let bx = 0; bx < cellsPerBlock; bx++) {
                        const histBase = ((y + by) * cellsX + (x + bx)) * orientations;
                        for (let bin = 0; bin < orientations; bin++) {
                            block.push(cellHist[histBase + bin]);
                        }
                    }
                }

                this.normalizeBlock(block, epsilon);
                for (let i = 0; i < block.length; i++) {
                    if (block[i] > clipValue) {
                        block[i] = clipValue;
                    }
                }
                this.normalizeBlock(block, epsilon);
                for (const value of block) {
                    blockFeatures.push(value);
                }
            }
        }

        return blockFeatures;
    }

    normalizeBlock(block, epsilon) {
        let sumSquares = 0;
        for (const value of block) {
            sumSquares += value * value;
        }
        const norm = Math.sqrt(sumSquares + epsilon * epsilon);
        for (let i = 0; i < block.length; i++) {
            block[i] /= norm;
        }
    }

    scaleFeatures(features) {
        const means = this.scaler.means;
        const stds = this.scaler.stds;
        return features.map((value, index) => {
            const std = stds[index] === 0 ? 1 : stds[index];
            return (value - means[index]) / std;
        });
    }

    sigmoid(value) {
        const clipped = Math.max(-50, Math.min(50, value));
        return 1 / (1 + Math.exp(-clipped));
    }

    predict(imageData) {
        if (!this.weights || !this.scaler) {
            throw new Error('HOG+SVM model is not loaded');
        }

        const features = this.extractFeatures(imageData);
        const scaledFeatures = this.scaleFeatures(features);
        let score = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            score += this.weights[i] * scaledFeatures[i];
        }

        const shiftedScore = score - this.decisionThreshold;
        const startProbability = this.sigmoid(this.calibration.coef * shiftedScore + this.calibration.intercept);
        const prediction = score >= this.decisionThreshold ? 1 : 0;
        const probability = prediction === 1 ? startProbability : 1 - startProbability;

        return {
            class: this.classNames[prediction],
            probability,
            startProbability,
            prediction,
            score
        };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = HOGSVMImagePredictor;
}
