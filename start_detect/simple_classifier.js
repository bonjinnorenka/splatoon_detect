// Simple Image Classifier - JavaScript Port
// Port of simple_classifier.py for browser/Node.js compatibility

class SimpleClassifier {
    constructor() {
        this.isNode = typeof window === 'undefined';
        this.canvas = null;
        this.ctx = null;
        
        if (!this.isNode) {
            this.canvas = document.createElement('canvas');
            this.ctx = this.canvas.getContext('2d');
        }
    }

    // Utility functions for array operations
    mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    std(arr) {
        const m = this.mean(arr);
        return Math.sqrt(arr.reduce((sq, n) => sq + Math.pow(n - m, 2), 0) / arr.length);
    }

    min(arr) {
        return Math.min(...arr);
    }

    max(arr) {
        return Math.max(...arr);
    }

    // Convert image to grayscale pixel array
    async imageToGrayscale(imageSrc, width = 32, height = 32) {
        if (this.isNode) {
            // Node.js implementation would need jimp or similar
            throw new Error('Node.js image processing not implemented in this version');
        }

        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                try {
                    this.canvas.width = width;
                    this.canvas.height = height;
                    this.ctx.drawImage(img, 0, 0, width, height);
                    
                    const imageData = this.ctx.getImageData(0, 0, width, height);
                    const pixels = imageData.data;
                    const grayscale = [];
                    
                    // Convert RGBA to grayscale
                    for (let i = 0; i < pixels.length; i += 4) {
                        const r = pixels[i];
                        const g = pixels[i + 1];
                        const b = pixels[i + 2];
                        // Standard grayscale conversion
                        const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                        grayscale.push(gray);
                    }
                    
                    resolve(grayscale);
                } catch (error) {
                    reject(error);
                }
            };
            img.onerror = reject;
            img.src = imageSrc;
        });
    }

    // Extract basic statistical features from grayscale image
    extractBasicFeatures(pixels) {
        const features = {
            mean: this.mean(pixels),
            std: this.std(pixels),
            min: this.min(pixels),
            max: this.max(pixels)
        };
        return features;
    }

    // Create histogram features
    createHistogram(pixels, bins = 8) {
        const histogram = new Array(bins).fill(0);
        const binSize = 256 / bins;
        
        for (const pixel of pixels) {
            const binIndex = Math.min(Math.floor(pixel / binSize), bins - 1);
            histogram[binIndex]++;
        }
        
        // Normalize histogram
        const total = pixels.length;
        return histogram.map(count => count / total);
    }

    // Simple edge detection (Sobel-like)
    detectEdges(pixels, width, height) {
        const edges = [];
        const threshold = 50;
        
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                
                // Sobel X
                const gx = (
                    -pixels[(y-1)*width + (x-1)] + pixels[(y-1)*width + (x+1)] +
                    -2*pixels[y*width + (x-1)] + 2*pixels[y*width + (x+1)] +
                    -pixels[(y+1)*width + (x-1)] + pixels[(y+1)*width + (x+1)]
                ) / 8;
                
                // Sobel Y
                const gy = (
                    -pixels[(y-1)*width + (x-1)] - 2*pixels[(y-1)*width + x] - pixels[(y-1)*width + (x+1)] +
                    pixels[(y+1)*width + (x-1)] + 2*pixels[(y+1)*width + x] + pixels[(y+1)*width + (x+1)]
                ) / 8;
                
                const magnitude = Math.sqrt(gx * gx + gy * gy);
                edges.push(magnitude > threshold ? 1 : 0);
            }
        }
        
        return edges;
    }

    // Calculate edge density
    calculateEdgeDensity(pixels, width, height) {
        const edges = this.detectEdges(pixels, width, height);
        return edges.reduce((sum, edge) => sum + edge, 0) / edges.length;
    }

    // Extract comprehensive features from image
    extractFeatures(pixels, width = 32, height = 32) {
        const basic = this.extractBasicFeatures(pixels);
        const histogram = this.createHistogram(pixels, 8);
        const edgeDensity = this.calculateEdgeDensity(pixels, width, height);
        
        // Downsample pixels (every 32nd pixel like in Python version)
        const downsampled = pixels.filter((_, i) => i % 32 === 0);
        
        // Combine all features
        const features = [
            basic.mean,
            basic.std,
            basic.min,
            basic.max,
            edgeDensity,
            ...histogram,
            ...downsampled
        ];
        
        return features;
    }

    // Simple threshold-based classifier
    trainThresholdClassifier(features, labels, featureIndex = 0) {
        const class0Features = [];
        const class1Features = [];
        
        for (let i = 0; i < features.length; i++) {
            if (labels[i] === 0) {
                class0Features.push(features[i][featureIndex]);
            } else {
                class1Features.push(features[i][featureIndex]);
            }
        }
        
        const allFeatures = [...class0Features, ...class1Features];
        const minVal = this.min(allFeatures);
        const maxVal = this.max(allFeatures);
        
        let bestThreshold = minVal;
        let bestAccuracy = 0;
        
        // Try different thresholds
        for (let i = 0; i < 100; i++) {
            const threshold = minVal + (maxVal - minVal) * i / 99;
            let correct = 0;
            
            for (let j = 0; j < features.length; j++) {
                const predicted = features[j][featureIndex] > threshold ? 1 : 0;
                if (predicted === labels[j]) {
                    correct++;
                }
            }
            
            const accuracy = correct / features.length;
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestThreshold = threshold;
            }
        }
        
        return {
            threshold: bestThreshold,
            accuracy: bestAccuracy,
            featureIndex: featureIndex
        };
    }

    // Predict using trained threshold classifier
    predict(features, classifier) {
        const featureValue = features[classifier.featureIndex];
        return featureValue > classifier.threshold ? 1 : 0;
    }

    // Test different simple classifiers
    testSimpleClassifiers(features, labels) {
        const results = {};
        
        // Test mean-based classifier (feature index 0)
        const meanClassifier = this.trainThresholdClassifier(features, labels, 0);
        results.meanPixel = meanClassifier;
        
        // Test std-based classifier (feature index 1)
        const stdClassifier = this.trainThresholdClassifier(features, labels, 1);
        results.stdDev = stdClassifier;
        
        // Test edge density classifier (feature index 4)
        const edgeClassifier = this.trainThresholdClassifier(features, labels, 4);
        results.edgeDensity = edgeClassifier;
        
        return results;
    }

    // Calculate accuracy
    calculateAccuracy(predictions, labels) {
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] === labels[i]) {
                correct++;
            }
        }
        return correct / predictions.length;
    }

    // Split data into train/test sets
    splitData(features, labels, testRatio = 0.3) {
        const shuffled = features.map((f, i) => ({ features: f, label: labels[i] }));
        
        // Simple shuffle
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        
        const splitIndex = Math.floor(shuffled.length * (1 - testRatio));
        
        return {
            trainFeatures: shuffled.slice(0, splitIndex).map(d => d.features),
            trainLabels: shuffled.slice(0, splitIndex).map(d => d.label),
            testFeatures: shuffled.slice(splitIndex).map(d => d.features),
            testLabels: shuffled.slice(splitIndex).map(d => d.label)
        };
    }

    // Main analysis function
    async analyzeImages(imageData) {
        console.log('Starting image analysis...');
        
        const features = [];
        const labels = [];
        
        // Process each image
        for (const item of imageData) {
            const pixels = await this.imageToGrayscale(item.src);
            const imageFeatures = this.extractFeatures(pixels);
            features.push(imageFeatures);
            labels.push(item.label);
        }
        
        console.log(`Processed ${features.length} images`);
        console.log(`Feature vector length: ${features[0].length}`);
        
        // Split data
        const split = this.splitData(features, labels);
        
        // Train simple classifiers
        const classifiers = this.testSimpleClassifiers(split.trainFeatures, split.trainLabels);
        
        // Test on test set
        const results = {};
        for (const [name, classifier] of Object.entries(classifiers)) {
            const predictions = split.testFeatures.map(f => this.predict(f, classifier));
            const accuracy = this.calculateAccuracy(predictions, split.testLabels);
            
            results[name] = {
                trainAccuracy: classifier.accuracy,
                testAccuracy: accuracy,
                threshold: classifier.threshold,
                featureIndex: classifier.featureIndex
            };
        }
        
        return results;
    }
}

// Export for both browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SimpleClassifier;
} else {
    window.SimpleClassifier = SimpleClassifier;
}