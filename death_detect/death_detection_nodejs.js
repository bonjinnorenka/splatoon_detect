import fs from 'fs';
import { createCanvas, loadImage } from 'canvas';

class DeathDetectionModel {
    constructor() {
        this.model = null;
        this.isLoaded = false;
    }

    async loadModel(modelPath = 'death_detection_model.json') {
        try {
            const modelData = fs.readFileSync(modelPath, 'utf8');
            this.model = JSON.parse(modelData);
            this.isLoaded = true;
            console.log('Model loaded successfully');
            console.log(`Image size: ${this.model.image_size[0]}x${this.model.image_size[1]}`);
            console.log(`Number of features: ${this.model.num_features}`);
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    preprocessImage(imageData, width, height) {
        // Resize image to 64x64 and normalize
        const targetSize = 64;
        
        // Create resized image data array
        const resizedImageData = new Uint8ClampedArray(targetSize * targetSize * 4);
        
        // Simple nearest neighbor resizing
        for (let y = 0; y < targetSize; y++) {
            for (let x = 0; x < targetSize; x++) {
                const srcX = Math.floor(x * width / targetSize);
                const srcY = Math.floor(y * height / targetSize);
                
                const srcIndex = (srcY * width + srcX) * 4;
                const destIndex = (y * targetSize + x) * 4;
                
                resizedImageData[destIndex] = imageData[srcIndex];     // R
                resizedImageData[destIndex + 1] = imageData[srcIndex + 1]; // G
                resizedImageData[destIndex + 2] = imageData[srcIndex + 2]; // B
                resizedImageData[destIndex + 3] = 255; // A
            }
        }
        
        // Flatten RGB values (ignore alpha channel)
        const flattened = [];
        for (let i = 0; i < resizedImageData.length; i += 4) {
            flattened.push(resizedImageData[i]);     // R
            flattened.push(resizedImageData[i + 1]); // G
            flattened.push(resizedImageData[i + 2]); // B
        }
        
        return flattened;
    }

    standardizeFeatures(features) {
        // Apply standardization: (x - mean) / scale
        const standardized = [];
        for (let i = 0; i < features.length; i++) {
            standardized.push((features[i] - this.model.mean[i]) / this.model.scale[i]);
        }
        return standardized;
    }

    sigmoid(z) {
        // Prevent overflow by clamping z
        if (z > 500) return 1.0;
        if (z < -500) return 0.0;
        return 1 / (1 + Math.exp(-z));
    }

    predict(imageData, width, height) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        // Preprocess image
        const features = this.preprocessImage(imageData, width, height);
        
        if (features.length !== this.model.num_features) {
            throw new Error(`Feature count mismatch. Expected ${this.model.num_features}, got ${features.length}`);
        }
        
        // Standardize features
        const standardizedFeatures = this.standardizeFeatures(features);
        
        // Calculate logistic regression: sigmoid(w * x + b)
        let z = this.model.intercept;
        for (let i = 0; i < standardizedFeatures.length; i++) {
            z += this.model.coefficients[i] * standardizedFeatures[i];
        }
        
        const probability = this.sigmoid(z);
        const prediction = probability > 0.5 ? 1 : 0;
        
        return {
            prediction: prediction,
            probability: probability,
            label: prediction === 1 ? 'Death' : 'Not Death',
            confidence: prediction === 1 ? probability : 1 - probability,
            logit: z
        };
    }

    async predictFromFile(imagePath) {
        try {
            // Load image using canvas
            const image = await loadImage(imagePath);
            
            // Create canvas and get image data
            const canvas = createCanvas(image.width, image.height);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0);
            
            const imageData = ctx.getImageData(0, 0, image.width, image.height);
            
            // Make prediction
            const result = this.predict(imageData.data, image.width, image.height);
            
            return {
                ...result,
                imagePath: imagePath,
                imageSize: [image.width, image.height]
            };
        } catch (error) {
            console.error(`Error processing image ${imagePath}:`, error);
            throw error;
        }
    }

    // Batch prediction for multiple images
    async predictBatch(imagePaths, showProgress = true) {
        const results = [];
        
        for (let i = 0; i < imagePaths.length; i++) {
            if (showProgress && i % 10 === 0) {
                console.log(`Processing image ${i + 1}/${imagePaths.length}...`);
            }
            
            try {
                const result = await this.predictFromFile(imagePaths[i]);
                results.push(result);
            } catch (error) {
                console.error(`Failed to process ${imagePaths[i]}:`, error.message);
                results.push({
                    imagePath: imagePaths[i],
                    error: error.message,
                    prediction: null
                });
            }
        }
        
        return results;
    }
}

export default DeathDetectionModel;