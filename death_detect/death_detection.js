class DeathDetectionModel {
    constructor() {
        this.model = null;
        this.isLoaded = false;
    }

    async loadModel(modelPath = 'death_detection_model.json') {
        try {
            const response = await fetch(modelPath);
            this.model = await response.json();
            this.isLoaded = true;
            console.log('Model loaded successfully');
            console.log(`Image size: ${this.model.image_size[0]}x${this.model.image_size[1]}`);
            console.log(`Number of features: ${this.model.num_features}`);
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    preprocessImage(imageData, width, height) {
        // Resize image to 64x64 and normalize
        const targetSize = 64;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = targetSize;
        canvas.height = targetSize;
        
        // Create ImageData for resizing
        const resizedImageData = ctx.createImageData(targetSize, targetSize);
        
        // Simple nearest neighbor resizing
        for (let y = 0; y < targetSize; y++) {
            for (let x = 0; x < targetSize; x++) {
                const srcX = Math.floor(x * width / targetSize);
                const srcY = Math.floor(y * height / targetSize);
                
                const srcIndex = (srcY * width + srcX) * 4;
                const destIndex = (y * targetSize + x) * 4;
                
                resizedImageData.data[destIndex] = imageData[srcIndex];     // R
                resizedImageData.data[destIndex + 1] = imageData[srcIndex + 1]; // G
                resizedImageData.data[destIndex + 2] = imageData[srcIndex + 2]; // B
                resizedImageData.data[destIndex + 3] = 255; // A
            }
        }
        
        // Flatten RGB values (ignore alpha channel)
        const flattened = [];
        for (let i = 0; i < resizedImageData.data.length; i += 4) {
            flattened.push(resizedImageData.data[i]);     // R
            flattened.push(resizedImageData.data[i + 1]); // G
            flattened.push(resizedImageData.data[i + 2]); // B
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
        return 1 / (1 + Math.exp(-z));
    }

    predict(imageData, width, height) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        // Preprocess image
        const features = this.preprocessImage(imageData, width, height);
        
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
            confidence: prediction === 1 ? probability : 1 - probability
        };
    }

    async predictFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    canvas.width = img.width;
                    canvas.height = img.height;
                    
                    ctx.drawImage(img, 0, 0);
                    const imageData = ctx.getImageData(0, 0, img.width, img.height);
                    
                    try {
                        const result = this.predict(imageData.data, img.width, img.height);
                        resolve(result);
                    } catch (error) {
                        reject(error);
                    }
                };
                img.onerror = reject;
                img.src = e.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
}

// Usage example:
// const model = new DeathDetectionModel();
// await model.loadModel();
// const result = await model.predictFromFile(fileInput.files[0]);
// console.log(result);

// Export for Node.js if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeathDetectionModel;
}