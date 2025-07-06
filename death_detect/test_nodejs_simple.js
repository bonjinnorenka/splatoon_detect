import fs from 'fs';
import path from 'path';

class SimpleDeathDetectionModel {
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

    predict(features) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

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

    // Test with synthetic data that matches the training distribution
    testWithSyntheticData() {
        console.log('\n=== Testing with Synthetic Data ===');
        
        // Generate test vectors with different characteristics
        const testCases = [
            {
                name: "High intensity (death-like)",
                features: Array(this.model.num_features).fill(0).map(() => Math.random() * 50 + 200), // High RGB values
                expected: "Death"
            },
            {
                name: "Low intensity (not-death-like)",
                features: Array(this.model.num_features).fill(0).map(() => Math.random() * 100 + 50), // Low RGB values
                expected: "Not Death"
            },
            {
                name: "Medium intensity",
                features: Array(this.model.num_features).fill(0).map(() => Math.random() * 150 + 100), // Medium RGB values
                expected: "Unknown"
            },
            {
                name: "Random noise",
                features: Array(this.model.num_features).fill(0).map(() => Math.random() * 255), // Random values
                expected: "Unknown"
            },
            {
                name: "All zeros",
                features: Array(this.model.num_features).fill(0),
                expected: "Not Death"
            },
            {
                name: "All max values",
                features: Array(this.model.num_features).fill(255),
                expected: "Death"
            }
        ];

        const results = [];
        
        testCases.forEach((testCase, index) => {
            console.log(`\nTest ${index + 1}: ${testCase.name}`);
            console.log(`Expected: ${testCase.expected}`);
            
            try {
                const result = this.predict(testCase.features);
                console.log(`Predicted: ${result.label}`);
                console.log(`Probability: ${result.probability.toFixed(6)}`);
                console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
                console.log(`Logit: ${result.logit.toFixed(6)}`);
                
                results.push({
                    testCase: testCase.name,
                    expected: testCase.expected,
                    predicted: result.label,
                    probability: result.probability,
                    confidence: result.confidence,
                    logit: result.logit
                });
            } catch (error) {
                console.error(`Error: ${error.message}`);
                results.push({
                    testCase: testCase.name,
                    error: error.message
                });
            }
        });
        
        return results;
    }

    // Compare with Python model using the same input
    async compareWithPythonModel() {
        console.log('\n=== Comparing with Python Model ===');
        
        // Create a test case and save it for Python to process
        const testFeatures = Array(this.model.num_features).fill(0).map(() => Math.random() * 255);
        
        // Save test features to file for Python script to use
        fs.writeFileSync('test_features.json', JSON.stringify({
            features: testFeatures,
            timestamp: new Date().toISOString()
        }, null, 2));
        
        console.log('Test features saved to test_features.json');
        console.log('Feature vector length:', testFeatures.length);
        console.log('First 10 features:', testFeatures.slice(0, 10).map(f => f.toFixed(2)));
        
        // Make prediction with JavaScript model
        const jsResult = this.predict(testFeatures);
        console.log('\nJavaScript model result:');
        console.log(`- Prediction: ${jsResult.label}`);
        console.log(`- Probability: ${jsResult.probability.toFixed(8)}`);
        console.log(`- Logit: ${jsResult.logit.toFixed(8)}`);
        
        return {
            testFeatures,
            jsResult
        };
    }
}

async function validateModelConsistency() {
    console.log('=== Validating Model Consistency ===');
    
    try {
        // Load and verify model parameters
        const modelData = JSON.parse(fs.readFileSync('death_detection_model.json', 'utf8'));
        
        console.log('Model validation:');
        console.log(`- Coefficients: ${modelData.coefficients.length}`);
        console.log(`- Intercept: ${modelData.intercept}`);
        console.log(`- Mean values: ${modelData.mean.length}`);
        console.log(`- Scale values: ${modelData.scale.length}`);
        console.log(`- Expected features: ${modelData.num_features}`);
        console.log(`- Image size: ${modelData.image_size[0]}x${modelData.image_size[1]}`);
        
        // Verify dimensions match
        const expectedFeatures = modelData.image_size[0] * modelData.image_size[1] * 3;
        console.log(`\nDimension check:`);
        console.log(`- Expected features (64x64x3): ${expectedFeatures}`);
        console.log(`- Model num_features: ${modelData.num_features}`);
        console.log(`- Coefficients length: ${modelData.coefficients.length}`);
        console.log(`- Mean length: ${modelData.mean.length}`);
        console.log(`- Scale length: ${modelData.scale.length}`);
        
        const allMatch = (
            expectedFeatures === modelData.num_features &&
            modelData.coefficients.length === modelData.num_features &&
            modelData.mean.length === modelData.num_features &&
            modelData.scale.length === modelData.num_features
        );
        
        console.log(`\nAll dimensions match: ${allMatch ? 'Yes ✓' : 'No ✗'}`);
        
        // Check for any NaN or infinite values
        const hasInvalidCoefficients = modelData.coefficients.some(c => !isFinite(c));
        const hasInvalidMean = modelData.mean.some(m => !isFinite(m));
        const hasInvalidScale = modelData.scale.some(s => !isFinite(s) || s === 0);
        
        console.log(`\nValue validation:`);
        console.log(`- Invalid coefficients: ${hasInvalidCoefficients ? 'Yes ✗' : 'No ✓'}`);
        console.log(`- Invalid mean values: ${hasInvalidMean ? 'Yes ✗' : 'No ✓'}`);
        console.log(`- Invalid scale values: ${hasInvalidScale ? 'Yes ✗' : 'No ✓'}`);
        console.log(`- Intercept is finite: ${isFinite(modelData.intercept) ? 'Yes ✓' : 'No ✗'}`);
        
        const isValid = allMatch && !hasInvalidCoefficients && !hasInvalidMean && !hasInvalidScale && isFinite(modelData.intercept);
        
        return isValid;
        
    } catch (error) {
        console.error('Error in model validation:', error);
        throw error;
    }
}

async function runAllTests() {
    console.log('Starting Node.js Death Detection Model Tests (Simple Version)');
    console.log('='.repeat(60));
    
    try {
        // Validate model consistency first
        const modelValid = await validateModelConsistency();
        if (!modelValid) {
            throw new Error('Model validation failed');
        }
        
        // Create and test model
        const model = new SimpleDeathDetectionModel();
        await model.loadModel();
        
        // Test with synthetic data
        const syntheticResults = model.testWithSyntheticData();
        
        // Compare with Python model
        const comparisonResults = await model.compareWithPythonModel();
        
        console.log('\n' + '='.repeat(60));
        console.log('All tests completed successfully!');
        
        // Summary
        console.log('\nTest Summary:');
        console.log(`- Model validation: ${modelValid ? 'PASSED' : 'FAILED'}`);
        console.log(`- Synthetic data tests: ${syntheticResults.length} tests completed`);
        console.log(`- Cross-platform comparison: READY (see test_features.json)`);
        
        // Show some predictions
        console.log('\nSample predictions:');
        syntheticResults.slice(0, 3).forEach(result => {
            if (!result.error) {
                console.log(`- ${result.testCase}: ${result.predicted} (${(result.confidence * 100).toFixed(1)}%)`);
            }
        });
        
    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runAllTests();
}