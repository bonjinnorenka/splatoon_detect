import DeathDetectionModel from './death_detection_nodejs.js';
import fs from 'fs';
import path from 'path';

async function testSingleImage() {
    console.log('=== Testing Single Image Prediction ===');
    
    const model = new DeathDetectionModel();
    
    try {
        // Load model
        console.log('Loading model...');
        await model.loadModel();
        
        // Get a sample image from each category
        const deathDir = 'data/death';
        const notDir = 'data/not';
        
        const deathFiles = fs.readdirSync(deathDir).filter(f => f.endsWith('.png'));
        const notFiles = fs.readdirSync(notDir).filter(f => f.endsWith('.png'));
        
        if (deathFiles.length === 0 || notFiles.length === 0) {
            throw new Error('No test images found');
        }
        
        // Test one image from each category
        const deathImage = path.join(deathDir, deathFiles[0]);
        const notImage = path.join(notDir, notFiles[0]);
        
        console.log(`\nTesting death image: ${deathImage}`);
        const deathResult = await model.predictFromFile(deathImage);
        console.log('Result:', deathResult);
        
        console.log(`\nTesting not-death image: ${notImage}`);
        const notResult = await model.predictFromFile(notImage);
        console.log('Result:', notResult);
        
        return { deathResult, notResult };
        
    } catch (error) {
        console.error('Error in single image test:', error);
        throw error;
    }
}

async function testBatchPrediction() {
    console.log('\n=== Testing Batch Prediction ===');
    
    const model = new DeathDetectionModel();
    
    try {
        await model.loadModel();
        
        // Get sample images for batch testing
        const deathDir = 'data/death';
        const notDir = 'data/not';
        
        const deathFiles = fs.readdirSync(deathDir)
            .filter(f => f.endsWith('.png'))
            .slice(0, 5)  // Take first 5 images
            .map(f => path.join(deathDir, f));
            
        const notFiles = fs.readdirSync(notDir)
            .filter(f => f.endsWith('.png'))
            .slice(0, 5)  // Take first 5 images
            .map(f => path.join(notDir, f));
        
        const allTestImages = [...deathFiles, ...notFiles];
        
        console.log(`Testing batch prediction on ${allTestImages.length} images...`);
        const results = await model.predictBatch(allTestImages);
        
        // Analyze results
        let correctPredictions = 0;
        let totalPredictions = 0;
        
        results.forEach((result, index) => {
            if (result.error) {
                console.log(`Error on image ${index + 1}: ${result.error}`);
                return;
            }
            
            const isDeathImage = result.imagePath.includes('/death/');
            const expectedLabel = isDeathImage ? 1 : 0;
            const isCorrect = result.prediction === expectedLabel;
            
            if (isCorrect) correctPredictions++;
            totalPredictions++;
            
            console.log(`Image ${index + 1}: ${result.imagePath}`);
            console.log(`  Expected: ${expectedLabel === 1 ? 'Death' : 'Not Death'}`);
            console.log(`  Predicted: ${result.label} (${result.confidence.toFixed(4)})`);
            console.log(`  Correct: ${isCorrect ? 'Yes' : 'No'}`);
            console.log('');
        });
        
        const accuracy = correctPredictions / totalPredictions;
        console.log(`Batch Accuracy: ${correctPredictions}/${totalPredictions} = ${(accuracy * 100).toFixed(2)}%`);
        
        return { results, accuracy };
        
    } catch (error) {
        console.error('Error in batch test:', error);
        throw error;
    }
}

async function validateModelConsistency() {
    console.log('\n=== Validating Model Consistency ===');
    
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
        
        return allMatch;
        
    } catch (error) {
        console.error('Error in model validation:', error);
        throw error;
    }
}

async function runAllTests() {
    console.log('Starting Node.js Death Detection Model Tests');
    console.log('='.repeat(50));
    
    try {
        // Validate model consistency first
        const modelValid = await validateModelConsistency();
        if (!modelValid) {
            throw new Error('Model validation failed');
        }
        
        // Test single image prediction
        const singleTestResults = await testSingleImage();
        
        // Test batch prediction
        const batchTestResults = await testBatchPrediction();
        
        console.log('\n' + '='.repeat(50));
        console.log('All tests completed successfully!');
        console.log(`Single image tests: Death=${singleTestResults.deathResult.label}, Not Death=${singleTestResults.notResult.label}`);
        console.log(`Batch test accuracy: ${(batchTestResults.accuracy * 100).toFixed(2)}%`);
        
    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    runAllTests();
}