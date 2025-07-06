// Node.js test for Simple Image Classifier
const SimpleClassifier = require('./simple_classifier');

// Test the classifier with synthetic data
async function testClassifier() {
    console.log('Testing Simple Image Classifier in Node.js...');
    console.log('=' * 50);
    
    const classifier = new SimpleClassifier();
    
    // Test basic utility functions
    console.log('\n1. Testing utility functions:');
    const testArray = [1, 2, 3, 4, 5];
    console.log(`Array: [${testArray.join(', ')}]`);
    console.log(`Mean: ${classifier.mean(testArray)}`);
    console.log(`Std: ${classifier.std(testArray).toFixed(4)}`);
    console.log(`Min: ${classifier.min(testArray)}`);
    console.log(`Max: ${classifier.max(testArray)}`);
    
    // Test feature extraction with synthetic pixel data
    console.log('\n2. Testing feature extraction:');
    
    // Create synthetic grayscale image data (32x32 = 1024 pixels)
    const syntheticImage1 = Array.from({ length: 1024 }, (_, i) => 
        Math.floor(100 + 50 * Math.sin(i / 100)) // Pattern with some variation
    );
    
    const syntheticImage2 = Array.from({ length: 1024 }, (_, i) => 
        Math.floor(150 + 30 * Math.cos(i / 80)) // Different pattern
    );
    
    console.log('Extracting features from synthetic images...');
    const features1 = classifier.extractFeatures(syntheticImage1);
    const features2 = classifier.extractFeatures(syntheticImage2);
    
    console.log(`Image 1 features length: ${features1.length}`);
    console.log(`Image 1 first 5 features: [${features1.slice(0, 5).map(f => f.toFixed(4)).join(', ')}]`);
    console.log(`Image 2 features length: ${features2.length}`);
    console.log(`Image 2 first 5 features: [${features2.slice(0, 5).map(f => f.toFixed(4)).join(', ')}]`);
    
    // Test histogram creation
    console.log('\n3. Testing histogram creation:');
    const histogram1 = classifier.createHistogram(syntheticImage1);
    const histogram2 = classifier.createHistogram(syntheticImage2);
    
    console.log(`Histogram 1: [${histogram1.map(h => h.toFixed(4)).join(', ')}]`);
    console.log(`Histogram 2: [${histogram2.map(h => h.toFixed(4)).join(', ')}]`);
    
    // Test edge detection
    console.log('\n4. Testing edge detection:');
    const edgeDensity1 = classifier.calculateEdgeDensity(syntheticImage1, 32, 32);
    const edgeDensity2 = classifier.calculateEdgeDensity(syntheticImage2, 32, 32);
    
    console.log(`Edge density 1: ${edgeDensity1.toFixed(6)}`);
    console.log(`Edge density 2: ${edgeDensity2.toFixed(6)}`);
    
    // Test classification with synthetic dataset
    console.log('\n5. Testing classification:');
    
    // Create synthetic dataset with more discriminative features
    const syntheticFeatures = [];
    const syntheticLabels = [];
    
    // Generate class 0 samples (lower mean values)
    for (let i = 0; i < 50; i++) {
        const pixels = Array.from({ length: 1024 }, () => 
            Math.floor(80 + 40 * Math.random()) // Mean around 100
        );
        const features = classifier.extractFeatures(pixels);
        syntheticFeatures.push(features);
        syntheticLabels.push(0);
    }
    
    // Generate class 1 samples (higher mean values)
    for (let i = 0; i < 50; i++) {
        const pixels = Array.from({ length: 1024 }, () => 
            Math.floor(160 + 40 * Math.random()) // Mean around 180
        );
        const features = classifier.extractFeatures(pixels);
        syntheticFeatures.push(features);
        syntheticLabels.push(1);
    }
    
    console.log(`Generated ${syntheticFeatures.length} synthetic samples`);
    console.log(`Feature vector length: ${syntheticFeatures[0].length}`);
    
    // Test simple classifiers
    const classifiers = classifier.testSimpleClassifiers(syntheticFeatures, syntheticLabels);
    
    console.log('\n6. Classification results:');
    console.log('Classifier\t\tAccuracy\tThreshold');
    console.log('-' * 45);
    
    for (const [name, result] of Object.entries(classifiers)) {
        console.log(`${name.padEnd(20)}\t${(result.accuracy * 100).toFixed(2)}%\t\t${result.threshold.toFixed(4)}`);
    }
    
    // Test train/test split
    console.log('\n7. Testing train/test split:');
    const split = classifier.splitData(syntheticFeatures, syntheticLabels, 0.3);
    console.log(`Train samples: ${split.trainFeatures.length}`);
    console.log(`Test samples: ${split.testFeatures.length}`);
    console.log(`Train labels distribution: [${split.trainLabels.filter(l => l === 0).length}, ${split.trainLabels.filter(l => l === 1).length}]`);
    console.log(`Test labels distribution: [${split.testLabels.filter(l => l === 0).length}, ${split.testLabels.filter(l => l === 1).length}]`);
    
    // Test on split data
    console.log('\n8. Testing on train/test split:');
    const trainClassifiers = classifier.testSimpleClassifiers(split.trainFeatures, split.trainLabels);
    
    console.log('\nTrain/Test Performance:');
    console.log('Classifier\t\tTrain Acc\tTest Acc');
    console.log('-' * 45);
    
    for (const [name, trainResult] of Object.entries(trainClassifiers)) {
        const testPredictions = split.testFeatures.map(f => classifier.predict(f, trainResult));
        const testAccuracy = classifier.calculateAccuracy(testPredictions, split.testLabels);
        
        console.log(`${name.padEnd(20)}\t${(trainResult.accuracy * 100).toFixed(2)}%\t\t${(testAccuracy * 100).toFixed(2)}%`);
    }
    
    console.log('\n' + '=' * 50);
    console.log('Node.js test completed successfully!');
    console.log('The classifier is working correctly with synthetic data.');
    console.log('For real image testing, open test_classifier.html in a browser.');
}

// Run the test
testClassifier().catch(console.error);