# Simple Image Classifier - JavaScript Port

This is a JavaScript port of `simple_classifier.py` that works in both browser and Node.js environments.

## Files

- `simple_classifier.js` - Main classifier implementation
- `test_classifier.html` - Browser-based testing interface
- `test_node.js` - Node.js testing script

## Features

- **Feature Extraction**: Mean, standard deviation, min/max values, histograms, edge density
- **Classification**: Simple threshold-based classifiers
- **Cross-platform**: Works in browser and Node.js
- **Real-time**: Process images in browser via Canvas API

## Browser Usage

1. Open `test_classifier.html` in a web browser
2. Upload images to "Class 0 (Not Start)" and "Class 1 (Start)" sections
3. Click "Analyze Images" to run classification
4. View results showing accuracy of different classifiers

## Node.js Usage

```bash
node test_node.js
```

This runs a comprehensive test with synthetic data to verify all functionality.

## Implementation Details

The JavaScript port implements:
- Image processing using Canvas API (browser)
- Feature extraction matching the Python version
- Simple threshold-based classifiers (Mean Pixel, Standard Deviation, Edge Density)
- Train/test data splitting
- Accuracy calculation and evaluation

## Performance

The classifier achieves similar results to the Python version:
- Mean pixel classifier typically performs best on synthetic data
- Edge density and standard deviation provide additional discriminative features
- Results depend on the visual distinguishability of the image classes

## Limitations

- Node.js version doesn't include image loading (would need additional libraries like jimp)
- Browser version requires user to upload images manually
- No advanced ML algorithms (SVM, Random Forest) - only simple threshold classifiers