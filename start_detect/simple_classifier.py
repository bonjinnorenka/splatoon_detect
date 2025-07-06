import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import random
import cv2

def load_and_analyze_data(data_dir, max_samples_per_class=100):
    """
    Load data and perform basic analysis
    """
    print("Loading and analyzing data...")
    
    images = []
    labels = []
    class_info = {}
    
    for class_idx, class_name in enumerate(['not', 'start']):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        
        # Random sample for faster processing
        if len(image_files) > max_samples_per_class:
            image_files = random.sample(image_files, max_samples_per_class)
        
        class_images = []
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            # Load as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize to small size for faster processing
                img_resized = cv2.resize(img, (32, 32))
                images.append(img_resized)
                labels.append(class_idx)
                class_images.append(img_resized)
        
        class_info[class_name] = {
            'count': len(class_images),
            'images': class_images[:5]  # Store first 5 for visualization
        }
        
        print(f"Loaded {len(class_images)} images for class '{class_name}'")
    
    return np.array(images), np.array(labels), class_info

def visualize_sample_images(class_info):
    """
    Visualize sample images from each class
    """
    print("Visualizing sample images...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images from Each Class', fontsize=16)
    
    for class_idx, (class_name, info) in enumerate(class_info.items()):
        for img_idx, img in enumerate(info['images']):
            axes[class_idx, img_idx].imshow(img, cmap='gray')
            axes[class_idx, img_idx].set_title(f'{class_name} - {img_idx+1}')
            axes[class_idx, img_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/ryokuryu/splat_2/private/start_detect/sample_images.png', dpi=300)
    plt.show()

def extract_features(images):
    """
    Extract various types of features from images
    """
    print("Extracting features...")
    
    features = []
    
    for img in images:
        # Flatten image
        flat_pixels = img.flatten()
        
        # Basic statistics
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        # Histogram features (reduced bins for simplicity)
        hist, _ = np.histogram(img, bins=8, range=(0, 256))
        hist_normalized = hist / np.sum(hist)
        
        # Edge features
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        # Texture features (simple)
        gray_co_matrix = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
        gray_co_matrix = gray_co_matrix / np.sum(gray_co_matrix)
        
        # Combine all features
        feature_vector = np.concatenate([
            [mean_val, std_val, min_val, max_val, edge_density],  # Basic stats
            hist_normalized,  # Histogram
            gray_co_matrix,  # Texture
            flat_pixels[::32]  # Downsampled pixels (every 32nd pixel)
        ])
        
        features.append(feature_vector)
    
    return np.array(features)

def analyze_feature_differences(features, labels, class_names=['not', 'start']):
    """
    Analyze differences between classes
    """
    print("Analyzing feature differences...")
    
    # Calculate mean features for each class
    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]
    
    mean_0 = np.mean(class_0_features, axis=0)
    mean_1 = np.mean(class_1_features, axis=0)
    
    # Calculate differences
    feature_diff = np.abs(mean_0 - mean_1)
    std_0 = np.std(class_0_features, axis=0)
    std_1 = np.std(class_1_features, axis=0)
    
    print(f"\nFeature Analysis:")
    print(f"Class '{class_names[0]}' samples: {len(class_0_features)}")
    print(f"Class '{class_names[1]}' samples: {len(class_1_features)}")
    print(f"Feature vector length: {len(mean_0)}")
    print(f"Max feature difference: {np.max(feature_diff):.4f}")
    print(f"Mean feature difference: {np.mean(feature_diff):.4f}")
    print(f"Std feature difference: {np.std(feature_diff):.4f}")
    
    # Show top differentiating features
    top_indices = np.argsort(feature_diff)[-10:]
    print(f"\nTop 10 differentiating features:")
    feature_names = (['mean', 'std', 'min', 'max', 'edge_density'] + 
                    [f'hist_{i}' for i in range(8)] + 
                    [f'texture_{i}' for i in range(8)] + 
                    [f'pixel_{i}' for i in range(len(mean_0)-21)])
    
    for idx in reversed(top_indices):
        if idx < len(feature_names):
            print(f"  {feature_names[idx]}: {feature_diff[idx]:.4f} "
                  f"(class0: {mean_0[idx]:.3f}±{std_0[idx]:.3f}, "
                  f"class1: {mean_1[idx]:.3f}±{std_1[idx]:.3f})")
    
    return feature_diff

def test_simple_models(features, labels):
    """
    Test various simple models
    """
    print("\nTesting simple models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    {cm}")
        
        # Classification report
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['not', 'start']))
    
    return results

def test_even_simpler_models(images, labels):
    """
    Test extremely simple approaches
    """
    print("\nTesting extremely simple models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    simple_results = {}
    
    # 1. Mean pixel value classifier
    print("\n1. Mean Pixel Value Classifier:")
    train_means = [np.mean(img) for img in X_train]
    test_means = [np.mean(img) for img in X_test]
    
    # Find optimal threshold
    thresholds = np.linspace(min(train_means), max(train_means), 100)
    best_threshold = None
    best_accuracy = 0
    
    for threshold in thresholds:
        pred_train = (np.array(train_means) > threshold).astype(int)
        accuracy = accuracy_score(y_train, pred_train)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Test with best threshold
    pred_test = (np.array(test_means) > best_threshold).astype(int)
    accuracy = accuracy_score(y_test, pred_test)
    
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, pred_test))
    
    simple_results['Mean Pixel'] = accuracy
    
    # 2. Standard deviation classifier
    print("\n2. Standard Deviation Classifier:")
    train_stds = [np.std(img) for img in X_train]
    test_stds = [np.std(img) for img in X_test]
    
    # Find optimal threshold
    thresholds = np.linspace(min(train_stds), max(train_stds), 100)
    best_threshold = None
    best_accuracy = 0
    
    for threshold in thresholds:
        pred_train = (np.array(train_stds) > threshold).astype(int)
        accuracy = accuracy_score(y_train, pred_train)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Test with best threshold
    pred_test = (np.array(test_stds) > best_threshold).astype(int)
    accuracy = accuracy_score(y_test, pred_test)
    
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, pred_test))
    
    simple_results['Std Dev'] = accuracy
    
    # 3. Edge density classifier
    print("\n3. Edge Density Classifier:")
    train_edges = [np.sum(cv2.Canny(img, 50, 150) > 0) / (img.shape[0] * img.shape[1]) for img in X_train]
    test_edges = [np.sum(cv2.Canny(img, 50, 150) > 0) / (img.shape[0] * img.shape[1]) for img in X_test]
    
    # Find optimal threshold
    thresholds = np.linspace(min(train_edges), max(train_edges), 100)
    best_threshold = None
    best_accuracy = 0
    
    for threshold in thresholds:
        pred_train = (np.array(train_edges) > threshold).astype(int)
        accuracy = accuracy_score(y_train, pred_train)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Test with best threshold
    pred_test = (np.array(test_edges) > best_threshold).astype(int)
    accuracy = accuracy_score(y_test, pred_test)
    
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, pred_test))
    
    simple_results['Edge Density'] = accuracy
    
    return simple_results

def main():
    """
    Main analysis function
    """
    print("Starting comprehensive simple model analysis...")
    print("=" * 60)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    data_dir = '/home/ryokuryu/splat_2/private/start_detect/data'
    images, labels, class_info = load_and_analyze_data(data_dir, max_samples_per_class=100)
    
    print(f"\nDataset summary:")
    print(f"Total images: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Visualize samples
    visualize_sample_images(class_info)
    
    # Extract features
    features = extract_features(images)
    print(f"Feature vector shape: {features.shape}")
    
    # Analyze differences
    feature_diff = analyze_feature_differences(features, labels)
    
    # Test simple models
    results = test_simple_models(features, labels)
    
    # Test extremely simple models
    simple_results = test_even_simpler_models(images, labels)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    
    print("\nSimple threshold-based classifiers:")
    for name, accuracy in simple_results.items():
        print(f"  {name}: {accuracy:.4f}")
    
    print("\nMachine learning models:")
    for name, result in results.items():
        print(f"  {name}: {result['accuracy']:.4f} (CV: {result['cv_mean']:.4f} ± {result['cv_std']:.4f})")
    
    # Determine if the problem is solvable
    max_simple_accuracy = max(simple_results.values())
    max_ml_accuracy = max([r['accuracy'] for r in results.values()])
    
    print(f"\nANALYSIS:")
    print(f"Best simple classifier accuracy: {max_simple_accuracy:.4f}")
    print(f"Best ML classifier accuracy: {max_ml_accuracy:.4f}")
    
    if max_simple_accuracy < 0.6:
        print("WARNING: Even simple classifiers perform poorly (<60%)")
        print("This suggests the classes may not be visually distinguishable")
        print("or the data may have fundamental issues.")
    elif max_simple_accuracy > 0.8:
        print("GOOD: Simple classifiers work well (>80%)")
        print("The problem is solvable with basic features.")
        print("CNN issues are likely due to overfitting or architecture.")
    else:
        print("MODERATE: Simple classifiers show modest performance (60-80%)")
        print("The problem has some signal but may need better features.")
    
    if max_ml_accuracy - max_simple_accuracy > 0.1:
        print("ML models significantly outperform simple thresholds.")
        print("Complex features are beneficial.")
    else:
        print("ML models don't significantly outperform simple thresholds.")
        print("The problem may be too simple or too noisy for complex models.")

if __name__ == "__main__":
    main()