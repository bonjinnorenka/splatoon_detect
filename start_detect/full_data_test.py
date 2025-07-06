import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

def load_all_data(data_dir):
    """
    Load ALL data from both classes
    """
    print("Loading ALL data...")
    start_time = time.time()
    
    images = []
    labels = []
    filenames = []
    
    for class_idx, class_name in enumerate(['not', 'start']):
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        
        print(f"Loading {len(image_files)} images from '{class_name}' class...")
        
        for i, img_file in enumerate(image_files):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(image_files)} images...")
            
            img_path = os.path.join(class_dir, img_file)
            # Load as grayscale and resize to small size
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (32, 32))
                images.append(img_resized)
                labels.append(class_idx)
                filenames.append(img_file)
        
        print(f"  Loaded {len([l for l in labels if l == class_idx])} images for class '{class_name}'")
    
    load_time = time.time() - start_time
    print(f"Data loading completed in {load_time:.2f} seconds")
    print(f"Total images: {len(images)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return np.array(images), np.array(labels), filenames

def calculate_simple_features(images):
    """
    Calculate the key features we found to be most discriminative
    """
    print("Calculating simple features...")
    start_time = time.time()
    
    features = []
    
    for i, img in enumerate(images):
        if i % 5000 == 0:
            print(f"  Processing image {i}/{len(images)}...")
        
        # The key features that worked best
        mean_val = np.mean(img)
        std_val = np.std(img)  # This was the best single feature!
        min_val = np.min(img)
        max_val = np.max(img)
        
        # Edge density
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        # Combine features
        feature_vector = [mean_val, std_val, min_val, max_val, edge_density]
        features.append(feature_vector)
    
    calc_time = time.time() - start_time
    print(f"Feature calculation completed in {calc_time:.2f} seconds")
    
    return np.array(features)

def test_std_threshold_classifier(images, labels):
    """
    Test the simple standard deviation threshold classifier on all data
    """
    print("\n" + "="*60)
    print("TESTING STANDARD DEVIATION THRESHOLD CLASSIFIER")
    print("="*60)
    
    # Calculate standard deviations for all images
    std_values = [np.std(img) for img in images]
    
    # Split data for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        std_values, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Find optimal threshold on training data
    print("Finding optimal threshold on training data...")
    thresholds = np.linspace(min(X_train), max(X_train), 1000)
    best_threshold = None
    best_accuracy = 0
    
    for threshold in thresholds:
        pred_train = (np.array(X_train) > threshold).astype(int)
        accuracy = accuracy_score(y_train, pred_train)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Best threshold found: {best_threshold:.4f}")
    print(f"Training accuracy: {best_accuracy:.4f}")
    
    # Test on test data
    pred_test = (np.array(X_test) > best_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, pred_test)
    
    print(f"\nTest Results:")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test samples: {len(y_test)}")
    
    # Detailed analysis
    cm = confusion_matrix(y_test, pred_test)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, pred_test, target_names=['not', 'start']))
    
    # Show standard deviation distributions
    std_class_0 = [std_values[i] for i in range(len(std_values)) if labels[i] == 0]
    std_class_1 = [std_values[i] for i in range(len(std_values)) if labels[i] == 1]
    
    print(f"\nStandard Deviation Analysis:")
    print(f"Class 'not' std: mean={np.mean(std_class_0):.2f}, std={np.std(std_class_0):.2f}, min={np.min(std_class_0):.2f}, max={np.max(std_class_0):.2f}")
    print(f"Class 'start' std: mean={np.mean(std_class_1):.2f}, std={np.std(std_class_1):.2f}, min={np.min(std_class_1):.2f}, max={np.max(std_class_1):.2f}")
    
    return test_accuracy, best_threshold

def test_logistic_regression(features, labels):
    """
    Test logistic regression on all data
    """
    print("\n" + "="*60)
    print("TESTING LOGISTIC REGRESSION CLASSIFIER")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training logistic regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test samples: {len(y_test)}")
    
    # Detailed analysis
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['not', 'start']))
    
    # Feature importance
    feature_names = ['mean', 'std', 'min', 'max', 'edge_density']
    coefficients = model.coef_[0]
    print(f"\nFeature Importance (coefficients):")
    for name, coef in zip(feature_names, coefficients):
        print(f"  {name}: {coef:.4f}")
    
    return accuracy

def analyze_class_separation(images, labels):
    """
    Analyze how well separated the classes are
    """
    print("\n" + "="*60)
    print("CLASS SEPARATION ANALYSIS")
    print("="*60)
    
    # Calculate key statistics for each class
    class_0_images = [images[i] for i in range(len(images)) if labels[i] == 0]
    class_1_images = [images[i] for i in range(len(images)) if labels[i] == 1]
    
    # Statistics for each class
    stats_0 = {
        'mean_pixel': [np.mean(img) for img in class_0_images],
        'std_pixel': [np.std(img) for img in class_0_images],
        'min_pixel': [np.min(img) for img in class_0_images],
        'max_pixel': [np.max(img) for img in class_0_images]
    }
    
    stats_1 = {
        'mean_pixel': [np.mean(img) for img in class_1_images],
        'std_pixel': [np.std(img) for img in class_1_images],
        'min_pixel': [np.min(img) for img in class_1_images],
        'max_pixel': [np.max(img) for img in class_1_images]
    }
    
    print(f"Class 'not' ({len(class_0_images)} samples):")
    print(f"Class 'start' ({len(class_1_images)} samples):")
    print()
    
    for stat_name in ['mean_pixel', 'std_pixel', 'min_pixel', 'max_pixel']:
        mean_0 = np.mean(stats_0[stat_name])
        std_0 = np.std(stats_0[stat_name])
        mean_1 = np.mean(stats_1[stat_name])
        std_1 = np.std(stats_1[stat_name])
        
        # Calculate separation (Cohen's d)
        pooled_std = np.sqrt(((len(class_0_images)-1)*std_0**2 + (len(class_1_images)-1)*std_1**2) / 
                           (len(class_0_images) + len(class_1_images) - 2))
        cohens_d = abs(mean_0 - mean_1) / pooled_std if pooled_std > 0 else 0
        
        print(f"{stat_name}:")
        print(f"  'not': {mean_0:.2f} ± {std_0:.2f}")
        print(f"  'start': {mean_1:.2f} ± {std_1:.2f}")
        print(f"  Cohen's d: {cohens_d:.3f} ({'Excellent' if cohens_d > 1.2 else 'Good' if cohens_d > 0.8 else 'Moderate' if cohens_d > 0.5 else 'Small'})")
        print()

def main():
    """
    Main function to test on ALL data
    """
    print("COMPREHENSIVE TEST ON ALL DATA")
    print("="*60)
    
    # Load ALL data
    data_dir = '/home/ryokuryu/splat_2/private/start_detect/data'
    images, labels, filenames = load_all_data(data_dir)
    
    # Analyze class separation
    analyze_class_separation(images, labels)
    
    # Test simple threshold classifier
    std_accuracy, threshold = test_std_threshold_classifier(images, labels)
    
    # Calculate features for ML
    features = calculate_simple_features(images)
    
    # Test logistic regression
    lr_accuracy = test_logistic_regression(features, labels)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Dataset size: {len(images)} images")
    print(f"Class distribution: not={np.sum(labels==0)}, start={np.sum(labels==1)}")
    print(f"Standard Deviation Threshold accuracy: {std_accuracy:.4f}")
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")
    
    if std_accuracy > 0.95:
        print("\nCONCLUSION: Problem is EASILY solvable with simple methods!")
        print("CNN failure was due to overengineering a simple problem.")
    elif std_accuracy > 0.85:
        print("\nCONCLUSION: Problem is well-suited for simple methods.")
        print("CNN may work but is unnecessarily complex.")
    else:
        print("\nCONCLUSION: Problem requires more sophisticated methods.")
        print("CNN failure may indicate fundamental data issues.")

if __name__ == "__main__":
    main()