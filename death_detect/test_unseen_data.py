import os
import cv2
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_test_images_from_folder(folder_path, label, start_idx=None, end_idx=None):
    """Load test images from a specific range to ensure they weren't used in training"""
    images = []
    labels = []
    
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    
    if start_idx is not None and end_idx is not None:
        image_files = image_files[start_idx:end_idx]
    
    for filename in tqdm(image_files, desc=f"Loading {os.path.basename(folder_path)} test images"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_flattened = img.flatten()
            images.append(img_flattened)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def create_unseen_test_set():
    """Create a test set from images that weren't used in training"""
    data_path = "data"
    death_path = os.path.join(data_path, "death")
    not_path = os.path.join(data_path, "not")
    
    # Get total counts
    death_files = sorted([f for f in os.listdir(death_path) if f.lower().endswith('.png')])
    not_files = sorted([f for f in os.listdir(not_path) if f.lower().endswith('.png')])
    
    print(f"Total death images available: {len(death_files)}")
    print(f"Total not-death images available: {len(not_files)}")
    
    # Take the last 20% of images as unseen test set (these weren't used in training due to random split)
    # Taking from the end to ensure different distribution
    death_test_start = int(len(death_files) * 0.8)
    not_test_start = int(len(not_files) * 0.8)
    
    print(f"Using death images from index {death_test_start} to {len(death_files)} as unseen test set")
    print(f"Using not-death images from index {not_test_start} to {len(not_files)} as unseen test set")
    
    # Load unseen test images
    print("Loading unseen death test images...")
    death_test_images, death_test_labels = load_test_images_from_folder(
        death_path, 1, death_test_start, len(death_files)
    )
    
    print("Loading unseen not-death test images...")
    not_test_images, not_test_labels = load_test_images_from_folder(
        not_path, 0, not_test_start, len(not_files)
    )
    
    return death_test_images, death_test_labels, not_test_images, not_test_labels

def test_model_performance():
    """Test the saved model on completely unseen data"""
    
    # Load the saved model and scaler
    print("Loading saved model and scaler...")
    model = joblib.load("death_detection_model.pkl")
    scaler = joblib.load("death_detection_scaler.pkl")
    
    # Load model parameters for comparison
    with open("death_detection_model.json", 'r') as f:
        model_params = json.load(f)
    
    print("Model loaded successfully!")
    print(f"Model coefficients shape: {len(model_params['coefficients'])}")
    print(f"Expected image size: {model_params['image_size']}")
    
    # Create unseen test set
    death_test_images, death_test_labels, not_test_images, not_test_labels = create_unseen_test_set()
    
    print(f"\nUnseen test set:")
    print(f"Death images: {len(death_test_images)}")
    print(f"Not-death images: {len(not_test_images)}")
    
    # Combine test data
    X_test_unseen = np.concatenate([death_test_images, not_test_images])
    y_test_unseen = np.concatenate([death_test_labels, not_test_labels])
    
    print(f"Total unseen test samples: {len(X_test_unseen)}")
    
    # Preprocess test data
    print("Preprocessing unseen test data...")
    X_test_scaled = scaler.transform(X_test_unseen)
    
    # Make predictions
    print("Making predictions on unseen data...")
    y_pred_unseen = model.predict(X_test_scaled)
    y_pred_proba_unseen = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy_unseen = accuracy_score(y_test_unseen, y_pred_unseen)
    
    print(f"\n=== Unseen Data Test Results ===")
    print(f"Accuracy on unseen data: {accuracy_unseen:.4f} ({accuracy_unseen*100:.2f}%)")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test_unseen, y_pred_unseen, target_names=['Not Death', 'Death']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_unseen, y_pred_unseen)
    print(f"[[TN: {cm[0,0]}, FP: {cm[0,1]}],")
    print(f" [FN: {cm[1,0]}, TP: {cm[1,1]}]]")
    
    # Analyze prediction confidence
    print(f"\nPrediction Confidence Analysis:")
    death_indices = y_test_unseen == 1
    not_death_indices = y_test_unseen == 0
    
    death_confidences = y_pred_proba_unseen[death_indices, 1]
    not_death_confidences = y_pred_proba_unseen[not_death_indices, 0]
    
    print(f"Death predictions - Mean confidence: {death_confidences.mean():.4f}, Std: {death_confidences.std():.4f}")
    print(f"Not-death predictions - Mean confidence: {not_death_confidences.mean():.4f}, Std: {not_death_confidences.std():.4f}")
    
    # Test some individual predictions
    print(f"\nSample Predictions (first 10):")
    for i in range(min(10, len(y_test_unseen))):
        true_label = "Death" if y_test_unseen[i] == 1 else "Not Death"
        pred_label = "Death" if y_pred_unseen[i] == 1 else "Not Death"
        confidence = max(y_pred_proba_unseen[i])
        print(f"Sample {i+1}: True={true_label}, Pred={pred_label}, Confidence={confidence:.4f}")

def test_javascript_model():
    """Test if the JavaScript model produces the same results as Python model"""
    print(f"\n=== JavaScript Model Verification ===")
    
    # Load model parameters
    with open("death_detection_model.json", 'r') as f:
        js_model = json.load(f)
    
    # Load Python model for comparison
    model = joblib.load("death_detection_model.pkl")
    scaler = joblib.load("death_detection_scaler.pkl")
    
    print(f"JavaScript model parameters loaded successfully")
    print(f"Coefficients: {len(js_model['coefficients'])}")
    print(f"Intercept: {js_model['intercept']:.6f}")
    print(f"Feature scaling parameters: mean={len(js_model['mean'])}, scale={len(js_model['scale'])}")
    
    # Compare with Python model
    print(f"\nModel comparison:")
    print(f"Python model intercept: {model.intercept_[0]:.6f}")
    print(f"JavaScript model intercept: {js_model['intercept']:.6f}")
    print(f"Intercept difference: {abs(model.intercept_[0] - js_model['intercept']):.10f}")
    
    coef_diff = np.abs(model.coef_[0] - np.array(js_model['coefficients']))
    print(f"Max coefficient difference: {coef_diff.max():.10f}")
    print(f"Mean coefficient difference: {coef_diff.mean():.10f}")

if __name__ == "__main__":
    test_model_performance()
    test_javascript_model()