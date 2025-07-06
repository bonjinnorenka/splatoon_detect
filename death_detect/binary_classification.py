import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_images_from_folder(folder_path, label, max_images=None):
    images = []
    labels = []
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    if max_images:
        image_files = image_files[:max_images]
    
    for filename in tqdm(image_files, desc=f"Loading {os.path.basename(folder_path)} images"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_flattened = img.flatten()
            images.append(img_flattened)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def main():
    data_path = "data"
    death_path = os.path.join(data_path, "death")
    not_path = os.path.join(data_path, "not")
    
    print("Loading death images...")
    death_images, death_labels = load_images_from_folder(death_path, 1)
    
    print("Loading not-death images...")
    not_images, not_labels = load_images_from_folder(not_path, 0)
    
    print(f"Death images: {len(death_images)}")
    print(f"Not-death images: {len(not_images)}")
    
    X = np.concatenate([death_images, not_images])
    y = np.concatenate([death_labels, not_labels])
    
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Death', 'Death']))
    
    print(f"\nClass distribution in test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        class_name = 'Not Death' if cls == 0 else 'Death'
        print(f"{class_name}: {count} samples")

if __name__ == "__main__":
    main()