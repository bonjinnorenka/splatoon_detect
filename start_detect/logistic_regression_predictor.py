import numpy as np
import os
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

class LogisticImagePredictor:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
    def extract_features(self, images):
        features = []
        for img in images:
            # 基本統計量
            mean_val = np.mean(img)
            std_val = np.std(img)
            min_val = np.min(img)
            max_val = np.max(img)
            
            # エッジ密度
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            
            # ヒストグラム
            hist, _ = np.histogram(img, bins=8, range=(0, 256))
            hist_normalized = hist / np.sum(hist)
            
            # 特徴量を結合
            feature_vector = np.concatenate([
                [mean_val, std_val, min_val, max_val, edge_density],
                hist_normalized
            ])
            features.append(feature_vector)
        
        return np.array(features)
    
    def load_data(self, data_dir, max_samples=100):
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(['not', 'start']):
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            
            for img_file in image_files[:max_samples]:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (32, 32))
                    images.append(img_resized)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def train(self, data_dir):
        # データ読み込み
        images, labels = self.load_data(data_dir)
        
        # 特徴量抽出
        features = self.extract_features(images)
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 学習
        self.model.fit(X_train_scaled, y_train)
        
        # テスト
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"精度: {accuracy:.4f}")
        print(f"分類レポート:")
        print(classification_report(y_test, y_pred, target_names=['not', 'start']))
        
        return accuracy
    
    def predict(self, image_path):
        # 画像読み込み
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (32, 32))
        
        # 特徴量抽出
        features = self.extract_features([img_resized])
        features_scaled = self.scaler.transform(features)
        
        # 予測
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        class_names = ['not', 'start']
        return class_names[prediction], probability
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']

def main():
    predictor = LogisticImagePredictor()
    
    # 学習
    data_dir = '/home/ryokuryu/splat_2/private/start_detect/data'
    accuracy = predictor.train(data_dir)
    
    # モデル保存
    predictor.save_model('logistic_model.pkl')
    
    print(f"\nモデルの学習が完了しました。精度: {accuracy:.4f}")
    print("モデルは 'logistic_model.pkl' に保存されました。")

if __name__ == "__main__":
    main()