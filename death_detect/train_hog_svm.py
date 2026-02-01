#!/usr/bin/env python3
"""
HOG + SVM Death Detection Training Script

OpenCVのHOG特徴量とSVM分類器を使用したdeath detection モデルの学習スクリプト
"""

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import json
import warnings
import time

warnings.filterwarnings('ignore')


# HOG特徴量パラメータ
HOG_PARAMS = {
    'winSize': (64, 64),
    'blockSize': (16, 16),
    'blockStride': (8, 8),
    'cellSize': (8, 8),
    'nbins': 9
}


def create_hog_descriptor():
    """HOGDescriptorを作成"""
    return cv2.HOGDescriptor(
        _winSize=HOG_PARAMS['winSize'],
        _blockSize=HOG_PARAMS['blockSize'],
        _blockStride=HOG_PARAMS['blockStride'],
        _cellSize=HOG_PARAMS['cellSize'],
        _nbins=HOG_PARAMS['nbins']
    )


def extract_hog_features(image, hog_descriptor):
    """
    画像からHOG特徴量を抽出
    
    Args:
        image: BGR画像（OpenCV形式）
        hog_descriptor: cv2.HOGDescriptor インスタンス
    
    Returns:
        HOG特徴量ベクトル（1次元配列）
    """
    # グレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # リサイズ
    gray = cv2.resize(gray, HOG_PARAMS['winSize'])
    
    # HOG特徴量を計算
    features = hog_descriptor.compute(gray)
    
    return features.flatten()


def load_images_from_folder(folder_path, label, hog_descriptor, max_images=None, shuffle=False):
    """
    フォルダから画像を読み込みHOG特徴量を抽出
    
    Args:
        folder_path: 画像フォルダのパス
        label: ラベル（0または1）
        hog_descriptor: HOGDescriptorインスタンス
        max_images: 読み込む最大画像数（Noneで全て）
        shuffle: 画像をシャッフルするかどうか
    
    Returns:
        特徴量配列、ラベル配列
    """
    features = []
    labels = []
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # シャッフルしてからサンプリング
    if shuffle:
        np.random.shuffle(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    for filename in tqdm(image_files, desc=f"Loading {os.path.basename(folder_path)} images"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            hog_features = extract_hog_features(img, hog_descriptor)
            features.append(hog_features)
            labels.append(label)
    
    return np.array(features), np.array(labels)


def train_svm_with_gridsearch(X_train, y_train, cv_folds=3):
    """
    GridSearchCVを使用してSVMをチューニング・学習
    
    Args:
        X_train: 学習用特徴量
        y_train: 学習用ラベル
        cv_folds: 交差検証の分割数
    
    Returns:
        学習済みSVMモデル
    """
    print("\nGridSearchCVによるハイパーパラメータ最適化...")
    
    # パラメータグリッド（簡素化版）
    param_grid = {
        'C': [1, 10],
        'gamma': ['scale', 0.01],
        'kernel': ['rbf']
    }
    
    # SVMモデル
    svm = SVC(probability=True, random_state=42)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=cv_folds,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nベストパラメータ: {grid_search.best_params_}")
    print(f"ベストスコア (F1): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def save_model(model, scaler, output_dir="."):
    """
    モデルとスケーラーを保存
    
    Args:
        model: 学習済みSVMモデル
        scaler: StandardScaler
        output_dir: 出力ディレクトリ
    """
    model_path = os.path.join(output_dir, "hog_svm_model.pkl")
    scaler_path = os.path.join(output_dir, "hog_svm_scaler.pkl")
    params_path = os.path.join(output_dir, "hog_svm_params.json")
    
    # モデルとスケーラーを保存
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # HOGパラメータとSVMパラメータをJSON形式で保存
    params = {
        'hog_params': {
            'winSize': list(HOG_PARAMS['winSize']),
            'blockSize': list(HOG_PARAMS['blockSize']),
            'blockStride': list(HOG_PARAMS['blockStride']),
            'cellSize': list(HOG_PARAMS['cellSize']),
            'nbins': HOG_PARAMS['nbins']
        },
        'svm_params': model.get_params()
    }
    
    # SVM paramsから保存できない項目を除外
    params['svm_params'] = {k: v for k, v in params['svm_params'].items() 
                            if isinstance(v, (int, float, str, bool, type(None)))}
    
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nモデルを保存しました:")
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")
    print(f"  - {params_path}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("HOG + SVM Death Detection Training")
    print("=" * 60)
    
    start_time = time.time()
    
    # 乱数シード固定
    np.random.seed(42)
    
    # データパス
    data_path = "data"
    death_path = os.path.join(data_path, "death")
    not_path = os.path.join(data_path, "not")
    
    # HOGDescriptorを作成
    hog = create_hog_descriptor()
    
    # サンプル画像でHOG特徴量の次元を確認
    sample_features = extract_hog_features(np.zeros((64, 64, 3), dtype=np.uint8), hog)
    print(f"\nHOG特徴量次元: {len(sample_features)}")
    
    # データ読み込み（1:1バランス）
    print("\n" + "-" * 40)
    print("データ読み込み中（1:1バランス）...")
    print("-" * 40)
    
    # まずdeath画像を全て読み込み
    death_features, death_labels = load_images_from_folder(death_path, 1, hog)
    num_death = len(death_features)
    
    # not画像はdeath画像と同数だけランダムサンプリング
    not_features, not_labels = load_images_from_folder(not_path, 0, hog, max_images=num_death, shuffle=True)
    
    print(f"\nDeath画像: {len(death_features)}枚")
    print(f"Not-death画像: {len(not_features)}枚")
    print(f"データバランス: 1:1")
    
    # データ結合
    X = np.concatenate([death_features, not_features])
    y = np.concatenate([death_labels, not_labels])
    
    print(f"\n総サンプル数: {len(X)}")
    print(f"特徴量次元: {X.shape[1]}")
    
    # データ分割
    print("\nデータ分割中...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練データ: {len(X_train)}サンプル")
    print(f"テストデータ: {len(X_test)}サンプル")
    
    # 特徴量の正規化
    print("\n特徴量を正規化中...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVMモデルの学習（GridSearchCV使用）
    print("\n" + "-" * 40)
    print("SVM モデル学習中...")
    print("-" * 40)
    
    model = train_svm_with_gridsearch(X_train_scaled, y_train)
    
    # 評価
    print("\n" + "-" * 40)
    print("モデル評価")
    print("-" * 40)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Death', 'Death']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN: {cm[0, 0]:5d}  FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d}  TP: {cm[1, 1]:5d}")
    
    # モデル保存
    print("\n" + "-" * 40)
    print("モデル保存中...")
    print("-" * 40)
    
    save_model(model, scaler)
    
    # 処理時間
    elapsed_time = time.time() - start_time
    print(f"\n総処理時間: {elapsed_time:.1f}秒")
    
    print("\n" + "=" * 60)
    print("学習完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
