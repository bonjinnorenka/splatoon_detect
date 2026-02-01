#!/usr/bin/env python3
"""
HOG + SVM Death Detection Evaluation Script

学習済みモデルの詳細評価と可視化を行うスクリプト
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import json
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# HOGパラメータ（デフォルト）
HOG_PARAMS = {
    'winSize': (64, 64),
    'blockSize': (16, 16),
    'blockStride': (8, 8),
    'cellSize': (8, 8),
    'nbins': 9
}


def load_hog_params(params_path):
    """HOGパラメータをJSONから読み込み"""
    global HOG_PARAMS
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
            hog_params = params.get('hog_params', {})
            HOG_PARAMS = {
                'winSize': tuple(hog_params.get('winSize', [64, 64])),
                'blockSize': tuple(hog_params.get('blockSize', [16, 16])),
                'blockStride': tuple(hog_params.get('blockStride', [8, 8])),
                'cellSize': tuple(hog_params.get('cellSize', [8, 8])),
                'nbins': hog_params.get('nbins', 9)
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
    """画像からHOG特徴量を抽出"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = cv2.resize(gray, HOG_PARAMS['winSize'])
    features = hog_descriptor.compute(gray)
    
    return features.flatten()


def load_images_from_folder(folder_path, label, hog_descriptor, max_images=None):
    """フォルダから画像を読み込みHOG特徴量を抽出"""
    features = []
    labels = []
    filenames = []
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    if max_images:
        image_files = image_files[:max_images]
    
    for filename in tqdm(image_files, desc=f"Loading {os.path.basename(folder_path)} images"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            hog_features = extract_hog_features(img, hog_descriptor)
            features.append(hog_features)
            labels.append(label)
            filenames.append(filename)
    
    return np.array(features), np.array(labels), filenames


def plot_confusion_matrix(cm, save_path=None):
    """混同行列を可視化"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Not Death', 'Death']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # テキストを追加
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混同行列を保存: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """ROC曲線を可視化"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC曲線を保存: {save_path}")
    
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, save_path=None):
    """Precision-Recall曲線を可視化"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {ap:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PR曲線を保存: {save_path}")
    
    plt.show()
    
    return ap


def plot_probability_distribution(y_true, y_scores, save_path=None):
    """予測確率の分布を可視化"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Not Death (y_true == 0) の確率分布
    not_death_probs = y_scores[y_true == 0]
    ax.hist(not_death_probs, bins=50, alpha=0.6, label='Not Death',
            color='blue', density=True)
    
    # Death (y_true == 1) の確率分布
    death_probs = y_scores[y_true == 1]
    ax.hist(death_probs, bins=50, alpha=0.6, label='Death',
            color='red', density=True)
    
    ax.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
    
    ax.set_xlabel('Predicted Death Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prediction Probability Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"確率分布を保存: {save_path}")
    
    plt.show()


def main():
    """メイン処理"""
    print("=" * 60)
    print("HOG + SVM Death Detection Evaluation")
    print("=" * 60)
    
    # モデルとパラメータを読み込み
    model_path = "hog_svm_model.pkl"
    scaler_path = "hog_svm_scaler.pkl"
    params_path = "hog_svm_params.json"
    
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        print("先に train_hog_svm.py を実行してください。")
        return
    
    # HOGパラメータを読み込み
    load_hog_params(params_path)
    
    # モデルとスケーラーを読み込み
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"モデル読み込み完了: {model_path}")
    
    # HOGDescriptorを作成
    hog = create_hog_descriptor()
    
    # データ読み込み
    print("\n" + "-" * 40)
    print("データ読み込み中...")
    print("-" * 40)
    
    data_path = "data"
    death_path = os.path.join(data_path, "death")
    not_path = os.path.join(data_path, "not")
    
    death_features, death_labels, death_files = load_images_from_folder(death_path, 1, hog)
    not_features, not_labels, not_files = load_images_from_folder(not_path, 0, hog)
    
    X = np.concatenate([death_features, not_features])
    y = np.concatenate([death_labels, not_labels])
    all_files = death_files + not_files
    
    # 同じ分割でテストデータを取得
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # テストデータを正規化
    X_test_scaled = scaler.transform(X_test)
    
    # 予測
    print("\n" + "-" * 40)
    print("評価中...")
    print("-" * 40)
    
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]  # Death確率
    
    # 基本メトリクス
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Death', 'Death']))
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]:5d}  FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d}  TP: {cm[1, 1]:5d}")
    
    # 追加メトリクス
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSensitivity (Recall for Death): {sensitivity:.4f}")
    print(f"Specificity (Recall for Not Death): {specificity:.4f}")
    
    # 可視化
    print("\n" + "-" * 40)
    print("可視化を生成中...")
    print("-" * 40)
    
    # 保存ディレクトリ
    os.makedirs("evaluation_results", exist_ok=True)
    
    # 混同行列
    plot_confusion_matrix(cm, save_path="evaluation_results/confusion_matrix.png")
    
    # ROC曲線
    roc_auc = plot_roc_curve(y_test, y_scores, save_path="evaluation_results/roc_curve.png")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # PR曲線
    ap = plot_precision_recall_curve(y_test, y_scores, save_path="evaluation_results/pr_curve.png")
    print(f"Average Precision: {ap:.4f}")
    
    # 確率分布
    plot_probability_distribution(y_test, y_scores, save_path="evaluation_results/probability_distribution.png")
    
    # 結果をJSONで保存
    results = {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc),
        'average_precision': float(ap),
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        },
        'test_samples': len(y_test),
        'death_samples': int(np.sum(y_test == 1)),
        'not_death_samples': int(np.sum(y_test == 0))
    }
    
    with open("evaluation_results/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("評価完了!")
    print(f"結果は evaluation_results/ ディレクトリに保存されました。")
    print("=" * 60)


if __name__ == "__main__":
    main()
