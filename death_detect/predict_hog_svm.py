#!/usr/bin/env python3
"""
HOG + SVM Death Detection Prediction Script

学習済みモデルを使用して画像のdeath判定を行う推論スクリプト
"""

import os
import sys
import cv2
import numpy as np
import joblib
import json
import argparse
from typing import Tuple, List, Optional


class DeathDetector:
    """HOG + SVM ベースのdeath detector クラス"""
    
    def __init__(self, model_dir: str = "."):
        """
        デス検出器を初期化
        
        Args:
            model_dir: モデルファイルがあるディレクトリ
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.hog = None
        self.hog_params = None
        
        self._load_model()
    
    def _load_model(self):
        """モデルとパラメータを読み込み"""
        model_path = os.path.join(self.model_dir, "hog_svm_model.pkl")
        scaler_path = os.path.join(self.model_dir, "hog_svm_scaler.pkl")
        params_path = os.path.join(self.model_dir, "hog_svm_params.json")
        
        # モデルとスケーラーの読み込み
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"スケーラーファイルが見つかりません: {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # HOGパラメータの読み込み
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                self.hog_params = params.get('hog_params', {})
        else:
            # デフォルトパラメータ
            self.hog_params = {
                'winSize': [64, 64],
                'blockSize': [16, 16],
                'blockStride': [8, 8],
                'cellSize': [8, 8],
                'nbins': 9
            }
        
        # HOGDescriptorを作成
        self.hog = cv2.HOGDescriptor(
            _winSize=tuple(self.hog_params['winSize']),
            _blockSize=tuple(self.hog_params['blockSize']),
            _blockStride=tuple(self.hog_params['blockStride']),
            _cellSize=tuple(self.hog_params['cellSize']),
            _nbins=self.hog_params['nbins']
        )
        
        print(f"モデル読み込み完了: {model_path}")
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        画像からHOG特徴量を抽出
        
        Args:
            image: BGR画像（OpenCV形式）
        
        Returns:
            HOG特徴量ベクトル
        """
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # リサイズ
        gray = cv2.resize(gray, tuple(self.hog_params['winSize']))
        
        # HOG特徴量を計算
        features = self.hog.compute(gray)
        
        return features.flatten()
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        単一画像の予測
        
        Args:
            image: BGR画像（OpenCV形式）
        
        Returns:
            (予測ラベル, 確信度)
            - 予測ラベル: 0=Not Death, 1=Death
            - 確信度: 0.0〜1.0（Death確率）
        """
        # 特徴量抽出
        features = self.extract_features(image)
        
        # 正規化
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 予測
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Death確率（クラス1の確率）
        death_prob = probability[1]
        
        return int(prediction), float(death_prob)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        複数画像のバッチ予測
        
        Args:
            images: 画像リスト
        
        Returns:
            [(予測ラベル, 確信度), ...] のリスト
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def predict_file(self, image_path: str) -> Tuple[int, float]:
        """
        画像ファイルから予測
        
        Args:
            image_path: 画像ファイルパス
        
        Returns:
            (予測ラベル, 確信度)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        return self.predict(image)
    
    def predict_folder(self, folder_path: str) -> List[Tuple[str, int, float]]:
        """
        フォルダ内の全画像を予測
        
        Args:
            folder_path: 画像フォルダパス
        
        Returns:
            [(ファイル名, 予測ラベル, 確信度), ...] のリスト
        """
        results = []
        
        image_files = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            try:
                prediction, probability = self.predict_file(image_path)
                results.append((filename, prediction, probability))
            except Exception as e:
                print(f"警告: {filename} の処理でエラー: {e}")
        
        return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='HOG + SVM Death Detection - 推論スクリプト'
    )
    parser.add_argument(
        'input',
        help='入力画像ファイルまたはフォルダのパス'
    )
    parser.add_argument(
        '--model-dir',
        default='.',
        help='モデルファイルがあるディレクトリ（デフォルト: カレントディレクトリ）'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Death判定の閾値（デフォルト: 0.5）'
    )
    
    args = parser.parse_args()
    
    # 検出器を初期化
    detector = DeathDetector(model_dir=args.model_dir)
    
    # 入力がファイルかフォルダか判定
    if os.path.isfile(args.input):
        # 単一ファイル
        prediction, probability = detector.predict_file(args.input)
        label = "Death" if prediction == 1 else "Not Death"
        print(f"\n結果: {label}")
        print(f"確信度: {probability:.4f} ({probability*100:.2f}%)")
        
        if probability >= args.threshold:
            print("判定: Death")
        else:
            print("判定: Not Death")
    
    elif os.path.isdir(args.input):
        # フォルダ
        results = detector.predict_folder(args.input)
        
        print(f"\n処理画像数: {len(results)}")
        print("-" * 60)
        
        death_count = 0
        for filename, prediction, probability in results:
            label = "Death" if prediction == 1 else "Not Death"
            if prediction == 1:
                death_count += 1
            print(f"{filename}: {label} (確信度: {probability:.4f})")
        
        print("-" * 60)
        print(f"Death検出: {death_count}/{len(results)}")
    
    else:
        print(f"エラー: 入力パスが見つかりません: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
