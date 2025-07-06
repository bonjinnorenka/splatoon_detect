#!/usr/bin/env python3
"""
隣り合うナンバリングの画像を比較して類似度が高い場合に削除するスクリプト
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm

def extract_number_from_filename(filename: str) -> int:
    """ファイル名から数字を抽出する"""
    match = re.search(r'output_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    
    match = re.search(r'output_2_(\d+)\.png', filename)
    if match:
        return int(match.group(1)) + 100000  # 2系列を区別するため大きな数を加算
    
    return 0

def load_image(image_path: str) -> np.ndarray:
    """画像を読み込む"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    return img

def calculate_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """2つの画像の類似度を計算する（構造類似度指数SSIMを使用）"""
    # グレースケールに変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 画像サイズを合わせる
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # SSIMを計算
    from skimage.metrics import structural_similarity
    similarity = structural_similarity(gray1, gray2)
    
    return similarity

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """ヒストグラムベースの類似度を計算する"""
    # HSVに変換
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # ヒストグラムを計算
    hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    
    # 相関係数を計算
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation

def get_sorted_image_files(directory: str) -> List[str]:
    """ディレクトリ内の画像ファイルを番号順にソートして取得"""
    image_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            image_files.append(filename)
    
    # 番号順にソート
    image_files.sort(key=extract_number_from_filename)
    
    return image_files

def remove_similar_images(directory: str, similarity_threshold: float = 0.95, 
                         histogram_threshold: float = 0.98, dry_run: bool = False) -> None:
    """類似度の高い隣接画像を削除する"""
    
    print(f"ディレクトリ: {directory}")
    print(f"類似度閾値 (SSIM): {similarity_threshold}")
    print(f"ヒストグラム類似度閾値: {histogram_threshold}")
    print(f"ドライラン: {dry_run}")
    print("-" * 50)
    
    image_files = get_sorted_image_files(directory)
    print(f"合計画像数: {len(image_files)}")
    
    removed_count = 0
    kept_count = 0
    
    for i in tqdm(range(len(image_files) - 1), desc="画像を比較中"):
        current_file = image_files[i]
        next_file = image_files[i + 1]
        
        current_path = os.path.join(directory, current_file)
        next_path = os.path.join(directory, next_file)
        
        # 現在のファイルが既に削除されている場合はスキップ
        if not os.path.exists(current_path):
            continue
            
        # 次のファイルが存在しない場合はスキップ
        if not os.path.exists(next_path):
            continue
        
        try:
            # 画像を読み込む
            img1 = load_image(current_path)
            img2 = load_image(next_path)
            
            # 類似度を計算
            ssim_similarity = calculate_similarity(img1, img2)
            hist_similarity = calculate_histogram_similarity(img1, img2)
            
            # 両方の類似度が閾値を超える場合は削除対象とする
            if ssim_similarity >= similarity_threshold and hist_similarity >= histogram_threshold:
                print(f"削除対象: {next_file} (SSIM: {ssim_similarity:.4f}, Hist: {hist_similarity:.4f})")
                
                if not dry_run:
                    os.remove(next_path)
                    print(f"削除しました: {next_file}")
                
                removed_count += 1
            else:
                kept_count += 1
                
        except Exception as e:
            print(f"エラー: {current_file} と {next_file} の比較中にエラーが発生しました: {e}")
            continue
    
    print("-" * 50)
    print(f"削除した画像数: {removed_count}")
    print(f"保持した画像数: {kept_count}")
    print(f"処理完了!")

def main():
    parser = argparse.ArgumentParser(description='隣接する類似画像を削除するスクリプト')
    parser.add_argument('directory', help='画像ディレクトリのパス')
    parser.add_argument('--ssim-threshold', type=float, default=0.95, 
                       help='SSIM類似度の閾値 (デフォルト: 0.95)')
    parser.add_argument('--hist-threshold', type=float, default=0.98,
                       help='ヒストグラム類似度の閾値 (デフォルト: 0.98)')
    parser.add_argument('--dry-run', action='store_true',
                       help='実際には削除せずに、削除対象のファイルを表示する')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"エラー: ディレクトリが存在しません: {args.directory}")
        return
    
    remove_similar_images(args.directory, args.ssim_threshold, args.hist_threshold, args.dry_run)

if __name__ == "__main__":
    main()