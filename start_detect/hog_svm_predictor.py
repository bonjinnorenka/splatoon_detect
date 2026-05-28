import argparse
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


CLASS_NAMES = ["not", "start"]


@dataclass(frozen=True)
class HOGConfig:
    image_size: int = 64
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 2
    block_norm_epsilon: float = 1e-6
    block_clip: float = 0.2


def sigmoid(value):
    value = float(np.clip(value, -50, 50))
    return 1.0 / (1.0 + math.exp(-value))


def image_to_gray_resized(image_or_path, image_size):
    if isinstance(image_or_path, (str, Path)):
        gray = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"画像を読み込めません: {image_or_path}")
    else:
        gray = image_or_path
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def extract_hog_features(gray_image, config=HOGConfig()):
    """Browser版と合わせた小さなHOG実装。unsigned gradient + L2-Hys blocks。"""
    image = image_to_gray_resized(gray_image, config.image_size)
    height, width = image.shape
    pixels_per_cell = config.pixels_per_cell
    orientations = config.orientations
    bin_width = 180.0 / orientations

    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]

    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = np.degrees(np.arctan2(gy, gx))
    angle = np.where(angle < 0, angle + 180.0, angle)
    angle = np.where(angle >= 180.0, angle - 180.0, angle)

    n_cells_y = height // pixels_per_cell
    n_cells_x = width // pixels_per_cell
    usable_h = n_cells_y * pixels_per_cell
    usable_w = n_cells_x * pixels_per_cell

    magnitude = magnitude[:usable_h, :usable_w]
    angle = angle[:usable_h, :usable_w]

    bin_pos = angle / bin_width
    lower_bin = np.floor(bin_pos).astype(np.int32) % orientations
    upper_bin = (lower_bin + 1) % orientations
    upper_weight = bin_pos - np.floor(bin_pos)
    lower_weight = 1.0 - upper_weight

    y_indices = np.arange(usable_h, dtype=np.int32)[:, None] // pixels_per_cell
    x_indices = np.arange(usable_w, dtype=np.int32)[None, :] // pixels_per_cell
    y_indices = np.broadcast_to(y_indices, magnitude.shape)
    x_indices = np.broadcast_to(x_indices, magnitude.shape)

    cell_hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
    np.add.at(
        cell_hist,
        (y_indices.ravel(), x_indices.ravel(), lower_bin.ravel()),
        (magnitude * lower_weight).ravel(),
    )
    np.add.at(
        cell_hist,
        (y_indices.ravel(), x_indices.ravel(), upper_bin.ravel()),
        (magnitude * upper_weight).ravel(),
    )

    block_size = config.cells_per_block
    blocks = []
    for y in range(n_cells_y - block_size + 1):
        for x in range(n_cells_x - block_size + 1):
            block = cell_hist[y : y + block_size, x : x + block_size].ravel()
            norm = np.sqrt(np.sum(block * block) + config.block_norm_epsilon * config.block_norm_epsilon)
            block = block / norm
            block = np.minimum(block, config.block_clip)
            norm = np.sqrt(np.sum(block * block) + config.block_norm_epsilon * config.block_norm_epsilon)
            blocks.append(block / norm)

    return np.concatenate(blocks).astype(np.float32)


def collect_image_paths(data_dir, include_false_positives=False):
    data_dir = Path(data_dir)
    paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        class_paths = sorted(class_dir.glob("*.png"))
        paths.extend(class_paths)
        labels.extend([class_idx] * len(class_paths))

    if include_false_positives:
        wrong_dir = data_dir.parent / "result" / "wrong"
        if wrong_dir.exists():
            wrong_paths = sorted(wrong_dir.glob("*.png"))
            paths.extend(wrong_paths)
            labels.extend([0] * len(wrong_paths))

    return np.array(paths, dtype=object), np.array(labels, dtype=np.int32)


def extract_hog_feature_matrix(paths, config=HOGConfig(), progress=True):
    features = []
    started_at = time.time()
    for index, path in enumerate(paths, start=1):
        if progress and (index == 1 or index % 1000 == 0 or index == len(paths)):
            elapsed = time.time() - started_at
            print(f"  HOG抽出: {index}/{len(paths)} ({elapsed:.1f}s)")
        gray = image_to_gray_resized(path, config.image_size)
        features.append(extract_hog_features(gray, config))
    return np.vstack(features)


def optimize_score_threshold(scores, labels, metric="f1"):
    unique_scores = np.unique(scores)
    if len(unique_scores) > 2000:
        thresholds = np.quantile(unique_scores, np.linspace(0.0, 1.0, 2000))
    else:
        thresholds = unique_scores

    best_threshold = 0.0
    best_value = -1.0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int32)
        if metric == "balanced_accuracy":
            value = balanced_accuracy_score(labels, predictions)
        else:
            value = f1_score(labels, predictions, zero_division=0)
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)

    return best_threshold, best_value


def metrics_dict(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_not": float(precision[0]),
        "recall_not": float(recall[0]),
        "f1_not": float(f1[0]),
        "precision_start": float(precision[1]),
        "recall_start": float(recall[1]),
        "f1_start": float(f1[1]),
        "confusion_matrix": confusion_matrix(labels, predictions).astype(int).tolist(),
    }


class HOGSVMImagePredictor:
    def __init__(self, config=HOGConfig()):
        self.config = config
        self.scaler = StandardScaler()
        self.model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=20000,
            dual="auto",
        )
        self.calibrator = LogisticRegression(random_state=42, max_iter=1000)
        self.decision_threshold = 0.0
        self.weights = None
        self.bias = None
        self.feature_count = None
        self.metrics = None

    def fit(self, train_paths, train_labels):
        print("HOG特徴量を抽出しています...")
        x_all = extract_hog_feature_matrix(train_paths, self.config)
        self.feature_count = int(x_all.shape[1])
        x_fit, x_threshold, y_fit, y_threshold = train_test_split(
            x_all,
            train_labels,
            test_size=0.2,
            random_state=42,
            stratify=train_labels,
        )

        print("特徴量を標準化しています...")
        x_fit_scaled = self.scaler.fit_transform(x_fit)
        x_threshold_scaled = self.scaler.transform(x_threshold)

        print("Linear SVMを学習しています...")
        self.model.fit(x_fit_scaled, y_fit)
        threshold_scores = self.model.decision_function(x_threshold_scaled)
        self.decision_threshold, best_f1 = optimize_score_threshold(threshold_scores, y_threshold, metric="f1")
        print(f"validation上の最適score threshold: {self.decision_threshold:.6f} (F1={best_f1:.4f})")

        shifted_scores = (threshold_scores - self.decision_threshold).reshape(-1, 1)
        self.calibrator.fit(shifted_scores, y_threshold)

        self.weights = self.model.coef_[0].astype(float)
        self.bias = float(self.model.intercept_[0])
        return self

    def decision_function_from_features(self, features):
        scaled = self.scaler.transform(features)
        return np.dot(scaled, self.weights) + self.bias

    def predict_from_scores(self, scores):
        return (np.asarray(scores) >= self.decision_threshold).astype(np.int32)

    def predict_proba_start_from_scores(self, scores):
        shifted_scores = (np.asarray(scores) - self.decision_threshold).reshape(-1, 1)
        return self.calibrator.predict_proba(shifted_scores)[:, 1]

    def evaluate(self, test_paths, test_labels, verbose=True):
        print("評価用HOG特徴量を抽出しています...")
        x_test = extract_hog_feature_matrix(test_paths, self.config)
        scores = self.decision_function_from_features(x_test)
        predictions = self.predict_from_scores(scores)
        result = metrics_dict(test_labels, predictions)
        self.metrics = result

        if verbose:
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Balanced accuracy: {result['balanced_accuracy']:.4f}")
            print("Confusion Matrix:")
            print(np.array(result["confusion_matrix"]))
            print("Classification Report:")
            print(classification_report(test_labels, predictions, target_names=CLASS_NAMES, zero_division=0))

        return result

    def train(self, data_dir, test_size=0.2, random_state=42, include_false_positives=False):
        paths, labels = collect_image_paths(data_dir, include_false_positives)
        train_paths, test_paths, y_train, y_test = train_test_split(
            paths,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        self.fit(train_paths, y_train)
        return self.evaluate(test_paths, y_test)

    def predict_image(self, image_or_path):
        gray = image_to_gray_resized(image_or_path, self.config.image_size)
        features = extract_hog_features(gray, self.config).reshape(1, -1)
        score = float(self.decision_function_from_features(features)[0])
        start_probability = float(self.predict_proba_start_from_scores([score])[0])
        prediction = int(score >= self.decision_threshold)
        probability = start_probability if prediction == 1 else 1.0 - start_probability
        return {
            "class": CLASS_NAMES[prediction],
            "prediction": prediction,
            "probability": probability,
            "start_probability": start_probability,
            "score": score,
        }

    def to_json_data(self):
        if self.weights is None:
            raise RuntimeError("モデルが未学習です")

        scaler_stds = self.scaler.scale_.astype(float)
        scaler_stds = np.where(scaler_stds == 0, 1.0, scaler_stds)
        calibrator_coef = float(self.calibrator.coef_[0][0])
        calibrator_intercept = float(self.calibrator.intercept_[0])

        return {
            "model_type": "hog_linear_svm",
            "class_names": CLASS_NAMES,
            "hog_config": asdict(self.config),
            "feature_count": self.feature_count,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "decision_threshold": self.decision_threshold,
            "scaler": {
                "means": self.scaler.mean_.astype(float).tolist(),
                "stds": scaler_stds.tolist(),
            },
            "calibration": {
                "coef": calibrator_coef,
                "intercept": calibrator_intercept,
                "input": "decision_score_minus_threshold",
            },
            "recommended_probability_threshold": 0.5,
            "metrics": self.metrics,
        }

    def save_json(self, filepath):
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.to_json_data(), file, ensure_ascii=False, indent=2)

    def save_pickle(self, filepath):
        payload = {
            "config": self.config,
            "scaler": self.scaler,
            "model": self.model,
            "calibrator": self.calibrator,
            "decision_threshold": self.decision_threshold,
            "weights": self.weights,
            "bias": self.bias,
            "feature_count": self.feature_count,
            "metrics": self.metrics,
        }
        with open(filepath, "wb") as file:
            pickle.dump(payload, file)

    def load_pickle(self, filepath):
        with open(filepath, "rb") as file:
            payload = pickle.load(file)
        self.config = payload["config"]
        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.calibrator = payload["calibrator"]
        self.decision_threshold = payload["decision_threshold"]
        self.weights = payload["weights"]
        self.bias = payload["bias"]
        self.feature_count = payload["feature_count"]
        self.metrics = payload.get("metrics")
        return self


def build_arg_parser():
    parser = argparse.ArgumentParser(description="start_detect用 HOG + Linear SVM")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="HOG+SVMモデルを学習して保存")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--model-json", default="hog_svm_model.json")
    train_parser.add_argument("--model-pkl", default="hog_svm_model.pkl")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--include-false-positives", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="保存済みpklモデルで1枚推論")
    predict_parser.add_argument("image")
    predict_parser.add_argument("--model-pkl", default="hog_svm_model.pkl")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    command = args.command or "train"

    if command == "train":
        base_dir = Path(__file__).resolve().parent
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = base_dir / data_dir
        model_json = Path(args.model_json)
        model_pkl = Path(args.model_pkl)
        if not model_json.is_absolute():
            model_json = base_dir / model_json
        if not model_pkl.is_absolute():
            model_pkl = base_dir / model_pkl

        predictor = HOGSVMImagePredictor()
        metrics = predictor.train(
            data_dir,
            test_size=args.test_size,
            include_false_positives=args.include_false_positives,
        )
        predictor.save_json(model_json)
        predictor.save_pickle(model_pkl)
        print(f"\nHOG+SVMモデルを保存しました: {model_json}")
        print(f"HOG+SVM pickleを保存しました: {model_pkl}")
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        return

    if command == "predict":
        predictor = HOGSVMImagePredictor().load_pickle(args.model_pkl)
        print(json.dumps(predictor.predict_image(args.image), ensure_ascii=False, indent=2))
        return

    parser.error(f"unknown command: {command}")


if __name__ == "__main__":
    main()
