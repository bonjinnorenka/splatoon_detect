import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hog_svm_predictor import (
    HOGSVMImagePredictor,
    CLASS_NAMES,
    collect_image_paths,
    image_to_gray_resized,
)


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
        "precision_start": float(precision[1]),
        "recall_start": float(recall[1]),
        "f1_start": float(f1[1]),
        "confusion_matrix": confusion_matrix(labels, predictions).astype(int).tolist(),
    }


def optimize_threshold(train_values, train_labels, metric="accuracy"):
    values = np.asarray(train_values, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int32)
    thresholds = np.unique(values)
    if len(thresholds) > 2000:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 2000))

    best = {"threshold": float(thresholds[0]), "direction": ">", "score": -1.0}
    for direction in [">", "<"]:
        for threshold in thresholds:
            if direction == ">":
                predictions = (values > threshold).astype(np.int32)
            else:
                predictions = (values < threshold).astype(np.int32)

            if metric == "balanced_accuracy":
                score = balanced_accuracy_score(labels, predictions)
            else:
                score = accuracy_score(labels, predictions)
            if score > best["score"]:
                best = {
                    "threshold": float(threshold),
                    "direction": direction,
                    "score": float(score),
                }
    return best


def apply_threshold(values, threshold_config):
    values = np.asarray(values, dtype=np.float32)
    if threshold_config["direction"] == ">":
        return (values > threshold_config["threshold"]).astype(np.int32)
    return (values < threshold_config["threshold"]).astype(np.int32)


def extract_std_values(paths, image_size):
    values = []
    for index, path in enumerate(paths, start=1):
        if index == 1 or index % 5000 == 0 or index == len(paths):
            print(f"  std特徴量: {index}/{len(paths)}")
        image = image_to_gray_resized(path, image_size)
        values.append(float(np.std(image)))
    return np.array(values, dtype=np.float32)


def extract_logistic_features(paths, image_size):
    features = []
    for index, path in enumerate(paths, start=1):
        if index == 1 or index % 5000 == 0 or index == len(paths):
            print(f"  logistic特徴量: {index}/{len(paths)}")
        image = image_to_gray_resized(path, image_size)
        mean_val = np.mean(image)
        std_val = np.std(image)
        min_val = np.min(image)
        max_val = np.max(image)
        edges = cv2.Canny(image.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        hist, _ = np.histogram(image, bins=8, range=(0, 256))
        hist_normalized = hist / np.sum(hist)
        features.append(
            np.concatenate(
                [
                    [mean_val, std_val, min_val, max_val, edge_density],
                    hist_normalized,
                ]
            )
        )
    return np.vstack(features).astype(np.float32)


def evaluate_majority_baseline(train_labels, test_labels):
    majority = int(np.bincount(train_labels).argmax())
    predictions = np.full_like(test_labels, majority)
    return metrics_dict(test_labels, predictions)


def evaluate_std_threshold(train_paths, test_paths, y_train, y_test):
    print("\n[1/3] Standard deviation threshold")
    train_values = extract_std_values(train_paths, image_size=32)
    test_values = extract_std_values(test_paths, image_size=32)
    threshold_config = optimize_threshold(train_values, y_train, metric="accuracy")
    predictions = apply_threshold(test_values, threshold_config)
    result = metrics_dict(y_test, predictions)
    result["threshold"] = threshold_config
    return result


def evaluate_logistic_regression(train_paths, test_paths, y_train, y_test):
    print("\n[2/3] Logistic Regression (stats + histogram)")
    x_train = extract_logistic_features(train_paths, image_size=32)
    x_test = extract_logistic_features(test_paths, image_size=32)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)
    return metrics_dict(y_test, predictions)


def evaluate_hog_svm(train_paths, test_paths, y_train, y_test, model_json, model_pkl):
    print("\n[3/3] HOG + Linear SVM")
    predictor = HOGSVMImagePredictor()
    predictor.fit(train_paths, y_train)
    result = predictor.evaluate(test_paths, y_test)
    predictor.metrics = result
    predictor.save_json(model_json)
    predictor.save_pickle(model_pkl)
    return result


def write_markdown_report(report_path, results, dataset_summary):
    rows = []
    for name, metrics in results.items():
        rows.append(
            [
                name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['balanced_accuracy']:.4f}",
                f"{metrics['precision_start']:.4f}",
                f"{metrics['recall_start']:.4f}",
                f"{metrics['f1_start']:.4f}",
                str(metrics["confusion_matrix"]),
            ]
        )

    lines = [
        "# start_detect HOG+SVM comparison",
        "",
        f"- Dataset: {dataset_summary['total']} images",
        f"- Class distribution: not={dataset_summary['not']}, start={dataset_summary['start']}",
        f"- Split: stratified train/test, test_size={dataset_summary['test_size']}, random_state=42",
        "",
        "| Method | Accuracy | Balanced accuracy | Start precision | Start recall | Start F1 | Confusion matrix [[not, start], ...] |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Confusion matrix rows are true labels `[not, start]`, columns are predicted labels `[not, start]`.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(results):
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(
        f"{'Method':38s} {'Accuracy':>10s} {'Bal Acc':>10s} "
        f"{'Start P':>10s} {'Start R':>10s} {'Start F1':>10s}"
    )
    for name, metrics in results.items():
        print(
            f"{name:38s} {metrics['accuracy']:10.4f} {metrics['balanced_accuracy']:10.4f} "
            f"{metrics['precision_start']:10.4f} {metrics['recall_start']:10.4f} {metrics['f1_start']:10.4f}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="start_detectの検出器精度比較")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--include-false-positives", action="store_true")
    parser.add_argument("--model-json", default="hog_svm_model.json")
    parser.add_argument("--model-pkl", default="hog_svm_model.pkl")
    parser.add_argument("--report-json", default="hog_svm_comparison.json")
    parser.add_argument("--report-md", default="hog_svm_comparison.md")
    return parser


def main():
    args = build_arg_parser().parse_args()
    base_dir = Path(__file__).resolve().parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    model_json = Path(args.model_json)
    model_pkl = Path(args.model_pkl)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    if not model_json.is_absolute():
        model_json = base_dir / model_json
    if not model_pkl.is_absolute():
        model_pkl = base_dir / model_pkl
    if not report_json.is_absolute():
        report_json = base_dir / report_json
    if not report_md.is_absolute():
        report_md = base_dir / report_md

    paths, labels = collect_image_paths(data_dir, include_false_positives=args.include_false_positives)
    train_paths, test_paths, y_train, y_test = train_test_split(
        paths,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    dataset_summary = {
        "total": int(len(labels)),
        "not": int(np.sum(labels == 0)),
        "start": int(np.sum(labels == 1)),
        "train": int(len(y_train)),
        "test": int(len(y_test)),
        "test_size": args.test_size,
        "include_false_positives": bool(args.include_false_positives),
    }

    print("Dataset summary:")
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))

    results = {
        "Majority baseline": evaluate_majority_baseline(y_train, y_test),
        "Std threshold": evaluate_std_threshold(train_paths, test_paths, y_train, y_test),
        "Logistic Regression (stats+hist)": evaluate_logistic_regression(train_paths, test_paths, y_train, y_test),
        "HOG + Linear SVM": evaluate_hog_svm(train_paths, test_paths, y_train, y_test, model_json, model_pkl),
    }

    output = {
        "dataset": dataset_summary,
        "class_names": CLASS_NAMES,
        "results": results,
        "hog_svm_model_json": str(model_json),
        "hog_svm_model_pkl": str(model_pkl),
    }
    report_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(report_md, results, dataset_summary)
    print_summary(results)
    print(f"\n比較結果を書き出しました: {report_json}")
    print(f"Markdownレポートを書き出しました: {report_md}")
    print(f"HOG+SVMモデルを書き出しました: {model_json}")


if __name__ == "__main__":
    main()
