# start_detect HOG+SVM comparison

- Dataset: 16671 images
- Class distribution: not=16532, start=139
- Split: stratified train/test, test_size=0.2, random_state=42

| Method | Accuracy | Balanced accuracy | Start precision | Start recall | Start F1 | Confusion matrix [[not, start], ...] |
|---|---:|---:|---:|---:|---:|---|
| Majority baseline | 0.9916 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | [[3307, 0], [28, 0]] |
| Std threshold | 0.9949 | 0.8558 | 0.6897 | 0.7143 | 0.7018 | [[3298, 9], [8, 20]] |
| Logistic Regression (stats+hist) | 0.9988 | 0.9286 | 1.0000 | 0.8571 | 0.9231 | [[3307, 0], [4, 24]] |
| HOG + Linear SVM | 0.9997 | 0.9998 | 0.9655 | 1.0000 | 0.9825 | [[3306, 1], [0, 28]] |

Confusion matrix rows are true labels `[not, start]`, columns are predicted labels `[not, start]`.
