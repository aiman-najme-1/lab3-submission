import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    x_test = np.asarray(data.test_features, dtype=float)
    weights = np.zeros(x_train.shape[1])
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        errors = x_train @ weights + bias - y_train
        weights -= data.learning_rate * 2 * (x_train.T @ errors) / len(y_train)
        bias -= data.learning_rate * 2 * errors.mean()
        # Сохраняем MSE после обновления весов и смещения.
        errors = x_train @ weights + bias - y_train
        loss_history.append(float(np.mean(errors ** 2)))

    predictions = x_test @ weights + bias
    return RegressionResult(weights, float(bias), loss_history, predictions)


def _sigmoid(values):
    probabilities = np.empty_like(values, dtype=float)
    positive = values >= 0
    probabilities[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1 + exp_values)
    return probabilities


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    x_test = np.asarray(data.test_features, dtype=float)
    weights = np.zeros(x_train.shape[1])
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        probabilities = _sigmoid(x_train @ weights + bias)
        errors = probabilities - y_train
        weights -= data.learning_rate * (x_train.T @ errors) / len(y_train)
        bias -= data.learning_rate * errors.mean()
        scores = x_train @ weights + bias
        # Устойчивая запись BCE без вычисления log(0).
        loss = np.mean(np.logaddexp(0, scores) - y_train * scores)
        loss_history.append(float(loss))

    probabilities = _sigmoid(x_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights, float(bias), loss_history, probabilities, predictions
    )


def _classification_metrics(target, probabilities, predictions):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).reshape(-1)
    x_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).reshape(-1)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = model.predict(x_test)
        metrics[name] = _classification_metrics(y_test, probabilities, predictions)
    return BaselineComparisonResult(**metrics)


def run_experiments():
    directory = Path(__file__).resolve().parent
    students = pd.read_csv(directory / "Student_Performance.csv")
    features = students.drop(columns="Performance Index").copy()
    features["Extracurricular Activities"] = features["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    target = students["Performance Index"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    regression_data = RegressionInput(x_train, y_train, x_test, y_test)
    regression = train_linear_regression(regression_data)
    test_mse = float(np.mean((regression.predictions - y_test.to_numpy()) ** 2))
    slow_regression = train_linear_regression(
        RegressionInput(x_train, y_train, x_test, y_test, learning_rate=0.01)
    )

    credit = pd.read_csv(directory / "german.csv", sep=";")
    features = credit.iloc[:, 1:]
    target = credit.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    classification_data = ClassificationInput(x_train, y_train, x_test, y_test)
    own_logistic = train_logistic_classifier(classification_data)
    own_metrics = _classification_metrics(
        y_test, own_logistic.probabilities, own_logistic.predictions
    )
    baselines = compare_classical_models(classification_data)
    slow_logistic = train_logistic_classifier(
        ClassificationInput(x_train, y_train, x_test, y_test, learning_rate=0.01)
    )
    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": asdict(own_metrics),
            **asdict(baselines),
        },
    }
    (directory / "experiment_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(results, indent=2))
    print("Student data: rows, features, missing, duplicates:", len(students),
          students.shape[1] - 1, int(students.isna().sum().sum()),
          int(students.duplicated().sum()))
    print("Credit data: rows, features, missing, duplicates:", len(credit),
          credit.shape[1] - 1, int(credit.isna().sum().sum()),
          int(credit.duplicated().sum()))
    print("Credit train classes:", y_train.value_counts().sort_index().to_dict())
    print("Credit test classes:", y_test.value_counts().sort_index().to_dict())
    for name, losses, checkpoints in [
        ("Linear lr=0.05", regression.loss_history, [1, 10, 100, 800]),
        ("Linear lr=0.01", slow_regression.loss_history, [1, 10, 100, 800]),
        ("Logistic lr=0.1", own_logistic.loss_history, [1, 100, 500, 1000]),
        ("Logistic lr=0.01", slow_logistic.loss_history, [1, 100, 500, 1000]),
    ]:
        print(name, {epoch: losses[epoch - 1] for epoch in checkpoints})
    print("Versions:", "Python", platform.python_version(), "NumPy", np.__version__,
          "Pandas", pd.__version__, "scikit-learn", sklearn.__version__)


if __name__ == "__main__":
    run_experiments()
