"""
evaluate.py

Вычисление метрик, per-class отчёт (сверка с Table II статьи),
сравнение воспроизведённых результатов с заявленными в статье (Table I).

Заявленные в статье метрики (paper_metrics) больше не хранятся здесь как
единый хардкод-словарь — каждая статья заявляет разные числа под одними и
теми же именами методов (например, статья про latent vector representation
переиспользует "baseline" под ResNet-50, а не ResNet-18), поэтому
paper_metrics читается из секции paper_metrics: конкретного yaml-конфига
(см. train.py) и передаётся сюда через compare_to_paper()/update_results().
"""

import json
import os
from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import CLASS_NAMES


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Macro-averaged Accuracy/Precision/Recall/F1 — та же схема усреднения, что в статье."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_report(y_true, y_pred, class_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Per-class Accuracy/Precision/Recall/F1 — для сверки с Table II статьи.

    Accuracy на класс здесь считается как one-vs-rest accuracy (доля верно
    классифицированных объектов при бинаризации "этот класс vs остальные"),
    именно так подобные таблицы обычно строятся в wafer defect литературе.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = {}
    for idx, cls_name in enumerate(class_names):
        y_true_bin = (y_true == idx).astype(int)
        y_pred_bin = (y_pred == idx).astype(int)

        report[cls_name] = {
            "accuracy": accuracy_score(y_true_bin, y_pred_bin),
            "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        }

    return report


def print_per_class_report(report: Dict[str, Dict[str, float]]):
    print(f"{'Class':<12}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    for cls_name, metrics in report.items():
        print(
            f"{cls_name:<12}"
            f"{metrics['accuracy']*100:>9.2f}%"
            f"{metrics['precision']*100:>11.2f}%"
            f"{metrics['recall']*100:>9.2f}%"
            f"{metrics['f1']*100:>9.2f}%"
        )


def compare_to_paper(results: Dict[str, float], paper_results: Dict[str, float]) -> Dict[str, float]:
    """
    Сравнение воспроизведённых метрик с заявленными в статье.
    Возвращает разницу (reproduced - paper) в процентных пунктах для каждой метрики.
    """
    diff = {}
    print(f"{'Metric':<12}{'Paper':>10}{'Reproduced':>14}{'Diff':>10}")
    for metric in ("accuracy", "precision", "recall", "f1"):
        paper_val = paper_results.get(metric)
        repro_val = results.get(metric)
        if paper_val is None or repro_val is None:
            continue
        d = (repro_val - paper_val) * 100
        diff[metric] = d
        print(
            f"{metric:<12}"
            f"{paper_val*100:>9.2f}%"
            f"{repro_val*100:>13.2f}%"
            f"{d:>+9.2f}pp"
        )
    return diff


def update_results(method_name: str, method_results: dict, path: str):
    """
    Обновляет сводный JSON-файл результатов: файл хранит словарь вида
    {"baseline": {...}, "mean_teacher": {...}, ...}, и каждый вызов
    перезаписывает только запись для method_name, не трогая остальные
    методы, уже сохранённые ранее в этом файле.

    Если файла ещё нет или он повреждён/в старом формате (один прогон
    без вложенности по методам) — создаёт новый сводный словарь.
    """
    all_results = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and "method" not in existing:
                # уже в сводном формате {"baseline": {...}, ...}
                all_results = existing
            elif isinstance(existing, dict) and "method" in existing:
                # старый формат (одиночный прогон) — мигрируем под его собственным именем метода
                all_results = {existing["method"]: existing}
        except (json.JSONDecodeError, OSError):
            all_results = {}

    all_results[method_name] = method_results

    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Результаты метода '{method_name}' обновлены в {path}")


def save_results(results: dict, path: str):
    """Оставлено для обратной совместимости — перезаписывает файл целиком одним прогоном."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Результаты сохранены в {path}")


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)