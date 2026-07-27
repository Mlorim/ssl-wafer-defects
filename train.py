"""
train.py

Точка входа для обучения. Пример запуска:

    python train.py --config configs/mean_teacher_supcon.yaml --method baseline
    python train.py --config configs/mean_teacher_supcon.yaml --method mean_teacher
    python train.py --config configs/mean_teacher_supcon.yaml --method supcon
    python train.py --config configs/mean_teacher_supcon.yaml --method mean_teacher_supcon
    python train.py --config configs/latent_vector_representation.yaml --method baseline
    python train.py --config configs/latent_vector_representation.yaml --method latent_vector_representation
    python train.py --config configs/cbam_cnn.yaml --method cbam_cnn
    python train.py --config configs/hybrid_cnn_vit.yaml --method hybrid_cnn_vit
    python train.py --config configs/mm_wae.yaml --method mm_wae
    python train.py --config configs/efficient_cnn.yaml --method efficient_cnn

Флаг --method переопределяет method.name из конфига. Список допустимых
значений берётся из METHOD_REGISTRY — при добавлении нового метода (новый
файл в methods/ + запись в registry.py) сюда ничего добавлять не нужно.

train.py не знает, из скольких фаз состоит обучение конкретного метода —
он просто вызывает method.fit(data, loader_factory, ...) и получает
{"best_f1", "best_metrics"}. Однофазные методы (baseline/mean_teacher/
supcon/mean_teacher_supcon) используют дефолтную реализацию fit() из
methods.base.SSLMethod; LatentVectorMethod переопределяет её полностью
(VAE pretrain -> teacher -> pseudo-labeling -> student -> fine-tune).
"""

import argparse
import os

import yaml
import torch

from datasets import prepare_data_for_method, DataLoaderFactory, NUM_CLASSES, CLASS_NAMES, set_seed
from models import build_model
from methods import build_method, METHOD_REGISTRY
from evaluate import compute_metrics, per_class_report, print_per_class_report, compare_to_paper, update_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Путь к YAML-конфигу")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=list(METHOD_REGISTRY.keys()),
        help="Переопределяет method.name из конфига",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_optimizer_factory(train_config: dict):
    """
    Возвращает optimizer_factory(params, lr_override=None) -> Adam/AdamW optimizer.

    Методам с несколькими фазами (например LatentVectorMethod) нужно строить
    отдельные optimizer'ы для teacher/student/fine-tune с разными LR, сохраняя
    общие betas/weight_decay из конфига — отсюда lr_override вместо готового
    optimizer'а. Статья Wei et al. про latent vector representation использует
    нестандартный beta2=0.99 (вместо дефолтных 0.999 в PyTorch), поэтому betas
    читаются из train.adam_beta1/adam_beta2 конфига (по умолчанию — 0.9/0.999,
    как раньше, когда betas не переопределялись вовсе). Статья SemiWaferNet
    (hybrid_cnn_vit) явно использует AdamW — train.optimizer: "adamw" в
    конфиге переключает на torch.optim.AdamW (decoupled weight decay), иначе
    используется Adam как раньше.
    """
    beta1 = train_config.get("adam_beta1", 0.9)
    beta2 = train_config.get("adam_beta2", 0.999)
    default_lr = train_config["lr"]
    weight_decay = train_config["weight_decay"]
    optimizer_cls = torch.optim.AdamW if train_config.get("optimizer", "adam").lower() == "adamw" else torch.optim.Adam

    def factory(params, lr_override=None):
        return optimizer_cls(
            params,
            lr=lr_override if lr_override is not None else default_lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )

    return factory


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested_device


def main():
    args = parse_args()
    config = load_config(args.config)

    method_name = args.method or config["method"]["name"]
    print(f"=== Метод: {method_name} ===")

    seed = config["dataset"]["seed"]
    set_seed(seed)

    device = resolve_device(config["train"]["device"])
    print(f"Device: {device}")

    print("Подготовка данных...")
    data = prepare_data_for_method(method_name, config, seed)
    if method_name == "efficient_cnn":
        print(
            f"Development: {len(data['development_indices'])}, "
            f"Test: {len(data['test_indices'])}"
        )
    else:
        print(
            f"Labeled: {len(data['labeled_images'])}, "
            f"Unlabeled: {len(data['unlabeled_images'])}, "
            f"Test: {len(data['test_images'])}"
        )

    model = build_model(method_name, num_classes=NUM_CLASSES, config=config["model"])
    optimizer_factory = make_optimizer_factory(config["train"])
    method = build_method(method_name, model, optimizer_factory, device=device, config=config)

    loader_factory = DataLoaderFactory(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
    )

    result = method.fit(
        data=data,
        loader_factory=loader_factory,
        epochs=config["train"]["epochs"],
        eval_every=config["eval"]["eval_every"],
        checkpoint_path=config["output"]["checkpoint_path"],
    )
    print(f"\nЛучший F1 во время обучения: {result['best_f1']*100:.2f}%")
    if result.get("optimal_epochs") is not None:
        print(f"Оптимальное число эпох (early stopping по валидации): {result['optimal_epochs']}")

    print("\n=== Финальная оценка (best checkpoint) ===")
    method.load_state_dict(torch.load(config["output"]["checkpoint_path"]))
    test_loader = method.build_eval_loader(data, loader_factory)
    y_true, y_pred = method.evaluate_per_class(test_loader)

    final_metrics = compute_metrics(y_true, y_pred)
    report = per_class_report(y_true, y_pred, CLASS_NAMES)
    print_per_class_report(report)

    print("\n=== Сравнение со статьёй ===")
    paper_result = config.get("paper_metrics", {}).get(method_name)
    if paper_result is not None:
        compare_to_paper(final_metrics, paper_result)

    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    result_payload = {
        "overall_metrics": final_metrics,
        "per_class_metrics": report,
        "paper_metrics": paper_result,
        "epochs_trained": config["train"]["epochs"],
    }
    if result.get("pseudo_label_diagnostics") is not None:
        result_payload["pseudo_label_diagnostics"] = result["pseudo_label_diagnostics"]

    update_results(
        method_name,
        result_payload,
        config["output"]["results_path"],
    )


if __name__ == "__main__":
    main()
