"""Validation-only tuning for the Stage-1 HybridCNN-ViT teacher.

The test split is loaded by the shared dataset pipeline but is never evaluated
or used for selection in this script. A separate pseudo-evaluation split is
used only after model selection to estimate pseudo-label quality and coverage.
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from datasets import (
    CLASS_NAMES,
    DataLoaderFactory,
    NUM_CLASSES,
    prepare_data_for_method,
    set_seed,
)
from evaluate import compute_metrics
from losses import inverse_sqrt_class_weights
from models import build_model
from train import resolve_device


def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())
    return compute_metrics(labels, predictions)


def train_trial(data, base_config, trial, device):
    config = copy.deepcopy(base_config)
    config["model"]["pre_transformer_dropout"] = trial["pre_dropout"]
    config["model"]["transformer_dropout"] = trial["transformer_dropout"]
    set_seed(trial["seed"])

    model = build_model("hybrid_cnn_vit", NUM_CLASSES, config["model"]).to(device)
    loaders = DataLoaderFactory(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
    )
    train_loader = loaders.onehot_loader(
        data["labeled_images"], data["labeled_labels"], transform_mode="weak"
    )
    val_loader = loaders.onehot_eval_loader(data["val_images"], data["val_labels"])
    weights = inverse_sqrt_class_weights(
        torch.as_tensor(data["labeled_labels"]), NUM_CLASSES
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=trial["max_epochs"], eta_min=trial["lr"] * 0.05
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, trial["max_epochs"] + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                **metrics,
            }
        )
        print(
            f"trial={trial['name']} epoch={epoch} "
            f"loss={history[-1]['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f}",
            flush=True,
        )
        if metrics["f1"] > best_f1 + 1e-4:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= trial["patience"]:
                break

    model.load_state_dict(best_state)
    return model, {
        "trial": trial,
        "best_epoch": best_epoch,
        "best_metrics": evaluate(model, val_loader, device),
        "history": history,
    }


@torch.no_grad()
def pseudo_eval(model, data, config, device):
    loaders = DataLoaderFactory(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
    )
    loader = loaders.onehot_eval_loader(
        data["pseudo_eval_images"], data["pseudo_eval_labels"]
    )
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        all_probs.append(F.softmax(model(images.to(device)), dim=1).cpu())
        all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    thresholds = [0.50, 0.70, 0.80, 0.90, 0.94]
    rows = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        rows.append(
            {
                "threshold": threshold,
                "coverage": float(accepted.mean()),
                "accepted": int(accepted.sum()),
                "macro_f1": (
                    float(
                        f1_score(
                            labels[accepted],
                            predictions[accepted],
                            labels=np.arange(NUM_CLASSES),
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if accepted.any()
                    else None
                ),
            }
        )
    return {
        "size": int(len(labels)),
        "overall": compute_metrics(labels, predictions),
        "mean_confidence": float(confidence.mean()),
        "thresholds": rows,
    }


def main():
    with open("configs/hybrid_cnn_vit.yaml") as stream:
        config = yaml.safe_load(stream)
    device = resolve_device(config["train"]["device"])
    print(f"device={device}", flush=True)
    data = prepare_data_for_method(
        "hybrid_cnn_vit", config, config["dataset"]["seed"]
    )
    print(
        f"train={len(data['labeled_images'])} val={len(data['val_images'])} "
        f"pseudo_eval={len(data['pseudo_eval_images'])}; test is not evaluated",
        flush=True,
    )

    # Small, predeclared search space: enough to test the dominant optimization
    # choices without repeatedly adapting the grid to validation outcomes.
    search_space = [
        {
            "name": name,
            "lr": lr,
            "pre_dropout": pre,
            "transformer_dropout": transformer,
            "max_epochs": 12,
            "patience": 3,
            "seed": config["dataset"]["seed"],
        }
        for name, lr, pre, transformer in (
            ("conservative", 1e-4, 0.1, 0.1),
            ("standard", 3e-4, 0.1, 0.1),
            ("low_dropout", 3e-4, 0.0, 0.1),
            ("scratch_fast", 1e-3, 0.0, 0.1),
        )
    ]

    results = []
    best_model, best_result = None, None
    for trial in search_space:
        model, result = train_trial(data, config, trial, device)
        results.append(result)
        if (
            best_result is None
            or result["best_metrics"]["f1"] > best_result["best_metrics"]["f1"]
        ):
            best_model, best_result = model, result

    pseudo_metrics = pseudo_eval(best_model, data, config, device)
    output = {
        "selection_metric": "validation_macro_f1",
        "test_used_for_selection": False,
        "best": best_result,
        "pseudo_evaluation": pseudo_metrics,
        "trials": results,
        "class_names": CLASS_NAMES,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/hybrid_teacher_tuning.json", "w") as stream:
        json.dump(output, stream, indent=2)
    torch.save(
        {"model": best_model.state_dict()},
        "results/hybrid_teacher_best_checkpoint.pt",
    )
    print(json.dumps({"best": best_result, "pseudo_evaluation": pseudo_metrics}, indent=2))


if __name__ == "__main__":
    main()
