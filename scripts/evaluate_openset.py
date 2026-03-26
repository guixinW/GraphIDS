"""
Open-Set Recognition Evaluation Script for GraphIDS.

Compares GraphIDS (baseline) vs GraphIDS-CL (contrastive learning) under
different openness levels to evaluate open-set recognition capability.

Usage:
    # Evaluate all openness levels on one dataset:
    python scripts/evaluate_openset.py \
        --data_dir data/ \
        --dataset NF-CSE-CIC-IDS2018-v2 \
        --config configs/NF-CSE-CIC-IDS2018-v2.yaml

    # Evaluate a specific openness level with a data fraction for fast debugging:
    python scripts/evaluate_openset.py \
        --data_dir data/ \
        --dataset NF-CSE-CIC-IDS2018-v2 \
        --config configs/NF-CSE-CIC-IDS2018-v2.yaml \
        --fraction 0.05 \
        --openness_levels low
"""

import argparse
import math
import os
import sys
import warnings

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch_geometric.loader import LinkNeighborLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.graphids import GraphIDS
from utils.dataloaders import NetFlowDataset
from utils.trainers import find_threshold, test, validate

warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage"
)

try:
    if torch.cuda.is_available():
        torch.cuda.init()
        device = "cuda"
    else:
        device = "cpu"
except Exception:
    device = "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# Openness Level Definitions per Dataset Family
# ═══════════════════════════════════════════════════════════════════════════════

# NF-CSE-CIC-IDS2018-v2 attack names (space-separated)
_IDS2018_V2_GROUPS = {
    "DoS": [
        "DoS attacks-Hulk",
        "DoS attacks-GoldenEye",
        "DoS attacks-SlowHTTPTest",
        "DoS attacks-Slowloris",
    ],
    "DDoS": [
        "DDOS attack-HOIC",
        "DDoS attacks-LOIC-HTTP",
        "DDOS attack-LOIC-UDP",
    ],
    "BruteForce": [
        "SSH-Bruteforce",
        "FTP-BruteForce",
    ],
    "Bot": ["Bot"],
    "Infilteration": ["Infilteration"],
    "Web": ["Brute Force -Web", "Brute Force -XSS"],
    "SQL": ["SQL Injection"],
}

# NF-CSE-CIC-IDS2018-v3 attack names (underscore-separated)
_IDS2018_V3_GROUPS = {
    "DoS": [
        "DoS_attacks-Hulk",
        "DoS_attacks-GoldenEye",
        "DoS_attacks-SlowHTTPTest",
        "DoS_attacks-Slowloris",
    ],
    "DDoS": [
        "DDOS_attack-HOIC",
        "DDoS_attacks-LOIC-HTTP",
        "DDOS_attack-LOIC-UDP",
    ],
    "BruteForce": [
        "SSH-Bruteforce",
        "FTP-BruteForce",
    ],
    "Bot": ["Bot"],
    "Infilteration": ["Infilteration"],
    "Web": ["Brute_Force_-Web", "Brute_Force_-XSS"],
    "SQL": ["SQL_Injection"],
}

# NF-UNSW-NB15 (v2 and v3 share the same names)
_NB15_GROUPS = {
    "Exploits": ["Exploits"],
    "Fuzzers": ["Fuzzers"],
    "Generic": ["Generic"],
    "Reconnaissance": ["Reconnaissance"],
    "DoS": ["DoS"],
    "Analysis": ["Analysis"],
    "Backdoor": ["Backdoor"],
    "Shellcode": ["Shellcode"],
    "Worms": ["Worms"],
}


def _flatten(groups, keys):
    """Flatten selected group keys into a single attack list."""
    result = []
    for k in keys:
        result.extend(groups[k])
    return result


def get_openness_configs(dataset_name):
    """Return {level_name: known_attacks_list} for the given dataset."""
    if "CSE-CIC-IDS2018" in dataset_name:
        groups = _IDS2018_V3_GROUPS if "v3" in dataset_name else _IDS2018_V2_GROUPS
        configs = {
            "low": _flatten(
                groups,
                ["DoS", "DDoS", "BruteForce", "Bot", "Infilteration"],
            ),
            "medium": _flatten(groups, ["DoS", "DDoS"]),
            "high": _flatten(groups, ["DoS"]),
        }
    elif "UNSW-NB15" in dataset_name:
        groups = _NB15_GROUPS
        configs = {
            "low": _flatten(
                groups,
                [
                    "Exploits",
                    "Fuzzers",
                    "Generic",
                    "Reconnaissance",
                    "DoS",
                    "Analysis",
                    "Backdoor",
                ],
            ),
            "medium": _flatten(
                groups, ["Exploits", "Fuzzers", "Generic", "Reconnaissance"]
            ),
            "high": _flatten(groups, ["Exploits", "Fuzzers"]),
        }
    else:
        raise ValueError(f"Unknown dataset family: {dataset_name}")
    return configs


def compute_openness(num_train_classes, num_test_classes):
    """Compute openness score."""
    return 1.0 - math.sqrt(2.0 * num_train_classes / (num_test_classes + num_train_classes))


def compute_fpr_at_tpr(labels, scores, target_tpr=0.95):
    """Compute FPR at a given TPR threshold (e.g., FPR@95TPR).

    Here labels=1 means 'unknown' (positive), scores = reconstruction errors.
    """
    labels = np.array(labels)
    scores = np.array(scores)

    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        if tpr >= target_tpr:
            return fp / n_neg
    return fp / n_neg


def load_yaml_config(path):
    """Load a YAML config file and return a flat dict of values."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "value" in val:
            config[key] = val["value"]
        else:
            config[key] = val
    return config


def evaluate_model(
    model, dataset, cfg, known_attacks, openness_level, model_name,
):
    """Evaluate a single model on an open-set scenario.

    Returns a dict with all metrics.
    """
    fanout = cfg.get("fanout", -1)
    fanout_list = [fanout] if fanout != -1 else [-1]
    shuffle = cfg.get("positional_encoding", "None") == "None"
    batch_size = cfg.get("batch_size", 16384)
    ae_batch_size = cfg.get("ae_batch_size", 64)
    window_size = cfg["window_size"]

    cpu_count = os.cpu_count()
    num_workers = min(cpu_count, 6) if cpu_count is not None else 0

    # Val loader (for threshold fitting)
    val_loader = LinkNeighborLoader(
        data=dataset.val_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.val_graph.edge_index,
        edge_label=dataset.val_graph.edge_labels,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    # Test loader
    test_loader = LinkNeighborLoader(
        data=dataset.test_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.test_graph.edge_index,
        edge_label=dataset.test_graph.edge_labels,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    # Fit threshold on val set
    _, val_errors, val_labels = validate(
        model, val_loader, ae_batch_size, window_size, device
    )
    threshold = find_threshold(val_errors, val_labels, method="supervised")

    # Evaluate on test set
    test_f1, test_pr_auc, errors, labels, pred_time = test(
        model, test_loader, ae_batch_size, window_size, device, threshold
    )

    # Separate known vs unknown errors/labels using attack type info
    test_attacks = dataset.test_attack_labels
    known_set = set(known_attacks)

    errors_np = errors.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Build per-sample: is_unknown (1 if unknown attack type, 0 if known or benign)
    is_unknown = np.array(
        [1 if (a != "Benign" and a not in known_set) else 0 for a in test_attacks]
    )
    is_known_attack = np.array(
        [1 if a in known_set else 0 for a in test_attacks]
    )
    is_benign = np.array([1 if a == "Benign" else 0 for a in test_attacks])

    # Predictions
    preds = (errors_np > threshold).astype(int)

    # --- Metrics ---
    # Overall F1 (binary: benign vs all attacks)
    overall_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

    # PR-AUC (binary: benign vs all attacks)
    overall_pr_auc = average_precision_score(labels_np, errors_np)

    # Known attack F1 (only on known attack + benign samples)
    known_mask = (is_unknown == 0)
    if known_mask.sum() > 0:
        known_f1 = f1_score(
            labels_np[known_mask], preds[known_mask], average="macro", zero_division=0
        )
    else:
        known_f1 = float("nan")

    # Unknown detection: can the model detect unknown attacks as anomalies?
    # Binary: unknown=1 vs benign=0 (we exclude known attacks here)
    unknown_vs_benign_mask = (is_unknown == 1) | (is_benign == 1)
    if is_unknown[unknown_vs_benign_mask].sum() > 0 and is_benign[unknown_vs_benign_mask].sum() > 0:
        unknown_auroc = roc_auc_score(
            is_unknown[unknown_vs_benign_mask],
            errors_np[unknown_vs_benign_mask],
        )
        unknown_fpr95 = compute_fpr_at_tpr(
            is_unknown[unknown_vs_benign_mask],
            errors_np[unknown_vs_benign_mask],
            target_tpr=0.95,
        )
    else:
        unknown_auroc = float("nan")
        unknown_fpr95 = float("nan")

    # Unknown F1: among unknown samples, how many are detected as anomalies?
    unknown_mask = (is_unknown == 1)
    if unknown_mask.sum() > 0:
        unknown_recall = preds[unknown_mask].sum() / unknown_mask.sum()
    else:
        unknown_recall = float("nan")

    # Count attack types
    all_attack_types = set(a for a in test_attacks if a != "Benign")
    unknown_types = all_attack_types - known_set
    num_train_classes = 1  # only Benign
    num_test_classes = 1 + len(all_attack_types)  # Benign + all attack types
    openness = compute_openness(num_train_classes, num_test_classes)

    return {
        "model": model_name,
        "dataset": dataset.name,
        "openness_level": openness_level,
        "openness": f"{openness:.2%}",
        "num_known_attacks": len(known_set),
        "num_unknown_attacks": len(unknown_types),
        "overall_f1": overall_f1,
        "overall_pr_auc": overall_pr_auc,
        "known_f1": known_f1,
        "unknown_auroc": unknown_auroc,
        "unknown_recall": unknown_recall,
        "fpr@95tpr": unknown_fpr95,
        "threshold": threshold,
        "pred_time_s": pred_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Open-Set Recognition Evaluation: GraphIDS vs GraphIDS-CL"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "NF-CSE-CIC-IDS2018-v2",
            "NF-CSE-CIC-IDS2018-v3",
            "NF-UNSW-NB15-v2",
            "NF-UNSW-NB15-v3",
        ],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config (defines model architecture)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Root checkpoint directory (expects GraphIDS/ and GraphIDS-CL/ subdirs)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of data to use (for fast debugging)",
    )
    parser.add_argument(
        "--openness_levels",
        type=str,
        nargs="+",
        default=["low", "medium", "high"],
        choices=["low", "medium", "high"],
        help="Which openness levels to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    openness_configs = get_openness_configs(args.dataset)

    # Model definitions: (display name, checkpoint subdir, has_projector)
    models_to_eval = [
        ("GraphIDS", "GraphIDS", False),
        ("GraphIDS-CL", "GraphIDS-CL", True),
    ]

    all_results = []

    for level in args.openness_levels:
        known_attacks = openness_configs[level]
        print("=" * 70)
        print(f"OPENNESS LEVEL: {level.upper()}")
        print(f"  Known attacks:   {known_attacks}")
        unknown_types = []
        all_configs = get_openness_configs(args.dataset)
        # All attacks = union of all levels' known attacks (the low level has the most)
        all_attacks = set(all_configs["low"]) | set(all_configs["medium"]) | set(all_configs["high"])
        # For display: find all attacks in the dataset family
        if "CSE-CIC-IDS2018" in args.dataset:
            groups = _IDS2018_V3_GROUPS if "v3" in args.dataset else _IDS2018_V2_GROUPS
        else:
            groups = _NB15_GROUPS
        all_attacks_flat = []
        for v in groups.values():
            all_attacks_flat.extend(v)
        unknown_display = [a for a in all_attacks_flat if a not in known_attacks]
        print(f"  Unknown attacks: {unknown_display}")
        print("=" * 70)

        # Load dataset once per openness level
        print(f"\nLoading dataset {args.dataset} (known_attacks filtering)...")
        dataset = NetFlowDataset(
            name=args.dataset,
            data_dir=args.data_dir,
            fraction=args.fraction,
            data_type="benign",
            seed=args.seed,
            known_attacks=known_attacks,
        )

        if dataset.test_attack_labels is None:
            print(
                "ERROR: No attack_labels.pt found. "
                "Re-process the dataset with --reload_dataset first."
            )
            sys.exit(1)

        for model_name, ckpt_subdir, has_projector in models_to_eval:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                ckpt_subdir,
                f"GraphIDS_{args.dataset}_{args.seed}.ckpt",
            )
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {model_name}: checkpoint not found at {ckpt_path}")
                continue

            print(f"\n--- Evaluating: {model_name} ---")
            print(f"  Checkpoint: {ckpt_path}")

            # Build model
            edim_in = dataset.num_edge_features
            proj_dim = cfg.get("proj_dim", 128)
            model = GraphIDS(
                ndim_in=edim_in,
                edim_in=edim_in,
                edim_out=cfg["edim_out"],
                embed_dim=cfg["ae_embedding_dim"],
                num_heads=4,
                num_layers=cfg["num_layers"],
                window_size=cfg["window_size"],
                dropout=cfg.get("dropout", 0.0),
                ae_dropout=cfg.get("ae_dropout", 0.0),
                positional_encoding=cfg.get("positional_encoding", "None"),
                agg_type=cfg.get("agg_type", "mean"),
                mask_ratio=cfg.get("mask_ratio", 0.15),
                proj_dim=proj_dim,
            ).to(device)

            # Load checkpoint (strict=False for baseline without projector)
            chk = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(chk["model_state_dict"], strict=has_projector)
            print(f"  Loaded (epoch={chk.get('epoch', '?')})")

            result = evaluate_model(
                model, dataset, cfg, known_attacks, level, model_name,
            )
            all_results.append(result)

            # Print key metrics
            print(f"  Overall F1:      {result['overall_f1']:.4f}")
            print(f"  Overall PR-AUC:  {result['overall_pr_auc']:.4f}")
            print(f"  Known F1:        {result['known_f1']:.4f}")
            print(f"  Unknown AUROC:   {result['unknown_auroc']:.4f}")
            print(f"  Unknown Recall:  {result['unknown_recall']:.4f}")
            print(f"  FPR@95TPR:       {result['fpr@95tpr']:.4f}")

    # ── Summary Table ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY: Open-Set Recognition Results")
    print("=" * 90)
    header = (
        f"{'Model':<14} {'Level':<8} {'Openness':<10} "
        f"{'F1':<8} {'PR-AUC':<9} {'Known-F1':<10} "
        f"{'Unk-AUROC':<11} {'Unk-Recall':<12} {'FPR@95':<8}"
    )
    print(header)
    print("-" * 90)
    for r in all_results:
        line = (
            f"{r['model']:<14} {r['openness_level']:<8} {r['openness']:<10} "
            f"{r['overall_f1']:<8.4f} {r['overall_pr_auc']:<9.4f} {r['known_f1']:<10.4f} "
            f"{r['unknown_auroc']:<11.4f} {r['unknown_recall']:<12.4f} {r['fpr@95tpr']:<8.4f}"
        )
        print(line)
    print("=" * 90)

    # ── Save to CSV ────────────────────────────────────────────────────────
    results_dir = "openset_results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"openset_{args.dataset}.csv")

    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
