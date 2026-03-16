"""
Cross-Dataset Evaluation Script for GraphIDS.

Purpose: Load a model checkpoint trained on dataset A and evaluate it on dataset B
to measure generalization capability across different network environments.

Usage:
    python scripts/evaluate_cross.py \
        --data_dir data/ \
        --source_config configs/NF-UNSW-NB15-v3.yaml \
        --target_dataset NF-CSE-CIC-IDS2018-v3 \
        --checkpoint checkpoints/GraphIDS_NF-UNSW-NB15-v3_42.ckpt

    This loads the model trained on NB15-v3 and evaluates it on IDE2018-v3's test set.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
)
from torch_geometric.loader import LinkNeighborLoader

# Add project root to path so we can import from utils/ and models/
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


def load_yaml_config(path):
    """Load a YAML config file and return a flat dict of values."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    # The YAML format uses {key: {value: actual_value}}
    config = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "value" in val:
            config[key] = val["value"]
        else:
            config[key] = val
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Evaluation for GraphIDS"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--source_config",
        type=str,
        required=True,
        help="Path to the YAML config used to TRAIN the model (defines architecture)",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        required=True,
        choices=[
            "NF-UNSW-NB15-v3",
            "NF-CSE-CIC-IDS2018-v3",
            "NF-UNSW-NB15-v2",
            "NF-CSE-CIC-IDS2018-v2",
        ],
        help="Target dataset to evaluate on (different from the training dataset)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--target_fraction",
        type=float,
        default=None,
        help="Fraction of target dataset to use (for faster debugging)",
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="transfer",
        choices=["transfer", "refit"],
        help=(
            "'transfer': use the threshold from the source checkpoint directly. "
            "'refit': re-compute the threshold on the target validation set."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── 1. Load source config (defines model architecture) ─────────────────
    src_cfg = load_yaml_config(args.source_config)
    source_dataset = src_cfg["dataset"]
    print(f"Source model trained on: {source_dataset}")
    print(f"Target evaluation on:   {args.target_dataset}")
    print(f"Threshold mode:         {args.threshold_mode}")
    print(f"Device:                 {device}")
    print("-" * 60)

    # ── 2. Load target dataset ─────────────────────────────────────────────
    target_ds = NetFlowDataset(
        name=args.target_dataset,
        data_dir=args.data_dir,
        fraction=args.target_fraction,
        data_type="benign",  # doesn't matter for eval, just for cache path
        seed=args.seed,
    )

    target_edim_in = target_ds.num_edge_features
    source_edim_in = target_edim_in  # Assuming same NF schema
    print(f"Target dataset features: {target_edim_in}")

    # ── 3. Build model with source architecture ────────────────────────────
    proj_dim = src_cfg.get("proj_dim", 128)
    model = GraphIDS(
        ndim_in=target_ds.num_node_features,
        edim_in=target_edim_in,
        edim_out=src_cfg["edim_out"],
        embed_dim=src_cfg["ae_embedding_dim"],
        num_heads=4,
        num_layers=src_cfg["num_layers"],
        window_size=src_cfg["window_size"],
        dropout=src_cfg.get("dropout", 0.0),
        ae_dropout=src_cfg.get("ae_dropout", 0.0),
        positional_encoding=src_cfg.get("positional_encoding", "None"),
        agg_type=src_cfg.get("agg_type", "mean"),
        mask_ratio=src_cfg.get("mask_ratio", 0.15),
        proj_dim=proj_dim,
    ).to(device)

    # ── 4. Load checkpoint ─────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    chk = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Use strict=False to handle projector keys that may or may not be present
    model.load_state_dict(chk["model_state_dict"], strict=False)
    source_threshold = chk.get("threshold", None)
    source_epoch = chk.get("epoch", "?")
    print(f"  Trained for {source_epoch} epochs")
    print(f"  Source threshold: {source_threshold}")
    print("-" * 60)

    # ── 5. Set up data loaders ─────────────────────────────────────────────
    fanout = src_cfg.get("fanout", -1)
    fanout_list = [fanout] if fanout != -1 else [-1]
    shuffle = src_cfg.get("positional_encoding", "None") == "None"
    batch_size = src_cfg.get("batch_size", 16384)
    ae_batch_size = src_cfg.get("ae_batch_size", 64)
    window_size = src_cfg["window_size"]

    cpu_count = os.cpu_count()
    num_workers = min(cpu_count, 6) if cpu_count is not None else 0

    # Validation loader (used if threshold_mode == "refit")
    val_loader = LinkNeighborLoader(
        data=target_ds.val_graph,
        num_neighbors=fanout_list,
        edge_label_index=target_ds.val_graph.edge_index,
        edge_label=target_ds.val_graph.edge_labels,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    test_loader = LinkNeighborLoader(
        data=target_ds.test_graph,
        num_neighbors=fanout_list,
        edge_label_index=target_ds.test_graph.edge_index,
        edge_label=target_ds.test_graph.edge_labels,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    # ── 6. Determine threshold ─────────────────────────────────────────────
    if args.threshold_mode == "transfer":
        threshold = source_threshold
        print(f"Using transferred threshold from source: {threshold}")
    else:
        print("Re-fitting threshold on target validation set...")
        _, val_errors, val_labels = validate(
            model, val_loader, ae_batch_size, window_size, device
        )
        val_pr_auc = average_precision_score(val_labels.cpu(), val_errors.cpu())
        threshold = find_threshold(val_errors, val_labels, method="supervised")
        print(f"  Target val PR-AUC: {val_pr_auc:.4f}")
        print(f"  Re-fitted threshold: {threshold}")
    print("-" * 60)

    # ── 7. Evaluate on target test set ─────────────────────────────────────
    print("Evaluating on target test set...")
    test_f1, test_pr_auc, errors, labels, pred_time = test(
        model, test_loader, ae_batch_size, window_size, device, threshold
    )

    # Also compute with re-fitted threshold for comparison
    refit_threshold = find_threshold(errors, labels, method="supervised")
    refit_pred = (errors > refit_threshold).int()
    refit_f1 = f1_score(labels.cpu(), refit_pred.cpu(), average="macro", zero_division=0)

    print("=" * 60)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Source:  {source_dataset}")
    print(f"Target:  {args.target_dataset}")
    print("-" * 60)
    print(f"PR-AUC (threshold-free):     {test_pr_auc:.4f}")
    print(f"F1 (transferred threshold):  {test_f1:.4f}")
    print(f"F1 (oracle re-fit on test):  {refit_f1:.4f}")
    print(f"Prediction time:             {pred_time:.4f}s")
    print("-" * 60)

    # Detailed classification report
    test_pred = (errors > threshold).int()
    print("\nClassification Report (transferred threshold):")
    print(
        classification_report(
            labels.cpu(),
            test_pred.cpu(),
            target_names=["Benign", "Malicious"],
            zero_division=0,
        )
    )

    # Save results to a file for easy comparison
    results_dir = "cross_eval_results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(
        results_dir,
        f"{source_dataset}_to_{args.target_dataset}.txt",
    )
    with open(result_file, "w") as f:
        f.write(f"Source: {source_dataset}\n")
        f.write(f"Target: {args.target_dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Threshold mode: {args.threshold_mode}\n")
        f.write(f"PR-AUC: {test_pr_auc:.4f}\n")
        f.write(f"F1 (transferred): {test_f1:.4f}\n")
        f.write(f"F1 (oracle refit): {refit_f1:.4f}\n")
        f.write(f"Prediction time: {pred_time:.4f}s\n")
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
