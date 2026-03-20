import argparse
import os
import sys

import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.graphids import GraphIDS

def load_yaml_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "value" in val:
            config[key] = val["value"]
        else:
            config[key] = val
    return config

def main():
    parser = argparse.ArgumentParser(description="Calculate GraphIDS model statistics")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # Note: These values should match your dataset features
    # NB15-V3 has 49, IDS2018-V3 has 49.
    # We use 49 as a representative default if not specified.
    edim_in = 49 
    ndim_in = 49

    import inspect
    sig = inspect.signature(GraphIDS.__init__)
    
    model_args = {
        "ndim_in": ndim_in,
        "edim_in": edim_in,
        "edim_out": cfg["edim_out"],
        "embed_dim": cfg["ae_embedding_dim"],
        "num_heads": 4,
        "num_layers": cfg["num_layers"],
        "window_size": cfg["window_size"],
        "dropout": cfg.get("dropout", 0.0),
        "ae_dropout": cfg.get("ae_dropout", 0.0),
        "positional_encoding": cfg.get("positional_encoding", "None"),
        "agg_type": cfg.get("agg_type", "mean"),
        "mask_ratio": cfg.get("mask_ratio", 0.15),
    }

    # Only add proj_dim if the model constructor supports it (CL branch)
    if "proj_dim" in sig.parameters:
        model_args["proj_dim"] = cfg.get("proj_dim", 128)

    model = GraphIDS(**model_args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Each parameter is float32 (4 bytes)
    param_size_mb = (total_params * 4) / (1024 * 1024)
    
    print("-" * 50)
    print(f"Model Statistics for config: {os.path.basename(args.config)}")
    print("-" * 50)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Model Size (Static):   {param_size_mb:.2f} MB")
    print("-" * 50)
    
    # Breakdown by component
    print("\nComponent Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:15} : {params:10,}")

    print("\nNote: Memory usage during training will be significantly higher")
    print("due to activations, gradients, and optimizer states.")

if __name__ == "__main__":
    main()
