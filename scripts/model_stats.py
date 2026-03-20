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
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference estimation")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # 1. Parameter Statistics/Inference
    # Default to v3 (49)
    edim_in = 49 
    ndim_in = 49

    # Try to determine dimensions from checkpoint if provided
    loaded_chk = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading weights from: {args.checkpoint}")
        loaded_chk = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in loaded_chk:
            state_dict = loaded_chk["model_state_dict"]
            # encoder.fc_neigh.weight shape is [ndim_in, edim_in]
            if "encoder.fc_neigh.weight" in state_dict:
                ndim_in_chk, edim_in_chk = state_dict["encoder.fc_neigh.weight"].shape
                ndim_in, edim_in = ndim_in_chk, edim_in_chk
                print(f"Inferred dimensions from checkpoint: ndim_in={ndim_in}, edim_in={edim_in}")

    # If no checkpoint, try to infer from dataset name in config
    elif "dataset" in cfg:
        if "v2" in cfg["dataset"]:
            edim_in = 41
            ndim_in = 41
            print(f"Inferred v2 dimensions from config: edim_in=41, ndim_in=41")

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

    if "proj_dim" in sig.parameters:
        model_args["proj_dim"] = cfg.get("proj_dim", 128)

    model = GraphIDS(**model_args)

    if loaded_chk:
        model.load_state_dict(loaded_chk["model_state_dict"], strict=False)
        ckpt_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
    else:
        ckpt_size_mb = 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = (total_params * 4) / (1024 * 1024)
    
    print("-" * 50)
    print(f"Model Statistics (Config: {os.path.basename(args.config)})")
    print("-" * 50)
    if args.checkpoint:
        print(f"Checkpoint File Size:  {ckpt_size_mb:.2f} MB")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Static Weights Memory: {param_size_mb:.2f} MB")
    print("-" * 50)
    
    # 2. Inference Memory Estimation (Dynamic)
    window = cfg["window_size"]
    embed = cfg["ae_embedding_dim"]
    layers = cfg["num_layers"]
    batch = args.batch_size
    
    # Heuristic for activations and attention buffers
    activation_mem_layer = (batch * window * embed * 12 * 4) / (1024 * 1024) 
    attn_mem_mb = (batch * 4 * (window**2) * 4) / (1024 * 1024)
    total_dynamic_mb = (activation_mem_layer * layers) + attn_mem_mb
    
    print(f"\nInference Memory Estimation (Batch Size: {batch}):")
    print(f"  Approx. Activation Memory: ~{total_dynamic_mb:.2f} MB")
    print(f"  Total Expected VRAM:       ~{param_size_mb + total_dynamic_mb + 50:.2f} MB") 
    print("  (Estimated for GPU inference setup)")
    
    print("\nComponent Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:15} : {params:10,}")

if __name__ == "__main__":
    main()
