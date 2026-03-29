import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,  # type: ignore[attr-defined]
        torch_geometric.data.data.DataTensorAttr,  # type: ignore[attr-defined]
        torch_geometric.data.storage.GlobalStorage,  # type: ignore[attr-defined]
    ]
)


def collate_fn(batch):
    sequences, masks = zip(*batch, strict=False)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return sequences_padded, masks_padded


class SequentialDataset(TorchDataset):
    def __init__(self, data, window, device, step=None):
        self.data = data
        self.window = window
        self.device = device
        if step is None:
            self.step = window
        else:
            self.step = step

    def __getitem__(self, index):
        start_idx = index * self.step
        end_idx = min(start_idx + self.window, len(self.data))
        x = self.data[start_idx:end_idx].to(self.device)
        mask = torch.ones_like(x, dtype=torch.bool).to(self.device)
        return x, mask

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.step + 1)


class NetFlowDataset:
    def __init__(
        self,
        name,
        data_dir,
        force_reload=False,
        fraction=None,
        data_type="benign",
        seed=42,
        source_features=None,
        known_attacks=None,
    ):
        self.name = name
        self.data_dir = data_dir
        self.fraction = fraction
        self.data_type = data_type
        self.seed = seed
        self.source_features = source_features
        self.known_attacks = known_attacks

        # Setup directories
        graph_dir = os.path.join(data_dir, "pyg_graph_data")
        if fraction is not None:
            assert 0 < fraction < 1
            fraction_str = str(fraction).replace(".", "_")
            self.processed_dir = os.path.join(graph_dir, f"{name}_{fraction_str}")
        else:
            self.processed_dir = os.path.join(graph_dir, name)

        self.raw_dir = os.path.join(data_dir, name)

        # Determine feature names from CSV header (needed for alignment even if cached)
        df_header = pd.read_csv(os.path.join(self.raw_dir, f"{self.name}.csv"), nrows=0)
        x_cols = df_header.drop(columns=["Attack", "Label"])
        if "v3" in self.name:
            self.edge_feature_names = [
                col for col in x_cols.columns
                if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_END_MILLISECONDS", "FLOW_START_MILLISECONDS"]
            ]
        else:
            self.edge_feature_names = [
                col for col in x_cols.columns
                if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
            ]

        # Handle force reload
        if force_reload and os.path.exists(self.processed_dir):
            print(
                f"Force reload: Removing existing processed data at {self.processed_dir}"
            )
            shutil.rmtree(self.processed_dir)

            # Also remove the scaler for this dataset
            scaler_path = os.path.join("scalers", f"scaler_{self.name}.pkl")
            if os.path.exists(scaler_path):
                print(f"Removing old scaler: {scaler_path}")
                os.remove(scaler_path)

        # Check if we need to process
        if self._needs_processing():
            # Check for seed mismatch before processing
            seed_file = os.path.join(self.processed_dir, ".seed")
            if os.path.exists(seed_file):
                with open(seed_file) as f:
                    cached_seed = int(f.read().strip())
                if cached_seed != self.seed:
                    print(
                        f"Warning: Cached data was created with seed={cached_seed}, but current seed={self.seed}"
                    )
                    print(
                        "Run with --reload_dataset to recreate data with the new seed"
                    )

            self._process()
        else:
            # Check for seed mismatch when loading existing cache
            seed_file = os.path.join(self.processed_dir, ".seed")
            if os.path.exists(seed_file):
                with open(seed_file) as f:
                    cached_seed = int(f.read().strip())
                if cached_seed != self.seed:
                    print(
                        f"Warning: Cached data was created with seed={cached_seed}, but current seed={self.seed}"
                    )
                    print(
                        "Run with --reload_dataset to recreate data with the new seed"
                    )

        # Load the processed data
        self.train_graph = torch.load(os.path.join(self.processed_dir, "train.pt"))[0]
        self.val_graph = torch.load(os.path.join(self.processed_dir, "val.pt"))[0]
        self.test_graph = torch.load(os.path.join(self.processed_dir, "test.pt"))[0]

        # Load attack type labels if available
        self.test_attack_labels = None
        self.val_attack_labels = None
        attack_path = os.path.join(self.processed_dir, "attack_labels.pt")
        if os.path.exists(attack_path):
            attack_data = torch.load(attack_path, weights_only=False)
            self.val_attack_labels = attack_data.get("val", None)
            self.test_attack_labels = attack_data.get("test", None)
        elif self.known_attacks is not None:
            # Regenerate attack labels from raw CSV to avoid re-processing
            print("Regenerating attack labels from raw CSV...")
            self._regenerate_attack_labels(attack_path)

        # Open-set filtering: restrict val set to Benign + known attacks
        if self.known_attacks is not None and self.val_attack_labels is not None:
            known_set = set(self.known_attacks)
            keep_mask = torch.tensor(
                [a == "Benign" or a in known_set for a in self.val_attack_labels],
                dtype=torch.bool,
            )
            self.val_graph.edge_index = self.val_graph.edge_index[:, keep_mask]
            self.val_graph.edge_attr = self.val_graph.edge_attr[keep_mask]
            self.val_graph.edge_labels = self.val_graph.edge_labels[keep_mask]
            self.val_attack_labels = [
                a for a, k in zip(self.val_attack_labels, keep_mask.tolist()) if k
            ]

        # Apply feature alignment if requested
        if self.source_features is not None:
            self._align_features()
        else:
            self.edge_features = self.edge_feature_names

    def _regenerate_attack_labels(self, save_path):
        """Regenerate attack labels from raw CSV using the same split logic.

        This reproduces the same train_test_split as _process() to extract
        the Attack column for val/test sets without re-processing graphs.
        """
        df = pd.read_csv(os.path.join(self.raw_dir, f"{self.name}.csv"))

        if self.fraction is not None:
            df = df.groupby(by="Attack").sample(
                frac=self.fraction, random_state=self.seed
            )

        x = df.drop(columns=["Attack", "Label"])
        y = df[["Attack", "Label"]]

        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0)

        df = pd.concat([x, y], axis=1)

        df_train, df_val_test = train_test_split(
            df, test_size=0.2, random_state=self.seed, stratify=y["Attack"]
        )

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=self.seed,
            stratify=df_val_test["Attack"],
        )

        if "v3" in self.name:
            df_train = df_train.sort_values(by="FLOW_START_MILLISECONDS")
            df_val = df_val.sort_values(by="FLOW_START_MILLISECONDS")
            df_test = df_test.sort_values(by="FLOW_START_MILLISECONDS")

        attack_labels = {
            "train": df_train["Attack"].tolist(),
            "val": df_val["Attack"].tolist(),
            "test": df_test["Attack"].tolist(),
        }

        torch.save(attack_labels, save_path)
        self.val_attack_labels = attack_labels["val"]
        self.test_attack_labels = attack_labels["test"]
        print(f"  Saved attack labels to {save_path}")

    def _align_features(self):
        """Align edge_attr columns to match source_features architecture."""
        print(f"Aligning {self.name} features to source schema...")
        target_feat_to_idx = {name: i for i, name in enumerate(self.edge_feature_names)}
        projection_indices = []
        for name in self.source_features:
            if name in target_feat_to_idx:
                projection_indices.append(target_feat_to_idx[name])
            else:
                projection_indices.append(-1)
        for graph in [self.train_graph, self.val_graph, self.test_graph]:
            old_attr = graph.edge_attr
            new_attr = torch.zeros((old_attr.size(0), len(self.source_features)), device=old_attr.device)
            for src_idx, tgt_idx in enumerate(projection_indices):
                if tgt_idx != -1:
                    new_attr[:, src_idx] = old_attr[:, tgt_idx]
            graph.edge_attr = new_attr
            graph.x = torch.ones(graph.num_nodes, len(self.source_features), dtype=torch.float)
        self.edge_features = list(self.source_features)

    def _needs_processing(self):
        """Check if processing is needed"""
        if not os.path.exists(self.processed_dir):
            return True

        required_files = ["train.pt", "val.pt", "test.pt"]
        for filename in required_files:
            if not os.path.exists(os.path.join(self.processed_dir, filename)):
                return True

        return False

    def _process(self):
        """Process the raw CSV data and create train/val/test splits"""
        print(f"Processing dataset {self.name}...")

        os.makedirs(self.processed_dir, exist_ok=True)

        df = pd.read_csv(os.path.join(self.raw_dir, f"{self.name}.csv"))
        df = df.dropna()

        if self.fraction is not None:
            df = df.groupby(by="Attack").sample(
                frac=self.fraction, random_state=self.seed
            )

        x = df.drop(columns=["Attack", "Label"])
        y = df[["Attack", "Label"]]

        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0)

        if "v3" in self.name:
            edge_features = [
                col
                for col in x.columns
                if col
                not in [
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                ]
            ]
        else:
            edge_features = [
                col
                for col in x.columns
                if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
            ]

        df = pd.concat([x, y], axis=1)

        df_train, df_val_test = train_test_split(
            df, test_size=0.2, random_state=self.seed, stratify=y["Attack"]
        )

        if self.data_type == "benign":
            df_train = df_train[df_train["Label"] == 0]

        scaler_path = os.path.join("scalers", f"scaler_{self.name}.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception as e:
                print(f"Failed to load scaler: {e}. Creating new one.")
                scaler = MinMaxScaler()
                scaler.fit(df_train[edge_features])
        else:
            scaler = MinMaxScaler()
            scaler.fit(df_train[edge_features])
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        df_train[edge_features] = scaler.transform(df_train[edge_features])
        df_val_test_scaled = scaler.transform(df_val_test[edge_features])
        df_val_test[edge_features] = np.clip(df_val_test_scaled, -10, 10)

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=self.seed,
            stratify=df_val_test["Attack"],
        )

        if "v3" in self.name:
            df_train = df_train.sort_values(by="FLOW_START_MILLISECONDS")
            df_val = df_val.sort_values(by="FLOW_START_MILLISECONDS")
            df_test = df_test.sort_values(by="FLOW_START_MILLISECONDS")

        unique_nodes = pd.concat(
            [
                df_train["IPV4_SRC_ADDR"],
                df_train["IPV4_DST_ADDR"],
                df_val["IPV4_SRC_ADDR"],
                df_val["IPV4_DST_ADDR"],
                df_test["IPV4_SRC_ADDR"],
                df_test["IPV4_DST_ADDR"],
            ]
        ).unique()
        node_map = {node: i for i, node in enumerate(unique_nodes)}
        num_nodes = len(node_map)

        datasets = {"train": df_train, "val": df_val, "test": df_test}
        attack_labels = {}  # Store attack type strings per split

        for split_name, df_split in datasets.items():
            src_nodes = np.array([node_map[ip] for ip in df_split["IPV4_SRC_ADDR"]])
            dst_nodes = np.array([node_map[ip] for ip in df_split["IPV4_DST_ADDR"]])
            edge_index = torch.tensor(
                np.array([src_nodes, dst_nodes]), dtype=torch.long
            )
            edge_attr = torch.tensor(df_split[edge_features].values, dtype=torch.float)
            edge_labels = torch.tensor(df_split["Label"].values, dtype=torch.long)
            x = torch.ones(num_nodes, edge_attr.shape[1], dtype=torch.float)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_labels=edge_labels,
                num_nodes=num_nodes,
            )

            # Save as list for compatibility
            torch.save([data], os.path.join(self.processed_dir, f"{split_name}.pt"))

            # Save attack type strings for open-set evaluation
            attack_labels[split_name] = df_split["Attack"].tolist()

        # Save attack labels for open-set evaluation
        torch.save(
            attack_labels,
            os.path.join(self.processed_dir, "attack_labels.pt"),
        )

        # Save seed information for cache validation
        seed_file = os.path.join(self.processed_dir, ".seed")
        with open(seed_file, "w") as f:
            f.write(str(self.seed))

        print("Done!")

    def __len__(self):
        # Return total number of graphs (for compatibility)
        return 3

    @property
    def num_node_features(self):
        return self.train_graph.x.shape[1]

    @property
    def num_edge_features(self):
        return self.train_graph.edge_attr.shape[1]

    @property
    def num_nodes(self):
        return self.train_graph.num_nodes
