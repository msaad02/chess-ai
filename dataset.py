from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import torch
import os


def get_dataset(base_path: str = "data/parallel"):
    """Returns X/Y data"""
    dfs = []
    for file in os.listdir(base_path):
        dfs.append(pd.read_parquet(base_path + "/" + file))

    df = pd.concat(dfs)

    x = df.drop(columns="next_move")
    y = df["next_move"]

    return x, y


def get_target_mappings(y):
    """Returns target id mappings based on targets"""
    # UCI = Universal Chess Interface. Used for moves, eg. e2e4 (move from e2 to e4)
    uci_set = set(y.unique())
    uci_to_id = {uci: idx for idx, uci in enumerate(sorted(uci_set))}
    id_to_uci = {v: k for k, v in uci_to_id.items()}

    return uci_to_id, id_to_uci
