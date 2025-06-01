"""
Contains LiChessDataset class which is a torch IterableDataset
used during model training
"""

from pathlib import Path
import json
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import torch


class LichessDataset(IterableDataset):
    """Streaming dataset for vectorized chess games"""
    def __init__(self, batches: list[Path], target_mappings: Path, batch_size: int):

        self.batches = batches
        self.batch_size = batch_size

        with open(target_mappings, "r", encoding="utf-8") as f:
            self.idx_to_move = json.load(f)

        self.move_to_idx = {move: idx for idx, move in self.idx_to_move.items()}

        self.get_target_id = np.vectorize(
            self.move_to_idx.__getitem__, otypes=[np.int32]
        )

    def _get_rows_from_batch(self, batch: Path | str):

        metadata = np.load(batch, mmap_mode="r")

        inputs = torch.as_tensor(metadata["inputs"], dtype=torch.float32)
        targets = torch.as_tensor(
            self.get_target_id(metadata["targets"]), dtype=torch.int64
        )

        for start in range(0, len(targets), self.batch_size):
            end = start + self.batch_size
            if end > len(targets):
                break

            yield inputs[start:end], targets[start:end]

    def __iter__(self):

        info = get_worker_info()

        if info:
            my_batches = self.batches[info.id :: info.num_workers]
        else:
            my_batches = self.batches

        for batch in my_batches:
            yield from self._get_rows_from_batch(batch)

    def __getitem__(self, idx):
        raise NotImplementedError()
