from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path
import numpy as np
import torch
import json
import random


class LichessStream(IterableDataset):
    def __init__(self, batches: list[Path], shuffle_batches: bool = False):

        self.shuffle_batches = shuffle_batches
        self.batches = batches

        distinct_moves = set()
        self._length = 0
        for batch in self.batches:
            with open(batch / "next_moves.json") as f:
                batch_targets = json.load(f)
                distinct_moves.update(batch_targets)
                self._length += len(batch_targets)

        self.idx_to_move = {
            idx: move for idx, move in enumerate(sorted(distinct_moves))
        }
        self.move_to_idx = {move: idx for idx, move in self.idx_to_move.items()}

    def __len__(self):
        return self._length

    def _yield_rows_from_batch(self, batch_dir):
        metadata = np.load(batch_dir / "board_vecs.npz", mmap_mode="r")
        shape, data = metadata["shape"], metadata["data"]
        bits_per_row = int(np.prod(shape[1:]))
        with open(batch_dir / "next_moves.json") as f:
            y_list = [self.move_to_idx[m] for m in json.load(f)]

        for row in range(shape[0]):
            byte0 = (row * bits_per_row) // 8
            byte1 = (row * bits_per_row + bits_per_row + 7) // 8
            bits = np.unpackbits(data[byte0:byte1], bitorder="little")[:bits_per_row]
            yield bits, y_list[row]

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            my_batches = self.batches
            rng = random.Random(42)
        else:
            num_w, wid = info.num_workers, info.id
            my_batches = self.batches[wid::num_w]
            rng = random.Random(42 + wid)

        while True:
            if self.shuffle_batches:
                rng.shuffle(my_batches)
            for b in my_batches:
                yield from self._yield_rows_from_batch(b)


def collate(batch):
    xs, ys = zip(*batch)
    # Casting here instead of LichessStream to save memory
    xs = torch.tensor(np.stack(xs), dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys
