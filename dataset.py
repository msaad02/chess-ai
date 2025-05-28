from pathlib import Path
import numpy as np
import torch
import json


class LichessDataset(torch.utils.data.Dataset):
    """
    Serves up data according to the batch, inherits pytorch Dataset
    which allows better parallelism and integration with pytorch
    """

    def __init__(
        self,
        root_dir: str | Path,
        minmax_batch: tuple[int, int] = (0, float("inf")),
        device: torch.device = None,
    ):
        """Initializes dataset and loads the first batch"""

        self.device = device

        self.batches = sorted(
            Path(root_dir).iterdir(), key=lambda p: int(p.name.split("_")[1])
        )

        batch_list = [
            p
            for p in self.batches
            if minmax_batch[0] <= int(p.name.split("_")[1]) < minmax_batch[1]
        ]

        self.batches = {i: p for i, p in enumerate(batch_list)}

        self.x_files = {
            id: Path(batch, "board_vecs.npz") for id, batch in self.batches.items()
        }
        self.y_files = {
            id: Path(batch, "next_moves.json") for id, batch in self.batches.items()
        }

        self.batch_ranges = {}
        distinct_moves = set()
        batch_start = 0
        for batch_id, batch_targets_file in self.y_files.items():
            with open(batch_targets_file) as f:
                batch_targets = json.load(f)

                self.batch_ranges[batch_id] = {
                    "min": batch_start,
                    "max": batch_start + len(batch_targets),
                }
                distinct_moves.update(batch_targets)

                batch_start += len(batch_targets)

        self.idx_to_move = {
            idx: move for idx, move in enumerate(sorted(distinct_moves))
        }
        self.move_to_idx = {move: idx for idx, move in self.idx_to_move.items()}

        self._length = batch_start

        self.curr_batch_id = 0
        self.curr_batch = self.load_new_batch(list(self.batches.keys())[0])

    def load_new_batch(
        self, batch_id: int
    ) -> dict[str : torch.Tensor, str : torch.Tensor]:
        """Load all X/Y values in batch"""

        metadata = np.load(self.x_files[batch_id], mmap_mode="r")
        shape, data = metadata["shape"], metadata["data"]
        bits = np.unpackbits(data, bitorder="little")[: np.prod(shape)]
        x = torch.from_numpy(bits.reshape(shape).astype(np.float32))

        # Load Y with new batch
        with open(self.y_files[batch_id]) as f:
            y = torch.tensor(
                [self.move_to_idx.get(x) for x in json.load(f)], dtype=torch.int16
            )

        if self.device:
            x = x.to(self.device)
            y = y.to(self.device)

        return {"x": x, "y": y}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):

        batch_info = {
            id: range
            for id, range in self.batch_ranges.items()
            if range["min"] <= idx < range["max"]
        }

        assert batch_info, IndexError

        batch_id = list(batch_info.keys())[0]
        if batch_id != self.curr_batch_id:

            self.curr_batch = self.load_new_batch(batch_id)
            self.curr_batch_id = batch_id

        idx_real = idx - batch_info[batch_id]["min"]

        return self.curr_batch["x"][idx_real], self.curr_batch["y"][idx_real]
