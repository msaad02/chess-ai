"""
Contains class that interacts with lichess bot repository. Move this
file into the lichess-bot folder as the `homemade.py` file and update
config.yml engine name according to class name.

https://github.com/lichess-bot-devs/lichess-bot
"""

import sys
import json
from pathlib import Path
from typing import override
import torch
import numpy as np
import chess
from chess.engine import PlayResult, Limit
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE

# So we can import from our actual source code repository
sys.path.append(str(Path(__file__).parent.parent.resolve().joinpath("chess-ai")))

from model import ImitationModel
from utils import board_to_vector


class ChessAIBot(MinimalEngine):
    """Class that implements model inference and usage"""

    def _init_model(self):
        """__init__ isn't working as expected. Manually init model as workaround"""

        model_path = Path("/home/msaad/workspace/chess-ai/models/model_v2/checkpoint.pt")
        target_mapping_path = Path(
            "/home/msaad/workspace/chess-ai/data/split_data/distinct_moves.json"
        )

        with open(Path(target_mapping_path), "r", encoding="utf-8") as f:
            target_map = json.load(f)

        self.model = ImitationModel(837, len(target_map))
        self.model.load_state_dict(torch.load(Path(model_path), weights_only=True)["model_state_dict"])

        self.model.eval()

        self.move_to_idx = {move: idx for idx, move in target_map.items()}
        self.idx_to_move = {int(v): k for k, v in self.move_to_idx.items()}

        self._get_target_id = np.vectorize(
            self.move_to_idx.__getitem__, otypes=[np.int32]
        )

    @override
    def search(
        self,
        board: chess.Board,
        time_limit: Limit,
        ponder: bool,
        draw_offered: bool,
        root_moves: MOVE,
        temperature: float = 0,
    ) -> PlayResult:
        """Output the model prediction"""

        if not hasattr(self, "model"):
            self._init_model()

        model_input = torch.as_tensor(board_to_vector(board), dtype=torch.float32)

        predictions = torch.softmax(self.model(model_input), dim=0)

        if 0 < temperature <= 1:
            random = torch.rand(len(self.move_to_idx)) / (1 - temperature)
            predictions += random

        legal_move_ids = torch.from_numpy(
            self._get_target_id(
                np.array([x.uci() for x in board.legal_moves], dtype="<U5")
            )
        )

        legal_predictions = predictions[legal_move_ids]

        best_move_id = legal_move_ids[legal_predictions.argmax()].item()

        pred_uci = self.idx_to_move[int(best_move_id)]

        return PlayResult(pred_uci, None)
