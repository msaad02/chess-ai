"""
The following functions are used in multiple scripts, hence
why they're stored here in the utils file
"""

import numpy as np
import chess


def board_to_vector(
    board: chess.Board,
    piece_tensor: np.ndarray = np.zeros((8, 8, 12), dtype=np.uint8),
    flattened: np.ndarray = np.empty(837, dtype=np.uint8),
) -> np.ndarray:
    """Converts chess.Board object to vector. Tensor parameters included to avoid recreation"""

    # Iterate over square, piece
    for sq, pc in board.piece_map().items():
        row, col = divmod(sq, 8)
        idx = (pc.piece_type - 1) + (6 if pc.color else 0)

        piece_tensor[7 - row, col, idx] = 1

    flattened[:768] = piece_tensor.ravel()
    flattened[768] = board.turn

    flattened[769:773] = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    flattened[773:] = 0
    if board.ep_square is not None:
        flattened[773 + board.ep_square] = 1

    return flattened  # shape: (837,)
