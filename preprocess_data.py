from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from tqdm import tqdm
import numpy as np
import chess.pgn
import json
import io


def include_game(pgn_str: str, min_elo: int = 1500) -> bool:
    """Returns True/False value whether to include game in data

    Used chess.pgn.Game before, but regex is more *far* more performant.
    """
    try:
        tmp = pgn_str.split("\n\n")[0].split("\n")
        tmp = [x.replace("[", "").replace("]", "") for x in tmp]
        headers = {k.strip(): v for k, v in [x.split('"')[:2] for x in tmp]}

        return all(
            [
                headers["Termination"] == "Normal",
                int(headers["WhiteElo"]) > min_elo,
                int(headers["BlackElo"]) > min_elo,
            ]
        )
    except:
        return False


def board_to_vector(
    board: chess.Board, piece_tensor: np.ndarray, flattened: np.ndarray
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


def process_game(game: chess.pgn.Game) -> tuple[np.ndarray, list[str]]:
    """Returns board state as tensor and next move for each move in the game"""
    board = game.board()

    # Initialize here for each time to avoid recreating in memory for each move
    pieces = np.zeros((8, 8, 12), dtype=np.uint8)
    flattened = np.empty(837, dtype=np.uint8)

    board_vecs = []
    next_moves = []

    for move in game.mainline_moves():
        pieces.fill(0)

        board_vecs.append(board_to_vector(board, pieces, flattened).copy())
        next_moves.append(move.uci())
        board.push(move)

    if board_vecs and next_moves:
        return np.stack(board_vecs, axis=0), next_moves
    else:
        return None


def process_batch(pgn_strs: list[str], output_dir: str, batch_id: int):
    """Apply pgn processing to a batch"""
    try:
        save_location = Path(output_dir, f"batch_{batch_id}")
        save_location.mkdir(parents=True, exist_ok=True)
        board_vecs = []
        next_moves = []
        for pgn in pgn_strs:
            if include_game(pgn):
                game = chess.pgn.read_game(io.StringIO(pgn))

                result = process_game(game)
                if result:
                    board_vecs.append(result[0])
                    next_moves.extend(result[1])

        if board_vecs and next_moves:
            vecs = np.concatenate(board_vecs, axis=0)
            packed = np.packbits(vecs.flatten(), bitorder="little")
            np.savez_compressed(
                Path(save_location, "board_vecs"), shape=vecs.shape, data=packed
            )

            with open(Path(save_location, "next_moves.json"), "w") as f:
                json.dump(next_moves, f, indent=4)
    except Exception as e:
        print(f"[Batch {batch_id}] Error: {e}")
    return len(pgn_strs)


def stream_pgns(pgn_path: str, batch_size: int = 100):
    """Parses .pgn data, into batches for processing"""
    with open(pgn_path, encoding="utf-8") as f:
        batch = []
        current_pgn_lines = []
        last_line = ""
        for line in f:
            current_pgn_lines.append(line)
            if line.strip() == "" and not last_line.strip().endswith("]"):
                current_pgn = "".join(current_pgn_lines)
                batch.append(current_pgn)
                current_pgn_lines = []
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            last_line = line
        if batch:
            yield batch


def process_pgn_parallel(
    pgn_path: str,
    output_dir: str,
    *,
    max_games: int | None = None,
    num_workers: int = 8,
    batch_size: int = 25_000,
):
    """Processes data in parallel using generator and pool"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_submitted = total_done = 0
    batch_iter = stream_pgns(pgn_path, batch_size=batch_size)
    pbar = tqdm(total=max_games, unit="games") if max_games else tqdm(unit="games")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pending = set()

        for batch_id, batch in enumerate(batch_iter):
            if max_games and total_submitted >= max_games:
                break
            if max_games:
                batch = batch[: max_games - total_submitted]

            pending.add(pool.submit(process_batch, batch, output_dir, batch_id))
            total_submitted += len(batch)

            # Back‑pressure: keep at most 2×workers futures alive
            while len(pending) >= num_workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    cnt = fut.result()
                    total_done += cnt
                    pbar.update(cnt)

        # Drain whatever is left
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                cnt = fut.result()
                total_done += cnt
                pbar.update(cnt)

    pbar.close()
    print(
        f"Finished {total_done} games in {batch_id + 1} batches using {num_workers} workers."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pgn_path", type=str, default="data/lichess_db_standard_rated_2025-04.pgn"
    )
    parser.add_argument("--output_dir", type=str, default="data/split_data")
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=25000)
    args = parser.parse_args()

    process_pgn_parallel(
        pgn_path=args.pgn_path,
        output_dir=args.output_dir,
        max_games=args.max_games,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )
