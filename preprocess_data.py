from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import chess.pgn
import io


def include_game(game: chess.pgn.Game, min_elo: int = 1500) -> bool:
    """Returns True/False value whether to include game in data"""
    try:
        return (
            game.headers.get("Termination") == "Normal"
            and int(game.headers.get("WhiteElo", 0)) > min_elo
            and int(game.headers.get("BlackElo", 0)) > min_elo
        )
    except:
        return False


def fen_to_vector(fen: str) -> np.ndarray:
    """Converts FEN string to vector"""
    board = chess.Board(fen)

    # Piece placement: 8x8x12
    piece_tensor = np.zeros((8, 8, 12), dtype=np.uint8)
    piece_to_idx = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece_to_idx[piece.symbol()]
        piece_tensor[row, col, idx] = 1

    # Turn: 1 bit
    turn_tensor = np.array([int(board.turn)], dtype=np.uint8)

    #  Castling rights: 4 bits
    castling_tensor = np.array(
        [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ],
        dtype=np.uint8,
    )

    # En passant: 64-bit one-hot
    ep_tensor = np.zeros(64, dtype=np.uint8)
    if board.ep_square is not None:
        ep_tensor[board.ep_square] = 1

    # Flatten and concatenate all parts
    full_vector = np.concatenate(
        [piece_tensor.flatten(), turn_tensor, castling_tensor, ep_tensor]
    )

    return full_vector  # shape: (837,)


def process_game(game: chess.pgn.Game) -> pd.DataFrame:
    """Applies vectorization to each game"""
    board = game.board()

    board_vecs = []
    next_moves = []

    for move in game.mainline_moves():
        board_vecs.append(fen_to_vector(board.fen()))
        next_moves.append(board.san(move))
        board.push(move)

    if board_vecs and next_moves:
        vecs = np.stack(board_vecs, axis=0)

        df = pd.DataFrame(vecs, dtype=np.int8)
        df["next_move"] = next_moves

        return df
    else:
        return pd.DataFrame()


def process_batch(pgn_strs: list[str], output_dir: str, batch_id: int):
    """Apply pgn processing to a batch"""
    try:
        data = []
        for pgn in pgn_strs:
            game = chess.pgn.read_game(io.StringIO(pgn))
            if include_game(game):
                df = process_game(game)
                if not df.empty:
                    data.append(df)
        if data:
            df = pd.concat(data)
            df.to_parquet(Path(output_dir) / f"chunk_{batch_id}.parquet", index=False)
    except Exception as e:
        print(f"[Batch {batch_id}] Error: {e}")
    return True


def stream_pgns(pgn_path: str, batch_size: int = 100):
    """Parses .pgn data, into batches for processing"""
    with open(pgn_path, encoding="utf-8") as f:
        batch, current_pgn, last_line = [], "", ""
        for line in f:
            current_pgn += line
            if line.strip() == "" and not last_line.strip().endswith("]"):
                batch.append(current_pgn)
                current_pgn = ""
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            last_line = line
        if batch:
            yield batch


def process_pgn_parallel(
    pgn_path: str,
    output_dir: str,
    max_games: int | None = None,
    num_workers: int = 8,
    batch_size: int = 25_000,
):
    """Processes data in parallel using generator and pool"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    game_count = 0
    batch_iter = stream_pgns(pgn_path, batch_size=batch_size)
    futures = []

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for batch_id, batch in enumerate(batch_iter):
            if max_games and game_count >= max_games:
                break

            # submit work
            fut = pool.submit(process_batch, batch, output_dir, batch_id)
            futures.append(fut)
            game_count += len(batch)

            # back‑pressure: don’t queue more than ~2× the worker count
            if len(futures) > num_workers * 2:
                _drain_completed(futures)

        # wait for the stragglers
        _drain_completed(futures, wait_all=True)

    print(f"Finished ~{game_count} games into {batch_id + 1} parquet files.")


def _drain_completed(futures, wait_all=False):
    """Pop completed futures, propagating exceptions immediately."""
    pending = []
    for fut in futures:
        if fut.done() or wait_all:
            fut.result()  # raises if worker failed
        else:
            pending.append(fut)
    futures[:] = pending


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pgn_path", type=str, default="data/lichess_db_standard_rated_2025-04.pgn"
    )
    parser.add_argument("--output_dir", type=str, default="data/parallel")
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
