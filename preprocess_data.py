import multiprocessing as mp
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


def process_game(game: chess.pgn.Game) -> list[dict]:
    """Applies vectorization to each game"""
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        moves.append(
            {
                **{k: v for k, v in enumerate(fen_to_vector(board.fen()).tolist())},
                "next_move": board.san(move),
            }
        )
        board.push(move)
    return moves


def downcast_ints_to_int8(df):
    """Convert to int8 for better storage (we are training in int8)"""
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        # Check if values fit in int8 range (-128 to 127)
        if df[col].min() >= -128 and df[col].max() <= 127:
            df[col] = df[col].astype(np.int8)
    return df


def process_batch(pgn_strs: list[str], output_dir: str, batch_id: int):
    """Apply pgn processing to a batch"""
    try:
        data = []
        for pgn in pgn_strs:
            game = chess.pgn.read_game(io.StringIO(pgn))
            if include_game(game):
                data.extend(process_game(game))
        if data:
            df = pd.DataFrame(data)
            df = downcast_ints_to_int8(df)
            df.to_parquet(Path(output_dir) / f"chunk_{batch_id}.parquet", index=False)
    except Exception as e:
        print(f"[Batch {batch_id}] Error: {e}")
    return None


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
    max_games: int = None,
    num_workers: int = 8,
    batch_size: int = 25000,
):
    """Execute parallel processing of pgn parsing"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pool = mp.Pool(num_workers)
    results = []
    game_count = 0

    for batch_id, batch in enumerate(stream_pgns(pgn_path, batch_size=batch_size)):
        if max_games and game_count >= max_games:
            break

        results.append(
            pool.apply_async(process_batch, args=(batch, output_dir, batch_id))
        )
        game_count += len(batch)

    # Finish remaining jobs
    for r in results:
        r.get()

    pool.close()
    pool.join()
    print(f"Finished processing ~{game_count} games into {batch_id + 1} parquet files.")


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