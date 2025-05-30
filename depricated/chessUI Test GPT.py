import chess
import pygame
import os

# Include the best_move() function, minimax() function, and the evaluate_board() function from the previous examples.
def evaluate_board(board):
    # A very simple evaluation function that counts material on the board.
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_value = piece_values.get(piece.piece_type, 0)
            score += piece_value if piece.color == chess.WHITE else -piece_value

    return score

def minimax(board, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval

def best_move(board, depth):
    best_eval = float('-inf')
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, False)
        board.pop()

        if eval > best_eval:
            best_eval = eval
            best_move = move

    return best_move

# Initialize the pygame library
pygame.init()

# Define colors
WHITE = (255, 255, 255)
GRAY = (240, 240, 240)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

pieces = {}
pieces["None"] = pygame.image.load(os.path.join("pieces", "white_None.png"))

piece_dict = {"rook": "r", "knight": "n", "bishop": "b", "queen": "q", "king": "k", "pawn": "p"}
# pieces["none"] = pygame.image.load(os.path.join("pieces", "black_None.png"))
for piece in chess.PIECE_NAMES:
    if piece is None:
        continue
    for color in ["white", "black"]:
        pieces[piece_dict[piece].upper() if color == "white" else piece_dict[piece].lower()] = pygame.image.load(
            os.path.join("pieces", f"{color}_{piece}.png")
        )

# Board parameters
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

# Initialize the display
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("Chess")

# Draw the chess board
def draw_board():
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else GRAY
            pygame.draw.rect(
                screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )

# Draw the pieces
def draw_pieces(board):
    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        piece = board.piece_at(square)
        if piece:
            screen.blit(pieces[str(piece)], (col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE))

# Main game loop
def main():
    board = chess.Board()
    running = True
    selected_square = None

    while running and not board.is_game_over():
        draw_board()
        draw_pieces(board)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                col, row = event.pos
                square = chess.square(col // SQUARE_SIZE, 7 - row // SQUARE_SIZE)
                piece = board.piece_at(square)
                print(piece)

                if selected_square is None and piece and piece.color == board.turn:
                    selected_square = square
                    pygame.draw.rect(
                        screen,
                        YELLOW,
                        (
                            (col // SQUARE_SIZE) * SQUARE_SIZE,
                            (row // SQUARE_SIZE) * SQUARE_SIZE,
                            SQUARE_SIZE,
                            SQUARE_SIZE,
                        ),
                        3,
                    )
                elif selected_square is not None:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                        if not board.is_game_over():
                            board.push(best_move(board, 3))
                    selected_square = None
                else:
                    print('broke')
                print("square is:", square, '   and selected square is:', selected_square, '   and piece is:', piece)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
