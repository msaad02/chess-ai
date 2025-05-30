import re
import numpy as np
import pandas as pd

def preprocess_scores(eval):
    score_dict = {
        '#+': 2000,
        '#-': -2000
    }

    try:
        eval = int(eval)
        eval = max(min(eval, 1500), -1500)
    except ValueError:
        eval = score_dict.get(eval[0:2], 0)

    return eval

# comment here

def vectorize(fen):
    data = re.split(" ", fen)
    rows= re.split("/", data[0])
    turn = data[1]
    can_castle = data[2]
    passant = data[3]
    half_moves = data[4]
    full_moves = data[5]
    
    bit_vector = np.zeros((13, 8, 8), dtype=np.int8)
    #print(bit_vector.shape)
    #what layer each piece is found on
    piece_to_layer = {
            'R': 1,
            'N': 2,
            'B': 3,
            'Q': 4,
            'K': 5,
            'P': 6,
            'p': 7,
            'k': 8,
            'q': 9,
            'b': 10,
            'n': 11,
            'r': 12
        }
    #find each piece based on type
    for r,value in enumerate(rows):
        colum = 0
        for piece in value:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece],r,colum] =1
                colum += 1
            else:
                colum += int(piece)
    
    if turn.lower() == 'w':
        bit_vector [0,7,4] =1
    else:
        bit_vector [0,0,4] =1
        
    #where each castle bit is located
    castle ={
        'k': (0,0),
        'q': (0,7),
        'K': (7,0),
        'Q': (7,7),
        }

    for value in can_castle:
        if value in castle:
            bit_vector[0,castle[value][0],castle[value][1]] = 1
    
    #put en-passant square in the vector
    if passant != '-':
        bit_vector[0,  5 if (int(passant[1])-1 == 3) else 2 , ord(passant[0]) - 97,] = 1
    
    return bit_vector.flatten()


def processDF(df):
    fen_strings = df.FEN
    evaluations = df.Evaluation.apply(preprocess_scores)

    # Vectorize the FEN strings
    vectorized_data = np.vstack([vectorize(fen) for fen in fen_strings])

    # Combine the vectorized data with the evaluations
    combined_data = np.hstack((vectorized_data, np.array(evaluations).reshape(-1, 1)))

    # Define the column names
    column_names = [i for i in range(832)] + ["Evaluation"]

    # Create the DataFrame
    df = pd.DataFrame(combined_data, columns=column_names, dtype=np.int8)
    
    return df

def processFEN(FEN):
    vectorized_data = np.vstack([vectorize(FEN)])
    column_names = [i for i in range(832)]
    df = pd.DataFrame(vectorized_data, columns=column_names, dtype=np.int8)

    return df

def evaluate(board, model):
    evaluation = model.predict_step(np.asarray(processFEN(board.fen())))
    return evaluation

def find_best_move(board, depth, maximizing_player, model):
    if maximizing_player:
        best_score = float('-inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth - 1, float('-inf'), float('inf'), False, model)
            print(move, score)
            if score > best_score:
                best_score = score
                best_move = move
            board.pop()
    else:
        best_score = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth - 1, float('-inf'), float('inf'), True, model)
            print(move, score)
            if score < best_score:
                best_score = score
                best_move = move
            board.pop()
    
    return best_move

# def minimax(board, depth, alpha, beta, maximizing_player, model):
#     if board.is_game_over() or depth == 0:
#         return evaluate(board, model)

def minimax(board, depth, alpha, beta, maximizing_player, model):
    if board.is_checkmate():
        return -10000 if maximizing_player else 10000
    if board.is_game_over(claim_draw=True):
        return 9000 if maximizing_player else -9000
    if depth == 0:
        return evaluate(board, model)
    
    if maximizing_player:
        best_score = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, False, model)
            score += depth
            board.pop()
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score
    else:
        best_score = float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, True, model)
            score -= depth
            board.pop()
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score