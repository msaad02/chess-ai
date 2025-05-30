import re
import numpy as np
import pandas as pd

def preprocess_scores(eval):
    score_dict = {
        '#+': 2501,
        '#-': -2501
    }

    try:
        eval = int(eval)
        eval = max(min(eval, 2501), -2501)
    except ValueError:
        eval = score_dict.get(eval[0:2], 0)

    return eval

def new_preprocess_scores(eval):
    if eval.startswith('#'):
        mating_value = int(eval[2:])
        if mating_value == 0:
            value = 2_500
        else:
            value = max(2_500 - (mating_value - 1) * 50, 2_000)

        if eval.startswith('#+'):
            value = value
        else:
            value = -1 * value

    else:
        eval = int(eval)
        scaleFactor = (5_000 - 1_000) / 1_000

        if -1_000 <= eval <= 1_000:
            value = eval
        elif eval < -1_000:
            value = (eval + 1_000)/scaleFactor - 1_000
        else:
            value = (eval - 1_000)/scaleFactor + 1_000

        value = max(min(value, 2_000), -2_000)

    return np.int16(value)

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
    
    return bit_vector.flatten().astype(bool)


def processDF(df):
    fen_strings = df.FEN
    evaluations = df.Evaluation.apply(preprocess_scores).astype(np.int16)

    # Vectorize the FEN strings
    vectorized_data = np.vstack([vectorize(fen) for fen in fen_strings])

    # Combine the vectorized data with the evaluations
    combined_data = np.hstack((vectorized_data, np.array(evaluations).reshape(-1, 1)))

    # Define the column names
    column_names = [i for i in range(832)] + ["Evaluation"]

    # Create the DataFrame
    df = pd.DataFrame(combined_data, columns=column_names, dtype=np.int16)
    
    return df

def processFEN(FEN):
    vectorized_data = np.vstack([vectorize(FEN)])
    column_names = [i for i in range(832)]
    df = pd.DataFrame(vectorized_data, columns=column_names, dtype=np.int8)

    return df