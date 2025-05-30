import chess
import berserk
import random

##### Requires "best_move()" function from other .py file. => See line 40.

# Replace YOUR_API_TOKEN with your personal Lichess API token
API_TOKEN = 'YOUR_API_TOKEN'

# Initialize the Lichess API client
client = berserk.Client(berserk.TokenSession(API_TOKEN))

# Choose a Lichess bot (replace 'BotUsername' with the bot's username)
BOT_USERNAME = 'BotUsername'

# Create a new game against the chosen bot
game = client.challenges.challenge_ai(BOT_USERNAME, clock_limit=60*10, clock_increment=5)
game_id = game['id']

# Connect to the game's event stream
stream = client.board.stream_game_state(game_id)

# Process events from the game stream
for event in stream:
    if event['type'] == 'gameFull':
        game_state = event['state']
    elif event['type'] == 'gameState':
        game_state = event

    if 'status' in game_state and game_state['status'] != 'started':
        print('Game over:', game_state['status'])
        break

    if 'fen' not in game_state:
        continue

    board = chess.Board(game_state['fen'])

    if board.turn == chess.WHITE:
        move = best_move(board, 3)
        if move is not None:
            client.board.make_move(game_id, move)
        else:
            print("No legal moves available")
            break