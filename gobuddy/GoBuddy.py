from flask import Flask
from flask_ask import Ask, question, statement
import pickle
import numpy as np
from Monte_Carlo3 import MonteCarlo

app = Flask(__name__)
ask = Ask(app, '/')

monte = None
state = None
coordinates = {"x": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
               "y": np.arange(19, step=-1) + 1}

@app.route('/')
def homepage():
    return "GoBuddy"


@ask.launch
def start_skill():
    global monte
    global state
    global coordinates
    state = None
    with open("mc_5x5.txt", "rb") as f:
        monte = pickle.load(f)
    assert isinstance(monte, MonteCarlo)
    monte.root.content.captures = 0
    coordinates["y"] = np.arange(monte.root.content.size, step=-1) + 1
    return question("I'll begin. Let me know when you are ready.")


@ask.intent("YesIntent")
def start_game():
    global monte
    global state
    if state is not None:
        return question("Sorry. Please try again.")
    state = monte.root.content
    result, move = monte.get_play()
    state = result[0]
    if move == "P":
        msg = "I'll pass."
    else:
        l_coord = coordinates["x"][move[1]]
        n_coord = coordinates["y"][move[0]]
        msg = "I'll play " + l_coord + " " + str(n_coord)
    state.paint()
    return question(msg)


@ask.intent("PassIntent")
def move_pass():
    global monte
    global state
    mv = "P"
    result = monte.update(mv)
    if result is False:
        return statement("That move was invalid. Please try again.")
    state = result.content
    state.paint()
    if state.winner() != 0:
        return end_game()
    result, move = monte.get_play()
    state = result[0]
    if move == "P":
        msg = "I'll pass."
    else:
        l_coord = coordinates["x"][move[1]]
        n_coord = coordinates["y"][move[0]]
        msg = "I'll play " + l_coord + " " + str(n_coord)
    state.paint()
    if state.winner() != 0:
        return end_game()
    return question(msg)



@ask.intent("MoveIntent")
def move(l_coord, n_coord):
    global monte
    global state
    r = coordinates["y"].index(n_coord)
    c = list(coordinates["x"]).index(l_coord)
    result = (r, c)
    result = monte.update(result)
    if result is False:
        return statement("That move was invalid. Please try again.")
    state = result.content
    state.paint()
    if state.winner() != 0:
        return end_game()
    result, move = monte.get_play()
    state = result[0]
    if move == "P":
        msg = "I'll pass."
    else:
        l_coord = coordinates["x"][move[1]]
        n_coord = coordinates["y"][move[0]]
        msg = "I'll play " + l_coord + " " + str(n_coord)
    state.paint()
    if state.winner() != 0:
        return end_game()
    return question(msg)


def end_game():
    global state
    scores = state.score()
    winner = state.winner()
    msg = ""
    if winner < 0:
        msg += "Tie reached."
    elif winner == 1:
        msg += "Computer wins."
    else:
        msg += "Player wins."
    msg += " The score was " + str(scores[0]) + "-" + str(scores[1]) + "."
    return statement(msg)


if __name__ == "__main__":
    app.run()