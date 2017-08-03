from flask import Flask
from flask_ask import Ask, question, statement
import pickle
import numpy as np
from Monte_Carlo3 import MonteCarlo, Node
import Go

app = Flask(__name__)
ask = Ask(app, '/')

monte = None
state = None
SIZE = 5
coordinates = {"x": "ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower(),
               "y": np.arange(SIZE, 0, step=-1)}


@app.route('/')
def homepage():
    return "GoBuddy"


@ask.launch
def start_skill():
    """
    Resets MCT for new game.

    Returns
    -------
    question
    """
    global monte
    global state
    global coordinates
    state = None
    monte.history = [monte.root]
    monte.root.content.captures = 0
    return question("I'll begin. Let me know when you are ready.")


@ask.intent("YesIntent")
def start_game():
    """
    Makes first move.

    Returns
    -------
    question and standard card
    """
    global coordinates
    global monte
    global state
    if state is not None:
        return question("Sorry. Please try again.")
    state = monte.root.content
    result, move = monte.get_play()
    state = result.content
    if move == "P":
        msg = "I'll pass."
    else:
        l_coord = coordinates["x"][move[1]]
        n_coord = coordinates["y"][move[0]]
        msg = "I'll play " + l_coord.upper() + " " + str(n_coord)
    img_link = state.paint()
    return question(msg) \
        .standard_card(title="GoBuddy Computer says...",
                       text=msg,
                       small_image_url=img_link,
                       large_image_url=img_link)


@ask.intent("PassIntent")
def move_pass():
    """
    Player passes, and computer moves (if applicable). Displays board in app.

    Returns
    -------
    question/statement and standard card
    """
    global monte
    global state
    mv = "P"
    result = monte.update(mv)
    if result is False:
        return statement("That move was invalid. Please try again.")
    state = result.content
    if state.winner() != 0:
        img_link = state.paint()
        return end_game() \
            .standard_card(title="GoBuddy Game Finished",
                           small_image_url=img_link,
                           large_image_url=img_link)
    result, move = monte.get_play()
    state = result[0]
    if move == "P":
        msg = "I'll pass."
    else:
        l_coord = coordinates["x"][move[1]]
        n_coord = coordinates["y"][move[0]]
        msg = "I'll play " + l_coord.upper() + " " + str(n_coord)
    img_link = state.paint()
    if state.winner() != 0:
        return end_game() \
            .standard_card(title="GoBuddy Game Finished",
                           small_image_url=img_link,
                           large_image_url=img_link)

    return question(msg) \
        .standard_card(title="GoBuddy Computer says...",
                       text=msg,
                       small_image_url=img_link,
                       large_image_url=img_link)


@ask.intent("MoveIntent")
def move(l_coord, n_coord):
    """
    Player moves, and computer moves in response (if applicable). Displays board in app.

    Parameters
    ----------
    l_coord: String
        lettter corresponding to the column value of move
    n_coord: String
        numerical string corresponding to the reverse of the row value of move

    Returns
    -------
    question/statement and standard card
    """
    global monte
    global state
    r = coordinates["y"].tolist().index(int(n_coord))
    c = list(coordinates["x"]).index(l_coord)
    result = (r, c)
    result = monte.update(result)
    if result is False:
        return statement("That move was invalid. Please try again.")
    state = result.content
    if state.winner() != 0:
        img_link = state.paint()
        return end_game() \
            .standard_card(title="GoBuddy Game Finished",
                           small_image_url=img_link,
                           large_image_url=img_link)
    result, move = monte.get_play()
    state = result.content
    if move == "P":
        msg = "I'll pass."
    else:
        l_coord = coordinates["x"][move[1]]
        n_coord = coordinates["y"][move[0]]
        msg = "I'll play " + l_coord.upper() + " " + str(n_coord)
    img_link = state.paint()
    if state.winner() != 0:
        return end_game() \
            .standard_card(title="GoBuddy Game Finished",
                           small_image_url=img_link,
                           large_image_url=img_link)
    return question(msg) \
        .standard_card(title="GoBuddy Computer says...",
                       text=msg,
                       small_image_url=img_link,
                       large_image_url=img_link)


def end_game():
    """
    Upon game ending, determine winner and return appropriate response.

    Returns
    -------
    statement
    """
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
    state = None
    return statement(msg)


if __name__ == "__main__":
    with open("mc_5x5.txt", "rb") as f:
        monte = pickle.load(f)
    print("Tree loaded.")
    assert isinstance(monte, MonteCarlo)
    app.run()