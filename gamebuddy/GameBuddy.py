from flask import Flask
from flask_ask import Ask, question, statement
import hangman
import word_association, ghost
import gensim
import time
from gensim.models.keyedvectors import KeyedVectors


app = Flask(__name__)
ask = Ask(app, '/')

game = None
game_state = None

t0 = time.time()
path = "glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)
t1 = time.time()
print("loaded word vectors in ", t1-t0)


@app.route('/')
def homepage():
    return "GameBuddy"


@ask.launch
def start_skill():
    global game
    global game_state
    game = None
    game_state = None
    return question("Hi. I'm Game Buddy. What would you like to play?")


@ask.intent("HangmanIntent")
def start_hangman():
    global game
    global game_state
    game = "Hangman"
    game_state = hangman.Hangman()
    return question(game_state.start_hint())


@ask.intent("LetterIntent")
def get_letter(letter):
    global game
    if game == "Hangman":
        if letter.lower() not in "abcdefghijklmnopqrstuvwxyz":
            return question(letter + " is not a letter. Try again.")
        m = game_state.guess_letter(letter.lower())
        print(m)
        if m[-1] == "!":
            game = None
            return statement(m)
        else:
            return question(m)
    return start_skill()

@ask.intent("WordAssocIntent")
def start_word_assoc(seed, level):
    global game
    global game_state
    global glove
    if level is None:
        level = 1
    game = "Word Association"
    game_state = word_association.Word_Association(seed=seed, level=level)
    return question(game_state.start())

@ask.intent("WordIntent")
def take_turn(word):
    global game
    if game == "Word Association":
        msg = game_state.take_turn(word)
        if msg[-1] == '!':
            return statement(msg)
        else:
            return question(msg)
    else:
        return("I'm sorry, what game are we playing?")

@ask.intent("GhostIntent")
def start_ghost():
    game = "Ghost"
    game_state = ghost.ghost()
    
    game_state.play()
    return question(game_state.start())

if __name__ == '__main__':
    app.run()