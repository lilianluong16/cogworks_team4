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

print("started!")

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
    if game == None:
        global game_state
        game = "Hangman"
        game_state = hangman.Hangman()
        print("made game")
        return question(game_state.start_hint())
    return question("I misheard you. Please try again")

@ask.intent("LetterIntent")
def get_letter(letter):
    global game
    print(game)
    print("you said " + letter)
    if game == "Hangman":
        letter = letter[0]
        print(letter)
        if letter.lower() not in "abcdefghijklmnopqrstuvwxyz":
            return question(letter + " is not a letter. Try again.")
        m = game_state.guess_letter(letter.lower())
        print(m)
        if m[-1] == "!":
            game = None
            return question(m + "What other game would you like to play?")
        else:
            return question(m)
    elif game == "Word Association":
        msg = game_state.take_turn(letter)
        print("I say " + msg)
        if len(letter) == 1:
            if letter == u:
                letter = "you"
            else:
                return(question(letter + "isn't a word. Try again. The word was " + word_association.word))
        if msg[-1] == '!':
            game = None
            return question(msg + "What other game would you like to play?")
        else:
            return question(msg)
    elif game == "Ghost":
        letter = letter[0]
        msg = game_state.take_turn(letter)
        if msg[-1] == "!":
            game = None
            return question(msg + "What other game would you like to play?")
        else:
            return question(msg)
    else:
        if letter == "nothing":
            print("stopping")
            return statement("Okay. Thanks anyway!")
        print(game)
        return question("I'm sorry; I'm not sure what game we're playing.")
    return start_skill()

@ask.intent("WordAssocIntent")
def start_word_assoc(seed, level):
    global game
    if game == None:
        global game_state
        global glove
        if level is None:
            level = 1
        game = "Word Association"
        game_state = word_association.Word_Association(seed=seed, level=level)
        print("made game")
        return question(game_state.start())
    return question("I misheard you. Please try again")

@ask.intent("GhostIntent")
def start_ghost():
    global game
    if game == None:
        global game_state
        game = "Ghost"
        game_state = ghost.ghost()
        print("made game")
        return question("Let's play Ghost! The first letter is " + game_state.first_turn())
    return question("I misheard you. Please try again")

@ask.intent("NoIntent")
def stop():
    print("stopping")
    return statement("Okay. Thanks anyway!")

if __name__ == '__main__':
    app.run(debug=True)