from flask import Flask
from flask_ask import Ask, question, statement
import hangman
import word_association

app = Flask(__name__)
ask = Ask(app, '/')

game = None
game_state = None


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

@ask.intent("WordAssocIntent"):
def start_word_assoc():
    game = "Word Association"
    game_state = word_association.Word_Association()
    return question(game_state.start())

@ask.intent("WordIntent")
def take_turn

if __name__ == '__main__':
    app.run()