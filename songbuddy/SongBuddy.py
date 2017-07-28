from flask import Flask
from flask_ask import Ask, statement, question
import requests
from Song_FP import *

app = Flask(__name__)
ask = Ask(app, '/')

initialize()
global database
database = retrieve_database()

@app.route('/')
def homepage():
    return "Launching song buddy"

@ask.launch
def start_skill():
    print("Starting song buddy")
    msg = "Please play your song."
    return question(msg)

@ask.intent("YesIntent")
def identify_song():
    print("Starting identify")
    global identitfy
    result = identify()
    print("Got result")
    if result is not None:
        msg = "This is {}".format(result)
        return statement(msg)
    msg = "I didn't find anything. Do you want to try again?"
    return question(msg)

@ask.intent("NoIntent")
def goodbye():
    return statement("Okay. Have a good day!")

if __name__ == '__main__':
    app.run(debug=True)
