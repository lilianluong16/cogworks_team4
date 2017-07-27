from flask import Flask
from flask_ask import Ask, question, statement

app = Flask(__name__)
ask = Ask(app, '/')


@app.route('/')
def homepage():
    return "NewsBuddy"


@ask.launch
def start_skill():
    welcome_msg = "Hi! I'm News Buddy. What would you like me to tell you?"
    return question(welcome_msg)

