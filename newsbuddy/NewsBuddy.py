from flask import Flask
from flask_ask import Ask, question, statement

import main

app = Flask(__name__)
ask = Ask(app, '/')


@app.route('/')
def homepage():
    return "NewsBuddy"


@ask.launch
def start_skill():
    welcome_msg = "Hi! I'm News Buddy. What would you like me to tell you?"
    return question(welcome_msg)


@ask.intent("GetTopic")
def question_one(topic):
    results = main.q1(topic)
    if results is None:
        return statement("I couldn't find anything.")
    return statement("I found this: " + results)


@ask.intent("GetEntity")
def question_two(entity, k=3):
    main.clear_database()
    main.collect_articles()
    top_results = [i[0] for i in main.q2(entity, k=k)]
    results = ", ".join(top_results)
    msg = "Most associated with " + entity + ": " + results
    return statement(msg)


if __name__ == '__main__':
    app.run()