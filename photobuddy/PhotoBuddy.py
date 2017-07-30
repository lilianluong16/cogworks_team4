from flask import Flask, render_template
from flask_ask import Ask, question, statement
import Face_Rec
import numpy as np

app = Flask(__name__)
ask = Ask(app, '/')


@app.route('/')
def homepage():
    return render_template("index.html")
# TODO: ADD OPTION FOR SAVING PHOTO IN DATABASE


@ask.launch
def start_skill():
    welcome_msg = "Hi! I'm Photo Buddy. Are you ready to take a photo?"
    return question(welcome_msg)


@ask.intent('YesIntent')
def photo():
    names, img = Face_Rec.go()
    names = np.array(names)
    print(names)
    identified_names = names[names != np.array(None)]
    print(identified_names)
    if len(identified_names) < 1 or identified_names[0] is None:
        return statement("I do not recognize anyone.")

    if len(identified_names) == 1:
        return statement("I see " + identified_names[0] + ".")
    msg = "I see " + ", ".join(identified_names[:-1]) + ", and " + identified_names[-1]
    return statement(msg)


if __name__ == '__main__':
    Face_Rec.initialize()
    app.run()