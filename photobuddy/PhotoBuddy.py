from flask import Flask, render_template
from flask_ask import Ask, question, statement
import Face_Rec
import numpy as np
import cloudinary
import cloudinary.api

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
    names, img, ul = Face_Rec.go()
    names = np.array(names)
    print(names)
    identified_names = names[names != np.array(None)]
    print(identified_names)
    if len(identified_names) < 1 or identified_names[0] is None:
        return statement("I do not recognize anyone.")

    if len(identified_names) == 1:
        return statement("I see " + identified_names[0] + ".") \
            .standard_card(title="I see...",
                           text=identified_names[0],
                           small_image_url=ul,
                           large_image_url=ul)
    msg = ", ".join(identified_names[:-1]) + ", and " + identified_names[-1]
    return statement("I see " + msg) \
        .standard_card(title="I see...",
                       text=msg,
                       small_image_url="https://e96a953b.ngrok.io/static/img.png",
                       large_image_url="https://e96a953b.ngrok.io/static/img.png")


if __name__ == '__main__':
    Face_Rec.initialize()
    app.run()