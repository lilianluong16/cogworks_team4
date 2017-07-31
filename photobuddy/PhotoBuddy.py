from flask import Flask, render_template
from flask_ask import Ask, question, statement
import Face_Rec
import numpy as np

app = Flask(__name__)
ask = Ask(app, '/')

add_flag = 0
descs = None
name = None


@app.route('/')
def homepage():
    return render_template("index.html")
# TODO: ADD OPTION FOR SAVING PHOTO IN DATABASE


@ask.launch
def start_skill():
    """
    Triggers when skill is called.
    :return:
    """
    add_flag = 0
    welcome_msg = "Hi! I'm Photo Buddy. Are you ready to take a photo?"
    return question(welcome_msg)


@ask.intent('YesIntent')
def photo():
    """
    Triggers when something synonymous to 'yes' is spoken.
    :return: Statement or question
    """
    if add_flag == 0:
        """
        If used at start of program to take photo.
        """
        global descs
        names, img, ul, descs = Face_Rec.go() # Take photo
        with open("test.txt", "w") as f:
            f.write(ul)

        names = np.array(names)
        nones = len(names) - np.count_nonzero(names) # Number of unidentifiable faces
        identified_names = names[names != np.array(None)] # Filters Nones out of list

        # Responses
        if len(names) < 1:
            return statement("I do not see anyone.")
        if len(identified_names) < 1 or identified_names[0] is None:
            if nones > 1:
                return statement("I see " + str(nones) + " unknown faces.")
            return statement("I see one unknown face.")
        if len(identified_names) == 1:
            if nones > 0:
                if nones > 1:
                    msg = identified_names[0] + " and " + str(nones) + "unknown faces."
                else:
                    msg = identified_names[0] + " and one unknown face."
                return statement("I see " + msg) \
                    .standard_card(title="I see...",
                                   text=msg,
                                   small_image_url=ul,
                                   large_image_url=ul)
            else:
                msg = identified_names[0]
                add_flag = 1
                global name
                name = msg
                print(ul)
                return question("I see " + msg + ". Would you like to add this to the database?") \
                    .standard_card(title="I see...",
                                   text=msg,
                                   small_image_url=ul,
                                   large_image_url=ul)
        if nones > 0:
            if nones > 1:
                msg = ", ".join(identified_names) + ", and " + str(nones) + "unknown faces."
            else:
                msg = ", ".join(identified_names) + ", and one unknown face."
        else:
            msg = ", ".join(identified_names[:-1]) + ", and " + identified_names[-1]
        return statement("I see " + msg) \
            .standard_card(title="I see...",
                           text=msg,
                           small_image_url=ul,
                           large_image_url=ul)
        # END RESPONSES
    else:
        """
        If 'yes' is said in response to being asked to be added to the database.
        """
        global add_flag
        add_flag = 0
        Face_Rec.add_image(descs, name=name)
        return statement("Photo added.")


@ask.intent("NoIntent")
def no():
    """
    Triggered when a word synonymous to "no" is said.
    Resets add_flag.
    :return:
    """
    global add_flag
    add_flag = 0
    return statement("Okay! Maybe next time.")


if __name__ == '__main__':
    add_flag = 0
    Face_Rec.initialize()
    app.run()