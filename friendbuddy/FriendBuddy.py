from flask import Flask
from flask_ask import Ask, question, statement
import profile
import numpy as np

app = Flask(__name__)
ask = Ask(app, '/')

# Global variables
name = None
state = None
temp = None


@app.route('/')
def homepage():
    return "FriendBuddy"


@ask.launch
def start_skill():
    """
    Upon starting the skill, a face is searched for using the camera and the global name variable is assigned.

    Returns
    -------
    question or statement
    """
    global name
    global temp
    global state
    prof = profile.find_profile()
    state = None
    if prof is False:
        # No face detected
        return statement("Nobody found. Please be ready to take a photo.")
    if isinstance(prof, int):
        # Many unknown faces detected
        return statement("Many unknown faces found. Please make sure you are alone.")
    if isinstance(prof, np.ndarray):
        # Single unknown face detected
        state = "Add profile"
        temp = prof
        return question("I don't know who you are. Do you want to be added to my database?")
    if isinstance(prof, list):
        # Multiple known faces detected
        state = "Choose profile"
        temp = prof
        # TODO: FIX UNSPEAKABLE NAMES AND CASES
        return question("I see " + ", ".join(prof[:-1]) + ", and " + prof[-1] + ". Who am I talking to?")
    name = prof
    return question("Welcome " + name + "! What would you like to do?")


@ask.intent("QueryPropertyIntent")
def query_property(prop):
    """
    Retrieves a property from the profile.

    Parameters
    ----------
    prop: String
        name of property

    Returns
    -------
    question
    """
    global temp
    global state
    result = profile.get_property(name, prop)
    if result is None:
        state = "Add value"
        temp = prop
        return question("I don't know your " + prop + ". What is it?")
    return question("Your " + prop + " is " + result + ". Anything else?")


@ask.intent("QueryListIntent")
def query_list(list_name):
    """
    Retrieves list from profile, and reads out formatted.

    Parameters
    ----------
    list_name: String
        name of list

    Returns
    -------
    string
    """
    result = profile.tell_list(name, list_name)
    return question(result + " Anything else?")


@ask.intent("PropertyIntent")
def define_property(value):
    """
    Responses to follow up on an asked question using an arbitrary string literal.

    Parameters
    ----------
    value: String
        the value/answer

    Returns
    -------
    question
    """
    global name
    global temp
    global state
    if state == "Choose profile":
        # If asked for a name when multiple faces are detected
        if value in temp:
            name = value
            return question("Welcome " + name + "! What would you like to do?")
        return question("Please pick one of the following names: " + ", ".join(temp))
    if state == "Add profile":
        # If asked for a name when an image is being added to the Face_Rec database
        name = profile.add_person(value, temp)
        profile.write_database()
        state = None
        temp = None
        return question("Welcome " + name + "! What would you like to do?")
    if state == "Inactive":
        # If called prior to starting the skill
        start_skill()
    if state == "Add value":
        # Answering in response to a value found to be None.
        profile.change_property(name, temp, value)
        profile.write_database()
        p = temp
        state = None
        temp = None
        return question(p + " has been set to " + value + ". Anything else?")
    if state == "Add list":
        # When adding a list requests the next object.
        profile.add_to_list(name, temp, [value])
        profile.write_database()
        return question("Added " + value + ". Anything else?")
    return question("I'm sorry, I don't understand. Please try again.")


@ask.intent("DeclarePropertyIntent")
def declare_property(prop, value):
    """
    Sets a property to a value directly.

    Parameters
    ----------
    prop: String
        name of property
    value: String
        the value that prop should be set to

    Returns
    -------
    question

    """
    profile.change_property(name, prop, value)
    profile.write_database()
    return question(prop + " has been set to " + value + ". Anything else?")


@ask.intent("AddListIntent")
def add_list(list_name):
    """
    Adds a list or items to one, if the list exists.

    Parameters
    ----------
    list_name: String
        name of list

    Returns
    -------
    question

    """
    global temp
    global state
    if list_name in profile.database[name] and not isinstance(profile.database[name][list_name], list):
        return question(list_name + " is not a list! Say delete " + list_name + " to reset it. Anything else?")
    temp = list_name
    state = "Add list"
    return question("Please tell me the first thing you want to add.")


@ask.intent("YesIntent")
def yes():
    """
    YesIntent triggers only if state == "Add profile"
    """
    global state
    if state == "Add profile":
        return question("What is your name?")


@ask.intent("NoIntent")
def no():
    """
    NoIntent triggers to reset global variables, close skill or cancel other function.

    Returns
    -------
    question (if continuing)
    statement (if not)
    """
    global temp
    global state
    if state is not None:
        temp = None
        state = None
        return question("Okay. Is there anything else you want to do?")
    else:
        profile.write_database()
        return statement("Alright! See you some other time.")


@ask.intent("DeleteIntent")
def delete_var(var_name):
    """

    Deletes variable from profile's part of database, whether or not it is an array or not.
    Parameters
    ----------
    var_name: String
        name of variable

    Returns
    -------
    question
    """
    profile.delete_variable(name, var_name)
    profile.write_database()
    return question("Deleted. Anything else?")


if __name__ == '__main__':
    state = "Inactive"
    app.run()
