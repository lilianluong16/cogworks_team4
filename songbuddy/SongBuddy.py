from flask import Flask
from flask_ask import Ask, statement, question
import requests
from Song_FP import *

@app.route('/')
def homepage():
    return "Launching analogies"

@ask.launch
def start_skill():
    msg = "What analogy would you like me to solve?"
    return question(msg)

def analogythis(thatone, thistwo, thattwo, k=3):
    """
    This is to that as thisone is to ?
    Returns:
    --------
    thatone: list(str)
    """
    query = glove.wv[thistwo] - glove.wv[thattwo] + glove.wv[thatone]
    thisone = np.array(glove.wv.similar_by_vector(query))[:k,0].tolist()
    print(thatone)
    return thisone

@ask.intent("ThisIntent")
def solvethis(thatone, thistwo, thattwo):
    thisone = ", or ".join(analogythis(thatone, thistwo, thattwo))
    answer_msg = "{}, is to {}, as {}, is to, {}".format(thisone, thatone, thistwo, thattwo)
    return statement(answer_msg)

def analogythat(thisone, thistwo, thattwo, k=3):
    """
    Returns:
    --------
    thatone: list(str)
    """
    query = glove.wv[thisone] + glove.wv[thattwo] - glove.wv[thistwo]
    thatone = np.array(glove.wv.similar_by_vector(query))[:k,0].tolist()
    print(thatone)
    return thatone

@ask.intent("ThatIntent")
def solvethis(thisone, thistwo, thattwo):
    thatone = ", or ".join(analogythat(thisone, thistwo, thattwo))
    answer_msg = "{}, is to {}, as {}, is to, {}".format(thisone, thatone, thistwo, thattwo)
    return statement(answer_msg)

if __name__ == '__main__':
    app.run(debug=True)
