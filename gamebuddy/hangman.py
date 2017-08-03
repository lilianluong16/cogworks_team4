import pickle
import numpy as np


class Hangman:
    def __init__(self):
        with open("most_common_words.txt", "rb") as f:
            common = pickle.load(f)
        self.common = np.array(common)
        self.word = np.random.choice(self.common)
        self.guess = ["_"] * len(self.word)
        self.tries = ["head", "torso", "left arm", "right arm", "left leg", "right leg"]
        self.letters = list("abcdefghijklmnopqrstuvwxyz")

    def start_hint(self):
        msg = "The word has " + str(len(self.word)) + " letters. What do you guess?"
        return msg

    def tell_word(self):
        msg = "Here's what you've got so far: "
        for c in self.guess:
            if c == "_":
                msg += "blank, "
            else:
                msg += c + ", "
        return msg

    def guess_letter(self, letter):
        if letter in self.letters:
            self.letters.remove(letter)
            if letter in self.word:
                for i in range(len(self.word)):
                    if self.word[i] == letter:
                        self.guess[i] = letter
                if "_" in self.guess:
                    return self.tell_word()
                else:
                    return self.end(None)
            else:
                part = self.tries.pop(0)
                msg = "Oops! You've lost your " + part + ". "
                if len(self.tries) < 1:
                    return self.end(msg)
                msg += self.tell_word()
                return msg
        else:
            return "You've already guessed " + letter

    def end(self, msg):
        if msg is None:
            m = "Congratulations! The word was " + self.word + ". Let's play again later!"
        else:
            m = msg + "The word was " + self.word + ". Try again next time!"
        return m
