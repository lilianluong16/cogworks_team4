import time, random, itertools
import numpy as np
from nltk.corpus import words
from collections import defaultdict, Counter
from nltk import word_tokenize


class ghost:
    
    
    def __init__(self):
        """
        Creates an instance of the ghost class.

        Plays Ghost, a game in which the first player (the computer)
        says a letter and the other players build off of it towards a word
        back and forth

        The computer or player loses when they make a word OR
        say a letter that prevents any word from being formed
        """
        with open("english_words.txt") as word_file:
            self.english_words = set(word.strip().lower() for word in word_file)
        with open("english_words.txt" , "r") as f:
            self.traintext  = f.read()
        self.lm = self.train_lm()
        self.string = ''
    
    def normalize(self, counter):
        """
        Normalizes the given counter by dividing each count by the sum of all the counts

        Parameters:
        -----------
        counter: Counter
            the Counter to be normalized

        Returns:
        --------
        dict
            the normalized counter, which now countains frequencies instead of counts
        """
        total = sum(counter.values())
        return [(char, cnt/total) for char, cnt in counter.most_common()]
    
    
    def train_lm(self):
        """
        Trains the model on all the words 
        Rather than using an n-gram model, this uses all of the previous letters in a word as the history

        Returns:
        --------
        lm: dict (str -> Counter)
            the dictionary mapping all possible histories to the frequency counter of probable next letter choices
        """
        text = ' '.join(list(itertools.filterfalse(lambda x: len(x) % 2 ==0, word_tokenize(self.traintext))))
        word_list = text.split()   
        #print(word_list)
        raw_lm = defaultdict(Counter)
        #print(raw_lm)
        word_num = 0
        first_letter = True

        for letter in text:
            word = word_list[word_num]
            #print(word)
            if letter == " ":
                word_num += 1
                first_letter = True
                continue
            #print(raw_lm)
            if first_letter:
                history = "~" + letter
            #print(history," ==> ", letter)
            raw_lm[history][letter] += 1
            #print(raw_lm)
            if not first_letter:
                history += letter
            #print(history)
            first_letter = False
        lm = { history : self.normalize(counter) for history, counter in raw_lm.items() }
        return lm
        
        
    def generate_letter(self, history):
        """
        Generates the next letter based on the supplied history

        Parameters:
        -----------
        history: str
            the word up to this point, preceded by a tilde

        Return:
        -------
        i: str
            the next letter, chosen using the normalized counter to weight the probablity
        """
        if not history in self.lm:
            return "~"
        letters, probs = zip(*self.lm[history])
        i = np.random.choice(letters, p=probs)
        print("generated " + i)
        return i
    
    def first_turn(self):
        """
        Completes the first turn

        Return:
        -------
        first_letter: str
            the randomly selected first letter, barring w, x, y, z
        """
        first_letter = random.choice('abcdefghijklmnopqrstuv')
        self.string += first_letter
        return first_letter

    def take_turn(self, letter):
        """
        Completes any turn other than the first, based on all the previous turns, the model, and the user's last letter

        Return:
        str: list
            all the letters in the word separated by ". " or a message declaring the winner and why they won
        """
        self.string += letter

        if self.string in self.english_words and len(self.string) > 3:
            return "Oops you spelt the word " + self.string + "! You lost!"

        not_word = True
        t0 = time.time()
        for word in sorted(list(self.english_words)):
            if word.startswith(self.string) and len(word) > 3:
                print(word)
                not_word = False
        t1=time.time()
        print(t1-t0)

        if not_word:
            return("That doesn't spell a word. You lost!")
        
        new_letter = self.generate_letter("~" + self.string)

        self.string += new_letter

        if new_letter == "~":
            print("got ", new_letter)
            return("That doesn't spell a word. You lost!")

        if self.string in self.english_words and len(self.string) > 3:
            return "Oops I spelt the word " + self.string + "! I lost!"
        print("took turn")
        print(self.string)
        return ". ". join(list(self.string)) + "."

    def reset(self):
        """
        Resets the game in order to replay it
        """
        self.string = ""
                    
game = ghost()
msg = game.first_turn()
print(msg)
while not msg[-1] == "!":
    guess = input("Guess a letter: ")
    msg  = game.take_turn(guess)
    print(msg)