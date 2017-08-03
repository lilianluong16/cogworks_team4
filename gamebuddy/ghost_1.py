import time, random, itertools
import numpy as np
from nltk.corpus import words
from collections import defaultdict, Counter
from nltk import word_tokenize


class ghost:
    
    
    def __init__(self):
        

        
        self.traintext = ""
        self.lm3 = {}
        self.lm5 = {}
        self.lm7 = {}
        
        self.english_words = ""
        
        with open("english_words.txt") as word_file:
            self.english_words = set(word.strip().lower() for word in word_file)
        
        with open("english_words.txt" , "r") as f:
            self.traintext  = f.read()
            
            
        self.lm3 = self.train_lm(3)
        
        self.lm5 = self.train_lm(5)
        
        self.lm7 = self.train_lm(7)
        self.lm = self.lm3

        self.string = ''
        
    def is_english_word(self, word):
        return word.lower() in self.english_words
        
        
    def unzip(self,pairs):

        return tuple(zip(*pairs))
    
    
    
    
    def normalize(self, counter):

        total = sum(counter.values())
        return [(char, cnt/total) for char, cnt in counter.most_common()]
    
    
    def train_lm(self, n):
   
        li = word_tokenize(self.traintext)

    
    
        text = ' '.join(list(itertools.filterfalse(lambda x: len(x) % 2 ==0, li)))
    

    
        raw_lm = defaultdict(Counter)
        history = "~" * (n - 1)
    
        for x in text:
            raw_lm[history][x] += 1
            history = history[1:] + x
    

        the_list = list(raw_lm.values())
        key_list = list(raw_lm.keys())
    
 
    
        for i in range(len(the_list)):
        
            if the_list[i].get(' ') is not None:
                del the_list[i][' ']

        final_lm = {}
        for key, val in zip(key_list,the_list):
            final_lm[key] = val
        

        lm = { history : self.normalize(counter) for history, counter in final_lm.items() }
    
        return lm
        
        
    def generate_letter(self, lm, history):

        if not history in lm:
            return "~"
        letters, probs = self.unzip(lm[history])
        i = np.random.choice(letters, p=probs)
        print("generated " + i)
        return i
    
    
    def first_turn(self):
        first_letter = random.choice('abcdefghijklmnopqrstuv')
        self.string += first_letter
        return first_letter

    def take_turn(self, letter):
        self.string += letter

        if self.is_english_word(self.string) and len(self.string) > 3:
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

        h_length = -2
        if len(self.string) >= 5:
            h_length = -4
            self.lm = self.lm5
        if len(self.string) >= 7:
            h_length = -6
            self.lm = self.lm7
        
        new_letter = self.generate_letter(self.lm,self.string[h_length:])

        self.string += new_letter

        if new_letter == "~":
            print("got ", new_letter)
            return("That doesn't spell a word. You lost!")

        if self.is_english_word(self.string) and len(self.string) > 3:
            return "Oops I spelt the word " + self.string + "! I lost!"
        print("took turn")
        print(self.string)
        return str(list(self.string))

    def play(self):
        
        print("in play")
        first_letter = random.choice('abcdefghijklmnopqrstuv')
        
        print("I will go first")
        
        print(first_letter)
        
        second_letter = input("Please enter the next letter: ")
        
        string = first_letter + second_letter
        
        print(string)
        history = string
        
        for i in range(12):
            
            if i == 0:
                
                letter = self.generate_letter(self.lm3,history)
                
                string += letter
                
                print(string)
                
                if self.is_english_word(string) and len(string) > 3:
                    print("Oops! I created the word ", string, ". I lose!")
                    break
                    
            if i % 2 == 1:
                
                letter = input("Please enter the next letter: ")
                
                string += letter
                
                print(string)
                
                if self.is_english_word(string) and len(string) > 3:
                    print("I'm sorry, but you created the word ", string,". You lose!")
                    break
                    
                    
            if i >= 2 and i % 2 == 0 and i < 4:
                letter = self.generate_letter(self.lm5,string[-4:])
                
                string += letter
                
                print(string)
                
                if self.is_english_word(string) and len(string) > 3:
                    print("Oops! I created the word ", string, ". I lose!")
                    break
                    
            if i >= 4 and i % 2 == 0:
                letter = self.generate_letter(self.lm7,string[-6:])
                
                string += letter
                
                print(string)
                
                if self.is_english_word(string) and len(string) > 3:
                    print("Oops! I created the word ", string, ". I lose!")
                    break
                    
game = ghost()
msg = game.first_turn()
print(msg)
while not msg[-1] == "!":
    letter = input("Guess a letter: ")
    msg  = game.take_turn(letter)
    print(msg)