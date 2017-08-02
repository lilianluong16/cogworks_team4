import time
import numpy as np
import random
import pickle

import Go


class MonteCarlo:
    def __init__(self, root, initial_time=120, calc_time=10, max_moves=1000, c=1.4):
        self.root = root
        self.calc_time = calc_time
        self.max_moves = max_moves
        self.c = 1.4
        self.history = [root]
        self.wins = {}
        self.plays = {}
        self.initial_time = initial_time
        self.initialize(initial_time)

    def update(self, game_move):
        current_state = self.history[-1]
        if len(current_state.children) < 1:
            current_state.get_children()

        if game_move not in current_state.moves:
            print(game_move)
            print(current_state.moves)
            return False
        new_state = current_state.children[current_state.moves.index(game_move)]
        self.history.append(new_state)
        return self.history[-1]

    def total_plays(self):
        return sum(self.plays.values())

    def win_prob(self, state):
        if state in self.plays:
            return self.wins.get(state, 0) / self.plays[state]
        return 0

    def uct(self, state):
        if state in self.plays:
            exploitation = self.wins.get(state, 0) / self.plays[state]
        else:
            exploitation = 0
        exploration = self.c * np.sqrt(np.log(self.total_plays()) / self.plays.get(state, 1))
        return exploitation + exploration

    def initialize(self, t=None):
        if t is None:
            t = self.initial_time
        state = self.history[-1]

        t0 = time.time()
        sims = 0
        while time.time() - t0 < t:
            self.search(state)
            sims += 1

        print("Total searches:", sims)
        return True

    def get_play(self, t=None):
        if t is None:
            t = self.calc_time
        state = self.history[-1]

        t0 = time.time()
        sims = 0
        while time.time() - t0 < t:
            self.search(state)
            sims += 1

        print("Total searches:", sims)
        new_move = max(state.children, key=lambda x: self.win_prob(x))
        move = state.moves[state.children.index(new_move)]
        print("Computer: I'll play ", move)
        self.history.append(new_move)
        return new_move

    def search(self, state):
        temp_hist = self.history[:]
        current = temp_hist[-1]
        player = current.content.get_player()

        for i in range(self.max_moves):
            if current.content.winner():
                break
            if len(current.children) > 0:
                # Selection
                current = max(current.children, key=lambda x: self.uct(x))
                temp_hist.append(current)
                continue
            # Expansion
            current.get_children()
            current = random.choice(current.children)

        if current.content.winner():
            # Backpropagation
            for item in temp_hist:
                if item not in self.plays:
                    self.plays[item] = 1
                else:
                    self.plays[item] += 1
                if current.content.winner() == player:
                    if item not in self.wins:
                        self.wins[item] = 1
                    else:
                        self.wins[item] += 1

    def winner(self):
        state = self.history[-1]
        return state.content.winner()


class Node:
    def __init__(self, content):
        self.content = content
        self.children = []
        self.moves = []

    def get_children(self):
        for state, move in self.content.gen_moves():
            self.children.append(Node(state))
            self.moves.append(move)


def start_game(reset=True):
    if reset:
        root = Node(Go.GameState())
        monte = MonteCarlo(root, 300)
        with open("mc_5x5.txt", "wb") as f:
            pickle.dump(monte, f)
    else:
        with open("mc_5x5.txt", "rb") as f:
            monte = pickle.load(f)
    print("Computer: I'll start.")
    root = monte.root
    root.content.captures = 0
    state = root.content
    while state.winner() == 0:
        state = monte.get_play().content
        state.paint()
        if state.winner() != 0:
            break
        result = False
        while result is False:
            r = ""
            c = ""
            while not (len(r) > 0 and (r.isdigit() or r.upper() == "P")):
                r = input("Enter row or P to pass: ")
            if r.upper() == "P":
                result = "P"
            else:
                while not (len(c) > 0 and c.isdigit()):
                    c = input("Enter column: ")
                result = (int(r), int(c))
            result = monte.update(result)
            if result is False:
                continue
            state = result.content
            state.paint()

    scores = state.score()
    winner = state.winner()
    if winner < 0:
        print("Tie reached.")
    elif winner == 1:
        print("Computer wins.")
    else:
        print("Player wins.")
    print("The score was " + str(scores[0]) + "-" + str(scores[1]) + ".")
    return winner


start_game()
