import time
import numpy as np
import random
import pickle
import sys

import Go


class MonteCarlo:
    """
    MonteCarlo class is used to represent a game tree built through playouts and simulations, and provides the basis
    for AI selection of moves.
    """
    def __init__(self, root, initial_time=120, calc_time=7, max_moves=1500, c=1.4, w=0.1, cnn=1):
        """
        Initialization of MonteCarlo class.

        Parameters
        ----------
        root: Node
            node representing initial empty GameState of game
        initial_time: int
            seconds to run initial simulations for
        calc_time: int
            seconds of playouts per turn
        max_moves: int
            maximum moves allowed per playout
        c: float
            constant coefficient used for UCT calculation
        """
        self.root = root
        self.calc_time = calc_time
        self.max_moves = max_moves
        self.c = c
        self.w = w
        self.cnn = cnn
        self.history = [root]
        self.wins = {}
        self.plays = {}
        self.initial_time = initial_time
        self.initialize(initial_time)

    def update(self, game_move):
        """
        Checks to ensure that a move is legal, and if so enacts it by adding the associated node to the game history.

        Parameters
        ----------
        game_move: 'P' for pass, or tuple(row, col) for move

        Returns
        -------
        False, if invalid move
        Node, if valid
        """
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
        """
        Sums all plays to return total number of plays, used for UCT calculation.

        Returns
        -------
        int
        """
        return sum(self.plays.values())

    def win_prob(self, this_state, state):
        """
        Returns the probability of winning given a node.

        Parameters
        ----------
        this_state: Node
            node representing current state before move is created
        state: Node
            node representing state for which win probability should be calculated

        Returns
        -------
        float
        """
        if state in self.plays:
            return self.wins.get(state, 0) / self.plays[state] + self.w * this_state.weights[this_state.children.index(state)]
        return self.w * this_state.weights[this_state.children.index(state)]

    def uct(self, this_state, state):
        """
        Calculates the UCT (Upper Confidence bound applied to Trees) of a specific state, defined as:

            UCT = wins/plays + c * sqrt(ln total_plays / plays)

        and adds weight * w.

        Parameters
        ----------
        this_state: Node
            node representing current state before move is created
        state: Node
            node representing state for which UCT should be calculated

        Returns
        -------
        float
        """
        if state in self.plays:
            exploitation = self.wins.get(state, 0) / self.plays[state]
        else:
            exploitation = 0
        exploration = self.c * np.sqrt(np.log(self.total_plays()) / self.plays.get(state, 1))
        return exploitation + exploration + self.w * this_state.weights[this_state.children.index(state)]

    def initialize(self, t=None):
        """
        Performs initial playouts and simulations to provide initial tree and values.

        Parameters
        ----------
        t: int
            seconds for which to run searches

        Returns
        -------
        True
        """
        if t is None:
            t = self.initial_time

        t0 = time.time()
        sims = 0
        while time.time() - t0 < t:
            self.search()
            sims += 1

        print("Total searches:", sims)
        return True

    def get_play(self, t=None):
        """
        Runs playouts for t or self.calc_time seconds, before selecting and playing move with highest win probability.

        Parameters
        ----------
        t: int
            seconds of playouts to run

        Returns
        -------
        tuple(Node, tuple(row, col))
            node for new move
            new move
        """
        if t is None:
            t = self.calc_time
        state = self.history[-1]

        t0 = time.time()
        sims = 0
        while time.time() - t0 < t:
            self.search()
            sims += 1

        print("Total searches:", sims)
        new_move = max(state.children, key=lambda x: self.win_prob(state, x))
        print(self.w * state.weights[state.children.index(new_move)])
        move = state.moves[state.children.index(new_move)]
        print("Computer: I'll play ", move)
        self.history.append(new_move)
        return new_move, move

    def search(self):
        """
        Performs a playout/simulation by continuously picking moves until a result is achieved, and updating values
        through backpropagation.

        Returns
        -------
        None
        """
        temp_hist = self.history[:]
        current = temp_hist[-1]
        player = current.content.get_player()

        for i in range(self.max_moves):
            if current.content.winner():
                break
            if len(current.children) > 0:
                # Selection
                current = max(current.children, key=lambda x: self.uct(current, x))
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
        """
        Returns winner at current state.

        Returns
        -------
        int: number of winner, or 0 if game unfinished
        """
        state = self.history[-1]
        return state.content.winner()


class Node:
    """
    Node class is used to build the Monte Carlo tree, and contains a GameState as well as a list of children and
    valid moves for easy access.
    """
    def __init__(self, content):
        """
        Initializes Node class.

        Parameters
        ----------
        content: GameState
        """
        self.content = content
        self.children = []
        self.moves = []
        self.weights = []

    def get_children(self):
        """
        Retrieves children and valid moves from GameState's gen_moves and adds them to class attributes.

        Returns
        -------
        None
        """
        results = self.content.gen_moves()
        for state, move, weight in results:
            self.children.append(Node(state))
            self.moves.append(move)
            self.weights.append(weight)


def create_game(init_time=120, filepath="mc.txt"):
    """
    Creates Monte Carlo and plays through for init_time seconds. Saves to txt file.

    Parameters
    ----------
    init_time: int
        seconds for which initialization will occur
    filepath: String
        path to txt file

    Returns
    -------
    None
    """
    root = Node(Go.GameState())
    monte = MonteCarlo(root, initial_time=init_time)
    with open(filepath, "wb") as f:
            pickle.dump(monte, f)


def start_game(reset=False, size=5, init_time=240):
    """
    Starts single player game, with computer going first, using python console.

    <<< USED FOR DEBUGGING PURPOSES, MAY NOT BE FULLY FUNCTIONAL >>>

    Parameters
    ----------
    reset: Boolean
        True: Create and train new Monte Carlo
        False: Load Monte Carlo from txt
    size: int
        size of board
    init_time: int
        seconds for which initialization will occur

    Returns
    -------
    int: winner
    """
    filepath = "mc_" + str(size) + "x" + str(size) + ".txt"
    sys.setrecursionlimit(6000)
    if reset:
        root = Node(Go.GameState(size))
        monte = MonteCarlo(root, initial_time=init_time)
        with open(filepath, "wb") as f:
            pickle.dump(monte, f)
    else:
        t0 = time.time()
        with open(filepath, "rb") as f:
            monte = pickle.load(f)
        print(time.time() - t0)
    print("Computer: I'll start.")
    root = monte.root
    root.content.captures = [0, 0]
    state = root.content
    while state.winner() == 0:
        state = monte.get_play()[0].content
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

# start_game(reset=True, size=5, init_time=300)
