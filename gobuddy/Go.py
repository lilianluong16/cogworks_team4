import numpy as np
from scipy.ndimage import label
from scipy.ndimage.morphology import generate_binary_structure, binary_dilation


class GameState:
    def __init__(self, size=5, board=None, captures=[0,0], moves_played=0, passes=0, move=None, prev_state=None, komi=0):
        self.size = size
        if board is None:
            self.board = np.zeros((size, size))
        else:
            self.board = board
        self.captures = captures
        self.moves_played = moves_played
        self.passes = passes
        self.move = move
        self.prev_state = prev_state
        self.komi = 0

    def __repr__(self):
        return self.board, self.captures, self.moves_played

    def __str__(self):
        return str(self.board)

    def paint(self):
        print(self)
        print("")

    def get_player(self):
        return self.moves_played % 2 + 1

    def get_groups(self, player=None):
        if player is not None:
            b = np.zeros((self.size, self.size))
            b[np.where(self.board == player)] = 1
            labels = label(b)
            return labels
        labels = []
        b = np.zeros((self.size, self.size))
        b[np.where(self.board == 1)] = 1
        labels.append(label(b))
        b = np.zeros((self.size, self.size))
        b[np.where(self.board == 2)] = 1
        labels.append(label(b))
        return labels

    def get_group_list(self, groups):
        l = []
        for i in range(1, groups[1] + 1):
            b = np.zeros((self.size, self.size))
            b[np.where(groups[0] == i)] = 1
            l.append(b)
        return l

    def find_perimeter(self, b):
        fp = generate_binary_structure(2, 1)
        mask = binary_dilation(b, fp).astype(b.dtype)
        perimeter = mask - b
        return perimeter

    def count_group_liberties(self, group):
        perimeter = self.find_perimeter(group)
        filled = perimeter * self.board
        liberties = np.sum(perimeter) - np.count_nonzero(filled)
        return int(liberties)

    def check_suicide(self, move):
        """
        Checks to see if a move was suicide in the current board.

        Parameters
        ----------
        move: tuple(row, col)
            index of last stone placed

        Returns
        -------
        Boolean: move is suicidal
        None if move wasn't made
        """
        group_list = self.get_group_list(self.get_groups(self.get_player() % 2 + 1))
        for g in group_list:
            if g[move]:
                if self.count_group_liberties(g) < 1:
                    return True
                return False
        return None

    def evaluate(self, num=None):
        groups = self.get_groups()
        one_index = (self.get_player()) % 2
        gl = [None, None]
        gl[0] = self.get_group_list(groups[self.get_player() - 1])
        gl[1] = self.get_group_list(groups[one_index])
        if num is None:
            for j, i in enumerate(gl):
                for g in i:
                    # Start by checking self (evaluate() is ran on new board)
                    if self.count_group_liberties(g) < 1:
                        # Add captures
                        self.captures[(j + self.get_player()) % 2] += int(np.sum(g))
                        # Remove group
                        mask = np.nonzero(g)
                        self.board[mask] = 0
        else:
            for g in gl[num]:
                if self.count_group_liberties(g) < 1:
                    self.captures[(num + self.get_player()) % 2] += int(np.sum(g))
                    mask = np.nonzero(g)
                    self.board[mask] = 0
        return self

    def score(self):
        """
        Computes score; assuming that a final position with all dead stones captured
        has been reached.

        Returns
        -------
        list([score_one, score_two])
        """
        s = np.copy(self.captures)
        b = np.zeros((self.size, self.size))
        b[np.where(self.board == 0)] = 1
        gl = self.get_group_list(label(b))
        for g in gl:
            perimeter = self.find_perimeter(g)
            surrounding = perimeter * self.board
            if 1 in surrounding and 2 not in surrounding:
                s[0] += int(np.sum(g))
            elif 2 in surrounding and 1 not in surrounding:
                s[1] += int(np.sum(g))
        return s

    def game_end(self):
        scores = self.score()
        if scores[0] == scores[1]:
            return -1, scores
        else:
            return np.argmax(np.array(scores)) + 1, scores

    def make_move(self, move):
        """
        Given a move index, check to see if it is valid, and if so make the move.

        Parameters
        ----------
        move: tuple(row, col) OR string
            index of cell to move to OR 'P' for pass

        Returns
        -------
        GameState of new move
        False, if move is invalid
        """
        if len(move) == 1:
            assert move == "P"
            new_board = GameState(size=self.size, board=np.copy(self.board), captures=np.copy(self.captures),
                                  moves_played=self.moves_played + 1, passes=self.passes + 1, prev_state=self,
                                  komi=self.komi)
            return new_board.evaluate()
        if self.board[move] != 0:
            # Cell already occupied
            # print("cell occupied")
            return False
        new_board = GameState(size=self.size, board=np.copy(self.board), captures=np.copy(self.captures),
                              moves_played=self.moves_played + 1, passes=0, prev_state=self, komi=self.komi)
        new_board.board[move] = self.get_player()
        new_board.evaluate(0)
        if new_board.check_suicide(move):
            # Move is suicidal
            # print("suicidal")
            return False
        new_board.evaluate(1)
        if self.prev_state is not None:
            matches = len(np.where(self.prev_state.board == new_board.board)[0])
            if matches == self.size**2:
                # Rule of Ko
                # print("Ko")
                return False
        return new_board

    def gen_moves(self):
        moves = [(self.make_move("P"), "P")]
        for r in range(self.size):
            for c in range(self.size):
                result = self.make_move((r, c))
                if result is not False:
                    moves.append((result, (r, c)))
        return moves

    def valid_states(self):
        return list(zip(*self.gen_moves()))

    def valid_moves(self):
        return [m[1] for m in self.gen_moves()]

    def winner(self):
        if self.passes > 1:
            return self.game_end()[0]
        return 0


def two_player_game(size=5):
    state = GameState(size=size)
    state.paint()

    while isinstance(state, GameState):
        row = input("Enter row, or P to pass: ")
        if row.upper() == "P":
            result = state.make_move("P")
        else:
            if len(row) == 0:
                continue
            r = int(row)
            if r < 0 or r >= state.size:
                continue
            col = input("Enter column: ")
            if len(col) == 0:
                continue
            c = int(col)
            if c < 0 or c >= state.size:
                continue
            result = state.make_move((r, c))
        if result is not False:
            state = result
            if isinstance(state, GameState):
                state.paint()

    result, scores = state
    if result == -1:
        print("It was a tie!")
    else:
        print("Player " + str(result) + " won!")
    print("Score was " + str(scores[0]) + "-" + str(scores[1]) + ".")
    return result
