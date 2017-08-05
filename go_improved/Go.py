import numpy as np
from scipy.ndimage import label
from scipy.ndimage.morphology import generate_binary_structure, binary_dilation
import matplotlib.pyplot as plt
import cloudinary
import cloudinary.uploader
import cloudinary.api

cloudinary.config(
        cloud_name="luong44976",
        api_key="165891819185365",
        api_secret="p2ib0QA6Rl2nK8CNxlBFQeJmoaM"
    )


class GameState:
    """
    GameState represents a single stage of the game, storing the board positions, capture count, moves played, passes,
    previous move and state (for reference and Ko), and board size.
    """
    def __init__(self, size=5, board=None, captures=[0,0], moves_played=0, passes=0, move=None, prev_state=None, komi=0):
        """
        Initializes GameState.

        Parameters
        ----------
        size: int (for one dimension)
        board: np.ndarray((size, size))
        captures: list length 2
        moves_played: int
        passes: int
        move: 'P' for pass, tuple(row, col) for move, or None
        prev_state: np.ndarray((size, size)) previous board positions
        komi: int
        """
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
        return str(self.board)

    def __str__(self):
        return str(self.board)

    def paint(self, save=True):
        """
        Prints out view of board, creates view using pyplot, and saves to cloudinary.

        Parameters
        ----------
        save: Boolean (default=True)
            Save and return image to cloudinary, for display

        Returns
        -------
        String: url of uploaded image
        """
        print(self)
        print("")
        if save:
            a = self.board[::-1, :]
            ones = np.where(a == 1)[::-1]
            twos = np.where(a == 2)[::-1]
            fig, ax = plt.subplots(figsize=(self.size, self.size))
            ax.set_xticklabels(np.array(list(" ABCDEFGHI")[:self.size + 1]))
            ax.set_yticklabels(np.array(list(" 123456789")[:self.size + 1]))
            ax.vlines(np.arange(1, self.size + 1), 0, self.size + 1)
            ax.hlines(np.arange(1, self.size + 1), 0, self.size + 1)
            ax.scatter(ones[0] + 1, ones[1] + 1, 1400, 'black', zorder=10)
            ax.scatter(twos[0] + 1, twos[1] + 1, 1400, 'white', edgecolors="black", zorder=10)
            ax.set_xlim(0.3, self.size + 0.7)
            ax.set_ylim(0.3, self.size + 0.7)
            fig.savefig("img.png")
            return cloudinary.uploader.upload('img.png')['secure_url']

    def get_player(self):
        """
        Returns number of player to move.

        Returns
        -------
        int (1 or 2)
        """
        return self.moves_played % 2 + 1

    def get_groups(self, player=None):
        """
        Labels individual groups of stones, according to player.

        Parameters
        ----------
        player: int (optional)
            the player for which the groups should be labeled

        Returns
        -------
        if player is None:
            list of tuples
        tuple(np.ndarray((self.size, self.size)), int)
            array has groups labeled 1, 2, ... etc.
            int is count of how many groups there are
        """
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
        """
        Splits a labeled group array into individual array masks displaying only the group and nothing else.

        Parameters
        ----------
        groups: tuple(np.ndarray((self.size, self.size)), int)
            The labeled groups tuple returned by get_groups()

        Returns
        -------
        List of arrays length groups[1]
        """
        l = []
        for i in range(1, groups[1] + 1):
            b = np.zeros((self.size, self.size))
            b[np.where(groups[0] == i)] = 1
            l.append(b)
        return l

    def find_perimeter(self, b):
        """
        Uses a binary structure to determine the perimeter cells of a group, discounting walls.

        Parameters
        ----------
        b: np.ndarray((self.size, self.size))
            array showing cells to calculate perimeter for.

        Returns
        -------
        np.ndarray((self.size, self.size))
            array with 1s at unoccupied cells neighboring occupied cells
        """
        fp = generate_binary_structure(2, 1)
        mask = binary_dilation(b, fp).astype(b.dtype)
        perimeter = mask - b
        return perimeter

    def count_group_liberties(self, group):
        """
        Counts the number of liberties a group has.

        Parameters
        ----------
        group: np.ndarray((self.size, self.size))
            array showing single group for which to count liberties

        Returns
        -------
        int
        """
        perimeter = self.find_perimeter(group)
        filled = perimeter * self.board
        liberties = np.sum(perimeter) - np.count_nonzero(filled)
        return int(liberties)

    def total_liberties(self, player):
        """
        Totals the number of liberties of a certain class of stones.

        Parameters
        ----------
        player: int
            1 or 2 depending on player for which total liberties should be computed

        Returns
        -------
        int
        """
        gl = self.get_group_list(self.get_groups(player))
        total = 0
        for g in gl:
            total += self.count_group_liberties(g)
        return total

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
        """
        Evaluates a state's board to execute captures in an ordered manner.

        Parameters
        ----------
        num: int
            order number to evaluate for (0 for self, 1 for opponent)

        Returns
        -------
        self
        """
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
        s[0] += np.count_nonzero(self.board == 1)
        s[1] += self.komi + np.count_nonzero(self.board == 2)
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
        """
        Computes winner of game.

        Returns
        -------
        tuple(winner, list([score_one, score_two]))
            winner: -1, if tie; number of winning player, if otherwise
        """
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
        GameState of new move, weight (int)
        False, if move is invalid
        """
        if len(move) == 1:
            assert move == "P"
            new_board = GameState(size=self.size, board=np.copy(self.board), captures=np.copy(self.captures),
                                  moves_played=self.moves_played + 1, passes=self.passes + 1, prev_state=self,
                                  komi=self.komi)
            return new_board.evaluate(), 0
        if self.board[move] != 0:
            # Cell already occupied
            # print("cell occupied")
            return False, None
        new_board = GameState(size=self.size, board=np.copy(self.board), captures=np.copy(self.captures),
                              moves_played=self.moves_played + 1, passes=0, prev_state=self, komi=self.komi)
        new_board.board[move] = self.get_player()
        new_board.evaluate(0)
        if new_board.check_suicide(move):
            # Move is suicidal
            # print("suicidal")
            return False, None
        new_board.evaluate(1)
        if self.prev_state is not None:
            matches = len(np.where(self.prev_state.board == new_board.board)[0])
            if matches == self.size**2:
                # Rule of Ko
                # print("Ko")
                return False, None
        my_lib_diff = self.total_liberties(self.get_player()) - new_board.total_liberties(self.get_player())
        their_lib_diff = self.total_liberties(new_board.get_player()) - new_board.total_liberties(new_board.get_player())
        capture_difference = np.array(new_board.captures) - np.array(self.captures)
        weight = capture_difference[self.get_player() - 1] - capture_difference[self.get_player() % 2]
        weight -= their_lib_diff - my_lib_diff
        return new_board, weight

    def gen_moves(self):
        """
        Generates list of valid moves from state.

        Returns
        -------
        List of tuple(GameState, Tuple[row, col], int)
            Resultant GameState of move
            tuple representing move
            weight of move
        """
        pass_move, pass_weight = self.make_move("P")
        moves = [(pass_move, "P", pass_weight)]
        for r in range(self.size):
            for c in range(self.size):
                result, weight = self.make_move((r, c))
                if result is not False:
                    moves.append((result, (r, c), weight))
        return moves

    def valid_moves(self):
        """
        Returns list of valid moves.

        Returns
        -------
        List of tuple(row, col)
        """
        return [m[1] for m in self.gen_moves()]

    def winner(self):
        """
        Returns winner at current state of game.

        Returns
        -------
        int: number of player, or 0 if game not finished
        """
        if self.passes > 1:
            return self.game_end()[0]
        return 0


def two_player_game(size=5):
    """
    Starts a human two-player game ran through Python console.

    <<< MAY NOT BE FULLY FUNCTIONAL >>>

    Parameters
    ----------
    size: int (one dimension of board)

    Returns
    -------
    int: winner
    """
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
