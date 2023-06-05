import numpy as np
import random

class TicTacToe:
    win_reward = 100
    lose_reward = -100

    # Creating the board
    def __init__(self):
        self.board = []
        self.q_values = np.zeros((3, 3, 9))
        #self.actions = list(range(1, 10))

    # append empty spots for each row-
    def create_board(self):
        self.board = [["-" for _ in range(3)] for _ in range(3)]

    # Randomly choosing which player goes first
    def get_random_first_player(self):
        return random.randint(0, 1)

    # choosing a spot
    def select_square(self, row, col, player):
        row = int(row) - 1
        col = int(col) - 1
        self.board[row][col] = player

    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    """def is_player_win(self, player):
        win_conditions = [
            [((0, 0), (0, 1), (0, 2))], # rows
            [((1, 0), (1, 1), (1, 2))],
            [((2, 0), (2, 1), (2, 2))],
            [((0, 0), (1, 0), (2, 0))], # columns
            [((0, 1), (1, 1), (2, 1))],
            [((0, 2), (1, 2), (2, 2))],
            [((0, 0), (1, 1), (2, 2))], # diagonals
            [((0, 2), (1, 1), (2, 0))]
        ]
        for condition in win_conditions:
            if all(self.board[r][c] == player for r, c in condition):
                return True
        return False"""

    def is_player_win(self, player):
        win = None

        n = len(self.board)

        # checking rows
        for i in range(n):
            win = True
            for j in range(n):
                if self.board[i][j] != player:
                    win = False
                    break
            if win:
                return win

        # checking columns
        for i in range(n):
            win = True
            for j in range(n):
                if self.board[j][i] != player:
                    win = False
                    break
            if win:
                return win

            # checking diagonals
        win = True
        for i in range(n):
            if self.board[i][i] != player:
                win = False
                break
        if win:
            return win

        win = True
        for i in range(n):
            if self.board[i][n - 1 - i] != player:
                win = False
                break
        if win:
            return win
        return False

    def get_next_action(self, epsilon):
        if np.random.random() < epsilon:
            valid_actions = []
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == "-":
                        valid_actions.append((i, j))
            return random.choice(valid_actions)
        else:
            q_value_masked_array = np.ma.masked_where(self.board != "-", self.q_values)
            q_value_masked_array = np.ma.masked_equal(q_value_masked_array, np.max(q_value_masked_array))
            return np.unravel_index(np.argmax(q_value_masked_array), self.q_values.shape[:2])

    def get_next_location(self, action, player):
        row, col = action
        self.select_square(row, col, player)

    """def get_next_action(self, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(self.q_values)
        else:
            return np.random.randint(9)

    def get_next_location(self, action_index, player):
        if TicTacToe.actions[action_index] == 1:
            row = 1
            col = 1
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 2:
            row = 1
            col = 2
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 3:
            row = 1
            col = 3
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 4:
            row = 2
            col = 1
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 5:
            row = 2
            col = 2
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 6:
            row = 2
            col = 3
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 7:
            row = 3
            col = 1
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 8:
            row = 3
            col = 2
            self.select_square(row, col, player)
        elif TicTacToe.actions[action_index] == 9:
            row = 3
            col = 3
            self.select_square(row, col, player)"""

    # Checking if the board is filled
    def is_board_filled(self):
        for row in self.board:
            for item in row:
                if item == '-':
                    return False
        return True

    # Function for displaying the board
    def board_display(self):
        for row in self.board:
            print(" ".join(row))

    # Switching turns
    def alternate_turn(self, player):
        return 'X' if player == 'O' else 'O'

    def q_learning(self, episodes=1000, epsilon=0.8, discount_factor=0.9, learning_rate=0.5):

        self.create_board()
        print(f"Training {episodes} times...")

        for episode in range(episodes):
            player = random.choice(["X", "O"])
            while True:
                action_index = self.get_next_action(epsilon)
                self.get_next_location(action_index, player)

                if self.is_player_win(player):
                    reward = TicTacToe.win_reward
                    break

                elif self.is_board_filled():
                    reward = 0
                    break

                else:
                    reward = 0

                player = self.alternate_turn(player)

                for i in range(9):
                    if self.actions[i] != "-":
                        indices = np.unravel_index(action_index, self.q_values.shape)
                        old_q_value = self.q_values[indices]
                        max_q_value = np.max(self.q_values)
                        temporal_difference = reward + (discount_factor * max_q_value) - old_q_value
                        new_q_value = old_q_value + (learning_rate * temporal_difference)
                        self.q_values[indices] = new_q_value

        print("Training Complete")

    def board_reset(self):
        self.create_board()

    def test(self):
        self.create_board()
        player = random.choice(["X", "O"])
        while True:
            action_index = self.get_next_action(epsilon=1)
            self.get_next_location(action_index, player)

            self.board_display()

            if self.is_player_win(player):
                self.board_display()
                print(f"Player {player} wins the game")
                break

            if self.is_board_filled():
                self.board_display()
                print("Draw")
                break

            player = self.alternate_turn(player)

    """def play(self):
        self.create_board()
        self.q_learning()
        self.board_reset()
        player = "X"
        if self.get_random_first_player() == 1:
            player = "O"
        else:
            player = "X"

        while True:
            self.board_display()

            if player == "X":
                row, col = input("Please enter row and column: ").split(" ")
                self.select_square(row, col, player)

                if self.is_player_win(player):
                    self.board_display()
                    print(f"Player {player} wins the game")
                    break

                if self.is_board_filled():
                    self.board_display()
                    print("Draw")
                    break
            elif player == "O":
                print("Opponent Turn")
                action_index = self.get_next_action(epsilon=1)
                self.get_next_location(action_index, player)

                if self.is_player_win(player):
                    self.board_display()
                    print(f"Player {player} wins the game")
                    break

                if self.is_board_filled():
                    self.board_display()
                    print("Draw")
                    break

            player = self.alternate_turn(player)"""

    def play(self):
        self.create_board()
        self.q_learning()
        self.board_reset()
        player = "X"
        if self.get_random_first_player() == 1:
            player = "O"
        else:
            player = "X"

        while True:
            self.board_display()

            if player == "X":
                while True:
                    row, col = input("Please enter row and column: ").split(" ")
                    row, col = int(row), int(col)
                    if self.board[row - 1][col - 1] == "-":
                        self.select_square(row, col, player)
                        break
                    else:
                        print("Invalid move. Please select an empty spot.")
            elif player == "O":
                print("Opponent Turn")
                action_index = self.get_next_action(epsilon=1)
                self.get_next_location(action_index, player)

            if self.is_player_win(player):
                self.board_display()
                print(f"Player {player} wins the game")
                break

            if self.is_board_filled():
                self.board_display()
                print("Draw")
                break

            player = self.alternate_turn(player)

    # Manual game code (non machine learning)
    def PvP(self):
        self.create_board()
        player = "X"
        if self.get_random_first_player() == 1:
            player = "O"
        else:
            player = "X"
        while True:
            self.board_display()
            row, col = input("Please enter row and column: ").split(" ")
            self.select_square(row, col, player)

            if self.is_player_win(player):
                self.board_display()
                print(f"Player {player} wins the game")
                break

            if self.is_board_filled():
                self.board_display()
                print("Draw")
                break

            player = self.alternate_turn(player)

tic_tac_toe = TicTacToe()
#tic_tac_toe.PvP()
tic_tac_toe.play()
#tic_tac_toe.test()