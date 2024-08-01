import numpy as np
import random

class TicTacToe:
    win_reward = 100
    lose_reward = -100

    def __init__(self):
        self.board = []
        self.q_values = np.random.uniform(low=0.0, high=1.0, size=(3, 3, 9))  # Q-values for each state-action pair
        self.create_board()  # Initialize the board

    def create_board(self):
        self.board = [["-" for _ in range(3)] for _ in range(3)]

    def get_random_first_player(self):
        return random.randint(0, 1)

    def select_square(self, row, col, player):
        self.board[row][col] = player

    def is_player_win(self, player):
        win_conditions = [
            [(0, 0), (0, 1), (0, 2)],  # rows
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],  # columns
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],  # diagonals
            [(0, 2), (1, 1), (2, 0)]
        ]
        for condition in win_conditions:
            if all(self.board[r][c] == player for r, c in condition):
                return True
        return False

    def get_next_action(self, epsilon):
        valid_actions = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == "-"]
        if not valid_actions:
            return None  # No valid actions available

        if np.random.random() < epsilon:
            # Exploration: pick a random valid action
            return random.choice(valid_actions)
        else:
            # Exploitation: choose the best action based on Q-values
            q_value_masked_array = np.ma.masked_where(self.board != "-", self.q_values)
            best_action_index = np.unravel_index(np.argmax(q_value_masked_array), self.q_values.shape[:2])
            if best_action_index in valid_actions:
                return best_action_index
            return random.choice(valid_actions)

    def get_next_location(self, action, player):
        if action:
            row, col = action
            self.select_square(row, col, player)

    def is_board_filled(self):
        for row in self.board:
            if "-" in row:
                return False
        return True

    def board_display(self):
        for row in self.board:
            print(" ".join(row))

    def alternate_turn(self, player):
        return 'X' if player == 'O' else 'O'

    def q_learning(self, episodes=50000, epsilon=0.5, discount_factor=0.9, learning_rate=0.01):
        print(f"Training {episodes} times...")

        for episode in range(episodes):
            self.create_board()  # Reset the board for each episode
            player = random.choice(["X", "O"])
            while True:
                action = self.get_next_action(epsilon)
                if action is None:
                    break  # No valid moves available, exit the loop

                self.get_next_location(action, player)

                if self.is_player_win(player):
                    reward = TicTacToe.win_reward
                elif self.is_board_filled():
                    reward = 0
                else:
                    reward = 0

                indices = action
                old_q_value = self.q_values[indices]
                max_q_value = np.max(self.q_values)
                temporal_difference = reward + (discount_factor * max_q_value) - old_q_value
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                self.q_values[indices] = new_q_value

                if reward != 0:
                    break

                player = self.alternate_turn(player)

        print("Training Complete")

    def board_reset(self):
        self.create_board()

    def test(self):
        self.create_board()
        player = random.choice(["X", "O"])
        while True:
            action = self.get_next_action(epsilon=1)
            self.get_next_location(action, player)

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

    def play(self):
        self.q_learning()
        self.board_reset()
        player = "X"
        if self.get_random_first_player() == 1:
            player = "O"

        while True:
            self.board_display()

            if player == "X":
                while True:
                    row, col = input("Please enter row and column: ").split(" ")
                    row, col = int(row) - 1, int(col) - 1
                    if self.board[row][col] == "-":
                        self.select_square(row, col, player)
                        break
                    else:
                        print("Invalid move. Please select an empty spot.")
            else:
                print("Opponent Turn")
                action = self.get_next_action(epsilon=0)  # Exploit during actual gameplay
                self.get_next_location(action, player)

            if self.is_player_win(player):
                self.board_display()
                print(f"Player {player} wins the game")
                break

            if self.is_board_filled():
                self.board_display()
                print("Draw")
                break

            player = self.alternate_turn(player)

    '''def PvP(self):
        self.create_board()
        player = "X"
        if self.get_random_first_player() == 1:
            player = "O"
        while True:
            self.board_display()
            row, col = input("Please enter row and column: ").split(" ")
            self.select_square(int(row) - 1, int(col) - 1, player)

            if self.is_player_win(player):
                self.board_display()
                print(f"Player {player} wins the game")
                break

            if self.is_board_filled():
                self.board_display()
                print("Draw")
                break

            player = self.alternate_turn(player)'''

# Example of playing a game
tic_tac_toe = TicTacToe()
tic_tac_toe.play()
