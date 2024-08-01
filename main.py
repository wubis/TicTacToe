import numpy as np
import random
import matplotlib.pyplot as plt

class TicTacToe:
    win_reward = 1
    lose_reward = -1
    draw_reward = 0

    def __init__(self):
        self.board = []
        self.q_values = np.random.uniform(low=0.0, high=1.0, size=(3, 3, 9))
        self.create_board()

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

    def get_valid_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == "-"]

    def get_next_action(self, epsilon):
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return None

        if np.random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            q_values = np.array([self.q_values[i, j, i*3 + j] for i, j in valid_actions])
            max_q_value_indices = np.where(q_values == np.max(q_values))[0]
            return valid_actions[random.choice(max_q_value_indices)]

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

    def q_learning(self, episodes=50000, epsilon=0.5, discount_factor=0.9, learning_rate=0.001, min_epsilon=0.1, decay_rate=0.9):
        print(f"Training {episodes} times...")
        
        # For monitoring convergence
        q_value_changes = []
        average_rewards = []
        
        for episode in range(episodes):
            self.create_board()
            player = random.choice(["X", "O"])
            states_actions_rewards = []

            episode_reward = 0
            q_value_change = 0
            
            while True:
                action = self.get_next_action(epsilon)
                if action is None:
                    break

                self.get_next_location(action, player)
                reward = 0

                if self.is_player_win(player):
                    reward = TicTacToe.win_reward
                elif self.is_board_filled():
                    reward = TicTacToe.draw_reward

                states_actions_rewards.append((action, reward))

                if reward != 0:
                    break

                player = self.alternate_turn(player)

            for i, (action, reward) in enumerate(states_actions_rewards):
                indices = action
                if i == len(states_actions_rewards) - 1:
                    future_q_value = 0
                else:
                    next_action = states_actions_rewards[i+1][0]
                    future_q_value = np.max(self.q_values[next_action])

                old_q_value = self.q_values[indices[0], indices[1], indices[0]*3 + indices[1]]
                temporal_difference = reward + (discount_factor * future_q_value) - old_q_value
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                q_value_change += abs(new_q_value - old_q_value)
                self.q_values[indices[0], indices[1], indices[0]*3 + indices[1]] = new_q_value

                episode_reward += reward

            q_value_changes.append(q_value_change)
            average_rewards.append(episode_reward / len(states_actions_rewards) if states_actions_rewards else 0)

            # Decay epsilon
            epsilon = max(min_epsilon, epsilon * decay_rate)

        print("Training Complete")
        
        # Plotting Q-value changes
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(q_value_changes, label='Q-value Changes')
        plt.xlabel('Episode')
        plt.ylabel('Total Q-value Change')
        plt.title('Q-value Convergence')
        plt.legend()
        
        # Plotting average rewards
        plt.subplot(1, 2, 2)
        plt.plot(average_rewards, label='Average Reward', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Reward Trend')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def save_q_values(self, filename='/Users/Justin Wang/Desktop/TicTacToeBot/q_values.npy'):
        np.save(filename, self.q_values)
        print(f"Q-values saved to {filename}")

    def load_q_values(self, filename='q_values.npy'):
        self.q_values = np.load(filename)
        print(f"Q-values loaded from {filename}")

    def board_reset(self):
        self.create_board()

    def play(self):
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

# Example usage
tic_tac_toe = TicTacToe()

# Train and save Q-values
#tic_tac_toe.q_learning()
#tic_tac_toe.save_q_values()

# To play later, load the Q-values
tic_tac_toe.load_q_values()
tic_tac_toe.play()
