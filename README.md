# TicTacToe
TicTacToe Reinforcement Learning Bot utilizing Q-Learning methods

# How It Works
### 1. Initialization
  - def init() initializes class TicTacToe and defines rewards for winning, losing, and drawing
    - Randomly initialized Q-values for each state action pair on the 3x3 board from 0 to 1 (3x3 board, 9 possible actions)
    - Board is created as 3x3 grid with "-" to indicate empty cell
  - def create_board() resets board for each new game
### 2. Game Setup
  - def get_random_first_player() randomly selects which player to start
  - def select_square() updates board with player move
### 3. Win Conditions
  - def is_player_win() defines all possible winning states, and if any of them are occupied then it returns true
  - def is_board_filled() checks whether or not the board is full or empty. IF full and no win, it returns a draw
### 4. Action Selections
  - def get_valid_actions() returns a list of all empty spots for the next move
  - def get_next_action() utilizes the epsilon greedy strategy to decide whether the next action will be randomized or based on the highest q-value
  - def get_next_location() gets player input to track what coordinates the human player wants to play
  - def board_display() displays current board state
### 5. Q-Learning
  - def q_learning() is defined with several parameters
    - episodes: number of games start to finish that the agent trains on
    - epsilon: variable that decides whether the agent's next action explores or exploits
    - min_epsilon: sets the lower bound for epsilon decay
    - decay_rate: the rate at which exploration/exploitation balance shifts towards exploiting maximum q-values
    - discount_factor: future rewards are discounted, forcing the agent to prioritize immediate gains
    - learning_rate: how quickly the agent abandons the previous Q-value in the Q-table for a given state-action pair for the new Q-table
  - while loop:
    - Resets the board after every episode where players either end in a win, loss, or draw
    - Assigns and appends rewards 
  - for loop:
    - Implements a Q-learning specific Bellman Equation recursively to continuously update Q-values (this is the "learning" part of the algorithm)
    - Q-value changes are tracked to track convergence (whether or not the algorithm is actually learning)
  - Epsilon decay is implemented to allow the algorithm to exploit more as the agent learns, making less randomized decisions
  - Q-values and rewards are then plotted to track learning
### 6. Training/Eval
  - def q_learning() is executed, creating a Q-table with optimal Q-values for playing against
  - Q-values are saved into a file to be called
### 7. Playing
  - def play() allows the human player to interact with the agent
  - Human player inputs moves while the agent inputs moves based on the saved Q-table
  - Actions are looped until either the player or the agent wins

# Future Improvements
### 1. Advanced Reward System
  - Rewarding advantageous positioning and punishing disadvantageous positioning
  - Examples:
    - Leaving open a human player position with two in a row -> punishment
    - Setting up agent position with two in a row -> reward
### 2. Set Start Positions
  - Setting up each training episode with a different designated cell spot each time instead of randomizing to allow for more efficient learning
  - Example: dividing training episodes into 9 parts, each at a different coordinate

### 3. Fine Tuning Parameters
  - Experimenting with different Q-Learning parameters to optimize convergence and rewards
