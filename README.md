# Poker Reinforcement Learning

This project implements a Deep Q-Network (DQN) based agent for playing Texas Hold'em Poker. The agent learns to make optimal decisions in a simplified two-player poker environment through reinforcement learning.

## Project Overview

This implementation features a DQN agent that learns poker strategies by playing against a basic rule-based opponent. The agent can:

- Make strategic decisions (fold, call/check, or raise)
- Learn from experience using a prioritized experience replay buffer
- Adapt its strategy based on the current hand strength and board texture

## Key Components

- **Poker Environment (`poker_env.py`)**: Simulates a two-player poker game with standard Texas Hold'em rules, including betting rounds, community cards, and hand evaluation.

- **DQN Agent (`dqn_agent.py`)**: Implements the reinforcement learning agent that makes decisions based on the current state and hand strength.

- **Training Script (`train.py`)**: Manages the training process, including experience collection, replay, and model saving.

- **Visualization (`visualize_game.py`)**: Provides a text-based interface to visualize the learned agent playing poker.

- **Utility Functions (`utils.py`)**: Helper functions for card encoding, hand strength evaluation, and data logging.

- **Configuration (`config.py`)**: Centralizes all hyperparameters and settings for the project.

## Installation and Setup

1. Ensure you have the following dependencies installed:
   - Python 3.7+
   - PyTorch
   - NumPy

2. Clone this repository:
   ```
   git clone <repository-url>
   cd pokerRL
   ```

## Usage

### Training the Agent

To start training from scratch:

```
python train.py
```

To resume training from a saved checkpoint:

```
python train.py --resume
```

Training progress is logged to `training.log` and models are periodically saved to disk.

### Visualizing Gameplay

To watch the trained agent play:

```
python visualize_game.py
```

This will use the best saved model to play a series of poker hands, displaying each action and decision.

## Model Architecture

The DQN model consists of:
- Input layer: State representation (107 features) + hand strength
- Hidden layers: Two fully connected layers with ReLU activation
- Output layer: Q-values for each possible action (fold, call/check, raise)

## Customization

You can modify various parameters in `config.py`, including:
- Stack sizes and blind amounts
- Neural network architecture
- Training hyperparameters (learning rate, discount factor, etc.)
- Replay buffer size

## Performance

The agent typically achieves a win rate of 45-55% against the rule-based opponent after approximately 5,000 episodes of training. Performance may vary based on random initialization and hyperparameters.

---

## Contributing

Contributions to this project are welcome! Here's how you can help:

- **Bug Reports**: If you find a bug, please open an issue with detailed steps to reproduce.
- **Feature Requests**: Have ideas for new features? Open an issue to discuss them.
- **Code Contributions**: 
  1. Fork the repository
  2. Create a feature branch (`git checkout -b feature/amazing-feature`)
  3. Commit your changes (`git commit -m 'Add some amazing feature'`)
  4. Push to the branch (`git push origin feature/amazing-feature`)
  5. Open a Pull Request

All contributions, including documentation improvements, code optimization, or new training approaches are appreciated.

You can also always reach out to me!

### Development Environment Setup

To set up a development environment:

```bash
# Clone your fork
git clone https://github.com/your-username/pokerRL.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
```

## Future Improvements

- Multi-player poker support
- More sophisticated opponent models
- Integration with graphical user interface
- Expanded action space (variable bet sizing)
- Advanced poker features (position awareness, opponent modeling)

---
## Credits

[Aashish Harishchandre](https://aashishharishchandre.netlify.app/)
