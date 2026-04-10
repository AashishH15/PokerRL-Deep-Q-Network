import argparse
import os

from config import Config
from dqn_agent import DQNAgent
from poker_env import PokerEnv
from utils import load_model


def evaluate(num_hands=100, model_path="best_model.pth"):
    env = PokerEnv()
    agent = DQNAgent(Config.STATE_DIM, Config.ACTION_DIM)
    agent.eps = 0

    load_model(agent.policy_net, model_path)


    wins = 0
    losses = 0
    ties = 0
    total_reward = 0
    total_stack_delta = 0

    for _ in range(num_hands):
        state = env.reset()
        initial_stack = env.players[0]["stack"]
        done = False
        hand_reward = 0

        while not done:
            hand_strength = env.get_hand_strength(state)
            action = agent.act(state, hand_strength)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            hand_reward += reward

        final_stack = env.players[0]["stack"]
        stack_delta = final_stack - initial_stack
        total_stack_delta += stack_delta
        total_reward += hand_reward

        if stack_delta > 0:
            wins += 1
        elif stack_delta < 0:
            losses += 1
        else:
            ties += 1

    print(f"Hands played: {num_hands}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Ties: {ties}")
    print(f"Win rate: {wins / num_hands:.3f}")
    print(f"Average reward: {total_reward / num_hands:.3f}")
    print(f"Average stack delta: {total_stack_delta / num_hands:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--model", type=str, default="best_model.pth")
    args = parser.parse_args()

    evaluate(num_hands=args.hands, model_path=args.model)
