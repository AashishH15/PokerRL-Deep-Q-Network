from poker_env import PokerEnv
from dqn_agent import DQNAgent
import torch
from utils import log_metrics, save_model, load_model
from config import Config
import numpy as np
import os
from collections import deque
import signal
import sys

if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU instead.")
else:
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(agent, total_rewards, win_rates, best_reward, episode):
    checkpoint = {
        'memory': agent.get_memory_for_save(),
        'total_rewards': total_rewards,
        'win_rates': win_rates,
        'best_reward': best_reward,
        'eps': agent.eps,
        'episode': episode
    }
    np.savez("checkpoint.npz", **checkpoint)
    save_model(agent.policy_net, "checkpoint_model.pth")
    print(f"\nCheckpoint saved at episode {episode}")

def signal_handler(signum, frame):
    print('\nInterrupt received. Saving checkpoint before exit...')
    if 'agent' in globals():
        save_checkpoint(agent, total_rewards, win_rates, best_reward, ep)
    sys.exit(0)

def adaptive_epsilon_decay(agent, win_rate):
    if win_rate < 0.35:
        return max(Config.EPS_END, agent.eps * 0.999)
    elif win_rate < 0.45:
        return max(Config.EPS_END, agent.eps * 0.995)
    else:
        return max(Config.EPS_END, agent.eps * 0.99)

def train(resume=False):
    global agent, total_rewards, win_rates, best_reward, ep
    signal.signal(signal.SIGINT, signal_handler)

    env = PokerEnv()
    agent = DQNAgent(Config.STATE_DIM, Config.ACTION_DIM)

    if resume:
        if os.path.exists("checkpoint.npz"):
            data = np.load("checkpoint.npz", allow_pickle=True)
            agent.load_memory_from_save(data['memory'].tolist())
            total_rewards = data['total_rewards'].tolist()
            win_rates = data['win_rates'].tolist()
            best_reward = data['best_reward'].item()
            agent.eps = data['eps'].item()
            start_episode = data['episode'].item() + 1
            
            if os.path.exists("checkpoint_model.pth"):
                load_model(agent.policy_net, "checkpoint_model.pth")
                load_model(agent.target_net, "checkpoint_model.pth")
        else:
            print("No checkpoint found. Starting from scratch.")
            total_rewards, win_rates, best_reward, start_episode = [], [], float('-inf'), 0
    else:
        total_rewards, win_rates, best_reward, start_episode = [], [], float('-inf'), 0

    episodes = Config.EPISODES

    for ep in range(start_episode, episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            hand_strength = env.get_hand_strength(state)
            action = agent.act(state, hand_strength)
            next_state, reward, done, _ = env.step(action)
            
            agent.add_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            agent.replay()
        
        total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards[-100:]) 

        if avg_reward > best_reward:
            best_reward = avg_reward
            save_model(agent.policy_net, "best_model.pth")

        if ep % 10 == 0:
            agent.update_target_net()
            win_rate = np.mean([1 if r > 0 else 0 for r in total_rewards[-100:]])
            win_rates.append(win_rate)
            log_metrics(ep, avg_reward, agent.eps, win_rate)

        if len(win_rates) > 1:
            agent.eps = adaptive_epsilon_decay(agent, win_rates[-1])

        if ep % 100 == 0:
            save_checkpoint(agent, total_rewards, win_rates, best_reward, ep)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    train(resume=args.resume)