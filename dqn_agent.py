import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from config import Config

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, Config.HIDDEN_DIM)
        self.fc2 = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.fc3 = nn.Linear(Config.HIDDEN_DIM, action_dim)
        
    def forward(self, state, hand_strength):
        x = torch.cat([state, hand_strength.unsqueeze(-1)], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = []
        self.priorities = np.array([])
        self.alpha = 0.6
        self.beta = 0.4 
        self.eps_prio = 1e-6 
        
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.eps = Config.EPS_START
        
    def act(self, state, hand_strength):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                hand_strength_tensor = torch.FloatTensor([hand_strength]).to(self.device)
                q_values = self.policy_net(state_tensor, hand_strength_tensor)
                return q_values.argmax().item()
    
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        max_prio = max(self.priorities, default=1.0)
        self.priorities = np.append(self.priorities, max_prio)

        if len(self.memory) > Config.REPLAY_BUFFER_SIZE:
            self.memory.pop(0)
            self.priorities = np.delete(self.priorities, 0)

    def prioritized_sample(self, batch_size):
        if len(self.memory) < batch_size:
            return [], [], []

        probs = self.priorities ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        samples = [self.memory[idx] for idx in indices]
        
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.eps_prio

    def replay(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return
            
        batch, indices, weights = self.prioritized_sample(Config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        hand_strengths = torch.FloatTensor([self.get_hand_strength(s) for s in states]).to(self.device)
        next_hand_strengths = torch.FloatTensor([self.get_hand_strength(s) for s in next_states]).to(self.device)
        
        current_q_values = self.policy_net(states, hand_strengths).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states, next_hand_strengths).max(1)[0]
        
        target_q_values = rewards + (1 - dones) * Config.GAMMA * next_q_values
        
        td_errors = (target_q_values.unsqueeze(1) - current_q_values).abs()
        
        self.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        loss = (weights.unsqueeze(1) * (current_q_values - target_q_values.unsqueeze(1))**2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        for target_param, policy_param in zip(
            self.target_net.parameters(), 
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                0.95 * target_param.data + 0.05 * policy_param.data
            )
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_hand_strength(self, state):
        return 0.5

    def get_memory_for_save(self):
        memory_list = []
        for state, action, reward, next_state, done in self.memory:
            memory_list.append({
                'state': state.tolist() if isinstance(state, np.ndarray) else state,
                'action': action,
                'reward': reward,
                'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
                'done': done
            })
        return memory_list

    def load_memory_from_save(self, memory_list):
        self.memory = deque(maxlen=Config.REPLAY_BUFFER_SIZE)
        for item in memory_list:
            state = np.array(item['state'])
            next_state = np.array(item['next_state'])
            self.memory.append((
                state,
                item['action'],
                item['reward'],
                next_state,
                item['done']
            ))