import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from config import Config
from utils import calculate_hand_strength

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

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def __len__(self):
        return self.n_entries

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def total_priority(self):
        return self.tree[0]

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority, data):
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                v -= self.tree[left_idx]
                parent_idx = right_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = SumTree(Config.REPLAY_BUFFER_SIZE)
        self.alpha = 0.6
        self.beta = 0.4 
        self.eps_prio = 1e-6 
        
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.eps = Config.EPS_START
        
    def act(self, state, hand_strength, valid_actions=None):
        if valid_actions is None:
            valid_actions = [0, 1, 2]

        if np.random.rand() <= self.eps:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                hand_strength_tensor = torch.FloatTensor([hand_strength]).to(self.device)
                q_values = self.policy_net(state_tensor, hand_strength_tensor)
                
                masked_q_values = q_values.clone()
                for action in range(self.action_size):
                    if action not in valid_actions:
                        masked_q_values[0, action] = -1e9
                
                return masked_q_values.argmax().item()
    
    def add_experience(self, state, hand_strength, action, reward, next_state, next_hand_strength, done):
        max_prio = np.max(self.memory.tree[Config.REPLAY_BUFFER_SIZE - 1 : Config.REPLAY_BUFFER_SIZE - 1 + self.memory.n_entries]) if self.memory.n_entries > 0 else 1.0
        experience = (state, hand_strength, action, reward, next_state, next_hand_strength, done)
        self.memory.add(max_prio, experience)

    def prioritized_sample(self, batch_size):
        if self.memory.n_entries < batch_size:
            return [], [], []

        batch = []
        indices = []
        priorities = []
        segment = self.memory.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.memory.get_leaf(v)
            priorities.append(p)
            indices.append(idx)
            batch.append(data)

        sampling_probabilities = np.array(priorities) / self.memory.total_priority
        weights = (self.memory.n_entries * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = float(np.abs(error).item()) + self.eps_prio
            priority = priority ** self.alpha
            self.memory.update(idx, priority)

    def replay(self):
        if self.memory.n_entries < Config.BATCH_SIZE:
            return
            
        batch, indices, weights = self.prioritized_sample(Config.BATCH_SIZE)
        states, hand_strengths, actions, rewards, next_states, next_hand_strengths, dones = zip(*batch)
        
        hand_strengths = torch.FloatTensor(hand_strengths).to(self.device)
        next_hand_strengths = torch.FloatTensor(next_hand_strengths).to(self.device)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
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
        state_array = np.asarray(state)
        hole_cards = self._decode_cards(state_array[:52])
        community_cards = self._decode_cards(state_array[52:104])
        return calculate_hand_strength(hole_cards, community_cards)

    def _decode_cards(self, encoded_cards):
        cards = []
        for idx, value in enumerate(encoded_cards):
            if value == 1:
                rank = idx // 4
                suit = idx % 4
                cards.append((rank, suit))
        return cards

    def get_memory_for_save(self):
        memory_list = []
        for i in range(self.memory.n_entries):
            idx = i + Config.REPLAY_BUFFER_SIZE - 1
            priority = self.memory.tree[idx]
            data = self.memory.data[i]
            if data is not None:
                state, hand_strength, action, reward, next_state, next_hand_strength, done = data
                memory_list.append({
                    'state': state.tolist() if isinstance(state, np.ndarray) else state,
                    'hand_strength': hand_strength,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
                    'next_hand_strength': next_hand_strength,
                    'done': done,
                    'priority': float(priority)
                })
        return memory_list

    def load_memory_from_save(self, memory_list):
        self.memory = SumTree(Config.REPLAY_BUFFER_SIZE)
        for item in memory_list:
            state = np.array(item['state'])
            next_state = np.array(item['next_state'])
            hand_strength = item.get('hand_strength', self.get_hand_strength(state))
            next_hand_strength = item.get('next_hand_strength', self.get_hand_strength(next_state))
            experience = (
                state,
                hand_strength,
                item['action'],
                item['reward'],
                next_state,
                next_hand_strength,
                item['done']
            )
            priority = item.get('priority', 1.0)
            self.memory.add(priority, experience)

        memory_len = len(self.memory)
        if memory_len == 0:
            self.priorities = np.array([])
        else:
            self.priorities = np.ones(memory_len, dtype=np.float32)