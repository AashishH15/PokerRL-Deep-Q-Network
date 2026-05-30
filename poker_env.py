import numpy as np
from collections import deque
from utils import encode_cards, get_valid_actions, calculate_hand_strength, get_opponent_action
from config import Config
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

class PokerEnv:
    def __init__(self):
        self.num_players = Config.NUM_PLAYERS
        self.init_stack = Config.INIT_STACK
        self.button_player = self.num_players - 1
        self.reset()
    
    def reset(self, player_stacks=None):
        self.deck = self._create_deck()
        self.community_cards = []
        
        self.button_player = (self.button_player + 1) % self.num_players
        
        if player_stacks is None:
            stacks = [Config.INIT_STACK] * self.num_players
        else:
            stacks = list(player_stacks)

        self.initial_stacks = list(stacks)

        if self.num_players == 2:
            sb_player = self.button_player
            bb_player = 1 - self.button_player
        else:
            sb_player = (self.button_player + 1) % self.num_players
            bb_player = (self.button_player + 2) % self.num_players
        
        stacks[sb_player] -= Config.SMALL_BLIND
        stacks[bb_player] -= Config.BIG_BLIND
        
        self.players = []
        for i in range(self.num_players):
            bet = 0
            if i == sb_player:
                bet = Config.SMALL_BLIND
            elif i == bb_player:
                bet = Config.BIG_BLIND
                
            self.players.append({
                'stack': stacks[i],
                'hand': [],
                'active': True,
                'current_bet': bet
            })
        
        self.pot = Config.SMALL_BLIND + Config.BIG_BLIND
        self.current_bet = Config.BIG_BLIND
        self.betting_round = 0
        self.raise_count = 0
        self.players_acted_this_round = set()
        
        if self.num_players == 2:
            self.active_player = self.button_player
        else:
            self.active_player = (self.button_player + 3) % self.num_players
        
        self.done = False
        self.reward = 0
        
        self._deal_hands()
        
        while self.active_player != 0:
            active_idx = self.active_player
            opp_strength = calculate_hand_strength(self.players[active_idx]['hand'], self.community_cards)
            opp_valid = get_valid_actions(self.current_bet, self.players[active_idx]['stack'], self.players[active_idx]['current_bet'])
            opp_action = get_opponent_action(
                opp_strength,
                self.current_bet,
                self.players[active_idx]['stack'],
                self.players[active_idx]['current_bet'],
                self.pot
            )
            if opp_action not in opp_valid:
                opp_action = 1
            self._apply_action(active_idx, opp_action)
            
            if self._is_hand_done():
                self.done = True
                active_count = sum(1 for p in self.players if p['active'])
                if active_count <= 1:
                    self.reward = self._handle_fold()
                else:
                    self.reward = self._run_showdown(is_all_in=True)
                break
                
            self._advance_turn()
            
        return self._get_state()

    def _create_deck(self):
        return [(rank, suit) for rank in range(13) for suit in range(4)]

    def _deal_hands(self):
        deck_copy = self.deck.copy()
        np.random.shuffle(deck_copy)
        for p in self.players:
            p['hand'] = deck_copy[:2]
            deck_copy = deck_copy[2:]
        self.deck = deck_copy

    def _get_state(self, player_idx=0):
        hand = self.players[player_idx]['hand']
        state = {
            'hand': hand,
            'community': self.community_cards,
            'pot': self.pot,
            'current_bet': self.current_bet,
            'stack': self.players[player_idx]['stack']
        }
        return self._encode_state(state)

    def _encode_state(self, state):
        encoded_state = np.zeros(Config.STATE_DIM)
        encoded_state[:52] = encode_cards(state['hand'])
        encoded_state[52:104] = encode_cards(state['community'])
        encoded_state[104] = state['pot'] / Config.INIT_STACK 
        encoded_state[105] = state['current_bet'] / Config.INIT_STACK
        encoded_state[106] = state['stack'] / Config.INIT_STACK
        return encoded_state

    def _decode_state(self, state_array):
        hand = state_array[:52]
        community = state_array[52:104]
        pot = state_array[104] * Config.INIT_STACK
        current_bet = state_array[105] * Config.INIT_STACK
        stack = state_array[106] * Config.INIT_STACK
        
        return {
            'hand': hand,
            'community': community,
            'pot': pot,
            'current_bet': current_bet,
            'stack': stack
        }

    def get_hand_strength(self, state):
        state_dict = self._decode_state(state)
        hole_cards = self._decode_cards(state_dict['hand'])
        community_cards = self._decode_cards(state_dict['community'])
        return calculate_hand_strength(hole_cards, community_cards)

    def _decode_cards(self, encoded_cards):
        cards = []
        for i in range(len(encoded_cards)):
            if encoded_cards[i] == 1:
                rank = i // 4
                suit = i % 4
                cards.append((rank, suit))
        return cards

    def _next_active_player(self, from_idx):
        idx = (from_idx + 1) % self.num_players
        while not self.players[idx]['active']:
            idx = (idx + 1) % self.num_players
        return idx

    def _apply_action(self, player_idx, action):
        valid_actions = get_valid_actions(
            self.current_bet, 
            self.players[player_idx]['stack'], 
            self.players[player_idx]['current_bet']
        )
        if action not in valid_actions:
            action = 1
            
        if action == 0:
            self.players[player_idx]['active'] = False
        elif action == 1:
            amount_to_call = self.current_bet - self.players[player_idx]['current_bet']
            if amount_to_call > 0:
                amount_to_call = min(amount_to_call, self.players[player_idx]['stack'])
                self.pot += amount_to_call
                self.players[player_idx]['stack'] -= amount_to_call
                self.players[player_idx]['current_bet'] += amount_to_call
            self.players_acted_this_round.add(player_idx)
        elif action == 2:
            raise_amount = min(Config.BIG_BLIND * 2, self.players[player_idx]['stack'])
            new_total_bet = self.current_bet + raise_amount
            amount_to_add = new_total_bet - self.players[player_idx]['current_bet']
            self.pot += amount_to_add
            self.players[player_idx]['stack'] -= amount_to_add
            self.players[player_idx]['current_bet'] = new_total_bet
            self.current_bet = new_total_bet
            self.raise_count += 1
            self.players_acted_this_round = {player_idx}

    def _is_hand_done(self):
        active_count = sum(1 for p in self.players if p['active'])
        if active_count <= 1:
            return True
        if any(p['stack'] <= 0 for p in self.players if p['active']):
            return True
        return False

    def _run_showdown(self, is_all_in=False):
        if is_all_in:
            needed_community = 5 - len(self.community_cards)
            if needed_community > 0:
                self.community_cards.extend(self.deck[:needed_community])
                self.deck = self.deck[needed_community:]
                
        player0_total_invested = self.initial_stacks[0] - self.players[0]['stack']
        
        strengths = {}
        for idx, p in enumerate(self.players):
            if p['active']:
                strengths[idx] = calculate_hand_strength(p['hand'], self.community_cards)
                
        max_strength = max(strengths.values())
        winners = [idx for idx, strength in strengths.items() if strength == max_strength]
        
        split_pot = self.pot / len(winners)
        for w in winners:
            self.players[w]['stack'] += split_pot
            
        if 0 in winners:
            reward = split_pot - player0_total_invested
        else:
            reward = -player0_total_invested
            
        return reward

    def _handle_fold(self):
        player0_total_invested = self.initial_stacks[0] - self.players[0]['stack']
        
        winner_idx = [idx for idx, p in enumerate(self.players) if p['active']][0]
        self.players[winner_idx]['stack'] += self.pot
        
        if winner_idx == 0:
            reward = self.pot - player0_total_invested
        else:
            reward = -player0_total_invested
            
        return reward

    def _advance_turn(self):
        active_indices = [i for i, p in enumerate(self.players) if p['active']]
        
        if all(i in self.players_acted_this_round for i in active_indices) and \
           all(self.players[i]['current_bet'] == self.current_bet for i in active_indices):
           
            if self.betting_round == 3:
                return True
            
            self.betting_round += 1
            self.current_bet = 0
            self.raise_count = 0
            self.players_acted_this_round = set()
            for p in self.players:
                p['current_bet'] = 0
            
            if self.betting_round == 1:
                self.community_cards.extend(self.deck[:3])
                self.deck = self.deck[3:]
            elif self.betting_round == 2:
                self.community_cards.append(self.deck[0])
                self.deck = self.deck[1:]
            elif self.betting_round == 3:
                self.community_cards.append(self.deck[0])
                self.deck = self.deck[1:]
            
            self.active_player = self._next_active_player(self.button_player)
        else:
            self.active_player = self._next_active_player(self.active_player)
            
        return False

    def step(self, action):
        if self.done:
            return self._get_state(0), self.reward, True, {}

        self._apply_action(0, action)
        
        if self._is_hand_done():
            self.done = True
            active_count = sum(1 for p in self.players if p['active'])
            if active_count <= 1:
                self.reward = self._handle_fold()
            else:
                self.reward = self._run_showdown(is_all_in=True)
            return self._get_state(0), self.reward, True, {}
            
        showdown_needed = self._advance_turn()
        if showdown_needed:
            self.done = True
            self.reward = self._run_showdown()
            return self._get_state(0), self.reward, True, {}
            
        while self.active_player != 0:
            active_idx = self.active_player
            opp_strength = calculate_hand_strength(self.players[active_idx]['hand'], self.community_cards)
            opp_valid = get_valid_actions(self.current_bet, self.players[active_idx]['stack'], self.players[active_idx]['current_bet'])
            
            opp_action = get_opponent_action(
                opp_strength,
                self.current_bet,
                self.players[active_idx]['stack'],
                self.players[active_idx]['current_bet'],
                self.pot
            )
            if opp_action not in opp_valid:
                opp_action = 1
                
            self._apply_action(active_idx, opp_action)
            
            if self._is_hand_done():
                self.done = True
                active_count = sum(1 for p in self.players if p['active'])
                if active_count <= 1:
                    self.reward = self._handle_fold()
                else:
                    self.reward = self._run_showdown(is_all_in=True)
                return self._get_state(0), self.reward, True, {}
                
            showdown_needed = self._advance_turn()
            if showdown_needed:
                self.done = True
                self.reward = self._run_showdown()
                return self._get_state(0), self.reward, True, {}
                
        player0_total_invested = self.initial_stacks[0] - self.players[0]['stack']
        self.reward = -player0_total_invested
        return self._get_state(0), self.reward, False, {}

    def calculate_reward(self):
        player0_total_invested = self.initial_stacks[0] - self.players[0]['stack']
        base_reward = self.pot if self.players[0]['active'] else -player0_total_invested
        hand_strength_bonus = self.get_hand_strength(self._get_state()) * 50
        return base_reward + hand_strength_bonus