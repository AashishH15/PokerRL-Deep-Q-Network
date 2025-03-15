import numpy as np
from collections import deque
from utils import encode_cards, get_valid_actions, calculate_hand_strength
from config import Config
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

class PokerEnv:
    def __init__(self):
        self.num_players = Config.NUM_PLAYERS
        self.init_stack = Config.INIT_STACK
        self.reset()
    
    def reset(self):
        self.deck = self._create_deck()
        self.community_cards = []
        self.pot = Config.SMALL_BLIND + Config.BIG_BLIND
        self.current_bet = Config.BIG_BLIND
        self.players = [
            {
                'stack': Config.INIT_STACK - Config.BIG_BLIND,
                'hand': [],
                'active': True,
                'current_bet': Config.BIG_BLIND
            },
            {
                'stack': Config.INIT_STACK - Config.SMALL_BLIND,
                'hand': [],
                'active': True,
                'current_bet': Config.SMALL_BLIND
            }
        ]
        self.betting_round = 0
        self._deal_hands()
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
        community = state_array[52:61]
        pot = state_array[61] * Config.INIT_STACK
        current_bet = state_array[62] * Config.BIG_BLIND 
        stack = state_array[63] * Config.INIT_STACK
        
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

    def _betting_round_complete(self):
        for player in self.players:
            if player['active'] and player['current_bet'] != self.current_bet:
                return False
        return True

    def step(self, action):
        reward = 0
        done = False
        initial_pot = self.pot
        
        if action == 0:  # Fold
            self.players[0]['active'] = False
            reward = -self.players[0]['current_bet']
            self.players[1]['stack'] += self.pot
            done = True
        elif action == 1:  # Call
            amount_to_call = self.current_bet - self.players[0]['current_bet']
            if amount_to_call > 0 and amount_to_call <= self.players[0]['stack']:
                self.pot += amount_to_call
                self.players[0]['stack'] -= amount_to_call
                self.players[0]['current_bet'] = self.current_bet
        elif action == 2:  # Raise
            raise_amount = min(
                Config.BIG_BLIND * 2,
                self.players[0]['stack'] 
            )
            new_total_bet = self.current_bet + raise_amount
            amount_to_add = new_total_bet - self.players[0]['current_bet']
            
            if amount_to_add <= self.players[0]['stack']:
                self.current_bet = new_total_bet
                self.pot += amount_to_add
                self.players[0]['stack'] -= amount_to_add
                self.players[0]['current_bet'] = new_total_bet

        if self.players[0]['stack'] <= 0 or self.players[1]['stack'] <= 0:
            done = True
            reward = self.pot if self.players[0]['stack'] > 0 else -self.pot
            return self._get_state(), reward, done, {}

        if not done and self.players[0]['active']:
            opp_action = np.random.choice([1, 2], p=[0.7, 0.3])
            if opp_action == 1:  # Call
                amount_to_call = self.current_bet - self.players[1]['current_bet']
                if amount_to_call <= self.players[1]['stack']:
                    self.pot += amount_to_call
                    self.players[1]['stack'] -= amount_to_call
                    self.players[1]['current_bet'] = self.current_bet
                else:  # All-in call
                    self.pot += self.players[1]['stack']
                    self.players[1]['current_bet'] += self.players[1]['stack']
                    self.players[1]['stack'] = 0
                    done = True
            elif opp_action == 2:  # Raise
                raise_amount = min(Config.BIG_BLIND * 2, self.players[1]['stack'])
                new_total_bet = self.current_bet + raise_amount
                amount_to_add = new_total_bet - self.players[1]['current_bet']
                
                if amount_to_add <= self.players[1]['stack']:
                    self.current_bet = new_total_bet
                    self.pot += amount_to_add
                    self.players[1]['stack'] -= amount_to_add
                    self.players[1]['current_bet'] = new_total_bet

        if not done and self._betting_round_complete():
            self.current_bet = 0
            for p in self.players:
                p['current_bet'] = 0

            if self.betting_round == 0:  # Pre-flop -> Flop
                self.community_cards.extend(self.deck[:3])
                self.deck = self.deck[3:]
                self.betting_round = 1
            elif self.betting_round == 1:  # Flop -> Turn
                self.community_cards.append(self.deck[0])
                self.deck = self.deck[1:]
                self.betting_round = 2
            elif self.betting_round == 2:  # Turn -> River
                self.community_cards.append(self.deck[0])
                self.deck = self.deck[1:]
                self.betting_round = 3
            elif self.betting_round == 3:  # River -> Showdown
                done = True
                if self.players[0]['active']:
                    player_strength = calculate_hand_strength(
                        self.players[0]['hand'],
                        self.community_cards
                    )
                    opponent_strength = calculate_hand_strength(
                        self.players[1]['hand'],
                        self.community_cards
                    )
                    if player_strength > opponent_strength:
                        reward = self.pot
                    elif player_strength < opponent_strength:
                        reward = -self.players[0]['current_bet']

        next_state = self._get_state()
        return next_state, reward, done, {}

    def calculate_reward(self):
        base_reward = self.pot if self.players[0]['active'] else -self.players[0]['total_invested']
        hand_strength_bonus = self.get_hand_strength(self._get_state()) * 50
        return base_reward + hand_strength_bonus