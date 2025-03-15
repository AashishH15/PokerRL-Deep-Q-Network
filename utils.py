import numpy as np
import torch
import logging
from config import Config

logging.basicConfig(filename='training.log', level=logging.INFO)

def encode_cards(cards, max_cards=52):
    """
    One-hot encode cards (hole cards + community cards).
    Example: [Ah, Kd] → 52-dim vector with 1s at indices 0 and 51.
    """
    encoded = np.zeros(max_cards, dtype=np.float32)
    for card in cards:
        # Convert (rank, suit) to unique index (0-51)
        idx = card[0] * 4 + card[1]
        encoded[idx] = 1.0
    return encoded

def get_valid_actions(current_bet, player_stack, min_raise=Config.BIG_BLIND):
    """
    Return mask for allowed actions in fixed-limit poker:
    0=Fold, 1=Check/Call, 2=Bet/Raise
    """
    valid_actions = [0, 1]  # Fold and Check/Call always allowed
    can_raise = (player_stack >= current_bet + min_raise and 
                 current_bet < Config.MAX_RAISES_PER_ROUND * Config.BIG_BLIND)
    if can_raise:
        valid_actions.append(2)
    return valid_actions

def log_metrics(episode, reward, eps, win_rate=None):
    """Log training progress to file and console"""
    logging.info(
        f"Episode {episode} | "
        f"Avg Reward: {reward:.2f} | "
        f"Epsilon: {eps:.3f} | "
        f"Win Rate: {win_rate*100:.2f}%" if win_rate else ""
    )
    print(f"Episode {episode} | Reward: {reward:.2f}")

def calculate_hand_strength(hole_cards, community_cards):
    """Calculate hand strength based on hole cards and community cards"""
    def get_rank_suit(card):
        return card[0], card[1]

    all_cards = hole_cards + community_cards
    if not all_cards:  # Handle empty card lists
        return 0.5

    ranks = [get_rank_suit(c)[0] for c in all_cards]
    suits = [get_rank_suit(c)[1] for c in all_cards]
    
    rank_count = {r: ranks.count(r) for r in set(ranks)}
    suit_count = {s: suits.count(s) for s in set(suits)}

    # Check for straight flush first (highest value)
    for suit in suits:
        suited_ranks = sorted([r for r, s in all_cards if s == suit])
        if len(suited_ranks) >= 5:
            for i in range(len(suited_ranks) - 4):
                if suited_ranks[i+4] - suited_ranks[i] == 4:
                    return 0.95 + (suited_ranks[i+4] / 100)

    # Then check four of a kind
    if 4 in rank_count.values():
        quad_rank = max(r for r in rank_count if rank_count[r] == 4)
        kickers = sorted([r for r in ranks if r != quad_rank], reverse=True)
        kicker = kickers[0] if kickers else 0
        return 0.9 + (quad_rank / 100) + (kicker / 1000)

    # Full house
    if 3 in rank_count.values() and 2 in rank_count.values():
        triplet = max(r for r in rank_count if rank_count[r] == 3)
        pair = max(r for r in rank_count if rank_count[r] == 2)
        return 0.85 + (triplet / 100) + (pair / 1000)

    # One pair
    if 2 in rank_count.values():
        pair = max(r for r in rank_count if rank_count[r] == 2)
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        kicker_score = 0
        if kickers:
            kicker_score = kickers[0] / 1000
        return 0.6 + (pair / 100) + kicker_score

    # High card (default case)
    sorted_ranks = sorted(ranks, reverse=True)
    return 0.5 + (sorted_ranks[0] / 100)

def action_mask_to_probs(action_mask):
    probs = np.array(action_mask, dtype=np.float32)
    return probs / probs.sum()

def save_model(model, path="poker_dqn.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="poker_dqn.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()

def visualize_game_state(state):
    # Example: Decode cards from one-hot vectors
    print(f"Player Hand: {decode_cards(state['hand'])}")
    print(f"Community Cards: {decode_cards(state['community'])}")
    print(f"Pot: {state['pot']} | Stack: {state['stack']}")

def decode_cards(encoded_cards):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    cards = []

    if isinstance(encoded_cards, list) and len(encoded_cards) > 0 and isinstance(encoded_cards[0], tuple):
        for rank, suit in encoded_cards:
            cards.append(f"{ranks[rank]}{suits[suit]}")
    else:
        for idx in np.where(np.array(encoded_cards) == 1)[0]:
            rank = idx // 4
            suit = idx % 4
            cards.append(f"{ranks[rank]}{suits[suit]}")
    
    return cards