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

def get_valid_actions(current_bet, player_stack, player_current_bet=0, min_raise=Config.BIG_BLIND):
    """
    Return mask for allowed actions in fixed-limit poker:
    0=Fold, 1=Check/Call, 2=Bet/Raise
    """
    valid_actions = [0, 1]  # Fold and Check/Call always allowed
    amount_to_call = current_bet - player_current_bet
    cost_to_raise = amount_to_call + min_raise
    can_raise = (player_stack >= cost_to_raise and 
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
    """
    Calculate normalized hand strength in [0.1, 1.0) based on standard poker rules.
    Combines hole cards and community cards to find the best 5-card combination.
    """
    def get_rank_suit(card):
        return card[0], card[1]

    all_cards = hole_cards + community_cards
    if not all_cards:  # Handle empty card lists
        return 0.5

    ranks = [get_rank_suit(c)[0] for c in all_cards]
    suits = [get_rank_suit(c)[1] for c in all_cards]
    
    rank_count = {r: ranks.count(r) for r in set(ranks)}
    suit_count = {s: suits.count(s) for s in set(suits)}

    # Helper to find the highest straight in a set of ranks
    def find_straight(ranks_set):
        # Check standard straights from Ace-high down to 6-high
        for high_rank in range(12, 3, -1):
            if all(r in ranks_set for r in range(high_rank - 4, high_rank + 1)):
                return high_rank
        # Check Ace-low straight (A, 2, 3, 4, 5) -> ranks (12, 0, 1, 2, 3)
        if 12 in ranks_set and all(r in ranks_set for r in range(4)):
            return 3  # High card is 5 (rank 3)
        return None

    # Helper to get base-15 kicker score
    def get_tie_breaker(tie_ranks):
        score = 0.0
        for idx, r in enumerate(tie_ranks):
            score += r / (15 ** (idx + 1))
        return score

    # 1. Straight Flush
    for suit, count in suit_count.items():
        if count >= 5:
            suited_ranks = set(r for r, s in all_cards if s == suit)
            straight_high = find_straight(suited_ranks)
            if straight_high is not None:
                return (9.0 + get_tie_breaker([straight_high])) / 10.0

    # 2. Four of a Kind
    if 4 in rank_count.values():
        quad_rank = max(r for r, count in rank_count.items() if count == 4)
        kickers = sorted([r for r in ranks if r != quad_rank], reverse=True)
        kicker = kickers[0] if kickers else 0
        return (8.0 + get_tie_breaker([quad_rank, kicker])) / 10.0

    # 3. Full House
    # Needs at least one triplet and one separate pair (or another triplet)
    triplets = [r for r, count in rank_count.items() if count >= 3]
    if triplets:
        triplet_rank = max(triplets)
        pairs = [r for r, count in rank_count.items() if count >= 2 and r != triplet_rank]
        if pairs:
            pair_rank = max(pairs)
            return (7.0 + get_tie_breaker([triplet_rank, pair_rank])) / 10.0

    # 4. Flush
    for suit, count in suit_count.items():
        if count >= 5:
            suited_ranks = sorted([r for r, s in all_cards if s == suit], reverse=True)
            return (6.0 + get_tie_breaker(suited_ranks[:5])) / 10.0

    # 5. Straight
    straight_high = find_straight(set(ranks))
    if straight_high is not None:
        return (5.0 + get_tie_breaker([straight_high])) / 10.0

    # 6. Three of a Kind
    if triplets:
        triplet_rank = max(triplets)
        kickers = sorted([r for r in ranks if r != triplet_rank], reverse=True)
        return (4.0 + get_tie_breaker([triplet_rank] + kickers[:2])) / 10.0

    # 7. Two Pair
    pairs = sorted([r for r, count in rank_count.items() if count >= 2], reverse=True)
    if len(pairs) >= 2:
        high_pair = pairs[0]
        low_pair = pairs[1]
        kickers = sorted([r for r in ranks if r != high_pair and r != low_pair], reverse=True)
        kicker = kickers[0] if kickers else 0
        return (3.0 + get_tie_breaker([high_pair, low_pair, kicker])) / 10.0

    # 8. One Pair
    if len(pairs) == 1:
        pair_rank = pairs[0]
        kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
        return (2.0 + get_tie_breaker([pair_rank] + kickers[:3])) / 10.0

    # 9. High Card
    sorted_ranks = sorted(ranks, reverse=True)
    return (1.0 + get_tie_breaker(sorted_ranks[:5])) / 10.0

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

def get_opponent_action(hand_strength, current_bet, opponent_stack, opponent_current_bet, pot):
    amount_to_call = current_bet - opponent_current_bet
    pot_odds = amount_to_call / (pot + amount_to_call) if (pot + amount_to_call) > 0 else 0
    
    can_raise = opponent_stack > amount_to_call + Config.BIG_BLIND
    
    if amount_to_call == 0:
        if hand_strength >= 0.3:
            if can_raise and np.random.rand() < 0.3:
                return 2
        return 1

    if hand_strength >= 0.5:
        if can_raise and np.random.rand() < 0.4:
            return 2
        return 1
        
    if hand_strength >= 0.3:
        if can_raise and np.random.rand() < 0.15:
            return 2
        return 1
        
    if hand_strength >= 0.2:
        if pot_odds < 0.15 or amount_to_call <= Config.BIG_BLIND:
            return 1
        if np.random.rand() < 0.70:
            return 1
        return 0
        
    if pot_odds < 0.10 or amount_to_call <= Config.SMALL_BLIND:
        if np.random.rand() < 0.35:
            return 1
        return 0
    if np.random.rand() < 0.05:
        return 1
    return 0