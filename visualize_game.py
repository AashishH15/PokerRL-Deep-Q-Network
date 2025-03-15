from poker_env import PokerEnv
from dqn_agent import DQNAgent
from utils import decode_cards, load_model
from config import Config
import torch
import time
import os

def visualize_game():
    env = PokerEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent1 = DQNAgent(Config.STATE_DIM, Config.ACTION_DIM)
    
    if os.path.exists("best_model.pth"):
        load_model(agent1.policy_net, "best_model.pth")
        agent1.eps = 0
    else:
        print("No trained model found!")
        return

    hand_number = 1
    hand_history = []
    MAX_HANDS = 10
    
    while hand_number <= MAX_HANDS and env.players[0]['stack'] > 0 and env.players[1]['stack'] > 0:
        print(f"\n=== Starting Hand #{hand_number}/{MAX_HANDS} ===")
        state = env.reset()
        done = False
        hand_actions = []
        initial_stacks = [env.players[0]['stack'], env.players[1]['stack']]
        
        while not done:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\n=== Poker Hand #{hand_number} ===")
            print("\nPlayer 1 (Agent)")
            print(f"Hand: {decode_cards(env.players[0]['hand'])}")
            print(f"Total Stack: {env.players[0]['stack']}")
            print(f"Current Bet: {env.players[0]['current_bet']}")
            
            print("\nPlayer 2 (Opponent)")
            print(f"Hand: {decode_cards(env.players[1]['hand'])}")
            print(f"Total Stack: {env.players[1]['stack']}")
            print(f"Current Bet: {env.players[1]['current_bet']}")
            
            print(f"\nPot: {env.pot}")
            print(f"Community Cards: {decode_cards(env.community_cards)}")
            print(f"Betting Round: {['Preflop', 'Flop', 'Turn', 'River'][env.betting_round]}")
            
            hand_strength = env.get_hand_strength(state)
            action = agent1.act(state, hand_strength)
            
            action_names = ['Fold', 'Call/Check', 'Raise']
            current_action = action_names[action]
            hand_actions.append(f"Player 1: {current_action}")
            print(f"\nPlayer 1 action: {current_action}")
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            
            time.sleep(1)
        
        final_stacks = [env.players[0]['stack'], env.players[1]['stack']]
        hand_result = {
            'hand_number': hand_number,
            'player1_hand': decode_cards(env.players[0]['hand']),
            'player2_hand': decode_cards(env.players[1]['hand']),
            'community_cards': decode_cards(env.community_cards),
            'pot': env.pot,
            'actions': hand_actions,
            'winner': 'Player 1' if reward > 0 else 'Player 2',
            'player1_stack_change': final_stacks[0] - initial_stacks[0],
            'player2_stack_change': final_stacks[1] - initial_stacks[1]
        }
        hand_history.append(hand_result)
        hand_number += 1

    print("\n=== POKER GAME HISTORY ===")
    total_p1_profit = 0
    total_p2_profit = 0
    
    for hand in hand_history:
        print(f"\nHand #{hand['hand_number']}:")
        print(f"Player 1: {hand['player1_hand']}")
        print(f"Player 2: {hand['player2_hand']}")
        print(f"Community: {hand['community_cards']}")
        print(f"Pot: ${hand['pot']}")
        print(f"Winner: {hand['winner']}")
        print(f"Stack Changes: P1: ${hand['player1_stack_change']:+}, P2: ${hand['player2_stack_change']:+}")
        print("Actions:")
        for action in hand['actions']:
            print(f"  {action}")
        total_p1_profit += hand['player1_stack_change']
        total_p2_profit += hand['player2_stack_change']
    
    print("\n=== FINAL RESULTS ===")
    print(f"Total Hands Played: {len(hand_history)}")
    print(f"Player 1 Final Stack: ${env.players[0]['stack']} (${total_p1_profit:+})")
    print(f"Player 2 Final Stack: ${env.players[1]['stack']} (${total_p2_profit:+})")
    print(f"Overall Winner: {'Player 1' if total_p1_profit > 0 else 'Player 2'}")

if __name__ == "__main__":
    visualize_game()