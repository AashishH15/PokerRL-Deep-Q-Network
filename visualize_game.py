from poker_env import PokerEnv
from dqn_agent import DQNAgent
from utils import decode_cards, load_model, get_valid_actions
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
    current_stacks = [Config.INIT_STACK] * env.num_players
    
    while hand_number <= MAX_HANDS and all(stack >= Config.BIG_BLIND for stack in current_stacks):
        print(f"\n=== Starting Hand #{hand_number}/{MAX_HANDS} ===")
        state = env.reset(player_stacks=current_stacks)
        done = False
        hand_actions = []
        initial_stacks = list(current_stacks)
        
        while not done:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\n=== Poker Hand #{hand_number} ===")
            for i in range(env.num_players):
                p_btn = " [BTN]" if env.button_player == i else ""
                p_label = "Agent" if i == 0 else f"Opponent {chr(64 + i)}"
                folded_str = " (Folded)" if not env.players[i]['active'] else ""
                
                print(f"\nPlayer {i + 1} ({p_label}){p_btn}{folded_str}")
                if env.players[i]['active']:
                    print(f"Hand: {decode_cards(env.players[i]['hand'])}")
                else:
                    print("Hand: [folded]")
                print(f"Total Stack: {env.players[i]['stack']}")
                print(f"Current Bet: {env.players[i]['current_bet']}")
            
            print(f"\nPot: {env.pot}")
            print(f"Community Cards: {decode_cards(env.community_cards)}")
            print(f"Betting Round: {['Preflop', 'Flop', 'Turn', 'River'][env.betting_round]}")
            
            if env.players[0]['active']:
                hand_strength = env.get_hand_strength(state)
                valid_actions = get_valid_actions(env.current_bet, env.players[0]['stack'], env.players[0]['current_bet'])
                action = agent1.act(state, hand_strength, valid_actions)
                
                action_names = ['Fold', 'Call/Check', 'Raise']
                current_action = action_names[action]
                hand_actions.append(f"Player 1: {current_action}")
                print(f"\nPlayer 1 action: {current_action}")
                
                next_state, reward, done, _ = env.step(action)
                state = next_state
            else:
                done = True
            
            time.sleep(1)
        
        final_stacks = [env.players[i]['stack'] for i in range(env.num_players)]
        hand_result = {
            'hand_number': hand_number,
            'community_cards': decode_cards(env.community_cards),
            'pot': env.pot,
            'actions': hand_actions,
            'winner': 'Player 1' if reward > 0 else 'Opponents' if reward < 0 else 'Split Pot',
            'player_hands': [decode_cards(env.players[i]['hand']) for i in range(env.num_players)],
            'player_stack_changes': [final_stacks[i] - initial_stacks[i] for i in range(env.num_players)]
        }
        hand_history.append(hand_result)
        current_stacks = final_stacks
        hand_number += 1

    print("\n=== POKER GAME HISTORY ===")
    total_profits = [0] * env.num_players
    
    for hand in hand_history:
        print(f"\nHand #{hand['hand_number']}:")
        for i in range(env.num_players):
            p_label = "Player 1 (Agent)" if i == 0 else f"Player {i + 1} (Opponent {chr(64 + i)})"
            print(f"{p_label}: {hand['player_hands'][i]}")
        print(f"Community: {hand['community_cards']}")
        print(f"Pot: ${hand['pot']}")
        print(f"Winner: {hand['winner']}")
        
        stack_changes_str = ", ".join(f"P{i+1}: ${hand['player_stack_changes'][i]:+}" for i in range(env.num_players))
        print(f"Stack Changes: {stack_changes_str}")
        print("Actions:")
        for action in hand['actions']:
            print(f"  {action}")
        for i in range(env.num_players):
            total_profits[i] += hand['player_stack_changes'][i]
    
    print("\n=== FINAL RESULTS ===")
    print(f"Total Hands Played: {len(hand_history)}")
    for i in range(env.num_players):
        p_label = "Player 1 (Agent)" if i == 0 else f"Player {i + 1} (Opponent {chr(64 + i)})"
        print(f"{p_label} Final Stack: ${current_stacks[i]} (${total_profits[i]:+})")
        
    for i in range(env.num_players):
        if current_stacks[i] < Config.BIG_BLIND:
            p_label = "Player 1" if i == 0 else f"Player {i + 1}"
            print(f"{p_label} went bankrupt!")
            
    print(f"Overall Winner: {'Player 1' if total_profits[0] > 0 else 'Opponents' if total_profits[0] < 0 else 'Split Pot'}")

if __name__ == "__main__":
    visualize_game()