import unittest

import numpy as np

from utils import calculate_hand_strength, decode_cards, encode_cards, get_valid_actions


class TestEncodeCards(unittest.TestCase):
    def test_single_card_sets_one_index(self):
        encoded = encode_cards([(12, 0)])  # Ace of hearts
        self.assertEqual(encoded.sum(), 1.0)
        self.assertEqual(encoded[48], 1.0)

    def test_two_cards_sets_two_indices(self):
        encoded = encode_cards([(12, 0), (11, 1)])
        self.assertEqual(encoded.sum(), 2.0)


class TestDecodeCards(unittest.TestCase):
    def test_tuple_cards(self):
        self.assertEqual(decode_cards([(12, 0), (11, 1)]), ["Ah", "Kd"])


class TestGetValidActions(unittest.TestCase):
    def test_fold_and_call_always_available(self):
        actions = get_valid_actions(current_bet=0, player_stack=1000, player_current_bet=0)
        self.assertIn(0, actions)
        self.assertIn(1, actions)

    def test_raise_available_with_stack(self):
        actions = get_valid_actions(current_bet=20, player_stack=1000, player_current_bet=0)
        self.assertIn(2, actions)


class TestHandStrength(unittest.TestCase):
    def test_empty_cards_returns_mid_strength(self):
        self.assertEqual(calculate_hand_strength([], []), 0.5)

    def test_pair_beats_high_card(self):
        pair = calculate_hand_strength([(12, 0), (12, 1)], [])
        high_card = calculate_hand_strength([(12, 0), (8, 2)], [])
        self.assertGreater(pair, high_card)


if __name__ == "__main__":
    unittest.main()
