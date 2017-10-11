import unittest
import random
import dueling_bandit
from collections import Counter


class Test(unittest.TestCase):
    def test_base_algorithm(self):
        def elo_duel(x, y):
            x_prob = 1 / (1 + 10 ** ((y - x) / 400))
            return 1 if random.random() < x_prob else -1
        counter = Counter()
        for _ in range(100):
            counter[dueling_bandit.find_max_element([1400, 1450, 1500, 1550, 1600], 200, elo_duel, better_tiebreak=False)] += 1
        print(counter)
        self.assertEqual(counter.most_common(1)[0][0], 1600, "failed to estimate max element.")

    def test_plus_algorithm(self):
        def elo_duel(x, y):
            x_prob = 1 / (1 + 10 ** ((y - x) / 400))
            return 1 if random.random() < x_prob else -1
        counter = Counter()
        for _ in range(100):
            counter[dueling_bandit.find_max_element([1400, 1450, 1500, 1550, 1600], 200, elo_duel, better_tiebreak=True)] += 1
        print(counter)
        self.assertEqual(counter.most_common(1)[0][0], 1600, "failed to estimate max element.")


if __name__ == '__main__':
    unittest.main()