import math
import numpy as np


def find_max_element(elements, rounds, duel, scale_factor=0.51, better_tiebreak=True):
    """
    :param elements: a list to find maximum element
    :param rounds: the number of duels during estimation
    :param duel: a noisy compare function. duel(a, b) = -1 suggests a > b, and duel(a, b) = 1 suggests a < b
    :param scale_factor:
    :param better_tiebreak: D-TS+ algorithm is used if true. if false, D-TS algorithm is used.
    :return: max element estimated by D-TS(+) algorithm
    """
    n = len(elements)
    duel_matrix = np.zeros((n, n))
    duel_matrix = _find_max_element(elements, rounds, duel, scale_factor, better_tiebreak, duel_matrix)
    coperand_scores = np.zeros((n,))
    for i in range(n):
        for j in range(n):
            if duel_matrix[i][j] > duel_matrix[j][i]:
                coperand_scores[i] += 1
    return elements[np.argmax(coperand_scores)]


def __kl_divergence(p, q):
    d = 0
    for pv, qv in zip([p, 1 - p], [q, 1 - q]):
        d += pv * np.log(pv / qv)
    return d


def _find_max_element(elements, rounds, duel, scale_factor, better_tiebreak, past_duels):
    """
    :param elements: a list to find maximum element
    :param rounds: the number of duels during estimation
    :param duel: a noisy compare function. duel(a, b) = -1 suggests a > b, and duel(a, b) = 1 suggests a < b
    :param scale_factor:
    :param better_tiebreak: D-TS+ algorithm is used if true. if false, D-TS algorithm is used.
    :param past_duels: ndarray with shape (n, n). element at (i, j) is the number of element i's win against element j
    :return: tuple of (estimated max element, duel result matrix)
    """
    n = len(elements)
    duel_history = np.zeros((n, n))
    upper_bounds = np.zeros((n, n))
    lower_bounds = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            duel_history[i][j] = past_duels[i][j]

    for _t in range(rounds):
        t = _t + 1
        for i in range(n):
            upper_bounds[i][i] = lower_bounds[i][i] = 1 / 2
            for j in range(n):
                if i == j:
                    continue
                total_matches = duel_history[i][j] + duel_history[j][i]
                if total_matches == 0:
                    center = 1
                    delta = 1
                else:
                    center = duel_history[i][j] / total_matches
                    delta = math.sqrt(scale_factor * math.log(t) / total_matches)
                upper_bounds[i][j] = center + delta
                lower_bounds[i][j] = center - delta

        upper_bound_scores = [len([ub for ub in upper_bound_row if ub > 1 / 2]) for upper_bound_row in upper_bounds]
        max_score = max(upper_bound_scores)
        first_candidates = set(i for i in range(n) if upper_bound_scores[i] == max_score)

        beta_samples = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sampled = np.random.beta(duel_history[i][j] + 1, duel_history[j][i] + 1)
                beta_samples[i][j] = sampled
                beta_samples[j][i] = 1 - sampled
        first_scores = [sum(sampled > 1 / 2 for sampled in row) / (n - 1) for row in beta_samples]
        max_first_score = max([sc for i, sc in enumerate(first_scores) if i in first_candidates])
        first_ties = [i for i in range(n) if first_scores[i] == max_first_score]
        if better_tiebreak:
            regrets = np.zeros((n,))
            for i in range(n):
                for j in range(n):
                    if beta_samples[i][j] == 1 / 2:
                        continue
                    regrets[i] += (max_first_score - (first_scores[i] + first_scores[j]) / 2) / __kl_divergence(beta_samples[i][j], 1 / 2)
            candidate_regrets = [regrets[idx] for idx in first_ties]
            first = first_ties[np.argmin(candidate_regrets)]
        else:
            first = np.random.choice(first_ties)

        second_score = np.zeros((n,))
        for i in range(n):
            if i == first:
                second_score[i] = 1 / 2
            elif lower_bounds[i][first] > 1 / 2:
                second_score[i] = -1
            else:
                second_score[i] = np.random.beta(duel_history[i][first] + 1, duel_history[first][i] + 1)
        second = np.argmax(second_score)

        outcome = duel(elements[first], elements[second])
        if outcome == 1:
            duel_history[first][second] += 1
        else:
            duel_history[second][first] += 1

    return duel_history