"""
calc_similarity.py

This module provides functions for computing similarity and distance between game data
dictionaries based on both numeric and non-numeric features. It supports two core comparison
metrics: cosine similarity and Euclidean distance. Additionally, it includes support for
handling list-based categorical features like 'top_players' using Jaccard-based similarity
and distance functions.

Key Components:
- top_players_similarity: Measures overlap of top players between two games.
- top_players_distance: Jaccard-style distance for top player lists.
- calc_cosine_similarity: Hybrid similarity score using cosine for numeric features and
  player overlap for non-numeric.
- calc_euclidean_distance: Hybrid distance score using Euclidean for numeric features and
  Jaccard distance for top players.
- recommend_games_cosine: Game recommender based on cosine similarity.
- recommend_games_euclidean: Game recommender based on Euclidean distance.

This module is designed to be extensible for future inclusion of additional non-numeric
or categorical features (e.g., teams, locations, player stats).

Dependencies:
- numpy
- pandas
"""
import numpy as np
import pandas as pd
from feature_extractor import NUM_OF_FEATURES, NUM_OF_NON_NUMERIC_FEATURES, NUM_OF_TOP_PLAYERS


def top_players_similarity(dict1, dict2):
    """
    Returns the number of player IDs in dict1['top_players'] that also appear in dict2['top_players'].

    Parameters:
        dict1 (dict): A dictionary containing a 'top_players' key with a list of player IDs.
        dict2 (dict): A dictionary containing a 'top_players' key with a list of player IDs.

    Returns:
        int: The count of player IDs in dict1['top_players'] that are also in dict2['top_players'].
    """
    players1 = set(dict1.get("top_players", []))
    players2 = set(dict2.get("top_players", []))

    if not players1 or not players2:
        return 0

    return len(players1 & players2) / NUM_OF_TOP_PLAYERS


def top_players_similarity_weighted(candidate, liked_games):
    """
    Calculates weighted overlap of top players in the candidate game
    with frequency counts from all liked games.
    """
    # Count total appearances of each player
    from collections import Counter

    total_count = 0
    player_counter = Counter()
    for game in liked_games:
        for p in game.get("top_players", []):
            player_counter[p] += 1
            total_count += 1

    if total_count == 0:
        return 0.0

    # Sum frequencies for candidate's players
    candidate_players = candidate.get("top_players", [])
    score = sum(player_counter[p] for p in candidate_players)

    return score / total_count


def teams_similarity(dict1, dict2):
    teams1 = set(dict1.get("teams", []))
    teams2 = set(dict2.get("teams", []))

    if not teams1 or not teams2:
        return 0

    return len(teams1 & teams2) / 2


def teams_similarity_weighted(candidate, liked_games):
    """
    Calculates weighted overlap of teams in the candidate game
    with frequency counts from all liked games.
    """
    from collections import Counter

    total_count = 0
    team_counter = Counter()
    for game in liked_games:
        for t in game.get("teams", []):
            team_counter[t] += 1
            total_count += 1

    if total_count == 0:
        return 0.0

    candidate_teams = candidate.get("teams", [])
    score = sum(team_counter[t] for t in candidate_teams)

    return score / total_count


def team_top_players_overlap(top_players_team, other_top_players1, other_top_players2):
    """
    Calculates overlap between a team's top players and the other game's top players.

    Returns:
        float: Fraction of players from top_players_team that appear in the union of the other game's top players.
    """
    players_team = set(top_players_team)
    others = set(other_top_players1) | set(other_top_players2)

    if not players_team or not others:
        return 0.0

    return len(players_team & others) / NUM_OF_TOP_PLAYERS


def team_top_players_overlap_weighted(candidate_team_top_players, liked_games):
    """
    Calculates weighted overlap between a candidate team's top players and
    the aggregate top players from all liked games.

    Returns:
        float: Fractional overlap score, weighted by player frequencies.
    """
    from collections import Counter

    # Flatten and count all top players from both teams in liked games
    player_counter = Counter()
    total_count = 0

    for game in liked_games:
        for p in game.get("top_players1", []):
            player_counter[p] += 1
            total_count += 1
        for p in game.get("top_players2", []):
            player_counter[p] += 1
            total_count += 1

    if total_count == 0:
        return 0.0

    # Count appearances of this candidate team's top players
    score = sum(player_counter[p] for p in candidate_team_top_players)

    return score / total_count


# def top_players_distance(dict1, dict2):
#     """
#     Returns a distance metric for 'top_players' feature:
#     1.0 means no overlap, 0.0 means perfect overlap.
#     """
#     players1 = set(dict1.get("top_players", []))
#     players2 = set(dict2.get("top_players", []))
#
#     if not players1 or not players2:
#         return 1.0  # Max distance if one of the lists is empty
#
#     intersection = len(players1 & players2)
#     union = len(players1 | players2)
#     return 1.0 - (intersection / union)  # Jaccard distance


def calc_cosine_similarity(param_dict1, param_dict2,
                           players_sim, teams_sim, team1_overlap, team2_overlap, alpha=1/NUM_OF_FEATURES):

    """
    Calculates a combined similarity score between two game parameter dictionaries,
    using cosine similarity for numeric features and Jaccard-style overlap for top_players.

    Args:
        param_dict1 (dict): Feature dictionary of first game.
        param_dict2 (dict): Feature dictionary of second game.
        alpha (float): Weight given to numeric features [0.0–1.0]. The rest is for non-numeric features.

    Returns:
        float: Combined similarity score between 0 and 1.
    """
    # Keys to exclude from numeric cosine similarity
    excluded_keys = {"game_id", "top_players", "teams", "top_players1", "top_players2"}

    # --- Cosine Similarity on numeric features ---
    numeric_keys = [k for k in param_dict1.keys() if k not in excluded_keys]
    filtered_dict1 = {k: param_dict1[k] for k in numeric_keys}
    filtered_dict2 = {k: param_dict2.get(k, 0) for k in numeric_keys}

    series1 = pd.Series(filtered_dict1)
    series2 = pd.Series(filtered_dict2)
    vec1, vec2 = series1.to_numpy(), series2.to_numpy()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    # # --- Similarity on 'top_players' ---
    # players_sim = top_players_similarity(param_dict1, param_dict2)
    # teams_sim = teams_similarity(param_dict1, param_dict2)
    #
    # # --- Team-specific top player overlap ---
    # team1_overlap = team_top_players_overlap(
    #     param_dict1.get("top_players1", []),
    #     param_dict2.get("top_players1", []),
    #     param_dict2.get("top_players2", [])
    # )
    # team2_overlap = team_top_players_overlap(
    #     param_dict1.get("top_players2", []),
    #     param_dict2.get("top_players1", []),
    #     param_dict2.get("top_players2", [])
    # )

    # --- Combined similarity ---
    # TODO: Magic number = 4 non-numeric features now

    combined_sim = ((1 - NUM_OF_NON_NUMERIC_FEATURES) * alpha) * cosine_sim \
                   + alpha * players_sim \
                   + alpha * teams_sim \
                   + alpha * team1_overlap \
                   + alpha * team2_overlap

    return combined_sim


# def calc_euclidean_distance(param_dict1, param_dict2, alpha= 1 / NUM_OF_FEATURES):
#     """
#     Compute a combined distance between two game parameter dictionaries,
#     using Euclidean distance for numeric features and a custom distance for 'top_players'.
#
#     Args:
#         param_dict1 (dict): First dictionary of parameters.
#         param_dict2 (dict): Second dictionary of parameters.
#         alpha (float): Weight for numeric distance [0.0–1.0]. The rest is for non-numeric.
#
#     Returns:
#         float: Combined distance metric between 0 and 1 (not strictly Euclidean).
#     """
#     excluded_keys = {"game_id", "top_players"}
#
#     # --- Numeric part ---
#     numeric_keys = [k for k in param_dict1 if k not in excluded_keys]
#     filtered_dict1 = {k: param_dict1[k] for k in numeric_keys}
#     filtered_dict2 = {k: param_dict2.get(k, 0) for k in numeric_keys}
#
#     vec1 = np.array([filtered_dict1[k] for k in numeric_keys])
#     vec2 = np.array([filtered_dict2[k] for k in numeric_keys])
#     euclidean_dist = np.linalg.norm(vec1 - vec2)
#
#     # --- Non-numeric part (top_players) ---
#     players_dist = top_players_distance(param_dict1, param_dict2)
#
#     # --- Combine distances ---
#     combined_dist = (1 - alpha) * euclidean_dist + alpha * players_dist
#     return combined_dist


def excitement_score(game):
    """
    Computes a heuristic score for a game based on its exciting features.
    Used when there are no liked games to compare against.

    Args:
        game (dict): Normalized feature dictionary of a game.

    Returns:
        float: Composite excitement score.
    """
    return (
        game.get("lead_changes", 0) +
        game.get("three_pt_count", 0) +
        game.get("dunk_count", 0) +
        game.get("block_count", 0) +
        game.get("star_score", 0) +
        (1 - game.get("close_score", 1)) +  # Lower close_score means closer game
        game.get("density_score", 0)
    )


def recommend_games_cosine(all_games, liked_games, num=3):
    """
    Recommends games from all_games based on cosine similarity to liked_games.

    Args:
        all_games (list of dict): List of game feature dictionaries from last certain range of time.
        liked_games (list of dict): List of liked game feature dictionaries.
        num (int): Number of recommendations to return.

    Returns:
        list of dict: Top `num` most similar game dictionaries.
    """
    if not all_games:
        return []

    # If player doesn't have favourite games, recommends based on excitement score.
    if not liked_games:
        scored_games = [(excitement_score(game), game) for game in all_games]
        scored_games.sort(reverse=True, key=lambda x: x[0])
        return [game for _, game in scored_games[:num]]

    similarity_scores = []

    for candidate_game in all_games:
        players_sim = top_players_similarity_weighted(candidate_game, liked_games)
        teams_sim = teams_similarity_weighted(candidate_game, liked_games)
        team1_overlap = team_top_players_overlap_weighted(
            candidate_game.get("top_players1", []),
            liked_games
        )
        team2_overlap = team_top_players_overlap_weighted(
            candidate_game.get("top_players2", []),
            liked_games
        )

        total_similarity = sum(
            calc_cosine_similarity(candidate_game, liked_game, players_sim, teams_sim, team1_overlap, team2_overlap)
            for liked_game in liked_games
        )
        avg_similarity = total_similarity / len(liked_games)

        similarity_scores.append((avg_similarity, candidate_game))

    similarity_scores.sort(reverse=True, key=lambda x: x[0])

    return [game for _, game in similarity_scores[:num]]


def recommend_games_cosine_max(all_games, liked_games, num=3):
    """
    Recommends games from all_games based on the highest cosine similarity
    between each candidate game and any of the liked_games.

    Args:
        all_games (list of dict): List of game feature dictionaries from a recent range of time.
        liked_games (list of dict): List of liked game feature dictionaries.
        num (int): Number of recommendations to return.

    Returns:
        list of dict: Top `num` most similar game dictionaries.
    """
    if not all_games:
        return []

    # If player doesn't have favourite games, recommend based on excitement score.
    if not liked_games:
        scored_games = [(excitement_score(game), game) for game in all_games]
        scored_games.sort(reverse=True, key=lambda x: x[0])
        return [game for _, game in scored_games[:num]]

    similarity_scores = []

    for candidate_game in all_games:

        players_sim = top_players_similarity_weighted(candidate_game, liked_games)
        teams_sim = teams_similarity_weighted(candidate_game, liked_games)
        team1_overlap = team_top_players_overlap_weighted(
            candidate_game.get("top_players1", []),
            liked_games
        )
        team2_overlap = team_top_players_overlap_weighted(
            candidate_game.get("top_players2", []),
            liked_games
        )

        # Compute the maximum similarity to any liked game
        max_similarity = max(
            calc_cosine_similarity(candidate_game, liked_game, players_sim, teams_sim, team1_overlap, team2_overlap)
            for liked_game in liked_games
        )
        similarity_scores.append((max_similarity, candidate_game))

    similarity_scores.sort(reverse=True, key=lambda x: x[0])

    return [game for _, game in similarity_scores[:num]]


def recommend_games_cosine_top_k(all_games, liked_games, num=3, k=3):
    """
    Recommends games based on average of top-k cosine similarities to liked_games.

    Args:
        all_games (list of dict): Games to recommend from.
        liked_games (list of dict): Games the user liked.
        num (int): Number of games to recommend.
        k (int): Number of most similar liked games to use for scoring.

    Returns:
        list of dict: Recommended games.
    """
    if not all_games:
        return []

    if not liked_games:
        scored_games = [(excitement_score(game), game) for game in all_games]
        scored_games.sort(reverse=True, key=lambda x: x[0])
        return [game for _, game in scored_games[:num]]

    similarity_scores = []

    for candidate_game in all_games:
        sims = []

        players_sim = top_players_similarity_weighted(candidate_game, liked_games)
        teams_sim = teams_similarity_weighted(candidate_game, liked_games)
        team1_overlap = team_top_players_overlap_weighted(candidate_game.get("top_players1", []), liked_games)
        team2_overlap = team_top_players_overlap_weighted(candidate_game.get("top_players2", []), liked_games)

        for liked_game in liked_games:
            sim = calc_cosine_similarity(
                candidate_game, liked_game,
                players_sim, teams_sim, team1_overlap, team2_overlap
            )
            sims.append(sim)

        sims.sort(reverse=True)
        avg_top_k = sum(sims[:k]) / min(k, len(sims))
        similarity_scores.append((avg_top_k, candidate_game))

    similarity_scores.sort(reverse=True, key=lambda x: x[0])
    return [game for _, game in similarity_scores[:num]]


