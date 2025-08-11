"""
calc_similarity.py

Utilities for computing similarity between NBA game feature dictionaries. Combines cosine
similarity on normalized numeric features with weighted overlaps on categorical lists
(e.g., top_players, teams, and per‑team top players) to produce a single hybrid score.

Key components:
- top_players_similarity_weighted: Frequency‑weighted overlap of candidate.top_players vs. liked games.
- teams_similarity_weighted: Frequency‑weighted team overlap across liked games.
- team_top_players_overlap_weighted: Per‑team top‑player overlap weighted by frequency.
- calc_cosine_similarity: Hybrid scorer = cosine(numerics) + weighted categorical overlaps.
- excitement_score: Heuristic fallback when no liked games are provided.
- recommend_games_cosine: Ranks candidates by average similarity to all liked games.
- recommend_games_cosine_max: Ranks candidates by best match to any single liked game (eval baseline).

Dependencies: numpy, pandas
"""
import numpy as np
import pandas as pd
from feature_extractor import NUM_OF_FEATURES, NUM_OF_NON_NUMERIC_FEATURES, NUM_OF_TOP_PLAYERS


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

    #calculating cosine similarity on the numeric feature vectors 
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    combined_sim = ((1 - NUM_OF_NON_NUMERIC_FEATURES) * alpha) * cosine_sim \
                   + alpha * players_sim \
                   + alpha * teams_sim \
                   + alpha * team1_overlap \
                   + alpha * team2_overlap

    return combined_sim


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
        (1 - game.get("three_pt_count", 1)) +
        game.get("dunk_count", 0) +
        game.get("block_count", 0) +
        game.get("misses_count", 0) +
        game.get("star_score", 0) +
        (1 - game.get("close_score", 1)) +  # Lower close_score means closer game
        game.get("density_score", 0)
    )


def recommend_games_cosine(all_games, liked_games, num=3):
    """
    Recommend games by averaging cosine similarity scores between each candidate 
    and all liked games, also factoring in weighted team and player overlaps.
    Falls back to excitement score if no liked games are given.

    Args:
        all_games (list of dict): List of game feature dictionaries from last certain range of time.
        liked_games (list of dict): List of liked game feature dictionaries.
        num (int): Number of recommendations to return.

    Returns:
        list of dict: Top `num` most similar game dictionaries.
    """
    if not all_games:
        return []

    # Cold-start case: no liked games → use excitement score ranking
    if not liked_games:
        scored_games = [(excitement_score(game), game) for game in all_games]
        scored_games.sort(reverse=True, key=lambda x: x[0])
        return [game for _, game in scored_games[:num]]

    similarity_scores = []

    for candidate_game in all_games:
        # Compute weighted similarities for categorical features
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
        
        # Average similarity with all liked games
        total_similarity = sum(
            calc_cosine_similarity(candidate_game, liked_game, players_sim, teams_sim, team1_overlap, team2_overlap)
            for liked_game in liked_games
        )
        avg_similarity = total_similarity / len(liked_games)

        similarity_scores.append((avg_similarity, candidate_game))

    # Rank candidates by highest average similarity
    similarity_scores.sort(reverse=True, key=lambda x: x[0])
    return [game for _, game in similarity_scores[:num]]


def recommend_games_cosine_max(all_games, liked_games, num=3):
    """
    Recommends games from all_games based on the highest cosine similarity
    between each candidate game and any of the liked_games.
    used as evaluation method.
    Args:
        all_games (list of dict): List of game feature dictionaries from a recent range of time.
        liked_games (list of dict): List of liked game feature dictionaries.
        num (int): Number of recommendations to return.

    Returns:
        list of dict: Top `num` most similar game dictionaries.
    """
    if not all_games:
        return []

    # Cold-start case: no liked games → use excitement score ranking
    if not liked_games:
        scored_games = [(excitement_score(game), game) for game in all_games]
        scored_games.sort(reverse=True, key=lambda x: x[0])
        return [game for _, game in scored_games[:num]]

    similarity_scores = []
  
    for candidate_game in all_games:
        # Compute weighted similarities for categorical features
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

        # Use the single highest similarity to any liked game
        max_similarity = max(
            calc_cosine_similarity(candidate_game, liked_game, players_sim, teams_sim, team1_overlap, team2_overlap)
            for liked_game in liked_games
        )
        similarity_scores.append((max_similarity, candidate_game))
    
    # Rank candidates by highest max similarity
    similarity_scores.sort(reverse=True, key=lambda x: x[0])
    return [game for _, game in similarity_scores[:num]]
