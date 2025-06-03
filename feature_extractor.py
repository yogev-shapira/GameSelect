"""
NBA Game Excitement Feature Extractor Module

This script is designed for the NBA game recommendation project.
It processes detailed play-by-play data for a single game and extracts key excitement and performance metrics,
including lead changes, 3-point makes, dunks, blocks, scoring density,
and closeness factors, as well as identifying top participating players. These normalized features can be used by the
recommendation engine to rank and suggest the most thrilling games for users.

Usage:
    python feature_extractor.py <game_id> <csv_file_path>

Outputs a dictionary of normalized metrics and identifiers, ready for ingestion by downstream recommendation components.
"""

import sys
import pandas as pd
import ast
from collections import Counter

NUM_OF_FEATURES = 12
NUM_OF_NON_NUMERIC_FEATURES = 4
NUM_OF_TOP_PLAYERS = 5

# Maximum values used for normalization of each metric
DUNK_COUNT_MAX = 30
BLOCK_COUNT_MAX = 30
LEAD_CHANGE_MAX = 35
MISS_COUNT_MAX = 170
THREE_PT_MAX = 50
CLOSE_GAME_MAX = 40
STAR_SCORE_MAX = 150


def count_lead_changes(game_data):
    """
    Count how many times the lead changes between home and away teams over the entire game.

    Args:
        game_data (DataFrame): Play-by-play data with 'homeScore' and 'awayScore' columns.
    Returns:
        int: Number of lead changes.
    """
    previous_leader = None
    lead_changes = 0
    # Iterate through each event row to detect changes in leader
    for _, row in game_data.iterrows():
        current_leader = "Away" if row['awayScore'] > row['homeScore'] else "Home"
        if current_leader != previous_leader:
            lead_changes += 1
        previous_leader = current_leader
    return lead_changes


def count_3pt_makes(game_data):
    """
    Count the total number of made 3-point shots in the game.

    Args:
        game_data (DataFrame): Play-by-play data with 'scoringPlay' and 'scoreValue' columns.
    Returns:
        int: Number of successful 3-point shots.
    """
    # Filter rows where a scoring play of value '3' occurs
    return len(game_data[(game_data['scoringPlay']) & (game_data['scoreValue'] == 3)])



def count_dunk_shots(game_data):
    """
    Count how many dunk attempts occurred, based on play type IDs.

    Args:
        game_data (DataFrame): Play-by-play data with 'type' column storing stringified dicts.
    Returns:
        int: Total number of dunks.
    """
    # Set of dunk-specific event type IDs
    dunk_type_ids = {'96', '115', '116', '118', '138', '150', '151', '152'}

    # Evaluate each 'type' entry, parse literal, and check ID membership
    num_dunks = sum(
        ast.literal_eval(x)['id'] in dunk_type_ids
        for x in game_data['type']
    )

    return num_dunks


def count_blocks(game_data):
    """
    Count number of block events by searching the description text.

    Args:
        game_data (DataFrame): Play-by-play data with 'text' descriptions.
    Returns:
        int: Number of blocks.
    """
    # Case-insensitive search for the word 'blocks' in play descriptions
    return game_data['text'].str.lower().str.contains("blocks").sum()

def count_misses(game_data):
    """
        Count the number of missed field goal attempts in the game.

        A missed shot is defined as a play where a shooting attempt was made
        ('shootingPlay' is True) but the shot was not successful ('scoringPlay' is False).

        Args:
            game_data (DataFrame): Play-by-play data containing 'shootingPlay' and 'scoringPlay' boolean columns.

        Returns:
            int: Total number of missed shots.
        """
    return len( game_data[(game_data['shootingPlay']) & (~game_data['scoringPlay'])] )


def calculate_density_score(game_data):
    """
    Compute scoring density: points scored per second between made shots, averaged across quarters.

    Args:
        game_data (DataFrame): Play-by-play data with 'scoringPlay', 'scoreValue', 'period', 'clockSec'.
    Returns:
        float: Average density score across quarters.
    """
    # Select only successful scoring plays
    made_shots = game_data[game_data['scoringPlay']].copy()

    # Group by quarter
    grouped_by_quarter = made_shots.groupby('period')

    density_scores = []
    for _, quarter_data in grouped_by_quarter:
        # Sort in chronological order (reverse clock)
        quarter_data = quarter_data.sort_values(by='clockSec', ascending=False).copy()
        # Compute time between consecutive shots
        quarter_data['TimeDiff'] = quarter_data['clockSec'].diff(periods=-1).fillna(0).abs()
        total_time_between_shots = quarter_data['TimeDiff'].sum()
        # Sum total points in the quarter
        total_points = quarter_data['scoreValue'].sum()
        if total_time_between_shots > 0:
            density_scores.append(total_points / total_time_between_shots)

    # Return average density or 0 if no scoring data
    return sum(density_scores) / len(density_scores) if density_scores else 0


def determine_close_game(game_data):
    """
    Measure how close the game was at the end of each period, including overtimes.

    Args:
        game_data (DataFrame): Play-by-play data with 'homeScore', 'awayScore', and 'type' column as dicts.
    Returns:
        float: Average point differential at end of periods; lower means closer game.
    """
    close_factors = []

    for _, row in game_data.iterrows():
        event_type = row.get('type')

        # Safely parse string if necessary
        if isinstance(event_type, str):
            try:
                event_type = ast.literal_eval(event_type)
            except Exception:
                continue  # Skip invalid entries
        # Checks cells with event of End Period
        if isinstance(event_type, dict) and event_type.get('id') == '412':
            home_score = row.get('homeScore')
            away_score = row.get('awayScore')
            if pd.notnull(home_score) and pd.notnull(away_score):
                close_factors.append(abs(home_score - away_score))

    return sum(close_factors) / len(close_factors) if close_factors else float('inf')


# Legacy regex-based method for extracting players (commented out)
"""
def extract_top_players(game_data, top_n=3):
    player_pattern = re.compile(r'\b([A-Z][a-z]+\s[A-Z][a-z]+)\b')  # Basic pattern for "First Last"
    player_mentions = []

    for text in game_data['text']:
        matches = player_pattern.findall(text)
        player_mentions.extend(matches)

    counter = Counter(player_mentions)
    return counter.most_common(top_n)
"""


def get_top_players(game_data, top_n=NUM_OF_TOP_PLAYERS):
    """
    Identify most frequently participating players based on 'participants' entries.

    Args:
        game_data (DataFrame): Play-by-play data with 'participants' column of lists.
        top_n (int): Number of top players to return.
    Returns:
        list: Tuples of (player_id, mention_count).
    """
    player_counter = Counter()

    # Iterate participant lists and count athlete IDs
    for participants in game_data['participants']:
        if pd.isna(participants):
            continue
        try:
            parsed_participants = ast.literal_eval(participants)
        except (ValueError, SyntaxError):
            continue

        if not isinstance(parsed_participants, list):
            continue

        for entry in parsed_participants:
            athlete = entry.get('athlete', {})
            athlete_id = athlete.get('id')
            if athlete_id:
                player_counter[athlete_id] += 1

    return player_counter.most_common(top_n)


def compute_star_score(top_players):
    """
    Compute a combined 'star' impact score by summing mention frequencies of top players.

    Args:
        top_players (list): Output from extract_top_players.
    Returns:
        int: Sum of mention counts.
    """
    return sum(freq for _, freq in top_players)


def get_team_ids(game_data):
    """
    Returns the first two unique team IDs from the 'team' column of the DataFrame.

    Assumes each entry in 'team' is a dict like {'id': '1'}.
    """
    seen_ids = []
    for raw_team in game_data['team']:
        team_dict = ast.literal_eval(raw_team) if isinstance(raw_team, str) else raw_team
        if isinstance(team_dict, dict):
            team_id = team_dict['id']
            if team_id not in seen_ids:
                seen_ids.append(team_id)
            if len(seen_ids) == 2:
                break

    return seen_ids


# TODO: check modification to include edge cases of block assists
def get_top_players_by_team(game_data, team_id, top_n=NUM_OF_TOP_PLAYERS):
    """
    Returns the top players from a given team based only on the first participant in each play.

    Args:
        game_data (DataFrame): Play-by-play data with 'team' and 'participants' columns.
                               'team' should be a dict like {'id': '1'}, and 'participants' a list of dicts.
        team_id (str): The ID of the team to focus on.
        top_n (int): Number of top players to return.

    Returns:
        list: Tuples of (player_id, mention_count), sorted by frequency.
    """
    player_counter = Counter()

    for _, row in game_data.iterrows():
        try:
            team = ast.literal_eval(row['team'])
            participants = ast.literal_eval(row['participants'])
        except (ValueError, SyntaxError):
            continue  # skip malformed rows

        if not isinstance(team, dict) or team.get('id') != team_id:
            continue

        if isinstance(participants, list) and participants:
            first = participants[0]
            if isinstance(first, dict) and 'athlete' in first and 'id' in first['athlete']:
                athlete_id = first['athlete']['id']
                player_counter[athlete_id] += 1

    return player_counter.most_common(top_n)


def normalize(value, min_value=0, max_value=1):
    """
    Scale a raw metric to a 0-1 range.

    Args:
        value (float): Raw metric.
        min_value (float): Expected minimum for normalization.
        max_value (float): Expected maximum for normalization.
    Returns:
        float: Normalized value between 0 and 1.
    """
    return (value - min_value) / (max_value - min_value) if max_value > min_value else 0


def get_features(game_id, game_data):
    # Extract raw features
    lead_changes = count_lead_changes(game_data)
    three_pt_count = count_3pt_makes(game_data)
    dunk_count = count_dunk_shots(game_data)
    block_count = count_blocks(game_data)
    misses_count = count_misses(game_data)
    density_score = calculate_density_score(game_data)
    close_score = determine_close_game(game_data)

    # Identify top players and compute star metric
    top_players = get_top_players(game_data)
    star_score = compute_star_score(top_players)

    team1_id, team2_id = get_team_ids(game_data)

    top_players1 = get_top_players_by_team(game_data, team1_id)
    top_players2 = get_top_players_by_team(game_data, team2_id)

    # Prepare normalized feature dictionary for recommendation input
    param_dict = {
        "game_id": game_id,
        "lead_changes": normalize(lead_changes, 0, LEAD_CHANGE_MAX),
        "three_pt_count": normalize(three_pt_count, 0, THREE_PT_MAX),
        "dunk_count": normalize(dunk_count, 0, DUNK_COUNT_MAX),
        "block_count": normalize(block_count, 0, BLOCK_COUNT_MAX),
        "misses_count": normalize(misses_count, 0, MISS_COUNT_MAX),

        "density_score": normalize(density_score),
        "close_score": normalize(close_score, 0, CLOSE_GAME_MAX),

        "star_score": normalize(star_score, 0, STAR_SCORE_MAX),
        "top_players": [player for player, _ in top_players],

        "teams": [team1_id, team2_id],
        "top_players1": [player for player, _ in top_players1],
        "top_players2": [player for player, _ in top_players2]
    }

    return param_dict


if __name__ == "__main__":
    # Load the game ID and CSV file path from command-line args
    game_id, file_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(file_path)

    get_features(game_id, df)



