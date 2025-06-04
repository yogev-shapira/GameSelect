"""
get_game_data.py

Utility module for fetching and processing NBA play-by-play data from ESPN's public API.

This script is part of the NBA recommendation project, enabling retrieval of raw game events
and conversion into a structured CSV format. It includes:
  - clock_to_seconds(clock_str): Helper to convert time strings to total seconds.
  - get_game_data(game_id): Fetch and process play-by-play data for a specific game ID,
    normalizing period and clock fields, computing a numerical clockSec column, and
    saving the result as a CSV for downstream feature extraction.

Usage example:
    from get_game_data import get_game_data
    # Retrieve and save data for game 401585306
    get_game_data(401585306)

Outputs:
    CSV file at data/espn_play_by_play_<game_id>.csv containing enriched play-by-play data.
"""

import requests
import pandas as pd


def clock_to_seconds(clock_str):
    """
    Convert a clock string (MM:SS or seconds as string/float) to total seconds.

    Args:
        clock_str (str): Time remaining in quarter, formatted as 'MM:SS' or numeric string.
    Returns:
        float or int or None: Total seconds, or None if parsing fails.
    """
    # Ensure input is a string before parsing
    if not isinstance(clock_str, str):
        return None
    # Handle minute:second format
    if ':' in clock_str:
        try:
            minutes, seconds = map(int, clock_str.split(":"))
            return minutes * 60 + seconds
        except:
            # Return None for any split/convert errors
            return None
    # Handle direct numeric strings (e.g., '12.5')
    elif '.' in clock_str:
        try:
            return float(clock_str)
        except:
            # Return None if float conversion fails
            return None


def get_game_data(game_id):
    """
    Fetch play-by-play events for a given NBA game ID from ESPN and save as CSV.

    Steps:
      1. Request summary JSON for the given game ID.
      2. Extract 'plays' list containing individual event dicts.
      3. Convert to pandas DataFrame.
      4. Normalize 'period' and 'clock' fields.
      5. Create 'clockSec' column with total seconds.
      6. Export the DataFrame to 'data/espn_play_by_play_<game_id>.csv'.

    Args:
        game_id (str or int): ESPN game identifier.
    """
    # Construct ESPN summary API URL for play-by-play data
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"

    # Use a common browser User-Agent to avoid potential blocking
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    # Check for successful HTTP response
    if response.status_code == 200:
        data = response.json()

        # Extract list of play dictionaries; key may vary by API version
        plays = data.get("plays", [])  # Adjust key if needed based on JSON structure

        if not plays:
            print(f"No play-by-play data found for game_id {game_id}. Game may have been canceled.")
            return  # Exit early if the game has no plays

        df = pd.DataFrame(plays)

        if df.empty:
            print(f"Play-by-play data is empty for game_id {game_id}.")
            return  # Extra safety check

        # If period info is nested dict, extract the 'number' field
        if 'period' in df.columns:
            df['period'] = df['period'].apply(
                lambda x: x.get('number') if isinstance(x, dict) else x
            )

        # Clean up clock display: extract 'displayValue' if nested
        if 'clock' in df.columns:
            df['clock'] = df['clock'].apply(
                lambda x: x.get('displayValue') if isinstance(x, dict) else x
            )

        # Compute numeric seconds remaining and add as new column
        df['clockSec'] = df['clock'].apply(clock_to_seconds)

        # Save enriched play-by-play DataFrame to CSV for downstream analysis
        df.to_csv(f"data/espn_play_by_play_{game_id}.csv", index=False)
        print("CSV file saved!")
    else:
        # Notify on failure to fetch data
        print("Failed to retrieve data")

