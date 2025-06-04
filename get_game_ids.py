"""
get_game_ids.py

Utility module for fetching NBA game identifiers and basic game metadata from ESPN's public API.

This script is part of the NBA recommendation project, supporting the collection of game IDs and
retrieval of core game details (teams, venue, date, and time) to feed into downstream analytics
and recommendation pipelines.

Primary functions:
  - get_game_ids_for_date(date): Return a list of ESPN game IDs for a specific date.
  - get_game_ids_for_range(start_date, end_date): Aggregate game IDs across a date range.
  - get_game_info_from_espn(game_id): Fetch home/away teams, location, and scheduled time for a game ID.

Usage examples:
    from get_game_ids import get_game_ids_for_range, get_game_info_from_espn

    # Fetch all game IDs between March 1 and March 25, 2025
    ids = get_game_ids_for_range("20250301", "20250325")
    # Retrieve metadata for the first game
    info = get_game_info_from_espn(ids[0])

"""
import requests
from datetime import datetime, timedelta


def get_game_ids_for_date(date):
    """Fetches game IDs for a specific date from ESPN's API. """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        events = data.get("events", [])
        # Return list of event IDs (game identifiers)
        return [event.get("id") for event in events]

    else:
        print(f"Failed to fetch data for {date}")
        return []


def get_game_ids_for_range(start_date, end_date):
    """Fetches game IDs for a range of dates from ESPN's API. """
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    game_ids = []

    # Iterate from start_date through end_date, inclusive
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        game_ids.extend(get_game_ids_for_date(date_str))
        current_date += timedelta(days=1)

    # Return aggregated game IDs
    return game_ids


def get_game_info_from_espn(game_id):
    """
    Fetches basic info about an NBA game from ESPN using its game ID.

    Parameters:
    - game_id (str or int): ESPN game ID (e.g., 401585306)

    Returns:
    - dict: Contains home_team, away_team, location (only stadium name), date, and hour.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from ESPN for game ID {game_id}")

    data = response.json()

    try:
        competitions = data["header"]["competitions"][0]

        # Determine home and away teams by checking 'homeAway' flag
        teams = {
            "home": competitions["competitors"][0]["team"]["displayName"]
            if competitions["competitors"][0]["homeAway"] == "home"
            else competitions["competitors"][1]["team"]["displayName"],

            "away": competitions["competitors"][0]["team"]["displayName"]
            if competitions["competitors"][0]["homeAway"] == "away"
            else competitions["competitors"][1]["team"]["displayName"]
        }

        # Retrieve venue details, with fallback to 'competitions' venue
        venue_info = data.get("gameInfo", {}).get("venue", {})
        if not venue_info:
            venue_info = competitions.get("venue", {})
        venue_name = venue_info.get("fullName")

        # Use only the venue name without appending city/state
        if not venue_name:
            venue_name = "Unknown Venue"

        # Parse UTC timestamp from ESPN and format date/hour
        date_str = competitions.get("date", "")
        date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

        return {
            "home_team": teams["home"],
            "away_team": teams["away"],
            "location": venue_name,  # only the stadium name now
            "date": date_obj.strftime("%Y-%m-%d"),
            "hour": date_obj.strftime("%H:%M UTC")
        }

    except Exception as e:
        # Propagate parsing errors with context
        raise Exception(f"Error parsing game info: {e}")


def get_game_string(game_info):
    """
    Converts a game info dictionary into a nicely formatted string.

    Parameters:
    - game_info (dict): Dictionary with keys 'home_team', 'away_team', 'location', 'date', and 'hour'.

    Returns:
    - str: A formatted string like "AwayTeam vs HomeTeam @ Location, Date, Hour"
    """
    try:
        return (f"{game_info['away_team']} vs {game_info['home_team']} @ "
                f"{game_info['location']}, {game_info['date']} {game_info['hour']}")
    except KeyError as e:
        raise ValueError(f"Missing expected key in game_info: {e}")


def get_games_info_by_date(game_date):
    game_strs = []
    game_ids = get_game_ids_for_date(game_date)
    for game_id in game_ids:
        game_info = get_game_info_from_espn(game_id)
        game_strs.append((game_id, get_game_string(game_info)))
    return game_strs

# Uncomment below for quick testing or example usage
# ids = get_game_ids_for_date("20250419")
# print(ids)
# for gid in ids:
#     print(get_game_info_from_espn(gid))
# print(get_game_ids_for_range("20250411", "20250420" ))
# ids = get_game_ids_for_range("20241015","20251022")
# print(ids)
