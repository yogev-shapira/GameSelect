import os
import csv
import ast
import pickle
import pandas as pd
from datetime import datetime, timedelta

from get_game_ids import get_game_ids_for_range, get_game_info_from_espn, get_game_string
from get_game_data import get_game_data
from feature_extractor import get_features

CACHE_PATH = "cached_game_features.pkl"


def _load_cache() -> dict:
    return pickle.load(open(CACHE_PATH, "rb")) if os.path.exists(CACHE_PATH) else {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)


def extract_final_scores(game_id, pbp_folder):
    pbp_path = os.path.join(pbp_folder, f"espn_play_by_play_{game_id}.csv")
    if not os.path.exists(pbp_path):
        return None, None

    try:
        pbp = pd.read_csv(pbp_path)
        for _, row in pbp[::-1].iterrows():
            event_type = row.get("type")
            if isinstance(event_type, str):
                try:
                    parsed_type = ast.literal_eval(event_type)
                    if isinstance(parsed_type, dict) and parsed_type.get("id") == '402':
                        return row.get("awayScore"), row.get("homeScore")
                except:
                    raise RuntimeError
    except Exception as e:
        print(f"Error reading scores for {game_id}: {e}")
    return None, None


def update_game_db_with_new_games(start_date, end_date,
                                  database_path,
                                  pbp_folder):
    """
    Fetch new game IDs between start_date and end_date, and update the game database CSV
    and play-by-play folder with new games only.

    Parameters:
    - start_date (str): Date in "YYYYMMDD" format.
    - end_date (str): Date in "YYYYMMDD" format.
    - database_path (str): Path to game_database.csv file.
    - pbp_folder (str): Directory where play-by-play files will be stored.
    """

    # Load existing game IDs from database (if exists)
    existing_game_ids = set()
    if os.path.exists(database_path):
        with open(database_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_game_ids.add(row["game_id"])

    feature_cache = _load_cache()
    cache_dirty = False

    # Get new game IDs from ESPN
    new_game_ids = get_game_ids_for_range(start_date, end_date)

    # Prepare to add rows
    new_rows = []

    for game_id in new_game_ids:
        if game_id in existing_game_ids:
            print(f"Skipping existing game {game_id}")
            continue

        try:
            # Get game info and string
            game_info = get_game_info_from_espn(game_id)
            game_string = get_game_string(game_info)

            # Save play-by-play data to folder
            get_game_data(game_id)

            # Extract scores after pbp file is saved
            away_score, home_score = extract_final_scores(game_id, pbp_folder)

            # Build new row
            row = {
                "game_id": game_id,
                "game_date": game_info["date"],
                "home_team": game_info["home_team"],
                "away_team": game_info["away_team"],
                "location": game_info["location"],
                "date": game_info["date"],
                "hour": game_info["hour"],
                "game_string": game_string,
                "away_score": away_score,
                "home_score": home_score
            }

            new_rows.append(row)
            print(f"Added new game {game_id}")

        except Exception as e:
            print(f"Failed to process game {game_id}: {e}")

        # ensure features are cached
        gid_str = str(game_id)
        if gid_str not in feature_cache:
            try:
                pbp_path = os.path.join(pbp_folder, f"espn_play_by_play_{game_id}.csv")
                if os.path.exists(pbp_path):
                    df = pd.read_csv(pbp_path)
                    feature_cache[gid_str] = get_features(game_id, df)
                    cache_dirty = True
                    print(f"   ↳ features cached for {game_id}")
            except Exception as fe:
                print(f"   ⚠️  could not cache features for {game_id}: {fe}")

    # Append new rows to the CSV database
    if new_rows:
        file_exists = os.path.exists(database_path)
        with open(database_path, mode="a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["game_id", "game_date", "home_team",
                                                   "away_team", "location", "date", "hour",
                                                   "game_string", "away_score", "home_score"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rows)
        print(f"Appended {len(new_rows)} new games to database.")
    else:
        print("No new games were added.")

    if cache_dirty:
        _save_cache(feature_cache)
        print(f"Feature cache updated — {len(feature_cache)} games total")


# update_game_database_with_new_games(
#     start_date="20250602",
#     end_date="20250603",
#     database_path=r"C:/Users/User/Desktop/University/year4/semA/FinalProject/game_database.csv",
#     pbp_folder=r"C:/Users/User/Desktop/University/year4/semA/FinalProject/data"
# )


update_game_db_with_new_games(
    start_date="20250611",
    end_date="20250620",
    database_path=r"C:/Users/User/Desktop/University/year4/semA/FinalProject/game_database.csv",
    pbp_folder=r"C:/Users/User/Desktop/University/year4/semA/FinalProject/data"
)