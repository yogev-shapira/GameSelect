import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from feature_extractor import get_features
from calc_similarity import recommend_games_cosine

CACHE_PATH = "cached_game_features.pkl"


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fh:
            return pickle.load(fh)
    return {}

def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)


app = Flask(__name__)

# CORS(app)
CORS(app, resources={r"/api/*": {"origins": "https://gameselect.onrender.com"}})

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/select-games')
def select_games():
    return render_template('select_games.html')

@app.route('/select-range')
def select_range():
    return render_template('select_range.html')

@app.route('/show-results')
def show_results():
    return render_template('show_results.html')


@app.route('/api/games_by_date')
def games_by_date():
    date = request.args.get('date')
    print("Raw input date:", date)
    if not date:
        return jsonify({"error": "Missing date parameter"}), 400

    try:
        # Parse the date string from frontend
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        target_date = date_obj.date()

        # Load the CSV and parse the mixed formats
        df = pd.read_csv("game_database.csv")
        df["date_parsed"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce").dt.date

        # Safety check
        if df["date_parsed"].isnull().all():
            return jsonify({"error": "Failed to parse any dates in the database"}), 500

        # Match rows by parsed date
        filtered = df[df["date_parsed"] == target_date]
        print(f"Games found for {target_date}: {len(filtered)}")

        if filtered.empty:
            return jsonify({"games": []})

        game_info = filtered[["game_id", "away_team", "home_team", "location", "hour", "away_score", "home_score"]].to_dict(orient="records")
        return jsonify({"games": game_info})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def get_games_in_range(start_date, end_date):
    df = pd.read_csv("game_database.csv")

    # ✅ Safe and correct for "12/05/2025" style
    df["date_parsed"] = pd.to_datetime(df["date"], format='mixed', dayfirst=True, errors='coerce').dt.date

    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    print("Looking for games between", start_date, "and", end_date)
    print("Sample parsed dates:", df["date_parsed"].head())

    mask = (df["date_parsed"] >= start_date) & (df["date_parsed"] <= end_date)
    filtered = df[mask]
    print("Filtered rows:", filtered.shape[0])
    return filtered


def get_game_dicts(game_ids):
    cache = _load_cache()
    dirty = False
    result = []

    for gid in map(str, game_ids):
        if gid not in cache:
            csv_path = f"data/espn_play_by_play_{gid}.csv"
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    cache[gid] = get_features(gid, df)
                    dirty = True
                    print(f"   ↳ features cached on-the-fly for {gid}")
                except Exception as e:
                    print(f"   ⚠️  could not compute features for {gid}: {e}")
            else:
                print(f"   ⚠️  missing CSV for {gid} – skipping")
                continue
        result.append(cache[gid])

    if dirty:
        _save_cache(cache)

    return result


@app.route('/api/recommender', methods=['POST'])
def recommender():
    try:
        data = request.get_json()

        liked_game_ids = data.get('liked_game_ids')
        num_days = int(data.get('days'))
        num_recommendations = int(data.get('games'))

        if num_days is None or num_recommendations is None:
            return jsonify({"error": "Missing liked_game_ids, days, or games in request body"}), 400

        # Define date range
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=num_days)
        print(start_date, end_date)
        # Get all games in range
        filtered_df = get_games_in_range(start_date, end_date)
        all_game_ids = filtered_df["game_id"].tolist()
        print(liked_game_ids)
        print(all_game_ids)
        # Remove liked games from possible recommendations
        candidate_game_ids = [gid for gid in all_game_ids if gid not in liked_game_ids]

        # Extract features
        all_game_dicts = get_game_dicts(candidate_game_ids)
        liked_game_dicts = get_game_dicts(liked_game_ids)

        # Recommend
        recommended_game_ids = recommend_games_cosine(
            all_game_dicts, liked_game_dicts, num_recommendations
        )

        recommended_ids = [str(game["game_id"]) for game in recommended_game_ids]
        print(recommended_ids)

        filtered_df["game_id"] = filtered_df["game_id"].astype(str)

        recommended_games_info = []

        for rid in recommended_ids:
            match_row = filtered_df[filtered_df["game_id"] == rid]
            if not match_row.empty:
                row = match_row.iloc[0]
                recommended_games_info.append({
                    "game_id": row["game_id"],
                    "away_team": row["away_team"],
                    "home_team": row["home_team"],
                    "location": row["location"],
                    "hour": row["hour"],
                    "date": row["date"]
                })

        # Return the results
        return jsonify({"recommended_games": recommended_games_info})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
