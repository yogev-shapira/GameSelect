import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from feature_extractor import get_features
from calc_similarity import recommend_games_cosine, recommend_games_cosine_max, recommend_games_cosine_top_k
import random


def recommend_games_random(all_games, num):
    return random.sample(all_games, min(num, len(all_games)))


responses_df = pd.read_csv("survey_responses_with_ids.csv")


def precompute_game_features(game_db_path="game_database.csv"):
    last_week_file = "cached_features_last_week.pkl"
    liked_range_file = "cached_features_liked_range.pkl"

    # Load from cache if both files exist
    if os.path.exists(last_week_file) and os.path.exists(liked_range_file):
        with open(last_week_file, 'rb') as f1, open(liked_range_file, 'rb') as f2:
            last_week_features = pickle.load(f1)
            liked_range_features = pickle.load(f2)
        print("✅ Loaded cached feature dictionaries.")
        return last_week_features, liked_range_features

    # Otherwise, compute from scratch
    games_df = pd.read_csv(game_db_path)
    games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')

    last_week_start = datetime(2025, 4, 27)
    last_week_end = datetime(2025, 5, 3)
    liked_range_start = datetime(2025, 3, 16)
    liked_range_end = datetime(2025, 4, 26)

    last_week_features = {}
    liked_range_features = {}

    for _, row in games_df.iterrows():
        game_id = str(row['game_id'])
        game_date = row['date']

        if not liked_range_start <= game_date <= last_week_end:
            continue

        path = f"data/espn_play_by_play_{game_id}.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        features = get_features(game_id, df)
        if last_week_start <= game_date <= last_week_end:
            last_week_features[game_id] = features
        elif liked_range_start <= game_date <= liked_range_end:
            liked_range_features[game_id] = features

    # Save to cache
    with open(last_week_file, 'wb') as f1, open(liked_range_file, 'wb') as f2:
        pickle.dump(last_week_features, f1)
        pickle.dump(liked_range_features, f2)

    print("✅ Feature dictionaries computed and cached.")
    return last_week_features, liked_range_features

def extract_liked_games(row, liked_game_dict):
    liked_games = []
    for cell in row[2:8]:  # Columns 2–7
        if pd.isna(cell):
            continue
        game_ids = [g.strip() for g in str(cell).split(',') if g.strip().upper() != "UNKNOWN"]
        for game_id in game_ids:
            if game_id in liked_game_dict:
                liked_games.append(liked_game_dict[game_id])
    return liked_games

def compute_recall_at_k(recommended, relevant, k):
    if not relevant or k == 0:
        return 0.0
    hits = sum([1 for g in recommended[:k] if g in relevant])
    return hits / len(relevant)

def compute_ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i in range(min(k, len(recommended))):
        if recommended[i] in relevant:
            dcg += 1 / np.log2(i + 2)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation():
    last_week_dict, liked_dict = precompute_game_features()
    all_games = list(last_week_dict.values())
    results = []

    for index, row in responses_df.iterrows():
        liked_games = extract_liked_games(row, liked_dict)
        if not liked_games:
            continue

        relevant_ids = []
        if pd.notna(row.iloc[1]):
            relevant_ids = [g.strip() for g in str(row.iloc[1]).split(',') if g.strip().upper() != "UNKNOWN"]

        R = len(relevant_ids)
        # recommendations = recommend_games_cosine(all_games, liked_games, num=17)
        # recommendations = recommend_games_random(all_games, 17)
        recommendations = recommend_games_cosine(all_games, [], num=17)

        recommended_ids = [g['game_id'] for g in recommendations]

        recall_at_3 = compute_recall_at_k(recommended_ids, relevant_ids, 3)
        recall_at_5 = compute_recall_at_k(recommended_ids, relevant_ids, 5)
        recall_at_10 = compute_recall_at_k(recommended_ids, relevant_ids, 10)
        recall_at_R = compute_recall_at_k(recommended_ids, relevant_ids, R)

        ndcg_at_1 = compute_ndcg_at_k(recommended_ids, relevant_ids, 1)
        ndcg_at_3 = compute_ndcg_at_k(recommended_ids, relevant_ids, 3)
        ndcg_at_5 = compute_ndcg_at_k(recommended_ids, relevant_ids, 5)
        ndcg_at_10 = compute_ndcg_at_k(recommended_ids, relevant_ids, 10)
        ndcg_at_R = compute_ndcg_at_k(recommended_ids, relevant_ids, R)

        print(f"\nUser {index}")
        print(f"  Relevant games: {relevant_ids}")
        print(f"  Recommended: {recommended_ids}")
        print(f"  Recall@3: {recall_at_3:.3f}, Recall@5: {recall_at_5:.3f},  Recall@10: {recall_at_10:.3f}, Recall@R: {recall_at_R:.3f}")
        print(f"  NDCG@1: {ndcg_at_1:.3f}, NDCG@3: {ndcg_at_3:.3f}, NDCG@5: {ndcg_at_5:.3f}, NDCG@10: {ndcg_at_10:.3f}, NDCG@R: {ndcg_at_R:.3f}")

        results.append({
            'user_id': index,
            'recall@3': recall_at_3,
            'recall@5': recall_at_5,
            'recall@10': recall_at_10,
            'recall@R': recall_at_R,
            'ndcg@1': ndcg_at_1,
            'ndcg@3': ndcg_at_3,
            'ndcg@5': ndcg_at_5,
            'ndcg@10': ndcg_at_10,
            'ndcg@R': ndcg_at_R
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("evaluation_metrics_exc.csv", index=False)
    print("\n✅ Evaluation complete. Metrics saved to 'evaluation_metrics.csv'.")


def run_evaluation_varying_k():
    last_week_dict, liked_dict = precompute_game_features()
    all_games = list(last_week_dict.values())
    print(all_games)
    k_values = list(range(1, 30))
    summary = []

    for k in k_values:
        print(f"\n=== Evaluating for k = {k} ===")
        results = []

        for index, row in responses_df.iterrows():
            liked_games = extract_liked_games(row, liked_dict)
            if not liked_games:
                continue

            relevant_ids = []
            if pd.notna(row.iloc[1]):
                relevant_ids = [g.strip() for g in str(row.iloc[1]).split(',') if g.strip().upper() != "UNKNOWN"]

            R = len(relevant_ids)
            recommendations = recommend_games_cosine_top_k(all_games, liked_games, num=17, k=k)
            recommended_ids = [g['game_id'] for g in recommendations]

            recall_at_3 = compute_recall_at_k(recommended_ids, relevant_ids, 3)
            recall_at_5 = compute_recall_at_k(recommended_ids, relevant_ids, 5)
            recall_at_10 = compute_recall_at_k(recommended_ids, relevant_ids, 10)
            recall_at_R = compute_recall_at_k(recommended_ids, relevant_ids, R)

            ndcg_at_1 = compute_ndcg_at_k(recommended_ids, relevant_ids, 1)
            ndcg_at_3 = compute_ndcg_at_k(recommended_ids, relevant_ids, 3)
            ndcg_at_5 = compute_ndcg_at_k(recommended_ids, relevant_ids, 5)
            ndcg_at_10 = compute_ndcg_at_k(recommended_ids, relevant_ids, 10)
            ndcg_at_R = compute_ndcg_at_k(recommended_ids, relevant_ids, R)

            results.append({
                'recall@3': recall_at_3,
                'recall@5': recall_at_5,
                'recall@10': recall_at_10,
                'recall@R': recall_at_R,

                'ndcg@1': ndcg_at_1,
                'ndcg@3': ndcg_at_3,
                'ndcg@5': ndcg_at_5,
                'ndcg@10': ndcg_at_10,
                'ndcg@R': ndcg_at_R
            })

        # Aggregate results for this k
        df = pd.DataFrame(results)
        avg_metrics = df.mean().to_dict()
        avg_metrics["k"] = k
        summary.append(avg_metrics)

    # Save results
    pd.DataFrame(summary).to_csv("evaluation_by_k.csv", index=False)
    print("\n✅ Evaluation across k values complete. Saved to 'evaluation_by_k.csv'.")



if __name__ == "__main__":
    run_evaluation()

