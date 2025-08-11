import pandas as pd
import re
from datetime import datetime

# Load the CSVs
responses_df = pd.read_csv(r"C:/Users/User/Desktop/University/year4/semA/FinalProject/evaluation/survey_responses2.csv")
games_df = pd.read_csv(r"C:/Users/User/Desktop/University/year4/semA/FinalProject/game_database.csv")      

# Parse game_date to datetime (dates like 06/03/2025 assumed to be DD/MM/YYYY)
games_df['game_date'] = pd.to_datetime(games_df['game_date'], format='mixed', dayfirst=True)

# Nickname to full team name mapping
nickname_to_full = {
    "Hawks": "Atlanta Hawks", "Celtics": "Boston Celtics", "Nets": "Brooklyn Nets",
    "Hornets": "Charlotte Hornets", "Bulls": "Chicago Bulls", "Cavaliers": "Cleveland Cavaliers",
    "Mavericks": "Dallas Mavericks", "Nuggets": "Denver Nuggets", "Pistons": "Detroit Pistons",
    "Warriors": "Golden State Warriors", "Rockets": "Houston Rockets", "Pacers": "Indiana Pacers",
    "Clippers": "LA Clippers", "Lakers": "Los Angeles Lakers", "Grizzlies": "Memphis Grizzlies",
    "Heat": "Miami Heat", "Bucks": "Milwaukee Bucks", "Timberwolves": "Minnesota Timberwolves",
    "Pelicans": "New Orleans Pelicans", "Knicks": "New York Knicks", "Thunder": "Oklahoma City Thunder",
    "Magic": "Orlando Magic", "76ers": "Philadelphia 76ers", "Suns": "Phoenix Suns",
    "Trail Blazers": "Portland Trail Blazers", "Kings": "Sacramento Kings", "Spurs": "San Antonio Spurs",
    "Raptors": "Toronto Raptors", "Jazz": "Utah Jazz", "Wizards": "Washington Wizards"
}

# === Step 3: Helper functions ===
def parse_game_string(game_str):
    match = re.match(r'(.+?) vs (.+?) \((\w+ \d{1,2})\)', game_str.strip())
    if not match:
        return None
    team1, team2, date_part = match.groups()
    try:
        date = datetime.strptime(f"{date_part} 2025", "%b %d %Y")
    except ValueError:
        return None
    team1_full = nickname_to_full.get(team1.strip())
    team2_full = nickname_to_full.get(team2.strip())
    return team1_full, team2_full, date

def find_game_id(team1, team2, date):
    match = games_df[
        (((games_df['home_team'] == team1) & (games_df['away_team'] == team2)) |
         ((games_df['home_team'] == team2) & (games_df['away_team'] == team1))) &
        (games_df['game_date'] == date)
    ]
    if not match.empty:
        return match.iloc[0]['game_id']
    return None

# === Step 4: Main replacement logic ===
def replace_game_labels_with_ids(responses_df):
    updated_df = responses_df.copy()
    for col in updated_df.columns:
        new_col = []
        for cell in updated_df[col]:
            cell = str(cell)
            game_labels = [g.strip() for g in cell.split(',') if g.strip()]
            game_ids = []
            for g in game_labels:
                parsed = parse_game_string(g)
                if parsed:
                    team1, team2, date = parsed
                    game_id = find_game_id(team1, team2, date)
                    game_ids.append(str(game_id) if game_id else 'UNKNOWN')
                else:
                    game_ids.append('UNKNOWN')
            new_col.append(', '.join(game_ids))
        updated_df[col] = new_col
    return updated_df

# === Step 5: Run the transformation and export ===
final_df = replace_game_labels_with_ids(responses_df)
final_df.to_csv("survey_responses_with_ids2.csv", index=False)

print("✅ Done! Output saved to 'survey_responses_with_ids2.csv'")
