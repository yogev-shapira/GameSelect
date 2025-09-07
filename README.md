# GameSelect — Find the NBA Game Worth Watching 🏀

> A personalized recommender that surfaces the most exciting NBA games to rewatch based on play-by-play data and your tastes.
---
![Project Cover Image](/static/GameSelect.png)
## Table of Contents
- [The Team](#the-team)
- [Project Description](#project-description)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing](#️-installing)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Repository Structure](#repository-structure)
- [Built With](#️-built-with)
- [Acknowledgments](#acknowledgments)

---

## 👥 The Team
- **Yogev Shapira:** yogev.shapira@mail.huji.ac.il  
-**Advisors:** Daniella Har Shalom, Gal Katzhendler

---

## ℹ️ Project Description
**Why:** Each season includes hundreds of NBA games- which can be too many for fans (especially those across different time zones) to know what’s worth rewatching.  
**What:** GameSelect recommends past games you’ll likely enjoy by analyzing *in-game action* (lead changes, dunks, blocks, 3PTs, scoring density, game closeness, star power) and combining it with your liked games and team/player overlap.  
**How (high level):**
- Parse ESPN play-by-play (PBP) into normalized feature vectors.
- If you have liked games → compute similarity (cosine on numeric features + weighted overlaps for teams/top players).  
- If not → fall back to an **excitement score** that prioritizes thrilling games.  
- Return the top-N games in your selected recent window via a simple Flask web app. ✨

---

## ⚡ Getting Started

### Prerequisites
- Python **3.10+**
- **pip**
- (Optional) **virtualenv** or **conda**
- Internet access for fetching PBP when updating the DB

### ️ Installing
```bash
git clone https://github.com/yogev-shapira/GameSelect.git
cd GameSelect

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```
---

### ️ 📈 Evaluation
We evaluate with **Recall@k** and **NDCG@k** using real user inputs (liked games vs. a recent candidate window). Results show our hybrid approach consistently outperforms random selection and is competitive against single-max similarity variants. See `evaluation/` for scripts and outputs (CSVs and plots). 📊

---

###  🌐 Deployment
- **Daily DB refresh**: GitHub Actions (in `.github/workflows/`) runs `update_db.py` to fetch new game IDs from ESPN, download PBP CSVs into `data/`, and update `game_database.csv` (including final scores when available).
- **Hosting**: Deployable on **Render** (or any Flask-friendly host). Configure CORS in `app.py` for your domain.

---

## 🧱 Repository Structure 
  ```text
.
├─ .github/workflows/        # CI: daily job to fetch PBP + update DB
├─ data/                     # Cached ESPN play-by-play CSVs (downloaded by updater)
├─ evaluation/               # Survey parsing, ID mapping, metrics & result exports
├─ static/                   # Static assets (CSS/images); logo under static/styles/
├─ templates/                # Flask templates: landing, select_games, select_range, show_results
├─ app.py                    # Flask app: routes, pages, and minimal JSON endpoints
├─ calc_similarity.py        # Cosine on numeric features + weighted team/player overlaps; excitement fallback
├─ feature_extractor.py      # PBP → normalized features (lead changes, 3PT, dunks, blocks, density, closeness, star score, top players)
├─ game_database.csv         # Master games table (ids, teams, datetime/location, optional scores)
├─ get_game_data.py          # Download PBP for a specific game id
├─ get_game_ids.py           # Fetch game metadata/ids for date ranges
├─ requirements.txt          # Python dependencies
├─ update_db.py              # Daily update: append new games + PBP + scores
└─ cached_game_features.pkl  # Cached feature vectors to speed up recommendations
```
---

## ⚙️ Built With
- **Python**, **Flask**, **pandas**, **NumPy**
- **HTML/CSS/JS**
- **GitHub Actions** (daily data updates)
- **Render** (optional hosting)

---

## Acknowledgments
- The Rachel & Selim Benin School of Computer Science and Engineering, HUJI  
- NBA fans who participated in our surveys 🧡  
- ESPN PBP data powering the feature extraction
