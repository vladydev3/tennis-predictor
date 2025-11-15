import os
from pathlib import Path
import pandas as pd
import numpy as np
import json



def map_round_to_num(r):
    if pd.isna(r):
        return np.nan
    s = str(r).strip().lower()
    # common mappings
    if '1st' in s or '1' == s:
        return 1
    if '2nd' in s:
        return 2
    if '3rd' in s:
        return 3
    if '4th' in s or 'r16' in s or 'round of 16' in s or 'r16' in s:
        return 4
    if 'quarter' in s:
        return 5
    if 'semi' in s:
        return 6
    if 'final' in s or 'the final' in s:
        return 7
    # fallback: try to extract number
    import re
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return np.nan


def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return np.nan


def preprocess(in_path: Path, out_dir: Path, drop_incomplete_score=True):
    df = pd.read_csv(in_path)

    print(f"Initial shape: {df.shape}")
    print(df.head(3).to_string())

    # basic checks
    print('\nNulls per column:')
    print(df.isnull().sum())
    print('\nDtypes:')
    print(df.dtypes)

    # Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Clean strings
    for c in ['Surface', 'Series', 'Round', 'Court', 'Tournament', 'Player_1', 'Player_2', 'Winner']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    # Score incomplete
    if 'Score' in df.columns:
        df['incomplete_score'] = df['Score'].isnull() | (df['Score'].astype(str).str.strip() == '')
        if drop_incomplete_score:
            df = df[~df['incomplete_score']]
            print('Dropped rows with incomplete score')

    # Preserve raw Surface for historical features (surface-specific ELO and winrate)
    if 'Surface' in df.columns:
        df['Surface_raw'] = df['Surface']

    # Categorical -> numeric (one-hot)
    one_hot_cols = [c for c in ['Surface', 'Series', 'Court'] if c in df.columns]
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, prefix=one_hot_cols, dummy_na=False)

    # Round mapping
    if 'Round' in df.columns:
        df['Round_num'] = df['Round'].apply(map_round_to_num)

    # Numeric features and cleaning
    # Replace placeholder missing values (common in this dataset: -1 or -1.0) with NaN
    for c in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df.loc[df[c] <= -1, c] = np.nan

    # Handle ranks missing -> use large number to indicate unranked
    for c in ['Rank_1', 'Rank_2']:
        if c in df.columns:
            df[c] = df[c].fillna(9999)

    if 'Odd_1' in df.columns and 'Odd_2' in df.columns:
        # set any non-positive odds to NaN
        df.loc[df['Odd_1'] <= 0, 'Odd_1'] = np.nan
        df.loc[df['Odd_2'] <= 0, 'Odd_2'] = np.nan
        # Do not drop rows without odds by default anymore; keep them so models that don't use odds
        # can still leverage all available matches. Print a summary for visibility.
        missing_odds = df['Odd_1'].isna() | df['Odd_2'].isna()
        print(f"Rows without valid odds: {missing_odds.sum()} (kept for training since odds are optional)")

    # 4.1 engineered features
    if 'Rank_1' in df.columns and 'Rank_2' in df.columns:
        df['rank_diff'] = df['Rank_1'] - df['Rank_2']

    if 'Pts_1' in df.columns and 'Pts_2' in df.columns:
        df['pts_diff'] = df['Pts_1'] - df['Pts_2']
        df['pts_sum'] = df['Pts_1'] + df['Pts_2']
        df['pts_ratio'] = df.apply(lambda r: safe_div(r['Pts_1'], r['pts_sum']) if pd.notna(r['pts_sum']) and r['pts_sum'] != 0 else np.nan, axis=1)

    if 'Odd_1' in df.columns and 'Odd_2' in df.columns:
        df['odd_diff'] = df['Odd_1'] - df['Odd_2']
        df['implied_prob_diff'] = df.apply(lambda r: safe_div(1.0, r['Odd_1']) - safe_div(1.0, r['Odd_2']) if pd.notna(r['Odd_1']) and pd.notna(r['Odd_2']) else np.nan, axis=1)

    # 6. target
    if 'Winner' in df.columns and 'Player_1' in df.columns and 'Player_2' in df.columns:
        df['target'] = (df['Winner'] == df['Player_1']).astype(int)

    # historical features and ELO rating
    # - recent N-match winrate (p1_lastN_winrate, p2_lastN_winrate)
    # - recent N-match winrate on same surface (p1_surf_lastN_winrate, p2_surf_lastN_winrate)
    # - H2H winrate (p1_h2h_winrate)
    # - Surface-specific ELO ratings (p1_elo_surface, p2_elo_surface, elo_surface_diff)
    def add_historical_features(df, N=5, K=32):
        # ensure sorted by date
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        from collections import deque, defaultdict

        last_results = defaultdict(lambda: deque(maxlen=N))  # store 1 for win, 0 for loss
        last_surface_results = defaultdict(lambda: defaultdict(lambda: deque(maxlen=N)))
        h2h_wins = defaultdict(lambda: defaultdict(int))  # h2h_wins[a][b] = wins of a vs b
        h2h_matches = defaultdict(lambda: defaultdict(int))
        elo_dict = {}  # keys: (player, surface) tuples, default value 1500

        p1_wr = []
        p2_wr = []
        p1_surf_wr = []
        p2_surf_wr = []
        p1_h2h = []
        p1_elo_surf = []
        p2_elo_surf = []
        elo_diff = []

        for _, row in df.iterrows():
            p1 = row['Player_1']
            p2 = row['Player_2']
            surface = row['Surface_raw'] if 'Surface_raw' in row and pd.notna(row['Surface_raw']) else None

            # overall recent winrate
            r1 = list(last_results[p1])
            r2 = list(last_results[p2])
            p1_wr.append(np.mean(r1) if len(r1) > 0 else np.nan)
            p2_wr.append(np.mean(r2) if len(r2) > 0 else np.nan)

            # same-surface winrate
            if surface is not None:
                s1 = list(last_surface_results[p1][surface])
                s2 = list(last_surface_results[p2][surface])
                p1_surf_wr.append(np.mean(s1) if len(s1) > 0 else np.nan)
                p2_surf_wr.append(np.mean(s2) if len(s2) > 0 else np.nan)
            else:
                p1_surf_wr.append(np.nan)
                p2_surf_wr.append(np.nan)

            # h2h winrate for p1 vs p2 (based on past matches)
            matches = h2h_matches[p1].get(p2, 0)
            wins = h2h_wins[p1].get(p2, 0)
            p1_h2h.append(wins / matches if matches > 0 else np.nan)

            # Surface-specific ELO ratings
            p1_elo_current = elo_dict.get((p1, surface), 1500) if surface is not None else np.nan
            p2_elo_current = elo_dict.get((p2, surface), 1500) if surface is not None else np.nan
            p1_elo_surf.append(p1_elo_current)
            p2_elo_surf.append(p2_elo_current)
            elo_diff.append(p1_elo_current - p2_elo_current if surface is not None else np.nan)

            # now update histories and ELO with current match outcome
            winner = row['Winner']
            if pd.notna(winner):
                w1 = 1 if winner == p1 else 0
                w2 = 1 if winner == p2 else 0

                last_results[p1].append(w1)
                last_results[p2].append(w2)

                if surface is not None:
                    last_surface_results[p1][surface].append(w1)
                    last_surface_results[p2][surface].append(w2)

                # update h2h counters
                h2h_matches[p1][p2] += 1
                h2h_matches[p2][p1] += 1
                if winner == p1:
                    h2h_wins[p1][p2] += 1
                elif winner == p2:
                    h2h_wins[p2][p1] += 1

                # Update surface-specific ELO ratings
                if surface is not None:
                    elo1 = elo_dict.get((p1, surface), 1500)
                    elo2 = elo_dict.get((p2, surface), 1500)
                    
                    # Expected scores using logistic formula
                    E1 = 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / 400.0))
                    E2 = 1.0 / (1.0 + 10.0 ** ((elo1 - elo2) / 400.0))
                    
                    # Update ELOs
                    new_elo1 = elo1 + K * (w1 - E1)
                    new_elo2 = elo2 + K * (w2 - E2)
                    
                    elo_dict[(p1, surface)] = new_elo1
                    elo_dict[(p2, surface)] = new_elo2

        df['p1_lastN_winrate'] = p1_wr
        df['p2_lastN_winrate'] = p2_wr
        df['p1_surf_lastN_winrate'] = p1_surf_wr
        df['p2_surf_lastN_winrate'] = p2_surf_wr
        df['p1_h2h_winrate'] = p1_h2h
        df['p1_elo_surface'] = p1_elo_surf
        df['p2_elo_surface'] = p2_elo_surf
        df['elo_surface_diff'] = elo_diff
        
        # Convert elo_dict with tuple keys to a JSON-serializable nested dict
        elo_json_dict = defaultdict(dict)
        for (player, surface), elo in elo_dict.items():
            if player and surface:
                elo_json_dict[player][surface] = elo
        
        return df, elo_json_dict

    df, elo_ratings = add_historical_features(df, N=5, K=32)

    if 'Surface_raw' in df.columns:
        df.drop(['Surface_raw'], axis=1, inplace=True)


    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'atp_preprocessed.csv'
    out_pkl = out_dir / 'atp_preprocessed.pkl'
    df.to_csv(out_csv, index=False)
    df.to_pickle(out_pkl)
    
    if elo_ratings:
        out_elo = out_dir / 'elo_ratings.json'
        with open(out_elo, 'w') as f:
            json.dump(elo_ratings, f, indent=2)
        print(f"Saved ELO ratings to {out_elo}")

    print(f"Saved preprocessed CSV to {out_csv} and pickle to {out_pkl}")
    return df


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]
    in_path = repo_root / 'atp_tennis.csv'
    out_dir = repo_root / 'data'
    df = preprocess(in_path, out_dir, drop_incomplete_score=True)
    print('\nPreprocessing done. Shape:', df.shape)
