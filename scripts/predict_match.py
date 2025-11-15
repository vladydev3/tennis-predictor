import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


def map_round_to_num(r):
    if pd.isna(r):
        return np.nan
    s = str(r).strip().lower()
    if '1st' in s or s == '1':
        return 1
    if '2nd' in s or s == '2':
        return 2
    if '3rd' in s or s == '3':
        return 3
    if 'r16' in s or 'round of 16' in s or '4th' in s:
        return 4
    if 'quarter' in s:
        return 5
    if 'semi' in s:
        return 6
    if 'final' in s:
        return 7
    import re
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return np.nan





def compute_hist_features(history_df, match_date, p1, p2, surface, N=20):
    # history_df should contain past matches (Date < match_date)
    dfh = history_df[history_df['Date'] < match_date]

    def lastN_winrate(player, surface=None):
        mask = (dfh['Player_1'] == player) | (dfh['Player_2'] == player)
        sub = dfh[mask].sort_values('Date', ascending=False)
        if surface is not None:
            # In preprocessed data the original 'Surface' column may have been one-hot-encoded and removed.
            # Support both cases: if 'Surface' exists, filter by it; otherwise look for a one-hot column like 'Surface_Hard'.
            if 'Surface' in sub.columns:
                sub = sub[sub['Surface'] == surface]
            else:
                one_hot_col = f"Surface_{surface}"
                if one_hot_col in sub.columns:
                    sub = sub[sub[one_hot_col] == 1]
                else:
                    # cannot filter by surface, keep all
                    pass
        sub = sub.head(N)
        if len(sub) == 0:
            return 0.5
        return (sub['Winner'] == player).sum() / len(sub)

    def h2h_winrate(a, b):
        mask = ((dfh['Player_1'] == a) & (dfh['Player_2'] == b)) | ((dfh['Player_1'] == b) & (dfh['Player_2'] == a))
        sub = dfh[mask]
        if len(sub) == 0:
            return 0.5
        wins = ((sub['Winner'] == a).sum())
        return wins / len(sub)

    return {
        'p1_lastN_winrate': lastN_winrate(p1, None),
        'p2_lastN_winrate': lastN_winrate(p2, None),
        'p1_surf_lastN_winrate': lastN_winrate(p1, surface),
        'p2_surf_lastN_winrate': lastN_winrate(p2, surface),
        'p1_h2h_winrate': h2h_winrate(p1, p2),
    }


def build_feature_row(df_full, match_info, elo_ratings=None):
    # df_full: preprocessed df (contains columns and one-hot categories)
    # match_info: dict with keys like Player_1, Player_2, Date (datetime), Surface, Rank_1, Rank_2, Pts_1, Pts_2, Round, Best of, Series, Court
    # elo_ratings: dict of dicts, e.g. {'Player Name': {'Surface': elo_score}}
    row = {}
    # basic numeric features
    row['rank_diff'] = match_info.get('Rank_1') - match_info.get('Rank_2') if match_info.get('Rank_1') is not None and match_info.get('Rank_2') is not None else 0
    p1 = match_info.get('Pts_1')
    p2 = match_info.get('Pts_2')
    if p1 is None or p2 is None:
        row['pts_diff'] = 0
        row['pts_ratio'] = 0.5
    else:
        row['pts_diff'] = p1 - p2
        s = p1 + p2
        row['pts_ratio'] = p1 / s if s != 0 else 0.5

    row['Round_num'] = map_round_to_num(match_info.get('Round'))
    row['Best of'] = match_info.get('Best of', None)

    # ELO features
    p1_name = match_info.get('Player_1')
    p2_name = match_info.get('Player_2')
    surface = match_info.get('Surface')
    
    p1_elo = 1500
    p2_elo = 1500
    if elo_ratings and p1_name and p2_name and surface:
        p1_elo = elo_ratings.get(p1_name, {}).get(surface, 1500)
        p2_elo = elo_ratings.get(p2_name, {}).get(surface, 1500)
    
    row['p1_elo_surface'] = p1_elo
    row['p2_elo_surface'] = p2_elo
    row['elo_surface_diff'] = p1_elo - p2_elo

    # one-hot features: take columns from df_full
    for c in df_full.columns:
        if c.startswith('Surface_'):
            val = c.replace('Surface_', '')
            row[c] = 1 if val == match_info.get('Surface') else 0
        if c.startswith('Series_'):
            val = c.replace('Series_', '')
            row[c] = 1 if val == match_info.get('Series') else 0
        if c.startswith('Court_'):
            val = c.replace('Court_', '')
            row[c] = 1 if val == match_info.get('Court') else 0

    # historical features computed from df_full
    hist = compute_hist_features(df_full, match_info.get('Date'), match_info.get('Player_1'), match_info.get('Player_2'), match_info.get('Surface'))
    row.update(hist)

    return row


def predict_from_dataset(df, model, match_filter=None):
    # match_filter: dict with keys to match exact row(s)
    q = pd.Series([True] * len(df))
    for k, v in (match_filter or {}).items():
        q = q & (df[k] == v)
    sub = df[q]
    if sub.empty:
        raise ValueError('No matching rows found in preprocessed dataset with the given filter')
    # pick first matching
    row = sub.iloc[0]
    # prepare feature vector, ensuring columns match model's expected features
    feature_cols = model.feature_names_in_
    X = row[feature_cols].to_frame().T
    
    # Impute missing values using the same strategy as in training
    for hist in ['p1_lastN_winrate', 'p2_lastN_winrate', 'p1_surf_lastN_winrate', 'p2_surf_lastN_winrate', 'p1_h2h_winrate']:
        if hist in X.columns:
            X[hist] = X[hist].fillna(0.5)
    for elo_col in ['p1_elo_surface', 'p2_elo_surface']:
        if elo_col in X.columns:
            X[elo_col] = X[elo_col].fillna(1500)
    if 'elo_surface_diff' in X.columns:
        X['elo_surface_diff'] = X['elo_surface_diff'].fillna(0)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None
    return pred, proba, row


def predict_custom(df, model, match_info, elo_ratings=None):
    feature_cols = model.feature_names_in_
    rowdict = build_feature_row(df, match_info, elo_ratings)
    X = pd.DataFrame([rowdict]).reindex(columns=feature_cols).fillna(0)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None
    return pred, proba, X


def parse_args():
    p = argparse.ArgumentParser(description='Predict a tennis match using trained model')
    p.add_argument('--mode', choices=['dataset', 'custom', 'demo'], default='demo')
    p.add_argument('--player1')
    p.add_argument('--player2')
    p.add_argument('--date')
    p.add_argument('--surface')
    p.add_argument('--rank1', type=float)
    p.add_argument('--rank2', type=float)
    p.add_argument('--pts1', type=float)
    p.add_argument('--pts2', type=float)
    p.add_argument('--round', dest='round_name')
    p.add_argument('--bestof', type=int, default=3)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    df_path = repo / 'data' / 'atp_preprocessed.pkl'
    model_path = repo / 'models' / 'rf_model.joblib'
    elo_path = repo / 'data' / 'elo_ratings.json'
    if not df_path.exists() or not model_path.exists():
        raise SystemExit('Preprocessed data or model not found. Run preprocess.py and train_model.py first.')
    df = pd.read_pickle(df_path)
    model = joblib.load(model_path)
    elo_ratings = None
    if elo_path.exists():
        import json
        with open(elo_path, 'r') as f:
            elo_ratings = json.load(f)

    if args.mode == 'demo':
        # pick a random recent match from dataset (not the very first)
        sample = df.sample(1).iloc[0]
        filt = {'Player_1': sample['Player_1'], 'Player_2': sample['Player_2'], 'Date': sample['Date']}
        pred, proba, row = predict_from_dataset(df, model, filt)
        print('Demo match:')
        print(row[['Date', 'Tournament', 'Player_1', 'Player_2', 'Winner']])
        print('Predicted winner is Player_1 (1) or Player_2 (0):', pred)
        print('Predicted probability Player_1 wins:', proba)

    elif args.mode == 'dataset':
        if not args.player1 or not args.player2 or not args.date:
            raise SystemExit('Provide --player1, --player2 and --date for dataset mode')
        dt = pd.to_datetime(args.date)
        filt = {'Player_1': args.player1, 'Player_2': args.player2, 'Date': dt}
        pred, proba, row = predict_from_dataset(df, model, filt)
        print('Row matched:')
        print(row[['Date', 'Tournament', 'Player_1', 'Player_2', 'Winner']])
        print('Predicted:', pred, 'Proba:', proba)

    else:  # custom
        if not args.player1 or not args.player2 or not args.date:
            raise SystemExit('Provide --player1, --player2 and --date for custom mode')
        match_info = {
            'Player_1': args.player1,
            'Player_2': args.player2,
            'Date': pd.to_datetime(args.date),
            'Surface': args.surface,
            'Rank_1': args.rank1,
            'Rank_2': args.rank2,
            'Pts_1': args.pts1,
            'Pts_2': args.pts2,
            'Round': args.round_name,
            'Best of': args.bestof,
            'Series': None,
            'Court': None,
        }
        pred, proba, X = predict_custom(df, model, match_info, elo_ratings)
        print('Features used:')
        print(X.to_dict(orient='records')[0])
        print('Predicted winner (1 means Player_1):', pred)
        print('Predicted probability Player_1 wins:', proba)
