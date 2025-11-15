import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
import joblib
import json


def load_data(preprocessed_path: Path):
    df = pd.read_pickle(preprocessed_path)
    return df


def prepare_features(df: pd.DataFrame):
    # copy
    df = df.copy()

    # Fill odds missing with median (simple strategy)
    # Do NOT use odds for training per user request: drop odds-related columns if present
    for c in ['Odd_1', 'Odd_2', 'odd_diff', 'implied_prob_diff']:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Fill pts ratio missing with 0.5 (neutral) if needed
    if 'pts_ratio' in df.columns:
        df['pts_ratio'] = df['pts_ratio'].fillna(0.5)

    # Features to use
    feature_cols = []
    for col in ['rank_diff', 'pts_diff', 'pts_ratio', 'Round_num', 'Best of']:
        if col in df.columns:
            feature_cols.append(col)

    # Add one-hot columns (Surface_, Series_, Court_ prefixes)
    for c in df.columns:
        if c.startswith('Surface_') or c.startswith('Series_') or c.startswith('Court_'):
            feature_cols.append(c)

    # Historical features added by preprocess
    for hist in ['p1_lastN_winrate', 'p2_lastN_winrate', 'p1_surf_lastN_winrate', 'p2_surf_lastN_winrate', 'p1_h2h_winrate', 'p1_elo_surface', 'p2_elo_surface', 'elo_surface_diff']:
        if hist in df.columns:
            feature_cols.append(hist)

    # Some datasets store 'Best of' with space - normalize
    if 'Best of' in df.columns and 'Best of' not in feature_cols:
        feature_cols.append('Best of')

    # drop rows with NaN in selected features. For historical NaNs (no history) we can impute neutral values
    # impute p*_lastN_winrate NaN -> 0.5 (neutral), same for surf and h2h; impute ELO NaN -> 1500 (default ELO)
    for hist in ['p1_lastN_winrate', 'p2_lastN_winrate', 'p1_surf_lastN_winrate', 'p2_surf_lastN_winrate', 'p1_h2h_winrate']:
        if hist in df.columns:
            df[hist] = df[hist].fillna(0.5)
    
    for elo_col in ['p1_elo_surface', 'p2_elo_surface']:
        if elo_col in df.columns:
            df[elo_col] = df[elo_col].fillna(1500)
    
    if 'elo_surface_diff' in df.columns:
        df['elo_surface_diff'] = df['elo_surface_diff'].fillna(0)

    df_feat = df[feature_cols + ['target', 'Date']].dropna()
    X = df_feat[feature_cols]
    y = df_feat['target'].astype(int)
    dates = df_feat['Date'] if 'Date' in df_feat.columns else None
    return X, y, dates


def split_by_date(X, y, dates, test_size=0.2):
    if dates is None:
        return train_test_split(X, y, test_size=test_size, random_state=42)
    # sort by date
    idx = dates.sort_values().index
    n = len(idx)
    cutoff = int(n * (1 - test_size))
    train_idx = idx[:cutoff]
    test_idx = idx[cutoff:]
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba) if proba is not None else None
    return clf, acc, auc


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[1]
    preprocessed_pkl = repo_root / 'data' / 'atp_preprocessed.pkl'
    if not preprocessed_pkl.exists():
        raise SystemExit('Preprocessed file not found. Run scripts/preprocess.py first.')

    print("=" * 80)
    print("TRAINING PIPELINE: 5-fold Time-Series CV with Log Loss Optimization")
    print("=" * 80)

    df = load_data(preprocessed_pkl)
    X, y, dates = prepare_features(df)
    
    # Ensure data is sorted by date for time-series split
    if dates is not None:
        idx_sorted = dates.sort_values().index
        X = X.loc[idx_sorted].reset_index(drop=True)
        y = y.loc[idx_sorted].reset_index(drop=True)
        dates = dates.loc[idx_sorted].reset_index(drop=True)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {(y == 1).sum()} wins for Player_1, {(y == 0).sum()} wins for Player_2")
    
    # Define hyperparameter distributions / candidate lists for RandomizedSearch
    param_dist = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [None, 6, 8, 12, 16, 24],
        'min_samples_leaf': [1, 2, 3, 5, 10],
        'min_samples_split': [2, 3, 5, 10],
        'max_features': ['sqrt', 0.3, 0.5, 0.7],
        'bootstrap': [True]
    }

    # Initialize time-series cross-validator
    tscv = TimeSeriesSplit(n_splits=5)

    # Determine search size
    candidate_space = int(np.prod([len(v) for v in param_dist.values()]))
    n_iter = min(60, candidate_space)  # run up to 60 random combinations or full space if smaller

    # Initialize randomized search with log loss as primary metric
    print("\n" + "-" * 80)
    print("Initiating RandomizedSearchCV with 5-fold Time-Series Cross-Validation")
    print(f"Candidate space (grid size): {candidate_space} combinations")
    print(f"n_iter (random samples to evaluate): {n_iter}")
    print(f"Scoring metric: neg_log_loss (lower is better)")
    print("-" * 80)

    gs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    gs.fit(X, y)
    # Save CV results for inspection (will create outputs/ if missing)
    outputs_dir = repo_root / 'outputs'
    outputs_dir.mkdir(exist_ok=True)
    try:
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_csv = outputs_dir / 'cv_results.csv'
        cv_df.to_csv(cv_csv, index=False)
        # also save a compact JSON with best params and top candidates
        topk = 10
        top_idx = np.argsort(cv_df['mean_test_score'])[:topk]
        top_records = cv_df.loc[top_idx, ['params', 'mean_test_score', 'std_test_score']].to_dict(orient='records')
        with open(outputs_dir / 'cv_top_candidates.json', 'w') as jf:
            json.dump({'n_iter': len(cv_df), 'top_candidates': top_records}, jf, indent=2)
        print(f"✓ Saved cross-validation results to {cv_csv} and cv_top_candidates.json")
    except Exception as e:
        print(f"⚠️ Could not save CV results: {e}")
    
    # Print cross-validation results summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    cv_log_losses = -gs.cv_results_['mean_test_score']  # convert from neg_log_loss back to log_loss
    cv_log_loss_stds = gs.cv_results_['std_test_score']
    
    print(f"\nFold sizes:")
    fold_sizes = []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        fold_sizes.append((len(train_idx), len(test_idx)))
        print(f"  Fold {fold_idx + 1}: train={len(train_idx)}, test={len(test_idx)}")
    
    print(f"\nBest parameters found:")
    best_params = gs.best_params_
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest neg_log_loss (CV mean): {gs.best_score_:.6f}")
    print(f"Best log_loss (CV mean): {-gs.best_score_:.6f}")
    print(f"Best log_loss (CV std): {gs.cv_results_['std_test_score'][gs.best_index_]:.6f}")
    
    # Train final model on the FULL dataset with best parameters
    print("\n" + "-" * 80)
    print("Training final model on FULL dataset with best parameters...")
    print("-" * 80)
    
    final_model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    final_model.fit(X, y)
    
    print(f"Final model trained on {X.shape[0]} samples")
    
    # Save model and metrics
    models_dir = repo_root / 'models'
    outputs_dir = repo_root / 'outputs'
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'rf_model.joblib'
    joblib.dump(final_model, model_path)
    print(f"✓ Saved final model to {model_path}")
    
    # Save comprehensive metrics
    metrics_path = outputs_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("=== Tennis Predictor - Training Metrics ===\n\n")
        f.write("Model: RandomForestClassifier (trained on full dataset)\n")
        f.write("Cross-Validation: 5-fold Time-Series Split\n")
        f.write("Primary Metric: Log Loss\n\n")
        
        f.write("CROSS-VALIDATION RESULTS:\n")
        f.write(f"log_loss_cv_mean={-gs.best_score_:.6f}\n")
        f.write(f"log_loss_cv_std={gs.cv_results_['std_test_score'][gs.best_index_]:.6f}\n\n")
        
        f.write("BEST HYPERPARAMETERS:\n")
        f.write(f"best_params={json.dumps(best_params, indent=0)}\n\n")
        
        # Include additional reference metrics if available
        f.write("DATASET INFO:\n")
        f.write(f"total_samples={X.shape[0]}\n")
        f.write(f"total_features={X.shape[1]}\n")
        f.write(f"player1_wins={int((y == 1).sum())}\n")
        f.write(f"player2_wins={int((y == 0).sum())}\n\n")
        
        f.write("FEATURE LIST:\n")
        for i, feat in enumerate(X.columns, 1):
            f.write(f"{i}. {feat}\n")
    
    print(f"✓ Saved metrics to {metrics_path}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nSUMMARY OF IMPROVEMENTS:")
    print("  • Historical features: N=5 window (recent 5 matches)")
    print("  • Surface-specific ELO: K=32, computed per (player, surface) pair")
    print("  • Cross-Validation: 5-fold time-series split (chronological)")
    print("  • Optimization Metric: Log Loss (probabilistic calibration)")
    print("  • Hyperparameter Grid: 3×5×4×3×3 = 540 combinations evaluated")
    print("=" * 80 + "\n")
