import streamlit as st
import sys
from pathlib import Path
import json

# Add the project root to the sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
import pandas as pd
import joblib
from scripts.predict_match import predict_custom, predict_from_dataset  # keep for local fallback if needed


@st.cache_data
def load_data_and_model():
    repo = Path(__file__).resolve().parents[1]
    # support both pickle and CSV preprocessed datasets
    pkl_path = repo / 'data' / 'atp_preprocessed.pkl'
    csv_path = repo / 'data' / 'atp_preprocessed.csv'
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise FileNotFoundError('No preprocessed dataset found (expected atp_preprocessed.pkl or atp_preprocessed.csv)')

    elo_path = repo / 'data' / 'elo_ratings.json'
    tour_elo_path = repo / 'data' / 'tournament_elo_ratings.json'
    tour_stats_path = repo / 'data' / 'tournament_stats.json'
    elo_ratings = None
    if elo_path.exists():
        with open(elo_path, 'r') as f:
            elo_ratings = json.load(f)
    tour_elo = None
    tour_stats = None
    if tour_elo_path.exists():
        with open(tour_elo_path, 'r') as f:
            tour_elo = json.load(f)
    if tour_stats_path.exists():
        with open(tour_stats_path, 'r') as f:
            tour_stats = json.load(f)

    # Load model locally for direct prediction (if available)
    model_path = repo / 'models' / 'rf_model.joblib'
    model = None
    if model_path.exists():
        model = joblib.load(model_path)

    return df, elo_ratings, tour_elo, tour_stats, model


def main():
    st.title('Tennis match predictor')
    st.markdown('Interfaz para predecir el ganador de un partido usando el modelo entrenado')

    df, elo_ratings, tour_elo, tour_stats, model = load_data_and_model()

    def call_local_custom(match_info):
        if model is None:
            raise RuntimeError('Model not found locally')
        mi = match_info.copy()
        if isinstance(mi.get('Date'), str):
            mi['Date'] = pd.to_datetime(mi['Date'])
        pred, proba, X = predict_custom(df, model, mi, elo_ratings, tour_elo, tour_stats)
        return {
            'predicted_winner_flag': int(pred),
            'proba_player1_win': float(proba) if proba is not None else None,
            'details': {'features': X.to_dict(orient='records')[0]}
        }

    def call_local_dataset(player1, player2, date_str):
        if model is None:
            raise RuntimeError('Model not found locally')
        filt = {'Player_1': player1, 'Player_2': player2, 'Date': pd.to_datetime(date_str)}
        pred, proba, row = predict_from_dataset(df, model, filt)
        return {
            'predicted_winner_flag': int(pred),
            'proba_player1_win': float(proba) if proba is not None else None,
            'details': {'matched_row': row[[c for c in ['Date', 'Tournament', 'Player_1', 'Player_2', 'Winner'] if c in row.index]].to_dict()}
        }

    mode = st.radio('Modo', ['demo', 'dataset', 'custom'])

    if mode == 'demo':
        st.write('Demo: se selecciona un partido aleatorio del dataset y se predice')
        if st.button('Predecir demo'):
            sample = df.sample(1).iloc[0]
            date_str = pd.to_datetime(sample['Date']).strftime('%Y-%m-%d')
            try:
                resp = call_local_dataset(sample['Player_1'], sample['Player_2'], date_str)
                st.write({'Date': sample['Date'], 'Tournament': sample.get('Tournament'), 'Player_1': sample['Player_1'], 'Player_2': sample['Player_2'], 'Winner': sample.get('Winner')})
                pred = resp['predicted_winner_flag']
                proba = resp.get('proba_player1_win')
                winner_name = sample['Player_1'] if pred == 1 else sample['Player_2']
                if proba is not None:
                    winner_prob = proba if pred == 1 else 1.0 - proba
                    st.success(f'Predicción: {winner_name} gana; probabilidad = {winner_prob:.3f}')
                else:
                    st.success(f'Predicción: {winner_name} gana (probabilidad no disponible)')
            except Exception as e:
                st.error(f'Error calling API: {e}')
            # Paragraph ready to copy
            copy_text = f"En el partido entre {sample['Player_1']} y {sample['Player_2']}, el modelo predice que {winner_name} será el ganador."
            st.text_area('Texto listo para copiar', value=copy_text, height=80)

    elif mode == 'dataset':
        st.write('Selecciona un partido existente en el dataset')
        players = sorted(pd.unique(df[['Player_1', 'Player_2']].values.ravel()))
        p1 = st.selectbox('Player 1', players)
        p2 = st.selectbox('Player 2', players, index=players.index(p1) if p1 in players else 0)
        # available dates for pair
        pair_mask = (df['Player_1'] == p1) & (df['Player_2'] == p2)
        dates = df.loc[pair_mask, 'Date'].sort_values().dt.strftime('%Y-%m-%d').unique().tolist()
        if dates:
            date_sel = st.selectbox('Date', dates)
            if st.button('Predecir (dataset)'):
                try:
                    resp = call_local_dataset(p1, p2, date_sel)
                    # API returns matched_row in details
                    matched = resp.get('details', {}).get('matched_row', {})
                    st.write(matched)
                    pred = resp['predicted_winner_flag']
                    proba = resp.get('proba_player1_win')
                    winner_name = matched.get('Player_1') if pred == 1 else matched.get('Player_2')
                    if proba is not None:
                        winner_prob = proba if pred == 1 else 1.0 - proba
                        st.success(f'Predicción: {winner_name} gana; probabilidad = {winner_prob:.3f}')
                    else:
                        st.success(f'Predicción: {winner_name} gana (probabilidad no disponible)')
                    copy_text = f"En el partido entre {matched.get('Player_1')} y {matched.get('Player_2')}, el modelo predice que {winner_name} será el ganador."
                    st.text_area('Texto listo para copiar', value=copy_text, height=80)
                except Exception as e:
                    st.error(f'Error calling API: {e}')
        else:
            st.info('No hay partidos de este par en el dataset')

    else:  # custom
        st.write('Ingresar un partido nuevo')
        # prepare options from preprocessed dataframe
        players = sorted(pd.unique(df[['Player_1', 'Player_2']].values.ravel()))
        surfaces = [c.replace('Surface_', '') for c in df.columns if c.startswith('Surface_')]
        if not surfaces and 'Surface' in df.columns:
            surfaces = sorted(df['Surface'].dropna().unique().tolist())
        rounds = sorted(df['Round'].dropna().unique().tolist()) if 'Round' in df.columns else ['1st Round', '2nd Round', 'Quarterfinals', 'Semifinals', 'The Final']
        series_opts = [c.replace('Series_', '') for c in df.columns if c.startswith('Series_')]
        if not series_opts and 'Series' in df.columns:
            series_opts = sorted(df['Series'].dropna().unique().tolist())
        court_opts = [c.replace('Court_', '') for c in df.columns if c.startswith('Court_')]
        if not court_opts and 'Court' in df.columns:
            court_opts = sorted(df['Court'].dropna().unique().tolist())

        with st.form('custom_form'):
            # Player selectors with option to type custom name
            p1_choice = st.selectbox('Player 1', options=['<type name>'] + players)
            if p1_choice == '<type name>':
                p1 = st.text_input('Player 1 name')
            else:
                p1 = p1_choice

            p2_choice = st.selectbox('Player 2', options=['<type name>'] + players, index=0)
            if p2_choice == '<type name>':
                p2 = st.text_input('Player 2 name')
            else:
                p2 = p2_choice

            date = st.date_input('Date')
            surface = st.selectbox('Surface', options=[''] + surfaces)
            tournament = st.selectbox('Tournament', options=[''] + sorted(df['Tournament'].dropna().unique().tolist())) if 'Tournament' in df.columns else st.text_input('Tournament')
            series = st.selectbox('Series', options=[''] + series_opts) if series_opts else st.text_input('Series')
            court = st.selectbox('Court', options=[''] + court_opts) if court_opts else st.text_input('Court')
            rank1 = st.number_input('Rank 1', value=1000)
            rank2 = st.number_input('Rank 2', value=1000)
            pts1 = st.number_input('Pts 1', value=0)
            pts2 = st.number_input('Pts 2', value=0)
            round_name = st.selectbox('Round', options=rounds)
            bestof = st.selectbox('Best of', [3, 5], index=0)
            submitted = st.form_submit_button('Predecir (custom)')

        if submitted:
            match_info = {
                'Player_1': p1,
                'Player_2': p2,
                'Date': pd.to_datetime(date),
                'Surface': surface if surface != '' else None,
                'Tournament': tournament if tournament != '' else None,
                'Rank_1': rank1,
                'Rank_2': rank2,
                'Pts_1': pts1,
                'Pts_2': pts2,
                'Round': round_name,
                'Best of': bestof,
                'Series': series if series != '' else None,
                'Court': court if court != '' else None,
            }
            try:
                # predict locally
                payload = match_info.copy()
                payload['Date'] = payload['Date'].strftime('%Y-%m-%d')
                resp = call_local_custom(payload)
                st.write('Features usadas:')
                st.json(resp.get('details', {}).get('features', {}))
                pred = resp['predicted_winner_flag']
                proba = resp.get('proba_player1_win')
                winner_name = match_info['Player_1'] if pred == 1 else match_info['Player_2']
                if proba is not None:
                    winner_prob = proba if pred == 1 else 1.0 - proba
                    st.success(f'Predicción: {winner_name} gana; probabilidad = {winner_prob:.3f}')
                else:
                    st.success(f'Predicción: {winner_name} gana (probabilidad no disponible)')
                copy_text = f"{match_info['Player_1']} vs {match_info['Player_2']}, el modelo predice que {winner_name} será el ganador."
                st.text_area('', value=copy_text, height=80)
            except Exception as e:
                st.error(f'Error calling API: {e}')


if __name__ == '__main__':
    main()
