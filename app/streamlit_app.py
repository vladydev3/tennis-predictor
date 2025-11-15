import streamlit as st
import sys
from pathlib import Path
import json

# Add the project root to the sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
import pandas as pd
import joblib
from scripts.predict_match import predict_custom, predict_from_dataset


@st.cache_data
def load_data_and_model():
    repo = Path(__file__).resolve().parents[1]
    df_path = repo / 'data' / 'atp_preprocessed.pkl'
    model_path = repo / 'models' / 'rf_model.joblib'
    elo_path = repo / 'data' / 'elo_ratings.json'
    df = pd.read_pickle(df_path)
    model = joblib.load(model_path)
    elo_ratings = None
    if elo_path.exists():
        with open(elo_path, 'r') as f:
            elo_ratings = json.load(f)
    return df, model, elo_ratings


def main():
    st.title('Tennis match predictor')
    st.markdown('Interfaz para predecir el ganador de un partido usando el modelo entrenado')

    df, model, elo_ratings = load_data_and_model()

    mode = st.radio('Modo', ['demo', 'dataset', 'custom'])

    if mode == 'demo':
        st.write('Demo: se selecciona un partido aleatorio del dataset y se predice')
        if st.button('Predecir demo'):
            sample = df.sample(1).iloc[0]
            filt = {'Player_1': sample['Player_1'], 'Player_2': sample['Player_2'], 'Date': sample['Date']}
            pred, proba, row = predict_from_dataset(df, model, filt)
            st.write(row[['Date', 'Tournament', 'Player_1', 'Player_2', 'Winner']])
            # Map prediction (1 => Player_1, 0 => Player_2) to actual player name and probability
            winner_name = row['Player_1'] if pred == 1 else row['Player_2']
            if proba is not None:
                winner_prob = proba if pred == 1 else 1.0 - proba
                st.success(f'Predicción: {winner_name} gana; probabilidad = {winner_prob:.3f}')
            else:
                st.success(f'Predicción: {winner_name} gana (probabilidad no disponible)')
            # Paragraph ready to copy
            copy_text = f"En el partido entre {row['Player_1']} y {row['Player_2']}, el modelo predice que {winner_name} será el ganador."
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
                pred, proba, row = predict_from_dataset(df, model, {'Player_1': p1, 'Player_2': p2, 'Date': pd.to_datetime(date_sel)})
                st.write(row[['Date', 'Tournament', 'Player_1', 'Player_2', 'Winner']])
                winner_name = row['Player_1'] if pred == 1 else row['Player_2']
                if proba is not None:
                    winner_prob = proba if pred == 1 else 1.0 - proba
                    st.success(f'Predicción: {winner_name} gana; probabilidad = {winner_prob:.3f}')
                else:
                    st.success(f'Predicción: {winner_name} gana (probabilidad no disponible)')
                copy_text = f"En el partido entre {row['Player_1']} y {row['Player_2']}, el modelo predice que {winner_name} será el ganador."
                st.text_area('Texto listo para copiar', value=copy_text, height=80)
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
                'Rank_1': rank1,
                'Rank_2': rank2,
                'Pts_1': pts1,
                'Pts_2': pts2,
                'Round': round_name,
                'Best of': bestof,
                'Series': series if series != '' else None,
                'Court': court if court != '' else None,
            }
            pred, proba, X = predict_custom(df, model, match_info, elo_ratings)
            st.write('Features usadas:')
            st.json(X.to_dict(orient='records')[0])
            winner_name = match_info['Player_1'] if pred == 1 else match_info['Player_2']
            if proba is not None:
                winner_prob = proba if pred == 1 else 1.0 - proba
                st.success(f'Predicción: {winner_name} gana; probabilidad = {winner_prob:.3f}')
            else:
                st.success(f'Predicción: {winner_name} gana (probabilidad no disponible)')
            copy_text = f"{match_info['Player_1']} vs {match_info['Player_2']}, el modelo predice que {winner_name} será el ganador."
            st.text_area('', value=copy_text, height=80)


if __name__ == '__main__':
    main()
