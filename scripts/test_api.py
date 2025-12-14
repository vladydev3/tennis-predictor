import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.api_main import load_resources, predict_custom_endpoint

class Req:
    Player_1='Novak Djokovic'
    Player_2='Rafael Nadal'
    Date='2023-05-01'
    Surface='Clay'
    Rank_1=1.0
    Rank_2=2.0
    Pts_1=12000.0
    Pts_2=10000.0
    Round='Final'
    Best_of=3
    Series='ATP'
    Court='Outdoor'

if __name__ == '__main__':
    load_resources()
    resp = predict_custom_endpoint(Req())
    print('Prediction custom:', resp.predicted_winner_flag, resp.proba_player1_win)
