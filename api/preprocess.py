import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = (
    [f'V{i}' for i in range(1, 29)] +
    ['Amount_log', 'hour', 'day',
     'is_night', 'is_round_amount', 'is_micro']
)

def preprocess(txn: dict, scaler: StandardScaler) -> np.ndarray:
    amount_log = np.log1p(txn['Amount'])

    hour     = int((txn['Time'] % 86400) // 3600)
    day      = int(txn['Time'] // 86400)
    is_night = int(0 <= hour <= 5)

    is_round  = int(txn['Amount'] % 1 == 0)
    is_micro  = int(txn['Amount'] < 1)

    v_feats = [txn[f'V{i}'] for i in range(1, 29)]
    row = np.array(v_feats + [
        amount_log, hour, day,
        is_night, is_round, is_micro
    ], dtype=float).reshape(1, -1)

    row[0, 28] = scaler.transform([[row[0, 28]]])[0][0]

    return row