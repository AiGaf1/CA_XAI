import numpy as np
import pandas as pd
from pathlib import Path


# Map CMU key names to pseudo-ASCII indices (consistent across all sessions)
def _build_key_map(key_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(sorted(set(key_names)))}


def load_cmu_as_aalto_format(csv_path: Path | str = None) -> dict:
    """
    Convert CMU DSL-StrongPasswordData.csv into the same nested dict format
    used by the Aalto dataset:

        Dict[user_id -> Dict[session_id -> np.ndarray (N, 3)]]

    Each row in the array is one keystroke: [hold_time, flight_time, key_idx]
    Hold and flight times are already in seconds in the CMU dataset.
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / 'raw/DSL-StrongPasswordData.csv'

    df = pd.read_csv(csv_path)

    hold_cols   = [c for c in df.columns if c.startswith("H.")]
    flight_cols = [c for c in df.columns if c.startswith("DD.")]

    key_names = [c.replace("H.", "") for c in hold_cols]
    key_map   = _build_key_map(key_names)

    data: dict = {}

    for _, row in df.iterrows():
        user_id    = row["subject"]
        session_id = (int(row["sessionIndex"]), int(row["rep"]))

        keystrokes = []
        for h_col, f_col, key_name in zip(hold_cols, flight_cols, key_names):
            hold   = float(row[h_col])
            flight = float(row[f_col])
            k_idx  = key_map[key_name]
            keystrokes.append([hold, flight, k_idx])

        session_array = np.array(keystrokes, dtype=np.float32)  # (N, 3)

        if user_id not in data:
            data[user_id] = {}
        data[user_id][session_id] = session_array

    return data


def save_cmu_npy(out_path: Path | str = None) -> None:
    if out_path is None:
        out_path = Path(__file__).parent / 'raw/cmu_dev_set.npy'
    data = load_cmu_as_aalto_format()
    np.save(out_path, data)
    n_users    = len(data)
    n_sessions = sum(len(s) for s in data.values())
    print(f"Saved {n_users} users, {n_sessions} sessions → {out_path}")


if __name__ == "__main__":
    save_cmu_npy()
