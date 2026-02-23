from pathlib import Path
import pandas as pd


def load_data(filename: str) -> pd.DataFrame:
    data_path = Path("data/raw") / filename