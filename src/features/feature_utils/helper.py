import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("../src")
import const
from utils import DataHandler

dh = DataHandler()


def save_features(df: pd.DataFrame, data_type: str = "train") -> None:
    save_path = const.FEATURE_CUSTOM_DATASET_DIR / "all.yml"

    if not save_path.exists():
        save_path.touch()
        feature_dict = {"features": []}
    else:
        feature_dict = dh.load(save_path)

    new_feature = sorted(set(feature_dict["features"] + df.columns.tolist()))
    feature_dict["features"] = new_feature
    dh.save(save_path, feature_dict)

    for col in df.columns:
        df[[col]].reset_index(drop=True).to_feather(
            const.FEATURE_DIR / f"{col}_{data_type}.feather"
        )
