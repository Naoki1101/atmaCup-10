import sys

import numpy as np
import pandas as pd

sys.path.append("../src")
import const


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")

    for col in const.ID_COLS + const.TARGET_COLS:
        train_df[[col]].to_feather(const.FEATURE_DIR / f"{col}.feather")

    train_df["likes_bin"] = pd.cut(np.log1p(train_df["likes"]), bins=10, labels=False)
    train_df[["likes_bin"]].to_feather(const.FEATURE_DIR / f"likes_bin.feather")


if __name__ == "__main__":
    main()
