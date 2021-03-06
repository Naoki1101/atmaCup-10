import sys

import pandas as pd
import numpy as np

sys.path.append("../src")
import const


def main():
    log_dir = const.LOG_DIR / "lightgbm_004_20210306203046_1.0272"
    oof = np.load(log_dir / "oof.npy")

    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train.csv")
    train_df["oof"] = oof
    train_df["diff"] = np.log1p(train_df["likes"]) - np.log1p(train_df["oof"])

    outlier_idx = train_df[train_df["diff"] >= 4.5][
        train_df["likes"] <= np.exp(8)
    ].index
    np.save(const.PROCESSED_DATA_DIR / "outlier.npy", outlier_idx)


if __name__ == "__main__":
    main()
