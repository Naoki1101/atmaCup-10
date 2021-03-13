import sys

import numpy as np
import pandas as pd

sys.path.append("../src")
import const


inference_model_name = "lightgbm_012_20210311224100_1.0207"


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train.csv")
    oof = np.load(const.LOG_DIR / inference_model_name / "oof.npy")

    train_df["residuals"] = np.log1p(train_df["likes"]) - np.log1p(oof.reshape(-1))
    train_df[["residuals"]].to_feather(const.FEATURE_DIR / "residuals.feather")


if __name__ == "__main__":
    main()
