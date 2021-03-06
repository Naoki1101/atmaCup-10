import sys

import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.append("../src")
import const
from factory import get_fold
from function import TargetEncoder
from feature_utils import save_features

cfg = edict(
    {
        "name": "KFold",
        "params": {"n_splits": 5, "shuffle": True, "random_state": 0},
        "split": {"y": "likes", "groups": None},
        "weight": "average",
    }
)


def get_features(train, test):
    train_features_df = pd.DataFrame()
    test_features_df = pd.DataFrame()

    fold_df = get_fold(cfg, train)

    for col in tqdm(const.CATEGORICAL_FEATURES):
        te = TargetEncoder(fold_df)
        train_features_df[f"te_{col}_by_likes"] = te.fit_transform(
            train[col], train["likes"]
        )
        test_features_df[f"te_{col}_by_likes"] = te.transform(test[col])

    return train_features_df, test_features_df


def main():
    train_df = pd.read_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df = pd.read_feather(const.INPUT_DATA_DIR / "test.feather")

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type="train")
    save_features(test_features_df, data_type="test")


if __name__ == "__main__":
    main()
