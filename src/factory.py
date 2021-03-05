import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("../src")
import const
import models
import metrics
import validation
from utils import reduce_mem_usage, DataHandler

dh = DataHandler()


def get_fold(cfg: Dict, df: pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()

    for col in [cfg.split.y, cfg.split.groups]:
        if col and col not in df_.columns:
            if cfg.name != "MultilabelStratifiedKFold":
                feat = dh.load(const.FEATURE_DIR / f"{col}.feather")
                df_[col] = feat[col]

            elif cfg.name == "MultilabelStratifiedKFold":
                col = getattr(const, col)
                for c in col:
                    feat = dh.load(const.FEATURE_DIR / f"{c}.feather")
                    df_[c] = feat[c]

    fold_df = pd.DataFrame(index=range(len(df_)))

    weight_list = get_fold_weights(cfg.params.n_splits, cfg.weight)

    fold = getattr(validation, cfg.name)(cfg)
    for fold_, (trn_idx, val_idx) in enumerate(fold.split(df_)):
        fold_df[f"fold_{fold_}"] = 0
        fold_df.loc[val_idx, f"fold_{fold_}"] = weight_list[fold_]
        if cfg.name == "GroupTimeSeriesKFold":
            fold_df.loc[val_idx[-1] + 1 :, f"fold_{fold_}"] = -1

    return fold_df


def get_fold_weights(n_splits: int, weight_type: str) -> List[float]:
    if weight_type == "average":
        weight_list = [1 / n_splits for i in range(n_splits)]

    elif weight_type == "accum_weight":
        sum_ = sum([i + 1 for i in range(n_splits)])
        weight_list = [(i + 1) / sum_ for i in range(n_splits)]

    assert len(weight_list) == n_splits

    return weight_list


def get_model(cfg: Dict):
    model = getattr(models, cfg.name)(cfg=cfg)
    return model


def get_metrics(cfg: Dict):
    evaluator = getattr(metrics, cfg)
    return evaluator


def fill_dropped(dropped_array: np.array, drop_idx: np.array) -> np.array:
    filled_array = np.zeros(len(dropped_array) + len(drop_idx))
    idx_array = np.arange(len(filled_array))
    use_idx = np.delete(idx_array, drop_idx)
    filled_array[use_idx] = dropped_array
    return filled_array


def get_features(features: List[str], cfg: Dict) -> pd.DataFrame:
    dfs = []
    for f in features:
        f_path = Path(const.FEATURE_DIR / f"{f}_{cfg.data_type}.feather")
        log_dir = Path(f"../logs/{f}")

        if f_path.exists():
            feat = dh.load(f_path)

        elif log_dir.exists():
            if cfg.data_type == "train":
                feat = dh.load(log_dir / "oof.npy")
                model_cfg = dh.load(log_dir / "config.yml")

                if model_cfg.common.drop:
                    drop_name_list = []
                    for drop_name in model_cfg.common.drop:
                        drop_name_list.append(drop_name)

                    drop_idxs = get_drop_idx(drop_name_list)
                    feat = fill_dropped(feat, drop_idxs)

            elif cfg.data_type == "test":
                feat = dh.load(log_dir / "raw_preds.npy")

            feat = pd.DataFrame(feat, columns=[f])

        dfs.append(feat)

    df = pd.concat(dfs, axis=1)
    if cfg.reduce:
        df = reduce_mem_usage(df)
    return df


def get_result(log_name: str, cfg: Dict) -> Tuple[np.array, np.array]:
    log_dir = Path(f"../logs/{log_name}")

    model_oof = dh.load(log_dir / "oof.npy")
    model_cfg = dh.load(log_dir / "config.yml")

    if model_cfg.common.drop:
        drop_name_list = []
        for drop_name in model_cfg.common.drop:
            drop_name_list.append(drop_name)

        drop_idxs = get_drop_idx(drop_name_list)
        model_oof = fill_dropped(model_oof, drop_idxs)

    model_preds = dh.load(log_dir / "raw_preds.npy")

    return model_oof, model_preds


def get_target(cfg: Dict) -> pd.DataFrame:
    target = pd.read_feather(const.FEATURE_DIR / f"{cfg.name}.feather")
    return target


def get_drop_idx(cfg: Dict) -> np.array:
    drop_idx_list = []
    for drop_name in cfg:
        drop_idx = np.load(const.PROCESSED_DATA_DIR / f"{drop_name}.npy")
        drop_idx_list.append(drop_idx)
    all_drop_idx = np.unique(np.concatenate(drop_idx_list))
    return all_drop_idx


def get_ad(
    cfg: Dict, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    whole_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    target = np.concatenate([np.zeros(len(train_df)), np.ones(len(test_df))])
    target_df = pd.DataFrame({f"{cfg.data.target.name}": target.astype(int)})

    fold_df = get_fold(cfg.validation, whole_df)
    if cfg.validation.single:
        col = fold_df.columns[-1]
        fold_df = fold_df[[col]]
        fold_df /= fold_df[col].max()

    return whole_df, target_df, fold_df


def get_lgb_objective(cfg: Dict):
    if cfg:
        obj = lambda x, y: getattr(loss, cfg.name)(x, y, **cfg.params)
    else:
        obj = None
    return obj


def get_lgb_feval(cfg: Dict):
    if cfg:
        feval = lambda x, y: getattr(metrics, cfg.name)(x, y, **cfg.params)
    else:
        feval = None
    return feval