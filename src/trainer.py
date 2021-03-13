import sys
import logging
import dataclasses
from typing import List, Dict

import optuna
import numpy as np
import pandas as pd

sys.path.append("../src")
import const
import factory
from utils import DataHandler


@dataclasses.dataclass
class Trainer:
    cfg: Dict

    def __post_init__(self):
        self.oof = None
        self.raw_preds = None
        self.target_columns = []
        self.weights = []
        self.weights = []
        self.models = []
        self.scores = []
        self.feature_importance_df = pd.DataFrame(columns=["feature", "importance"])
        self.dh = DataHandler()
        self.thresh_hold = 0.5

        if self.cfg.data.features.cat_features is None:
            self.cat_features = []
        else:
            self.cat_features = self.cfg.data.features.cat_features

    def train(
        self, train_df: pd.DataFrame, target_df: pd.DataFrame, fold_df: pd.DataFrame
    ) -> float:
        self.target_columns = target_df.columns.tolist()
        self.oof = np.zeros((len(train_df), len(self.target_columns)))

        target_df[self.target_columns] = self._convert(
            target_df.values, self.cfg.data.target.convert_type
        )
        self._transform_category(train_df)

        for fold_, col in enumerate(fold_df.columns):
            print(
                f"\n========================== FOLD {fold_ + 1} / {len(fold_df.columns)} ... ==========================\n"
            )
            logging.debug(
                f"\n========================== FOLD {fold_ + 1} / {len(fold_df.columns)} ... ==========================\n"
            )

            self._train_fold(
                train_df=train_df, target_df=target_df, fold_df=fold_df, fold_num=fold_
            )

        cv = np.mean(self.scores)

        print("\n\n===================================\n")
        print(f"CV: {cv:.6f}")
        print("\n===================================\n\n")
        logging.debug("\n\n===================================\n")
        logging.debug(f"CV: {cv:.6f}")
        logging.debug("\n===================================\n\n")

        return cv

    def _train_fold(
        self,
        train_df: pd.DataFrame,
        target_df: pd.DataFrame,
        fold_df: pd.DataFrame,
        fold_num: int,
    ) -> None:
        fold = fold_df[fold_df.columns[fold_num]]
        tr_x, va_x = train_df[fold == 0], train_df[fold > 0]
        tr_y, va_y = target_df[fold == 0], target_df[fold > 0]
        weight = fold.max()
        self.weights.append(weight)

        model = factory.get_model(self.cfg.model)
        model.fit(tr_x, tr_y, va_x, va_y, self.cat_features)
        va_pred = model.predict(va_x, self.cat_features)

        va_y = self._reconvert(va_y, self.cfg.data.target.reconvert_type)
        va_pred = self._reconvert(va_pred, self.cfg.data.target.reconvert_type)

        self.models.append(model)
        if self.cfg.data.target.name != "rank_class":
            self.oof[va_x.index, :] = va_pred.copy()
        else:
            self.oof[va_x.index, :] = va_pred.copy()[:, 0]
            va_pred = va_pred[:, 0]

        if va_y is not None:
            score = factory.get_metrics(self.cfg.common.metrics.name)(va_y, va_pred)
            print(f"\n{self.cfg.common.metrics.name}: {score:.6f}\n")
            self.scores.append(score)

        if self.cfg.model.name in ["lightgbm", "catboost", "xgboost"]:
            importance_fold_df = pd.DataFrame()
            fold_importance = model.extract_importances()
            importance_fold_df["feature"] = train_df.columns
            importance_fold_df["importance"] = fold_importance
            self.feature_importance_df = pd.concat(
                [self.feature_importance_df, importance_fold_df], axis=0
            )

    def predict(self, test_df: pd.DataFrame) -> np.array:
        self._transform_category(test_df)

        preds = np.zeros((len(test_df), len(self.target_columns)))
        for fold_, model in enumerate(self.models):
            pred = model.predict(test_df)
            pred = self._reconvert(pred, self.cfg.data.target.reconvert_type)
            preds += pred.copy() * self.weights[fold_]

        self.raw_preds = preds.copy()

        return preds

    def save(self, run_name: str) -> None:
        log_dir = const.LOG_DIR / run_name
        self.dh.save(log_dir / "oof.npy", self.oof)
        self.dh.save(log_dir / "raw_preds.npy", self.raw_preds)
        self.dh.save(log_dir / "importance.csv", self.feature_importance_df)
        self.dh.save(log_dir / "model_weight.pkl", self.models)

    def _transform_category(self, df: pd.DataFrame) -> None:
        if self.cat_features is not None:
            df[self.cat_features] = df[self.cat_features].astype("category")

    def _convert(self, array: np.array, convert_type: str) -> np.array:
        if convert_type is not None:
            converted_array = getattr(np, self.cfg.data.target.convert_type)(array)
        else:
            converted_array = array.copy()
        return converted_array

    def _reconvert(self, array: np.array, reconvert_type: str) -> np.array:
        if reconvert_type is not None:
            converted_array = getattr(np, self.cfg.data.target.reconvert_type)(array)
            converted_array = np.where(converted_array >= 0, converted_array, 0)
        else:
            converted_array = array.copy()
        return converted_array


def opt_ensemble_weight(
    cfg: Dict, y_true: np.array, oof_list: List, metric: str
) -> List[float]:
    def objective(trial):
        p_list = [0 for i in range(len(oof_list))]
        for i in range(len(oof_list) - 1):
            p_list[i] = trial.suggest_discrete_uniform(
                f"p{i}", 0.0, 1.0 - sum(p_list), 0.01
            )
        p_list[-1] = round(1 - sum(p_list[:-1]), 2)

        y_pred = np.zeros(len(y_true))
        for i in range(len(oof_list)):
            y_pred += oof_list[i] * p_list[i]

        return metric(y_true, y_pred)

    study = optuna.create_study(direction=cfg.opt_params.direction)
    if hasattr(cfg.opt_params, "n_trials"):
        study.optimize(objective, n_trials=cfg.opt_params.n_trials)
    elif hasattr(cfg.opt_params, "timeout"):
        study.optimize(objective, timeout=cfg.opt_params.timeout)
    else:
        raise (NotImplementedError)
    best_params = list(study.best_params.values())
    best_weight = best_params + [round(1 - sum(best_params), 2)]

    return best_weight