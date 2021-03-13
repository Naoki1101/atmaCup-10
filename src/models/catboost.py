import shutil
import logging
import dataclasses
from typing import List, Dict, Optional
from abc import ABCMeta, abstractmethod

from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from pathlib import Path
import pandas as pd

from .base import Model


@dataclasses.dataclass
class _BaseCB(Model, metaclass=ABCMeta):
    cfg: Dict

    def fit(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.Series,
        va_x: Optional[pd.DataFrame] = None,
        va_y: Optional[pd.Series] = None,
        cat_features: List = [],
        feval=None,
    ) -> None:
        if cat_features is not None:
            tr_x[cat_features] = tr_x[cat_features].astype("category")
            va_x[cat_features] = va_x[cat_features].astype("category")

        validation = va_x is not None
        cb_train = Pool(tr_x, label=tr_y, cat_features=cat_features)
        if validation:
            cb_valid = Pool(va_x, label=va_y, cat_features=cat_features)

        self.model = self.cb.fit(
            cb_train,
            eval_set=cb_valid,
            use_best_model=True,
            verbose_eval=self.cfg.verbose_eval,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            plot=False,
        )

        self._log_evaluation(period=self.cfg.verbose_eval)

    @abstractmethod
    def predict(self, te_x: pd.DataFrame, cat_features: List = []):
        pass

    def extract_importances(self):
        return self.model.feature_importances_

    def _log_evaluation(self, period: int = 1, level=logging.DEBUG) -> None:
        info_dir = Path("./catboost_info")
        learn_df = pd.read_csv(info_dir / "learn_error.tsv", sep="\t")
        test_df = pd.read_csv(info_dir / "learn_error.tsv", sep="\t")

        logger = logging.getLogger("main")
        for iteration in range(len(learn_df)):
            if period > 0 and learn_df is not None and (iteration + 1) % period == 0:
                metrics = self.cfg.params.eval_metric
                train_loss = learn_df.iloc[iteration, 1]
                val_loss = test_df.iloc[iteration, 1]
                result = f"train-{metrics}: {train_loss:.6f}\ttvalid-{metrics}: {val_loss:.6f}"
                logger.log(level, "[{}]\t{}".format(iteration + 1, result))

        shutil.rmtree(info_dir)


@dataclasses.dataclass
class CBBinaryClassifier(_BaseCB):
    cfg: Dict

    def __post_init__(self):
        self.cb = CatBoostClassifier(**self.cfg.params)

    def predict(self, te_x: pd.DataFrame, cat_features: List = []):
        # if cat_features is not None:
        #     te_x[cat_features] = te_x[cat_features].astype("category")
        predict_array = self.model.predict_proba(te_x)[:, 1]
        return predict_array


@dataclasses.dataclass
class CBClassifier(_BaseCB):
    cfg: Dict

    def __post_init__(self):
        self.cb = CatBoostClassifier(**self.cfg.params)

    def predict(self, te_x: pd.DataFrame, cat_features: List = []):
        # if cat_features is not None:
        #     te_x[cat_features] = te_x[cat_features].astype("category")
        predict_array = self.model.predict_proba(te_x)
        return predict_array


@dataclasses.dataclass
class CBRegressor(_BaseCB):
    cfg: Dict

    def __post_init__(self):
        self.cb = CatBoostRegressor(**self.cfg.params)

    def predict(self, te_x: pd.DataFrame, cat_features: List = []):
        # if cat_features is not None:
        #     te_x[cat_features] = te_x[cat_features].astype("category")
        predict_array = self.model.predict(te_x).reshape(-1, 1)
        return predict_array


def catboost(cfg: Dict):
    if cfg.task_type == "binary":
        return CBBinaryClassifier(cfg)
    elif cfg.task_type == "classification":
        return CBClassifier(cfg)
    elif cfg.task_type == "regression":
        return CBRegressor(cfg)
    else:
        raise (NotImplementedError)
