import argparse
import datetime
import logging
import warnings

import pandas as pd
from pathlib import Path

import const
import factory
from trainer import Trainer
from utils import DataHandler, Notificator, Timer, seed_everything, Git

warnings.filterwarnings("ignore")

# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument("--common", default="../configs/common/compe.yml")
parser.add_argument("--notify", default="../configs/common/notify.yml")
parser.add_argument("-b", "--debug", action="store_true")
parser.add_argument("-m", "--model")
parser.add_argument("-c", "--comment")
options = parser.parse_args()

dh = DataHandler()
cfg = dh.load(f"../configs/exp/{options.model}.yml")

notify_params = dh.load(options.notify)

features_params = dh.load(f"../configs/feature/{cfg.data.features.name}.yml")
features = features_params.features

comment = options.comment
model_name = options.model
now = datetime.datetime.now()
if cfg.model.task_type != "optuna":
    run_name = f"{model_name}_{now:%Y%m%d%H%M%S}"
else:
    run_name = f"{model_name}_optuna_{now:%Y%m%d%H%M%S}"

logger_path = Path(f"../logs/{run_name}")
debug = options.debug


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.common.seed)

    logger_path.mkdir(exist_ok=True)
    logging.basicConfig(filename=logger_path / "train.log", level=logging.DEBUG)

    dh.save(logger_path / "config.yml", cfg)
    dh.save(logger_path / "features.yml", features_params)

    with t.timer("load data"):
        train_x = factory.get_features(features, cfg.data.loader.train)
        test_x = factory.get_features(features, cfg.data.loader.test)
        train_y = factory.get_target(cfg.data.target)

        if debug:
            train_x = train_x.iloc[: int(len(train_x) * 0.1)]
            test_x = train_x.iloc[: int(len(test_x) * 0.1)]
            train_y = train_y.iloc[: int(len(train_x) * 0.1)]

    with t.timer("preprocess"):
        train_null_count = train_x.isnull().values.sum(axis=1)
        test_null_count = test_x.isnull().values.sum(axis=1)

        train_x["feature_null_count"] = train_null_count
        test_x["feature_null_count"] = test_null_count

        features.append("feature_null_count")

    with t.timer("add oof"):
        if cfg.data.features.oof.name is not None:
            oof, preds = factory.get_result(cfg.data.features.oof.name, cfg)

            train_x["oof"] = oof
            test_x["oof"] = preds
            features.append("oof")

    with t.timer("make folds"):
        fold_df = factory.get_fold(cfg.validation, train_x)
        if cfg.validation.single:
            col = fold_df.columns[-1]
            fold_df = fold_df[[col]]
            fold_df /= fold_df[col].max()

    with t.timer("drop index"):
        if cfg.common.drop is not None:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            train_x = train_x.drop(drop_idx, axis=0).reset_index(drop=True)
            train_y = train_y.drop(drop_idx, axis=0).reset_index(drop=True)
            fold_df = fold_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer("prepare for ad"):
        if cfg.data.adversarial_validation:
            train_x, train_y, fold_df = factory.get_ad(cfg, train_x, test_x)

    with t.timer("train and predict"):
        trainer = Trainer(cfg)
        cv = trainer.train(train_df=train_x, target_df=train_y, fold_df=fold_df)
        preds = trainer.predict(test_x)
        trainer.save(run_name)

        run_name_cv = f"{run_name}_{cv:.4f}"
        logger_path.rename(f"../logs/{run_name_cv}")
        logging.disable(logging.FATAL)

    with t.timer("make submission"):
        sub_df = pd.read_csv(const.SAMPLE_SUB_PATH)
        if debug:
            sub_df = sub_df.iloc[: int(len(test_x) * 0.1)]

        sub_df[const.TARGET_COLS[0]] = preds
        sub_df.to_csv(const.OUTPUT_DATA_DIR / f"{run_name_cv}.csv", index=False)

    with t.timer("notify"):
        process_minutes = t.get_processing_time()
        notificator = Notificator(
            run_name=run_name_cv,
            model_name=cfg.model.name,
            cv=round(cv, 4),
            process_time=round(process_minutes, 2),
            comment=comment,
            params=notify_params,
        )
        notificator.send_line()
        notificator.send_notion()

    with t.timer("git"):
        git = Git()
        git.push(comment=run_name_cv)


if __name__ == "__main__":
    main()
